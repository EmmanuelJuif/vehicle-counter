from __future__ import annotations

import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameGrabber:
    def __init__(self, source, live: Optional[bool] = None, maxlen: int = 32,
                 reconnect: bool = True, loop: bool = False, fps: float = 0.0,
                 reopen_after: int = 10, reopen_delay: float = 1.0):
        self.source = source
        self.is_live = self._guess_live(source) if live is None else live
        self.maxlen = 1 if self.is_live else maxlen
        self.q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max(1, self.maxlen))
        self.reconnect = reconnect
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = 25.0
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.drops = 0
        self.read_index = 0
        # reconnexion : seuil d'échecs avant réouverture + temporisation
        self.reopen_after = reopen_after
        self.reopen_delay = reopen_delay
        self.reconnects = 0
        self._open(fps)

    # ------------------------------------------------------------------
    @staticmethod
    def _guess_live(source) -> bool:
        if isinstance(source, int):
            return True
        s = str(source)
        return s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"))

    def _open(self, fps: float = 0.0) -> None:
        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if self.is_live else cv2.VideoCapture(src)
        if cap is None or not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {self.source}")
        self.cap = cap
        if self.is_live:
            # évite que `open()` bloque indéfiniment si le serveur ne répond pas
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            except Exception:
                pass
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        f = self.cap.get(cv2.CAP_PROP_FPS)
        f = float(f) if f and f > 1 and not np.isnan(f) else 0.0
        self.fps = fps or f or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # ------------------------------------------------------------------
    def start(self) -> "FrameGrabber":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _push(self, frame) -> None:
        if self.is_live:
            # on jette ce qui n'a pas été consommé : seule la dernière frame compte
            while True:
                try:
                    self.q.get_nowait()
                    self.drops += 1
                except queue.Empty:
                    break
            try:
                self.q.put_nowait(frame)
            except queue.Full:  # pragma: no cover
                pass
        else:
            # fichier : on bloque plutôt que de perdre une frame
            while not self._stop.is_set():
                try:
                    self.q.put(frame, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _reopen(self) -> None:
        """Réouverture de la source (flux live interrompu)."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        time.sleep(self.reopen_delay)
        try:
            self._open(self.fps)
            self.reconnects += 1
        except Exception:
            self.cap = None

    def _loop(self) -> None:
        consecutive_fail = 0
        while not self._stop.is_set():
            ok, frame = (self.cap.read() if self.cap is not None else (False, None))
            if ok and frame is not None:
                consecutive_fail = 0
                self.read_index += 1
                self._push(frame)
                continue

            # --- lecture en échec -------------------------------------
            if not self.is_live:
                if self.loop and self.cap is not None and self.frame_count > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.read_index = 0
                    continue
                break  # fin de fichier

            if not self.reconnect:
                break

            # Flux live : un échec est transitoire par nature (perte de
            # paquet, caméra redémarrée, serveur qui recadre le chemin...).
            # On ne sort JAMAIS de la boucle pour un simple échec de lecture,
            # sinon le sentinelle de fin tue le grabber définitivement.
            consecutive_fail += 1
            if consecutive_fail >= self.reopen_after:
                consecutive_fail = 0
                self._reopen()
            else:
                time.sleep(0.02)
        self._signal_end()

    def _signal_end(self) -> None:
        
        if self.is_live:
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            try:
                self.q.put_nowait(None)
            except queue.Full:  # pragma: no cover
                pass
            return

        while not self._stop.is_set():
            try:
                self.q.put(None, timeout=0.5)
                return
            except queue.Full:
                continue

    def read(self, timeout: float = 5.0) -> Tuple[bool, Optional[np.ndarray]]:
        if self._thread is None:
            return (self.cap.read() if self.cap is not None else (False, None))
        try:
            frame = self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None
        if frame is None:
            return False, None
        return True, frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
