"""Pipeline temps réel : acquisition -> YOLOv8 + ByteTrack -> Counting Zone -> rendu."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from .capture import FrameGrabber
from .config import Config, ID_TO_NAME
from .counting_zone import CountEvent, CountingZone, ZoneCounter
from .detector import VehicleDetector
from .motion_fallback import GhostDetector
from .visualizer import Visualizer


class VehicleCountingPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector: Optional[VehicleDetector] = None
        self.counter: Optional[ZoneCounter] = None
        self.ghost: Optional[GhostDetector] = None
        self.vis: Optional[Visualizer] = None
        self.zone: Optional[CountingZone] = None
        self.writer = None
        self.fps_hist: Deque[float] = deque(maxlen=30)
        self.avg_fps = 0.0
        self.frame_id = 0
        self._tiling_disabled = False
        self.trails: Dict[int, List[tuple]] = {}
        self._refine_id = 2_000_000_000  # préfixe pour les pistes issues de tuiles

    # ------------------------------------------------------------------
    def build(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.zone = CountingZone.from_config(self.cfg.zone, w, h)
        self.detector = VehicleDetector(self.cfg.detector, self.cfg.tracker)
        self.counter = ZoneCounter(
            self.zone, self.cfg.tracker, self.cfg.motorcycle, self.cfg.fast, w * h
        )
        self.ghost = GhostDetector(self.cfg.ghost, self.zone, w, h)
        self.vis = Visualizer(self.cfg.output, self.zone)
        print(f"[init] source {w}x{h} @ {self.cfg.source}")
        print(f"[init] zone : {self.zone.top:.0f} -> {self.zone.bottom:.0f} "
              f"({self.zone.height:.0f} px, axe {self.zone.axis})")
        print(f"[init] device : {self.detector.device}   tiling : {self.cfg.detector.tiled}")

    # ------------------------------------------------------------------
    def _open_writer(self, w: int, h: int) -> None:
        path = self.cfg.output.save_video
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, self.cfg.tracker.frame_rate, (w, h))
        if not self.writer.isOpened():
            print("[warn] writer indisponible")
            self.writer = None

    # ------------------------------------------------------------------
    def _tile_refine(self, frame: np.ndarray, primary: List[dict]) -> List[dict]:
        """Si tiling activé, raffine les pistes incertaines près de la zone."""
        if self.cfg.detector.tiled == "off" or self._tiling_disabled:
            return []
        if self.frame_id % max(1, self.cfg.detector.refine_interval) != 0:
            return []
        refined: List[dict] = []
        h, w = frame.shape[:2]
        for t in primary:
            if t["conf"] >= 0.30:
                continue
            box = t["box"]
            m = 60
            roi = (max(0, int(box[0]) - m), max(0, int(box[1]) - m),
                   min(w, int(box[2]) + m), min(h, int(box[3]) + m))
            extra = self.detector.tile_refine(frame, roi)
            if len(extra) == 0:
                continue
            cx, cy = t["center"]
            best, bd = None, 1e9
            for d in extra:
                d_cx, d_cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                dist = (d_cx - cx) ** 2 + (d_cy - cy) ** 2
                if dist < bd:
                    bd = dist
                    best = d
            if best is None:
                continue
            self._refine_id += 1
            bx = np.asarray(best[:4], dtype=float)
            refined.append({
                "id": self._refine_id,
                "box": bx,
                "cls": ID_TO_NAME.get(int(best[5]), "car"),
                "conf": float(best[4]),
                "center": ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0),
                "refined": True,
            })
        return refined

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> List[dict]:
        primary = self.detector.detect_and_track(frame)
        refined = self._tile_refine(frame, primary)
        tracks = primary + refined

        events = self.counter.update(tracks, self.frame_id)

        boxes = np.array([t["box"] for t in tracks], dtype=float).reshape(-1, 4) if tracks \
            else np.zeros((0, 4), dtype=float)
        ghost_vis = self.ghost.update(frame, boxes, self.frame_id, self.counter)

        for t in tracks:
            st = self.counter.states.get(t["id"])
            if st is not None and len(st.points) > 1:
                self.trails[t["id"]] = list(st.points)[-self.cfg.output.track_len:]
        for tid in list(self.trails.keys()):
            if tid not in {t["id"] for t in tracks}:
                self.trails.pop(tid, None)

        self.vis.draw(frame, tracks, self.counter, self.avg_fps, self.frame_id,
                      ghosts=ghost_vis, trails=self.trails if self.cfg.output.show_tracks else None)
        return tracks

    # ------------------------------------------------------------------
    def _adaptive(self) -> None:
        if not self.cfg.detector.adaptive or self._tiling_disabled:
            return
        if len(self.fps_hist) < 20:
            return
        if float(np.mean(self.fps_hist)) < self.cfg.detector.min_fps and self.cfg.detector.tiled != "off":
            self.cfg.detector.tiled = "off"
            self._tiling_disabled = True
            print(f"[adapt] FPS < {self.cfg.detector.min_fps} -> inférence en tuiles désactivée")

    # ------------------------------------------------------------------
    def run(self, max_frames: int = 0) -> dict:
        grabber = FrameGrabber(self.cfg.source)
        grabber.start()
        ok, frame = grabber.read(timeout=5.0)
        if not ok or frame is None:
            grabber.stop()
            raise RuntimeError("Aucune frame lue depuis la source.")
        if self.cfg.output.window_scale != 1.0:
            s = self.cfg.output.window_scale
            frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
        self.build(frame)
        self._open_writer(frame.shape[1], frame.shape[0])
        if self.cfg.tracker.frame_rate <= 0:
            self.cfg.tracker.frame_rate = int(grabber.fps)

        win = "Vehicle Counting - YOLOv8 + ByteTrack"
        t_prev = time.perf_counter()
        processed = 0

        try:
            while True:
                ok, frame = grabber.read(timeout=5.0)
                if not ok or frame is None:
                    break
                if self.cfg.output.window_scale != 1.0:
                    s = self.cfg.output.window_scale
                    frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))

                self.frame_id += 1
                self.process(frame)

                if self.writer is not None:
                    self.writer.write(frame)

                if self.cfg.output.display and (self.frame_id % max(1, self.cfg.output.every) == 0):
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                now = time.perf_counter()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    inst = 1.0 / dt
                    self.fps_hist.append(inst)
                    self.avg_fps = float(np.mean(self.fps_hist)) if self.fps_hist else inst
                processed += 1
                self._adaptive()
                if max_frames and processed >= max_frames:
                    break
        finally:
            grabber.stop()
            if self.writer is not None:
                self.writer.release()
            if self.cfg.output.display:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

        return self.finalize(processed, grabber)

    # ------------------------------------------------------------------
    def finalize(self, processed: int, grabber: FrameGrabber) -> dict:
        summary = self.counter.summary()
        summary.update({
            "frames": processed,
            "fps_mean": round(float(np.mean(self.fps_hist)), 2) if self.fps_hist else 0.0,
            "source": str(self.cfg.source),
            "zone": {"top": self.zone.top, "bottom": self.zone.bottom, "axis": self.zone.axis},
            "events": [
                {
                    "track_id": e.track_id, "class": e.cls, "direction": e.direction,
                    "frame": e.frame, "method": e.method, "conf": round(e.conf, 3),
                }
                for e in self.counter.events
            ],
        })

        if self.cfg.output.csv:
            os.makedirs(os.path.dirname(os.path.abspath(self.cfg.output.csv)), exist_ok=True)
            with open(self.cfg.output.csv, "w", newline="", encoding="utf-8") as fh:
                wr = csv.writer(fh)
                wr.writerow(["track_id", "class", "direction", "frame", "method", "conf"])
                for e in self.counter.events:
                    wr.writerow([e.track_id, e.cls, e.direction, e.frame, e.method, round(e.conf, 3)])
        if self.cfg.output.json:
            os.makedirs(os.path.dirname(os.path.abspath(self.cfg.output.json)), exist_ok=True)
            with open(self.cfg.output.json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        return summary