from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from .config import ID_TO_NAME, DetectorConfig, TrackerConfig, write_tracker_yaml
from .geometry import nms


class VehicleDetector:

    def __init__(self, cfg: DetectorConfig, tracker: "TrackerConfig | None" = None):
        from ultralytics import YOLO

        self.cfg = cfg
        self.tcfg = tracker or TrackerConfig()
        self.model = YOLO(cfg.weights)
        self.device = self._resolve_device(cfg.device)
        try:
            self.model.to(self.device)
        except Exception:  # pragma: no cover
            pass
        self.use_half = cfg.half and str(self.device).startswith("cuda")
        self.names = self.model.names
        self._frame_idx = 0
        self.last_tile_count = 0
        self._tracker_yaml: Optional[str] = None   # généré à la 1re frame
        self._warmup()

    # ------------------------------------------------------------------
    def _resolve_device(self, device: str) -> str:
        if device and device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _warmup(self) -> None:
        img = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(img, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                               iou=self.cfg.iou, classes=list(self.cfg.classes),
                               device=self.device, half=self.use_half, verbose=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _predict_raw(self, image: np.ndarray, conf: Optional[float] = None,
                     imgsz: Optional[int] = None, max_det: Optional[int] = None):
        return self.model.predict(
            image,
            imgsz=imgsz or self.cfg.imgsz,
            conf=self.cfg.conf if conf is None else conf,
            iou=self.cfg.iou,
            max_det=max_det or self.cfg.max_det,
            classes=list(self.cfg.classes),
            device=self.device,
            half=self.use_half,
            agnostic_nms=self.cfg.agnostic_nms,
            verbose=self.cfg.verbose,
        )

    @staticmethod
    def _to_array(results) -> np.ndarray:
        if results is None or len(results) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        data = boxes.data.detach().cpu().numpy().astype(np.float32)
        return data[:, :6]

    # ------------------------------------------------------------------
    def detect_and_track(self, frame: np.ndarray) -> List[dict]:
        """``model.track`` : détecte + suivi multi-objets en une seule passe.

        Le backend est choisi par ``TrackerConfig.name`` (``bytetrack`` ou
        ``botsort``). Retourne ``[{id, box, cls, conf, center}]`` ; les ``id``
        sont stables entre frames tant que le suiveur conserve la piste.
        """
        if self._tracker_yaml is None:
            self._tracker_yaml = write_tracker_yaml(self.tcfg)
        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_yaml,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            classes=list(self.cfg.classes),
            device=self.device,
            half=self.use_half,
            agnostic_nms=self.cfg.agnostic_nms,
            verbose=self.cfg.verbose,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.boxes.id is None:
            return []
        data = r.boxes.data.detach().cpu().numpy()
        ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.zeros(len(data), dtype=int)
        out: List[dict] = []
        for i, row in enumerate(data):
            # Colonnes Ultralytics : x1, y1, x2, y2, track_id, conf, cls
            box = np.asarray(row[:4], dtype=float)
            conf = float(row[5])
            cid = int(row[6])
            out.append({
                "id": int(ids[i]),
                "box": box,
                "cls": ID_TO_NAME.get(cid, "car"),
                "conf": conf,
                "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
            })
        return out

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Détection pleine image + tuiles (rattrapage petits objets).

        Retourne ``(N, 6)`` : ``x1, y1, x2, y2, conf, cls``.
        """
        h, w = frame.shape[:2]
        dets = self._to_array(self._predict_raw(frame))

        tiled = self.cfg.tiled
        if tiled != "off" and self._frame_idx % max(1, self.cfg.refine_interval) == 0:
            roi = region if tiled == "band" else None
            tile_dets = self._tiled(frame, roi)
            if len(tile_dets):
                dets = np.concatenate([dets, tile_dets], axis=0) if len(dets) else tile_dets
                dets = nms(dets, self.cfg.merge_iou)
        self._frame_idx += 1
        if len(dets):
            dets[:, 0] = np.clip(dets[:, 0], 0, w - 1)
            dets[:, 2] = np.clip(dets[:, 2], 0, w - 1)
            dets[:, 1] = np.clip(dets[:, 1], 0, h - 1)
            dets[:, 3] = np.clip(dets[:, 3], 0, h - 1)
        return dets

    # ------------------------------------------------------------------
    def tile_refine(self, frame: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Inférence en tuiles limitée à une région (approche = zoom + rappel)."""
        return self._tiled(frame, region)

    def _tiles(self, w: int, h: int, region) -> List[Tuple[int, int, int, int]]:
        size = int(self.cfg.tile_size)
        stride = max(64, int(size * (1.0 - self.cfg.tile_overlap)))
        if region is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
            x1, y1 = min(w, int(region[2])), min(h, int(region[3]))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return []
        xs = list(range(x0, max(x0 + 1, x1 - size + 1), stride))
        ys = list(range(y0, max(y0 + 1, y1 - size + 1), stride))
        if xs[-1] + size < x1:
            xs.append(max(x0, x1 - size))
        if ys[-1] + size < y1:
            ys.append(max(y0, y1 - size))
        xs = sorted(set(int(x) for x in xs))
        ys = sorted(set(int(y) for y in ys))
        return [(x, y, min(w, x + size), min(h, y + size)) for y in ys for x in xs]

    def _tiled(self, frame: np.ndarray, region) -> np.ndarray:
        h, w = frame.shape[:2]
        tiles = self._tiles(w, h, region)
        self.last_tile_count = len(tiles)
        if not tiles:
            return np.zeros((0, 6), dtype=np.float32)
        out: List[np.ndarray] = []
        for (tx0, ty0, tx1, ty1) in tiles:
            crop = frame[ty0:ty1, tx0:tx1]
            if crop.size == 0:
                continue
            d = self._to_array(self._predict_raw(crop, conf=self.cfg.zoom_conf))
            if len(d) == 0:
                continue
            d[:, [0, 2]] += tx0
            d[:, [1, 3]] += ty0
            out.append(d)
        if not out:
            return np.zeros((0, 6), dtype=np.float32)
        merged = np.concatenate(out, axis=0)
        return nms(merged, self.cfg.merge_iou)

    # ------------------------------------------------------------------
    def benchmark(self, frame: np.ndarray, n: int = 5) -> dict:
        t0 = time.perf_counter()
        for _ in range(n):
            self._predict_raw(frame)
        full = (time.perf_counter() - t0) / n
        tiled_ms = float("nan")
        if self.cfg.tiled != "off":
            h, w = frame.shape[:2]
            tiles = self._tiles(w, h, None)
            if tiles:
                t0 = time.perf_counter()
                for _ in range(max(1, n // 2)):
                    for (a, b, c, d_) in tiles[:4]:
                        self._predict_raw(frame[b:d_, a:c], conf=self.cfg.zoom_conf)
                tiled_ms = (time.perf_counter() - t0) / (max(1, n // 2) * min(4, len(tiles))) * 1000
        return {"full_frame_ms": full * 1000, "tile_ms": tiled_ms, "tiles": self.last_tile_count}