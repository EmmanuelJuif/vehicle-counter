from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import TrackerConfig, tracker_args
from .geometry import iou_matrix


# ---------------------------------------------------------------------------
# Adaptateur entre les détections ndarray et l'API Results attendue par
# `ultralytics.trackers.byte_tracker.BYTETracker.update(...)`.
# ---------------------------------------------------------------------------
class _DetResults:
    __slots__ = ("xywh", "conf", "cls", "_xyxy")

    def __init__(self, dets: np.ndarray):
        if dets is None or len(dets) == 0:
            self._xyxy = np.zeros((0, 4), dtype=np.float32)
            self.xywh = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
            self.cls = np.zeros((0,), dtype=np.float32)
            return
        d = np.asarray(dets, dtype=np.float32)
        self._xyxy = d[:, :4].astype(np.float32).copy()
        x1, y1, x2, y2 = self._xyxy.T
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.xywh = np.stack([cx, cy, x2 - x1, y2 - y1], axis=1).astype(np.float32)
        self.conf = d[:, 4].astype(np.float32)
        self.cls = d[:, 5].astype(np.float32)

    def __len__(self) -> int:
        return len(self.xywh)

    def __getitem__(self, idx):
        m = np.asarray(idx, dtype=bool)
        n = len(self.xywh)
        # Reconstruction à partir des coordonnées originales, pas des xywh recalculés.
        sub = np.concatenate(
            [self._xyxy[m], self.conf[m, None], self.cls[m, None]], axis=1
        )
        return _DetResults(sub)


# ---------------------------------------------------------------------------
# Suiveur IoU de secours (deux passes : haute puis basse confiance).
# ---------------------------------------------------------------------------
class FallbackTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: dict[int, dict] = {}
        self.next_id = 1
        self.max_missed = cfg.track_buffer

    def update(self, dets: np.ndarray, frame=None) -> List[dict]:
        dets = np.asarray(dets, dtype=np.float32)
        if len(dets) == 0:
            for t in self.tracks.values():
                t["missed"] += 1
            self.tracks = {k: v for k, v in self.tracks.items() if v["missed"] <= self.max_missed}
            return self._active()

        for t in self.tracks.values():
            t["pred"][:4] = t["box"] + np.array([t["v"][0], t["v"][1], t["v"][0], t["v"][1]])
            c = t["pred"][:4]
            t["center"] = ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)

        active = {k: v for k, v in self.tracks.items() if v["missed"] <= self.max_missed}
        out: List[dict] = []
        if active:
            ids = list(active.keys())
            preds = np.array([active[i]["pred"][:4] for i in ids], dtype=np.float32)
            iou = iou_matrix(preds, dets[:, :4])
            used_d, used_t = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[0]
            for ti, di in order:
                ti, di = int(ti), int(di)
                if ti in used_t or di in used_d or iou[ti, di] < 1 - self.cfg.match_thresh:
                    continue
                used_t.add(ti)
                used_d.add(di)
                self._assign(ids[ti], dets[di])
            for di in range(len(dets)):
                if di in used_d:
                    continue
                if dets[di][4] < self.cfg.new_track_thresh:
                    continue
                self._create(dets[di])
        else:
            for d in dets:
                if d[4] >= self.cfg.new_track_thresh:
                    self._create(d)

        for k, v in list(self.tracks.items()):
            if v["missed"] > self.max_missed:
                self.tracks.pop(k, None)
        return self._active()

    def _create(self, d):
        tid = self.next_id
        self.next_id += 1
        box = d[:4].astype(float)
        self.tracks[tid] = {
            "id": tid, "box": box.copy(), "pred": box.copy(), "cls": int(d[5]),
            "score": float(d[4]), "center": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            "v": np.zeros(2, float), "missed": 0, "hits": 1,
        }

    def _assign(self, tid, d):
        t = self.tracks[tid]
        new_center = ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
        step = np.array(new_center) - np.array(t["center"])
        t["v"] = 0.6 * t["v"] + 0.4 * step
        t["box"] = d[:4].astype(float)
        t["pred"] = t["box"].copy()
        t["cls"] = int(d[5])
        t["score"] = float(d[4])
        t["center"] = new_center
        t["missed"] = 0
        t["hits"] += 1

    def _active(self) -> List[dict]:
        res = []
        for t in self.tracks.values():
            if t["missed"] > 0 or t["hits"] < self.cfg.min_hits:
                continue
            res.append({
                "id": t["id"], "box": t["box"].copy(), "cls": int(t["cls"]),
                "conf": t["score"], "center": t["center"],
            })
        return res


# ---------------------------------------------------------------------------
# Wrapper ByteTrack (API Ultralytics)
# ---------------------------------------------------------------------------
class ByteTrackWrapper:
  
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.impl = None
        self.backend = "fallback"
        self._fallback = FallbackTracker(cfg)
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            args = tracker_args(cfg)
            self.impl = BYTETracker(args)
            self.backend = "bytetrack"
        except Exception as exc:  # pragma: no cover
            print(f"[tracker] BYTETracker indisponible ({exc}) -> suiveur de secours")
            self.impl = self._fallback

    # ------------------------------------------------------------------
    def update(self, dets: np.ndarray, frame: Optional[np.ndarray] = None) -> List[dict]:
        dets = np.asarray(dets, dtype=np.float32)
        if dets.size and dets.shape[1] < 6:
            raise ValueError("dets doit contenir 6 colonnes")
        if dets.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)

        if self.backend != "bytetrack":
            return self.impl.update(dets, frame)

        try:
            res_obj = _DetResults(dets)
            out = self.impl.update(res_obj, frame)
        except Exception as exc:
            print(f"[tracker] erreur BYTETracker ({exc}) -> bascule suiveur de secours")
            self.backend = "fallback"
            self.impl = self._fallback
            return self.impl.update(dets, frame)

        if out is None or len(out) == 0:
            return []

        # `out` : (N, 8) = xyxy(4) | track_id | score | cls | idx
        tracks: List[dict] = []
        for row in out:
            box = np.asarray(row[:4], dtype=float).copy()
            tid = int(row[4])
            score = float(row[5])
            cls_raw = row[6]
            try:
                cls = int(np.asarray(cls_raw).reshape(-1)[0])
            except Exception:
                cls = -1
            tracks.append({
                "id": tid, "box": box, "cls": cls, "conf": score,
                "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
            })
        return tracks