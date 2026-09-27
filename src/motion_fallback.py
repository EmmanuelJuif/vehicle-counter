from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, List, Optional

import cv2
import numpy as np

from .config import GhostConfig
from .counting_zone import CountingZone


@dataclass
class GhostBlob:
    gid: int
    center: tuple
    box: np.ndarray
    frames: int = 0
    first: tuple = (0.0, 0.0)
    max_area: float = 0.0
    history: Deque[tuple] = field(default_factory=lambda: __import__("collections").deque(maxlen=32))
    missed: int = 0
    done: bool = False
    last_frame: int = -1
    # --- suivi de la « qualité véhicule » (filtres anti-bruit) ---
    areas: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    first_frame: int = -1


class GhostDetector:
    def __init__(self, cfg: GhostConfig, zone: CountingZone, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.zone = zone
        self.fw, self.fh = frame_w, frame_h
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=True
        )
        self.blobs: dict[int, GhostBlob] = {}
        self.next_id = 10_000  # préfixe distinct des IDs ByteTrack
        self.warmup = 0
        self.kernel_o = np.ones((3, 3), np.uint8)
        self.kernel_c = np.ones((5, 5), np.uint8)
        # compteurs de rejet : indispensables pour calibrer les filtres
        self.stats = {"blobs": 0, "min_frames": 0, "displacement": 0, "filters": 0,
                      "rej_speed": 0, "rej_size": 0, "rej_fill": 0, "rej_axis": 0,
                      "rej_dir": 0, "counted": 0}
        self.trace: list[dict] = []   # relevé des candidats (mode calibrage)

        # région analysée : la bande + couloirs d'approche / de sortie
        m = 220
        if zone.axis == "y":
            self.roi = (0, max(0, int(zone.top) - m), frame_w, min(frame_h, int(zone.bottom) + m))
        else:
            self.roi = (max(0, int(zone.top) - m), 0, min(frame_w, int(zone.bottom) + m), frame_h)
        self.scale = cfg.scale

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, tracked_boxes: np.ndarray,
               frame_id: int, counter) -> List[dict]:
        if not self.cfg.enable:
            return []
        self.warmup += 1
        if self.warmup < 25:
            return []

        x0, y0, x1, y1 = self.roi
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []
        s = self.scale
        small = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                           interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.bg.apply(gray, learningRate=-1)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_o)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_c)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obs = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.cfg.min_area or a > self.cfg.max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 4 or bh < 4:
                continue
            ar = bw / max(1.0, bh)
            if ar < 0.25 or ar > 4.0:
                continue
            # retour en coords pleine image
            fx = x0 + bx / s
            fy = y0 + by / s
            fw_ = bw / s
            fh_ = bh / s
            box = np.array([fx, fy, fx + fw_, fy + fh_], dtype=float)
            center = (fx + fw_ / 2.0, fy + fh_ / 2.0)
            if self._covered(box, center, tracked_boxes):
                continue
            # taux de remplissage : un blob diffus (ombre, reflet) remplit mal
            # sa boîte englobante, un véhicule la remplit bien
            fill = float(a) / max(1.0, bw * bh)
            obs.append((center, box, a / (s * s), fill))

        self._associate(obs, frame_id)

        events = []
        for gid, b in list(self.blobs.items()):
            if b.done or gid in counter.ghost_ids:
                continue
            if frame_id - b.last_frame > 3:
                continue
            self.stats["blobs"] += 1
            if b.frames < self.cfg.min_frames:
                self.stats["min_frames"] += 1
                continue
            if len(b.history) < 2:
                continue
            p0, p1 = b.history[-2], b.history[-1]
            disp = float(np.hypot(p1[0] - b.first[0], p1[1] - b.first[1]))
            if disp < self.cfg.min_displacement:
                self.stats["displacement"] += 1
                continue
            d = self.zone.cross_direction(p0, p1, 0.5)
            if self.cfg.trace:
                # on enregistre systématiquement (avant filtres) pour le calibrage
                f = self._features(b, disp)
                f["displacement"] = disp
                f["frames"] = b.frames
                self.trace.append({"frame": frame_id, "gid": gid, "dir": d or "",
                                   "center": (float(p1[0]), float(p1[1])),
                                   "area": float(b.max_area), "feat": f})
            if self.cfg.strict:
                why = self._passes_filters(b, disp)
                if why is not True:
                    self.stats["filters"] += 1
                    self.stats[why] = self.stats.get(why, 0) + 1
                    continue
            if d:
                cls = self._classify(b)
                ev = counter.register_ghost(gid, cls, d, frame_id, p1)
                self.stats["counted"] += 1
                b.done = True
                events.append({"id": gid, "cls": cls, "box": b.box, "ghost": True, "conf": 0.0,
                               "center": p1, "event": ev})
        # nettoyage
        for gid in [k for k, v in self.blobs.items() if frame_id - v.last_frame > 20]:
            self.blobs.pop(gid, None)
        return events

    # ------------------------------------------------------------------
    def _covered(self, box, center, tracked: np.ndarray) -> bool:
        """Le blob correspond-il déjà à une piste YOLO ?"""
        if tracked is None or len(tracked) == 0:
            return False
        for tb in tracked:
            tb = np.asarray(tb, dtype=float)
            ox = max(0.0, min(box[2], tb[2]) - max(box[0], tb[0]))
            oy = max(0.0, min(box[3], tb[3]) - max(box[1], tb[1]))
            inter = ox * oy
            a1 = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
            if inter / a1 > 0.35:
                return True
            if tb[0] - 10 <= center[0] <= tb[2] + 10 and tb[1] - 10 <= center[1] <= tb[3] + 10:
                return True
        return False

    def _associate(self, obs, frame_id: int) -> None:
        used = set()
        for gid, b in self.blobs.items():
            best, bd = None, self.cfg.assoc_dist
            for i, (c, box, area, _fill) in enumerate(obs):
                if i in used:
                    continue
                d = float(np.hypot(c[0] - b.center[0], c[1] - b.center[1]))
                if d < bd:
                    best, bd = i, d
            if best is not None:
                c, box, area, fill = obs[best]
                used.add(best)
                b.center = c
                b.box = box
                b.frames += 1
                b.missed = 0
                b.max_area = max(b.max_area, area)
                b.areas.append(area)
                b.fills.append(fill)
                b.history.append(c)
                b.last_frame = frame_id
            else:
                b.missed += 1
        for i, (c, box, area, fill) in enumerate(obs):
            if i in used:
                continue
            if len(self.blobs) >= self.cfg.max_ghosts:
                break
            gid = self.next_id
            self.next_id += 1
            from collections import deque
            self.blobs[gid] = GhostBlob(
                gid=gid, center=c, box=box, frames=1, first=c, max_area=area,
                history=deque([c], maxlen=32), last_frame=frame_id,
                areas=[area], fills=[fill], first_frame=frame_id,
            )

    def _features(self, b: GhostBlob, disp: float) -> dict:
       
        span = max(1, b.last_frame - b.first_frame)
        feat = {"speed": disp / span, "size_var": 1.0, "fill": 1.0,
                "axis": 1.0, "dir": 1.0}

        if len(b.areas) >= 3:
            amin, amax = min(b.areas), max(b.areas)
            if amin > 0:
                feat["size_var"] = amax / amin
        if b.fills:
            feat["fill"] = sum(b.fills) / len(b.fills)

        pts = list(b.history)
        if len(pts) < 2:
            return feat
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        total = float(np.hypot(dx, dy)) + 1e-9
        along = abs(dy) if self.zone.axis == "y" else abs(dx)
        feat["axis"] = along / total

        steps = []
        for i in range(1, len(pts)):
            sx = pts[i][0] - pts[i - 1][0]
            sy = pts[i][1] - pts[i - 1][1]
            steps.append(sy if self.zone.axis == "y" else sx)
        steps = [s for s in steps if abs(s) > 0.5]
        if len(steps) >= 3:
            pos = sum(1 for s in steps if s > 0)
            feat["dir"] = max(pos, len(steps) - pos) / len(steps)
        return feat

    @staticmethod
    def _reject(feat: dict, cfg) -> str | None:
        """Applique les seuils à des caractéristiques déjà calculées."""
        if feat["speed"] < cfg.min_speed:
            return "rej_speed"
        if feat["size_var"] > cfg.max_size_var:
            return "rej_size"
        if feat["fill"] < cfg.min_fill:
            return "rej_fill"
        if feat["axis"] < cfg.min_axis_ratio:
            return "rej_axis"
        if feat["dir"] < cfg.min_dir_consistency:
            return "rej_dir"
        return None

    def _passes_filters(self, b: GhostBlob, disp: float):
        
        why = self._reject(self._features(b, disp), self.cfg)
        return True if why is None else why

    @staticmethod
    def _classify(b: GhostBlob) -> str:
        a = b.max_area
        if a < 2500:
            return "motorcycle"
        if a > 12000:
            return "truck"
        return "car"
