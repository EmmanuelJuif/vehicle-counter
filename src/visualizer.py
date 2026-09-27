"""Rendu OpenCV : boîtes, IDs, Counting Zone et panneau de statistiques."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .config import CLASS_COLORS, OutputConfig
from .counting_zone import CountingZone, ZoneCounter


class Visualizer:
    def __init__(self, cfg: OutputConfig, zone: CountingZone):
        self.cfg = cfg
        self.zone = zone
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ------------------------------------------------------------------
    def draw_zone(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        if self.zone.axis == "y":
            cv2.rectangle(overlay, (0, int(self.zone.top)), (w, int(self.zone.bottom)),
                          (60, 200, 255), -1)
            cv2.line(frame, (0, int(self.zone.top)), (w, int(self.zone.top)), (0, 140, 255), 2)
            cv2.line(frame, (0, int(self.zone.bottom)), (w, int(self.zone.bottom)), (0, 140, 255), 2)
            yc = int(self.zone.center)
            for x in range(0, w, 24):
                cv2.line(frame, (x, yc), (min(w, x + 14), yc), (255, 255, 255), 2)
            label = f"COUNTING ZONE - {int(self.zone.height)} px"
            self._text(frame, label, (10, max(18, int(self.zone.top) - 8)), (255, 255, 255), 0.55, 2)
        else:
            cv2.rectangle(overlay, (int(self.zone.top), 0), (int(self.zone.bottom), h),
                          (60, 200, 255), -1)
            cv2.line(frame, (int(self.zone.top), 0), (int(self.zone.top), h), (0, 140, 255), 2)
            cv2.line(frame, (int(self.zone.bottom), 0), (int(self.zone.bottom), h), (0, 140, 255), 2)
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray, tracks: List[dict], counter: ZoneCounter,
             fps: float, frame_id: int, ghosts: Optional[List[dict]] = None,
             trails: Optional[Dict[int, List[tuple]]] = None) -> np.ndarray:
        self.draw_zone(frame)

        for g in ghosts or []:
            b = np.asarray(g["box"], dtype=int)
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), CLASS_COLORS["ghost"], 1, cv2.LINE_AA)
            self._text(frame, f"?{g['id']} {g['cls']}", (b[0], max(12, b[1] - 4)),
                       CLASS_COLORS["ghost"], 0.42, 1)

        if self.cfg.show_tracks:
            for t in tracks:
                tid = int(t["id"])
                box = np.asarray(t["box"], dtype=int)
                cls = t.get("cls_name", "car")
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                counted = tid in counter.counted_ids
                thick = 2 if counted else 1
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thick, cv2.LINE_AA)
                if trails and tid in trails:
                    pts = trails[tid]
                    for i in range(1, len(pts)):
                        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                                 (int(pts[i][0]), int(pts[i][1])), color, 1, cv2.LINE_AA)
                tag = f"#{tid} {cls} {t['conf']:.2f}" + ("  [OK]" if counted else "")
                self._text(frame, tag, (box[0], max(12, box[1] - 5)), color, 0.46, 2)

        panel = self._panel(counter, fps, frame_id, len(tracks))
        self._stats(frame, panel)
        return frame

    # ------------------------------------------------------------------
    def _panel(self, counter: ZoneCounter, fps: float, frame_id: int, n_tracks: int) -> List[str]:
        s = counter.summary()
        keys = sorted(counter.total.keys())
        lines = [f"FPS {fps:5.1f}   frame {frame_id}   tracks {n_tracks}"]
        lines.append("-" * 34)
        for cls in ("car", "motorcycle", "truck", "bus"):
            per = s["by_class"].get(cls)
            if not per:
                continue
            up = per.get(keys[0], 0)
            dn = per.get(keys[1], 0)
            lines.append(f"{cls:<11} {keys[0][0].upper()}{up:>4} {keys[1][0].upper()}{dn:>4}   ={up + dn:>4}")
        lines.append("-" * 34)
        lines.append(f"TOTAL        {keys[0][0].upper()}{s['total'].get(keys[0], 0):>4} "
                     f"{keys[1][0].upper()}{s['total'].get(keys[1], 0):>4}   ={s['grand_total']:>4}")
        if s["ghosts"]:
            lines.append(f"recup. mouvement : {s['ghosts']}")
        return lines

    def _stats(self, frame: np.ndarray, lines: List[str]) -> None:
        pad, lh = 8, 20
        w = max(190, int(max(cv2.getTextSize(l, self.font, 0.5, 1)[0][0] for l in lines)) + 2 * pad)
        h = len(lines) * lh + 2 * pad
        x, y = 10, 10
        sub = frame[y:y + h, x:x + w]
        if sub.size:
            blk = np.zeros_like(sub)
            cv2.addWeighted(blk, 0.55, sub, 0.45, 0, sub)
            frame[y:y + h, x:x + w] = sub
        cv2.rectangle(frame, (x, y), (x + w, y + h), (90, 90, 90), 1)
        for i, l in enumerate(lines):
            col = (255, 255, 255) if i else (120, 255, 160)
            cv2.putText(frame, l, (x + pad, y + pad + (i + 1) * lh - 5),
                        self.font, 0.5, col, 1, cv2.LINE_AA)

    def _text(self, frame, txt, org, color, scale, thick) -> None:
        (tw, th), _ = cv2.getTextSize(txt, self.font, scale, thick)
        x, y = int(org[0]), int(org[1])
        x0, y0 = max(0, x - 2), max(0, y - th - 3)
        cv2.rectangle(frame, (x0, y0), (min(frame.shape[1], x + tw + 3), y + 3), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x, y), self.font, scale, color, thick, cv2.LINE_AA)
