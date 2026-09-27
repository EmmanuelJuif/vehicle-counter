from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import GhostConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--zone-top", type=int, default=380)
    ap.add_argument("--zone-bottom", type=int, default=580)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--every", type=int, default=25,
                    help="affiche le détail toutes les N frames")
    a = ap.parse_args()

    cfg = GhostConfig(enable=True)
    cap = cv2.VideoCapture(a.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("source illisible")
    h, w = frame.shape[:2]

    m = 220
    roi = (0, max(0, a.zone_top - m), w, min(h, a.zone_bottom + m))
    x0, y0, x1, y1 = roi
    s = cfg.scale
    bg = cv2.createBackgroundSubtractorMOG2(
        history=cfg.history, varThreshold=cfg.var_threshold, detectShadows=True)
    k_o = np.ones((3, 3), np.uint8)
    k_c = np.ones((5, 5), np.uint8)

    tot_raw = tot_kept = 0
    areas_all = []
    fid = 0
    print(f"source {a.source}  {w}x{h}  ROI {roi}  echelle {s}")
    print(f"filtres d'aire : {cfg.min_area} <= aire <= {cfg.max_area} (à l'échelle {s})")
    print()
    while True:
        if fid > 0:
            ok, frame = cap.read()
            if not ok:
                break
        crop = frame[y0:y1, x0:x1]
        small = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                           interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = bg.apply(gray, learningRate=-1)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_o)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        kept = [ar for ar in areas if cfg.min_area <= ar <= cfg.max_area]
        if fid > 25:  # warmup du modèle de fond
            tot_raw += len(areas)
            tot_kept += len(kept)
            areas_all.extend(kept)
        if fid % a.every == 0:
            top = sorted(areas, reverse=True)[:5]
            print(f"  f{fid:4d}  contours={len(areas):3d}  retenus={len(kept):3d}  "
                  f"aires_top={[int(t) for t in top]}  px_fg={int((mask > 0).sum())}")
        fid += 1
        if a.max_frames and fid >= a.max_frames:
            break
    cap.release()

    n = max(1, fid - 26)
    print(f"\n--- sonde MOG2 ---")
    print(f"  frames analysées (après warmup) : {n}")
    print(f"  contours bruts / frame          : {tot_raw / n:.1f}")
    print(f"  contours retenus / frame        : {tot_kept / n:.1f}")
    if areas_all:
        arr = np.array(areas_all)
        print(f"  aire retenue : min {arr.min():.0f}  médiane {np.median(arr):.0f}  "
              f"moy {arr.mean():.0f}  max {arr.max():.0f}")
        print(f"  déciles : {[int(q) for q in np.percentile(arr, [10, 25, 50, 75, 90, 99])]}")
    else:
        print("  AUCUN contour retenu -> la soustraction de fond ne voit rien ici")


if __name__ == "__main__":
    main()
