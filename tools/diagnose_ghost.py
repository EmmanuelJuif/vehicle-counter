from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FastVehicleConfig, GhostConfig, MotorcycleConfig, TrackerConfig
from src.counting_zone import CountingZone, ZoneCounter
from src.motion_fallback import GhostDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/eval.mp4")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--zone-top", type=int, default=380)
    ap.add_argument("--zone-bottom", type=int, default=580)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--lax", action="store_true",
                    help="désactive les filtres anti-bruit (mesure du bruit brut)")
    ap.add_argument("--min-frames", type=int, default=None,
                    help="surcharge GhostConfig.min_frames")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="surcharge GhostConfig.min_speed")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("source illisible")
    if a.scale != 1.0:
        frame = cv2.resize(frame, (int(frame.shape[1] * a.scale), int(frame.shape[0] * a.scale)))
    h, w = frame.shape[:2]

    zone = CountingZone(a.zone_top, a.zone_bottom, "y")
    counter = ZoneCounter(zone, TrackerConfig(min_hits=2, track_buffer=60),
                          MotorcycleConfig(), FastVehicleConfig(), w * h)
    kw = {"enable": True, "strict": not a.lax}
    if a.min_frames is not None:
        kw["min_frames"] = a.min_frames
    if a.min_speed is not None:
        kw["min_speed"] = a.min_speed
    ghost = GhostDetector(GhostConfig(**kw), zone, w, h)

    print(f"zone {zone.top:.0f}->{zone.bottom:.0f} (mediane {zone.center:.0f})  image {w}x{h}")
    print(f"ROI analyse : {ghost.roi}  echelle {ghost.scale}")

    stats = {"contours": 0, "kept": 0, "blobs": 0, "events": 0}
    quiet = bool(os.environ.get("QUIET"))
    fid = 0
    while True:
        if fid > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if a.scale != 1.0:
                frame = cv2.resize(frame, (w, h))
        ghost_vis = ghost.update(frame, np.zeros((0, 4)), fid, counter)
        stats["blobs"] = max(stats["blobs"], len(ghost.blobs))
        stats["events"] += len(ghost_vis)
        for g in ghost_vis:
            if quiet:
                continue
            print(f"  frame {fid:4d}  ghost #{g['id']}  {g['cls']:<10} "
                  f"aire={g['box'][2]-g['box'][0]:.0f}x{g['box'][3]-g['box'][1]:.0f}")
        fid += 1
        if a.max_frames and fid >= a.max_frames:
            break

    cap.release()
    dur = fid / 25.0
    print(f"\n--- bilan filet mouvement (sans YOLO) ---")
    print(f"  mode                 : {'PERMISSIF (filtres OFF)' if a.lax else 'STRICT (filtres ON)'}")
    print(f"  frames               : {fid} ({dur:.0f}s @ 25fps)")
    print(f"  blobs simultanés max : {stats['blobs']}")
    print(f"  franchissements      : {stats['events']}"
          f"  ({stats['events'] / max(1e-9, dur) * 60:.1f} / min)")
    s = ghost.stats
    print(f"  candidats évalués    : {s['blobs']}")
    print(f"  rejets min_frames    : {s['min_frames']}")
    print(f"  rejets déplacement   : {s['displacement']}")
    print(f"  rejets filtres       : {s['filters']}")
    for k in ("rej_speed", "rej_size", "rej_fill", "rej_axis", "rej_dir"):
        if s.get(k):
            print(f"      {k:<18} : {s[k]}")

    if a.gt:
        with open(a.gt, encoding="utf-8") as fh:
            gt = json.load(fh)
        print(f"  vérité terrain       : {gt['total']}")
        pred = stats["events"]
        print(f"  rappel du filet seul : {min(pred, gt['total']) / gt['total'] * 100:.0f}%")


if __name__ == "__main__":
    main()