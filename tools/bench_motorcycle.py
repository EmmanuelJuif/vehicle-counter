from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import DetectorConfig  # noqa: E402
from src.detector import VehicleDetector  # noqa: E402
from src.geometry import iou_matrix  # noqa: E402

# seuils de taille : côté (en px) de la boîte vérité terrain
SIZE_BINS = [(0, 16), (16, 24), (24, 32), (32, 48), (48, 64), (64, 10_000)]
# seuils de luminosité moyenne (0-255) du crop
LUM_BINS = [(0, 50), (50, 80), (80, 110), (110, 145), (145, 256)]


def bin_label(bins, v):
    for lo, hi in bins:
        if lo <= v < hi:
            return f"{lo}-{hi}" if hi < 10_000 and hi < 256 else f">={lo}"
    return "?"


def load_gt(img_dir: str, lbl_dir: str, cls: int = 3):
    """Boîtes vérité terrain en pixels absolus : {stem: [(x1,y1,x2,y2), ...]}."""
    out = {}
    for f in sorted(os.listdir(lbl_dir)):
        if not f.endswith(".txt"):
            continue
        stem = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(img_dir, stem + ".jpg"))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for line in open(os.path.join(lbl_dir, f), encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            if int(float(p[0])) != cls:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
        out[stem] = (img, boxes)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "visdrone"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--tiled", default="off", choices=["off", "band", "full"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU mini pour compter un TP")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    img_dir = os.path.join(a.data, a.split, "images")
    lbl_dir = os.path.join(a.data, a.split, "labels")
    gt = load_gt(img_dir, lbl_dir)
    if not gt:
        print(f"aucune donnée dans {img_dir}")
        return 1
    n_moto = sum(len(b) for _, b in gt.values())
    print(f"banc moto : {len(gt)} images, {n_moto} motos réelles "
          f"(imgsz={a.imgsz}, conf={a.conf}, tiled={a.tiled})")

    det = VehicleDetector(DetectorConfig(
        weights=a.weights, imgsz=a.imgsz, conf=a.conf, iou=0.65,
        tiled=a.tiled, device=a.device, adaptive=False,
    ))

    size_stats = {bin_label(SIZE_BINS, 0): [0, 0] for _, _ in SIZE_BINS}
    lum_stats = {bin_label(LUM_BINS, 0): [0, 0] for _, _ in LUM_BINS}
    tp_total = 0

    for stem, (img, boxes) in gt.items():
        preds = det.detect(img)
        det_boxes = preds[:, :4] if len(preds) else np.zeros((0, 4))
        det_cls = preds[:, 5].astype(int) if len(preds) else np.zeros(0, dtype=int)
        # on n'accepte que les détections étiquetées moto (classe COCO 3)
        keep = det_cls == 3
        det_boxes = det_boxes[keep] if len(det_boxes) else det_boxes

        if len(det_boxes) and len(boxes):
            m = iou_matrix(np.asarray(boxes, dtype=float), det_boxes.astype(float))
        else:
            m = np.zeros((len(boxes), max(1, len(det_boxes))))
        matched = (m >= a.iou).any(axis=1) if len(boxes) else np.zeros(0, bool)

        for b, ok in zip(boxes, matched):
            x1, y1, x2, y2 = b
            side = float(max(x2 - x1, y2 - y1))
            crop = img[int(max(0, y1)):int(min(img.shape[0], y2)),
                       int(max(0, x1)):int(min(img.shape[1], x2))]
            lum = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()) if crop.size else 0.0
            hit = int(bool(ok))
            tp_total += hit
            for stats, bins, v in ((size_stats, SIZE_BINS, side), (lum_stats, LUM_BINS, lum)):
                k = bin_label(bins, v)
                stats.setdefault(k, [0, 0])
                stats[k][0] += hit
                stats[k][1] += 1

    def table(title, stats, unit):
        print(f"\n{title}")
        print(f"  {unit:<12} {'motos':>7} {'détectées':>10} {'rappel':>8}")
        for k in stats:
            hit, tot = stats[k]
            if tot == 0:
                continue
            print(f"  {k:<12} {tot:>7} {hit:>10} {100.0 * hit / tot:>7.1f}%")

    table("--- RAPPEL PAR TAILLE (côté de la boîte, px) ---", size_stats, "taille")
    table("--- RAPPEL PAR LUMINOSITÉ (niveau de gris moyen) ---", lum_stats, "luminosité")
    print(f"\nRAPPEL GLOBAL : {tp_total}/{n_moto} = {100.0 * tp_total / max(1, n_moto):.1f}%")

    out = {"weights": a.weights, "imgsz": a.imgsz, "conf": a.conf, "tiled": a.tiled,
           "total": n_moto, "detected": tp_total,
           "recall": round(tp_total / max(1, n_moto), 4),
           "by_size": {k: {"hit": v[0], "total": v[1]} for k, v in size_stats.items()},
           "by_lum": {k: {"hit": v[0], "total": v[1]} for k, v in lum_stats.items()}}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nécrit : {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
