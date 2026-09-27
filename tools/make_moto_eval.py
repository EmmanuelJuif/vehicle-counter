from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np

W, H = 1280, 720
ZONE_TOP, ZONE_BOTTOM = 380, 580  # Counting Zone 200 px
NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _draw_road(bg: np.ndarray) -> None:
    for y in range(H):
        t = y / H
        v = int(70 + 45 * t)
        bg[y, :] = (v, v, v)
    cv2.rectangle(bg, (0, 0), (W, 160), (95, 110, 95), -1)
    cv2.rectangle(bg, (0, 130), (W, 160), (120, 120, 90), -1)
    for x in (340, 540, 740, 940):
        for y in range(160, H, 60):
            cv2.rectangle(bg, (x - 4, y), (x + 4, min(H, y + 34)), (200, 200, 200), -1)


def extract_visdrone_crops(root: str, cls: int, want: int, min_px: int = 16) -> list:
    """Crops réels d'une classe, du plus grand au plus petit (VisDrone = aérien)."""
    found = []
    for lf in sorted(glob.glob(os.path.join(root, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = os.path.join(root, "images", stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        im = cv2.imread(img_path)
        if im is None:
            continue
        ih, iw = im.shape[:2]
        boxes = []
        for line in open(lf, encoding="utf-8"):
            p = line.split()
            if len(p) < 5 or int(p[0]) != cls:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1 = int((cx - bw / 2) * iw)
            y1 = int((cy - bh / 2) * ih)
            x2 = int((cx + bw / 2) * iw)
            y2 = int((cy + bh / 2) * ih)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            boxes.append((x1, y1, x2, y2))
        if not boxes:
            continue
        # la plus grosse boîte de l'image
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        found.append(((x2 - x1) * (y2 - y1), im[y1:y2, x1:x2].copy()))

    found.sort(key=lambda kv: -kv[0])
    out, seen = [], set()
    for area, crop in found:
        h, w = crop.shape[:2]
        key = (w // 8, h // 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(crop)
        if len(out) >= want:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/moto.mp4")
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--visdrone", default="data/visdrone")
    ap.add_argument("--n-moto", type=int, default=10)
    ap.add_argument("--n-car", type=int, default=4)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    print("[1/3] extraction de motos réelles depuis VisDrone ...")
    motos = extract_visdrone_crops(a.visdrone, 3, a.n_moto)
    cars = extract_visdrone_crops(a.visdrone, 2, a.n_car)
    print(f"   motorcycle: {len(motos)} crops   car: {len(cars)} crops")
    if not motos:
        raise SystemExit("aucune moto extraite")
    crops = {"motorcycle": motos, **({"car": cars} if cars else {})}

    print(f"[2/3] composition de {a.out} ({a.frames} frames) ...")
    plan, t = [], 0
    while t < a.frames - 40:
        cls = "motorcycle" if rng.random() < 0.75 else (rng.choice(list(crops)))
        idx = rng.randrange(len(crops[cls]))
        fast = rng.random() < 0.35
        speed = rng.uniform(20, 30) if fast else rng.uniform(6, 13)
        scale = rng.uniform(0.9, 1.6) if fast else rng.uniform(1.0, 2.2)
        lane = rng.choice([230, 430, 630, 830])
        down = rng.random() < 0.6
        plan.append({"cls": cls, "idx": idx, "speed": speed, "scale": scale,
                     "lane": lane, "down": down, "start": int(t), "fast": fast})
        t += rng.uniform(8, 16)

    bg = np.zeros((H, W, 3), dtype=np.uint8)
    _draw_road(bg)
    noise = np.random.default_rng(a.seed + 1).normal(0, 4, (H, W, 3)).astype(np.int16)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))

    for f in range(a.frames):
        frame = bg.copy()
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.rectangle(frame, (0, ZONE_TOP), (W, ZONE_BOTTOM), (60, 90, 120), 1)
        for v in plan:
            k = f - v["start"]
            if k < 0:
                continue
            crop = crops[v["cls"]][v["idx"]]
            sh = crop.shape
            nw = max(8, int(sh[1] * v["scale"]))
            nh = max(8, int(sh[0] * v["scale"]))
            sprite = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            y = 60 + k * v["speed"] if v["down"] else H - 60 - k * v["speed"]
            if y < -nh or y > H + nh:
                continue
            x = max(0, min(W - nw, int(v["lane"] - nw / 2)))
            yy = int(y - nh / 2)
            if 0 <= yy < H - nh and 0 <= x < W - nw:
                frame[yy:yy + nh, x:x + nw] = sprite
            prev_y = y - v["speed"] if v["down"] else y + v["speed"]
            mid = (ZONE_TOP + ZONE_BOTTOM) / 2
            if k >= 1 and ((prev_y <= mid <= y) or (y <= mid <= prev_y)):
                v.setdefault("_counted_at", f)
        vw.write(frame)
    vw.release()

    gt = {"frames": a.frames, "fps": a.fps,
          "zone": {"top": ZONE_TOP, "bottom": ZONE_BOTTOM},
          "counts": {}, "items": []}
    for v in plan:
        if "_counted_at" in v:
            d = "down" if v["down"] else "up"
            gt["counts"].setdefault(v["cls"], {"up": 0, "down": 0})[d] += 1
            gt["items"].append({"cls": v["cls"], "dir": d, "frame": v["_counted_at"],
                                "fast": v["fast"]})
    gt["total"] = sum(sum(c.values()) for c in gt["counts"].values())

    gt_path = os.path.splitext(a.out)[0] + "_gt.json"
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2, ensure_ascii=False)

    print(f"[3/3] vidéo  : {a.out} ({a.frames} frames, {W}x{H})")
    print(f"      vérité : {gt_path}  -> total {gt['total']}")
    for k, v in sorted(gt["counts"].items()):
        print(f"        {k:<11} {v}")
    nfast = sum(1 for i in gt["items"] if i["fast"])
    print(f"      dont passages très rapides : {nfast}")


if __name__ == "__main__":
    main()
