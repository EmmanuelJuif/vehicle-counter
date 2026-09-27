from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np


W, H = 1280, 720
ZONE_TOP, ZONE_BOTTOM = 380, 580  # Counting Zone 200 px


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


def extract_crops(source: str, out_dir: str, samples: int = 60) -> dict:
    """Capture des véhicules réels depuis `source` (vidéo CCTV existante)."""
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    model = YOLO("yolov8s.pt")
    collected: dict = {n: [] for n in ("car", "truck", "bus", "motorcycle")}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fidx in np.linspace(0, max(1, n - 1), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, imgsz=640, conf=0.30, iou=0.65,
                         classes=[2, 3, 5, 7], device="cpu", verbose=False)
        for b in r[0].boxes:
            cid = int(b.cls.item())
            name = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}[cid]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                continue
            score = float(b.conf.item())
            collected[name].append((score, crop))
    cap.release()

    selected: dict = {}
    for name, items in collected.items():
        items.sort(key=lambda kv: -kv[0] * (kv[1].shape[0] * kv[1].shape[1]))
        chosen = []
        seen_hashes = set()
        for score, crop in items:
            h = (crop.shape[0], crop.shape[1])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chosen.append(crop)
            if len(chosen) >= 8:
                break
        selected[name] = chosen
        for i, c in enumerate(chosen):
            cv2.imwrite(os.path.join(out_dir, f"{name}_{i}.png"), c)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/sample5s.mp4")
    ap.add_argument("--out", default="data/eval.mp4")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    crops_dir = "data/crops"
    print(f"[1/2] extraction de véhicules réels depuis {a.source} ...")
    crops = extract_crops(a.source, crops_dir)
    for name, lst in crops.items():
        print(f"   {name}: {len(lst)} crops")
    if not any(crops.values()):
        raise SystemExit("Aucun véhicule détecté dans la source — vidéo inutilisable.")

    print(f"[2/2] composition de la vidéo d'évaluation ({a.frames} frames) ...")
    plan = []
    t = 0
    while t < a.frames - 30:
        available = [(cls, i, c) for cls, lst in crops.items() for i, c in enumerate(lst)]
        if not available:
            break
        cls, idx, crop = rng.choice(available)
        fast = rng.random() < 0.25
        speed = rng.uniform(22, 32) if fast else rng.uniform(7, 14)
        scale = rng.uniform(0.55, 1.0)
        lane = rng.choice([230, 430, 630, 830])
        down = lane < 600
        plan.append({
            "cls": cls, "idx": idx, "speed": speed, "scale": scale,
            "lane": lane, "down": down, "start": int(t),
            "fast": fast, "gid": len(plan) + 1,   # id vérité terrain stable
        })
        t += rng.uniform(7, 18)

    bg = np.zeros((H, W, 3), dtype=np.uint8)
    _draw_road(bg)
    noise = np.random.default_rng(a.seed + 1).normal(0, 4, (H, W, 3)).astype(np.int16)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (W, H))

    gt = {"frames": a.frames, "fps": a.fps,
          "zone": {"top": ZONE_TOP, "bottom": ZONE_BOTTOM},
          "counts": {}, "items": [],
          # --- trajectoires par frame : nécessaires aux métriques MOT -------
          # Sans elles on ne peut calculer ni MOTA ni IDF1 : ces métriques
          # comparent des BOÎTES ET des identités image par image, pas
          # seulement des franchissements. `tracks[f]` = [{id, cls, box}].
          "tracks": {}}

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
            x = int(v["lane"] - nw / 2)
            x = max(0, min(W - nw, x))
            yy = int(y - nh / 2)
            if 0 <= yy < H - nh and 0 <= x < W - nw:
                frame[yy:yy + nh, x:x + nw] = sprite
                gt["tracks"].setdefault(str(f), []).append({
                    "id": int(v["gid"]), "cls": v["cls"],
                    "box": [float(x), float(yy), float(x + nw), float(yy + nh)],
                })

            prev_y = y - v["speed"] if v["down"] else y + v["speed"]
            mid = (ZONE_TOP + ZONE_BOTTOM) / 2
            if k >= 1 and ((prev_y <= mid <= y) or (y <= mid <= prev_y)):
                v["_counted_at"] = v.get("_counted_at", f)
        vw.write(frame)

    vw.release()

    for v in plan:
        if "_counted_at" in v:
            d = "down" if v["down"] else "up"
            gt["counts"].setdefault(v["cls"], {"up": 0, "down": 0})[d] += 1
            gt["items"].append({"cls": v["cls"], "dir": d, "frame": v["_counted_at"]})
    gt["total"] = sum(sum(c.values()) for c in gt["counts"].values())

    gt_path = os.path.splitext(a.out)[0] + "_gt.json"
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=2, ensure_ascii=False)
    print(f"\nvidéo   : {a.out}  ({a.frames} frames, {W}x{H})")
    print(f"vérité  : {gt_path}  -> total {gt['total']}")
    for k, v in sorted(gt["counts"].items()):
        print(f"   {k:<11} {v}")


if __name__ == "__main__":
    main()