from __future__ import annotations

import argparse
import json
import math
import os
import random

import cv2
import numpy as np

W, H = 1280, 720
LANES = [230, 430, 630, 830]  # centres des voies (2 descendeuses / 2 montantes)


def _car_sprite(w, h, color, dark=False):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, int(h * 0.35)), (w - 1, h - 1), color, -1)
    cv2.rectangle(img, (int(w * 0.12), 0), (int(w * 0.88), int(h * 0.5)), color, -1)
    # pare-brise
    cv2.rectangle(img, (int(w * 0.18), int(h * 0.06)), (int(w * 0.82), int(h * 0.42)),
                  (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60)), -1)
    # roues
    cv2.rectangle(img, (int(w * 0.05), int(h * 0.72)), (int(w * 0.22), h - 1), (30, 30, 30), -1)
    cv2.rectangle(img, (int(w * 0.78), int(h * 0.72)), (int(w * 0.95), h - 1), (30, 30, 30), -1)
    if dark:
        img = (img * 0.35).astype(np.uint8)
    return img


def _moto_sprite(w, h, dark=True):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    body = (28, 28, 32) if dark else (200, 200, 205)
    cv2.rectangle(img, (int(w * 0.2), int(h * 0.35)), (int(w * 0.8), int(h * 0.95)), body, -1)
    cv2.circle(img, (int(w * 0.5), int(h * 0.22)), max(2, int(w * 0.16)), body, -1)  # casque
    cv2.circle(img, (int(w * 0.12), int(h * 0.85)), max(2, int(h * 0.13)), (45, 45, 45), -1)
    cv2.circle(img, (int(w * 0.88), int(h * 0.85)), max(2, int(h * 0.13)), (45, 45, 45), -1)
    return img


def _truck_sprite(w, h, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, int(h * 0.2)), (int(w * 0.72), h - 1), color, -1)   # remorque
    cv2.rectangle(img, (int(w * 0.72), int(h * 0.42)), (w - 1, h - 1), (max(0, color[0] - 40),
                  max(0, color[1] - 40), max(0, color[2] - 40)), -1)            # cabine
    cv2.rectangle(img, (int(w * 0.78), int(h * 0.5)), (int(w * 0.96), int(h * 0.78)), (60, 90, 120), -1)
    return img


def _bus_sprite(w, h, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, int(h * 0.15)), (w - 1, h - 1), color, -1)
    for i in range(5):
        x = int(w * (0.08 + i * 0.17))
        cv2.rectangle(img, (x, int(h * 0.28)), (x + int(w * 0.12), int(h * 0.55)), (70, 100, 130), -1)
    return img


def scale_for(y: float) -> float:
    """Perspective grossière : plus c'est haut dans l'image, plus c'est petit."""
    return 0.45 + 0.85 * (y / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traffic.mp4")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rng = random.Random(a.seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (W, H))

    # fond route
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        t = y / H
        v = int(70 + 45 * t)
        bg[y, :] = (v, v, v)
    cv2.rectangle(bg, (0, 0), (W, 150), (95, 110, 95), -1)          # décor haut
    cv2.rectangle(bg, (0, 120), (W, 150), (120, 120, 90), -1)
    for x in (330, 530, 730):                                        # lignes de voie
        for y in range(150, H, 60):
            cv2.rectangle(bg, (x - 4, y), (x + 4, min(H, y + 34)), (200, 200, 200), -1)
    noise = np.random.default_rng(a.seed).normal(0, 4, (H, W, 3)).astype(np.int16)

    # --- programme du trafic ----------------------------------------------
    # (classe, voie, sens, vitesse px/frame, frame d'apparition, teinte)
    plan = []
    t = 0
    while t < a.frames:
        cls = rng.choices(["car", "car", "motorcycle", "truck", "bus"],
                          weights=[5, 5, 4, 2, 1])[0]
        lane = rng.choice(LANES)
        down = lane < 530
        speed = {"car": rng.uniform(6, 12), "motorcycle": rng.uniform(8, 18),
                 "truck": rng.uniform(5, 8), "bus": rng.uniform(4, 7)}[cls]
        color = (rng.randint(40, 230), rng.randint(40, 230), rng.randint(40, 230))
        plan.append({"cls": cls, "lane": lane, "down": down, "speed": speed,
                     "start": int(t), "color": color,
                     "dark": rng.random() < 0.6})
        t += rng.uniform(6, 18)

    # cas pièges explicites
    plan.append({"cls": "motorcycle", "lane": 430, "down": True, "speed": 62.0,
                 "start": 60, "color": (30, 30, 34), "dark": True})   # moto très rapide
    plan.append({"cls": "motorcycle", "lane": 630, "down": False, "speed": 34.0,
                 "start": 150, "color": (22, 22, 26), "dark": True})  # moto sombre rapide
    plan.append({"cls": "car", "lane": 830, "down": False, "speed": 28.0,
                 "start": 200, "color": (60, 60, 70), "dark": True})  # voiture rapide

    gt = {"frames": a.frames, "fps": a.fps, "zone": {"top": 0, "bottom": 0},
          "counts": {}, "items": []}

    zone_top, zone_bottom = int(0.62 * H - 100), int(0.62 * H + 100)
    gt["zone"] = {"top": zone_top, "bottom": zone_bottom}

    for f in range(a.frames):
        frame = bg.copy()
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # bande de comptage dessinée pour repère visuel
        cv2.rectangle(frame, (0, zone_top), (W, zone_bottom), (60, 90, 120), 1)

        for v in plan:
            k = f - v["start"]
            if k < 0:
                continue
            y = 40 + k * v["speed"] if v["down"] else H - 40 - k * v["speed"]
            if y < -80 or y > H + 80:
                continue
            s = scale_for(y)
            cls = v["cls"]
            if cls == "car":
                w, h = int(150 * s), int(84 * s)
                spr = _car_sprite(w, h, v["color"], v["dark"])
            elif cls == "motorcycle":
                w, h = int(58 * s), int(70 * s)
                spr = _moto_sprite(w, h, v["dark"])
            elif cls == "truck":
                w, h = int(230 * s), int(120 * s)
                spr = _truck_sprite(w, h, v["color"])
            else:
                w, h = int(250 * s), int(130 * s)
                spr = _bus_sprite(w, h, v["color"])
            w, h = max(6, w), max(6, h)
            spr = cv2.resize(spr, (w, h), interpolation=cv2.INTER_AREA)
            x = int(v["lane"] - w / 2)
            x = max(0, min(W - w, x))
            yy = int(y - h / 2)
            if 0 <= yy < H - h and 0 <= x < W - w:
                roi = frame[yy:yy + h, x:x + w]
                m = spr > 8
                roi[m] = spr[m]
            # franchissement réel de la médiane (= vérité terrain)
            prev_y = y - v["speed"] if v["down"] else y + v["speed"]
            mid = (zone_top + zone_bottom) / 2
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
    print(f"vidéo   : {a.out} ({a.frames} frames, {W}x{H})")
    print(f"vérité  : {gt_path}  -> total {gt['total']}")
    for k, v in sorted(gt["counts"].items()):
        print(f"   {k:<11} {v}")


if __name__ == "__main__":
    main()
