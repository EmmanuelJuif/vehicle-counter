from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W, H = 1280, 720
ZONE_TOP, ZONE_BOTTOM = 380, 580


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/noise.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shadows", type=int, default=4,
                    help="nombre d'ombres mobiles (la principale source de FP)")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="amplitude du micro-jitter caméra en px")
    ap.add_argument("--grain", type=float, default=5.0,
                    help="écart-type du bruit capteur")
    ap.add_argument("--rain", type=int, default=-1,
                    help="gouttes de pluie par frame (stries rapides : gros pourvoyeur de FP)")
    ap.add_argument("--profile", choices=("doux", "moyen", "severe"), default="moyen",
                    help="intensité globale : doux / moyen / severe")
    ap.add_argument("--gain", type=float, default=-1.0,
                    help="amplitude du scintillement d'exposition (0.14 = ±14%%)")
    a = ap.parse_args()

    P = {"doux": dict(shadows=3, shadow_amp=(0.10, 0.20), shadow_spd=1.2, gain=0.06,
                      grain=4.0, jitter=0.8, rain=0),
         "moyen": dict(shadows=5, shadow_amp=(0.18, 0.38), shadow_spd=2.6, gain=0.14,
                       grain=7.0, jitter=1.6, rain=250),
         "severe": dict(shadows=8, shadow_amp=(0.25, 0.50), shadow_spd=4.0, gain=0.22,
                        grain=10.0, jitter=2.5, rain=600)}[a.profile]
    a.shadows = P["shadows"]
    if a.grain < 0:
        a.grain = P["grain"]
    if a.jitter < 0:
        a.jitter = P["jitter"]
    if a.rain < 0:
        a.rain = P["rain"]
    if a.gain < 0:
        a.gain = P["gain"]

    rng = np.random.default_rng(a.seed)
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    _draw_road(bg)
    bgf = bg.astype(np.float32)

    # ombres : ellipses molles qui dérivent lentement
    shadows = []
    amp_lo, amp_hi = P["shadow_amp"]
    spd = P["shadow_spd"]
    for _ in range(a.shadows):
        shadows.append({
            "x": rng.uniform(0, W), "y": rng.uniform(200, H),
            "rx": rng.uniform(90, 260), "ry": rng.uniform(50, 150),
            "vx": rng.uniform(-spd, spd), "vy": rng.uniform(-spd * 0.4, spd * 0.4),
            "amp": rng.uniform(amp_lo, amp_hi),
        })

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, fourcc, a.fps, (W, H))

    for f in range(a.frames):
        frame = bgf.copy()

        # 1) ombres mobiles
        for s in shadows:
            s["x"] += s["vx"]
            s["y"] += s["vy"]
            if s["x"] < -s["rx"] or s["x"] > W + s["rx"]:
                s["vx"] *= -1
            if s["y"] < 150 or s["y"] > H + s["ry"]:
                s["vy"] *= -1
            overlay = frame.copy()
            cv2.ellipse(overlay, (int(s["x"]), int(s["y"])),
                        (int(s["rx"]), int(s["ry"])), 0, 0, 360, (0, 0, 0), -1)
            cv2.addWeighted(overlay, s["amp"], frame, 1 - s["amp"], 0, dst=frame)

        # 2) scintillement d'exposition
        gain = 1.0 + a.gain * np.sin(f / 17.0) + (a.gain / 2) * np.sin(f / 3.1)
        frame *= gain

        # 3) grain capteur
        frame += rng.normal(0, a.grain, (H, W, 3))

        out = np.clip(frame, 0, 255).astype(np.uint8)

        # 3b) pluie : stries rapides, très petites mais nombreuses
        if a.rain > 0:
            n = int(a.rain * (0.6 + 0.4 * np.sin(f / 23.0)))
            xs = rng.integers(0, W, n)
            ys = rng.integers(0, H, n)
            lens = rng.integers(6, 15, n)
            for k in range(n):
                x, y, L = int(xs[k]), int(ys[k]), int(lens[k])
                cv2.line(out, (x, y), (x + 3, y + L), (235, 235, 235), 1, cv2.LINE_AA)

        # 4) micro-jitter caméra : déplace toute l'image
        if a.jitter > 0:
            jx = a.jitter * np.sin(f / 9.0)
            jy = a.jitter * np.cos(f / 13.0)
            M = np.float32([[1, 0, jx], [0, 1, jy]])
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

        # 5) artefacts de compression (quantification par blocs 8x8)
        q = cv2.resize(out, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        q = cv2.resize(q, (W, H), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(out, 0.85, q, 0.15, 0)

        cv2.rectangle(out, (0, ZONE_TOP), (W, ZONE_BOTTOM), (60, 90, 120), 1)
        vw.write(out)

    vw.release()
    dur = a.frames / a.fps
    print(f"vidéo de bruit : {a.out}")
    print(f"  {a.frames} frames @ {a.fps} fps = {dur:.0f}s, {W}x{H}, "
          f"{a.shadows} ombres mobiles, jitter {a.jitter} px, grain {a.grain}")
    print(f"  vérité terrain : 0 véhicule — tout franchissement compté est un FAUX POSITIF")


if __name__ == "__main__":
    main()
