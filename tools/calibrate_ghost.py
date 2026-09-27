from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FastVehicleConfig, GhostConfig, MotorcycleConfig, TrackerConfig
from src.counting_zone import CountingZone, ZoneCounter
from src.motion_fallback import GhostDetector


def collect(source: str, zone_top: int, zone_bottom: int, min_frames: int) -> tuple[list, float]:
    """Passe le filet en mode permissif et relève les candidats au franchissement."""
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"source illisible : {source}")
    h, w = frame.shape[:2]

    zone = CountingZone(zone_top, zone_bottom, "y")
    counter = ZoneCounter(zone, TrackerConfig(min_hits=2, track_buffer=60),
                          MotorcycleConfig(), FastVehicleConfig(), w * h)
    ghost = GhostDetector(
        GhostConfig(enable=True, strict=False, min_frames=min_frames, trace=True),
        zone, w, h,
    )

    fid = 0
    while True:
        if fid > 0:
            ok, frame = cap.read()
            if not ok:
                break
        ghost.update(frame, np.zeros((0, 4)), fid, counter)
        fid += 1
    cap.release()
    return ghost.trace, fid / 25.0


def label_positives(trace: list, gt_path: str, tol: int = 12) -> tuple[list, int]:
    """Apparie les franchissements relevés à la vérité terrain."""
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)
    items = [{"frame": it["frame"], "dir": it.get("dir", ""), "cls": it.get("cls", "")}
             for it in gt.get("items", [])]
    n_gt = gt.get("total", len(items))

    # la GT parle « down »/« up », cross_direction() renvoie +1/-1
    sgn = {"down": 1, "up": -1, "in": 1, "out": -1, "1": 1, "-1": -1}

    def norm(v):
        if isinstance(v, str):
            return sgn.get(v, 0)
        return int(v) if v else 0

    crossings = [t for t in trace if t["dir"]]
    used = set()
    out = []
    for t in sorted(crossings, key=lambda x: x["frame"]):
        best, bd = -1, 10 ** 9
        td = norm(t["dir"])
        for i, g in enumerate(items):
            if i in used:
                continue
            gd = norm(g["dir"])
            if gd and td and gd != td:
                continue
            d = abs(g["frame"] - t["frame"])
            if d < bd and d <= tol:
                best, bd = i, d
        if best >= 0:
            used.add(best)
            out.append({**t, "label": 1})
        else:
            out.append({**t, "label": 0})
    return out, n_gt


GRID = {
    "min_frames": [3, 4, 5, 6, 8],
    "min_speed": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
    "max_size_var": [1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 1e9],
    "min_fill": [0.0, 0.15, 0.25, 0.35, 0.50],
    "min_axis_ratio": [0.0, 0.3, 0.5, 0.7],
    "min_dir_consistency": [0.0, 0.5, 0.6, 0.7, 0.9],
}


def keep(t: dict, thr: dict) -> bool:
    f = t["feat"]
    if f["frames"] < thr["min_frames"]:
        return False
    if f["speed"] < thr["min_speed"]:
        return False
    if f["size_var"] > thr["max_size_var"]:
        return False
    if f["fill"] < thr["min_fill"]:
        return False
    if f["axis"] < thr["min_axis_ratio"]:
        return False
    if f["dir"] < thr["min_dir_consistency"]:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs="+", default=["data/eval.mp4"],
                    help="vidéos avec véhicules (une GT par vidéo, même ordre)")
    ap.add_argument("--gt", nargs="+", default=["data/eval_gt.json"])
    ap.add_argument("--neg", nargs="+", default=["data/noise.mp4"],
                    help="vidéos de bruit pur, sans véhicule")
    ap.add_argument("--zone-top", type=int, default=380)
    ap.add_argument("--zone-bottom", type=int, default=580)
    ap.add_argument("--tol", type=int, default=12)
    ap.add_argument("--min-frames-collect", type=int, default=3,
                    help="min_frames utilisé pendant le relevé (doit être <= grille)")
    ap.add_argument("--out", default=None, help="écrit le meilleur jeu de seuils en JSON")
    a = ap.parse_args()

    print("[1/3] relevé sur les vidéos AVEC véhicules ...")
    labelled: list = []
    n_gt = 0
    dur_pos = 0.0
    for vid, gtp in zip(a.pos, a.gt):
        with open(gtp, encoding="utf-8") as fh:
            z = json.load(fh).get("zone") or {}
        zt = int(z.get("top", a.zone_top))
        zb = int(z.get("bottom", a.zone_bottom))
        tpos, dur = collect(vid, zt, zb, a.min_frames_collect)
        lab, n = label_positives(tpos, gtp, a.tol)
        labelled.extend(lab)
        n_gt += n
        dur_pos += dur
        print(f"      {vid:<22} zone {zt}-{zb}  {len(lab)} franchissements, "
              f"{sum(1 for t in lab if t['label'] == 1)} appariés / {n} GT")
    n_tp = sum(1 for t in labelled if t["label"] == 1)
    print(f"      TOTAL {n_tp} vrais positifs pour {n_gt} attendus")

    print("[2/3] relevé sur les vidéos de BRUIT PUR ...")
    neg_cross: list = []
    dur_neg = 0.0
    for vid in a.neg:
        tneg, dur = collect(vid, a.zone_top, a.zone_bottom, a.min_frames_collect)
        nc = [t for t in tneg if t["dir"]]
        neg_cross.extend(nc)
        dur_neg += dur
        print(f"      {vid:<22} {len(nc)} franchissement(s) faux positif(s) sur {dur:.0f}s")
    print(f"      TOTAL {len(neg_cross)} faux positifs sur {dur_neg:.0f}s de bruit")

    # séparation des classes : que vaut chaque caractéristique ?
    pos_feats = [t["feat"] for t in labelled if t["label"] == 1]
    neg_feats = [t["feat"] for t in labelled if t["label"] == 0] + [t["feat"] for t in neg_cross]
    print(f"\n--- séparabilité des caractéristiques ---")
    print(f"  {'caract.':<10} {'vrais +':>12} {'faux +':>12}")
    for k in ("speed", "size_var", "fill", "axis", "dir"):
        p = np.array([f[k] for f in pos_feats]) if pos_feats else np.array([0.0])
        n = np.array([f[k] for f in neg_feats]) if neg_feats else np.array([0.0])
        print(f"  {k:<10} med={np.median(p):6.2f} p90={np.percentile(p, 90):6.2f}   "
              f"med={np.median(n):6.2f} p90={np.percentile(n, 90):6.2f}")

    print(f"\n[3/3] recherche de seuils ({np.prod([len(v) for v in GRID.values()]):.0f} combinaisons) ...")
    keys = list(GRID)
    best = None
    for combo in itertools.product(*(GRID[k] for k in keys)):
        thr = dict(zip(keys, combo))
        tp = sum(1 for t in labelled if t["label"] == 1 and keep(t, thr))
        fp = sum(1 for t in labelled if t["label"] == 0 and keep(t, thr)) \
            + sum(1 for t in neg_cross if keep(t, thr))
        rec = tp / max(1, n_gt)
        prec = tp / max(1, tp + fp)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        score = (f1, rec, -fp)
        if best is None or score > best[0]:
            best = (score, thr, tp, fp, rec, prec, f1)

    _, thr, tp, fp, rec, prec, f1 = best
    fp_min = fp / max(1e-9, dur_pos + dur_neg) * 60
    print(f"\n=== MEILLEUR JEUX DE SEUILS ===")
    for k in keys:
        v = thr[k]
        print(f"  {k:<20} : {v:.2f}" if isinstance(v, float) else f"  {k:<20} : {v}")
    print(f"\n  vrais positifs conservés : {tp}/{n_gt}  (rappel {rec * 100:.0f}%)")
    print(f"  faux positifs conservés  : {fp}  ({fp_min:.1f} / min)")
    print(f"  précision {prec * 100:.1f}%   F1 {f1 * 100:.1f}%")

    # comparaison avec les seuils actuels du code
    cur = {"min_frames": 5, "min_speed": 3.0, "max_size_var": 3.5, "min_fill": 0.30,
           "min_axis_ratio": 0.5, "min_dir_consistency": 0.7}
    ctp = sum(1 for t in labelled if t["label"] == 1 and keep(t, cur))
    cfp = sum(1 for t in labelled if t["label"] == 0 and keep(t, cur)) \
        + sum(1 for t in neg_cross if keep(t, cur))
    crec = ctp / max(1, n_gt)
    cprec = ctp / max(1, ctp + cfp)
    cf1 = 0.0 if (cprec + crec) == 0 else 2 * cprec * crec / (cprec + crec)
    print(f"\n  seuils ACTUELS du code : rappel {crec * 100:.0f}%  "
          f"précision {cprec * 100:.1f}%  F1 {cf1 * 100:.1f}%  "
          f"FP {cfp} ({cfp / max(1e-9, dur_pos + dur_neg) * 60:.1f}/min)")

    loose = {"min_frames": 3, "min_speed": 0.0, "max_size_var": 1e9, "min_fill": 0.0,
             "min_axis_ratio": 0.0, "min_dir_consistency": 0.0}
    ltp = sum(1 for t in labelled if t["label"] == 1 and keep(t, loose))
    lfp = sum(1 for t in labelled if t["label"] == 0 and keep(t, loose)) \
        + sum(1 for t in neg_cross if keep(t, loose))
    lrec = ltp / max(1, n_gt)
    lprec = ltp / max(1, ltp + lfp)
    lf1 = 0.0 if (lprec + lrec) == 0 else 2 * lprec * lrec / (lprec + lrec)
    print(f"  AUCUN filtre            : rappel {lrec * 100:.0f}%  "
          f"précision {lprec * 100:.1f}%  F1 {lf1 * 100:.1f}%  "
          f"FP {lfp} ({lfp / max(1e-9, dur_pos + dur_neg) * 60:.1f}/min)")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"thresholds": thr, "tp": tp, "fp": fp, "recall": rec,
                       "precision": prec, "f1": f1, "fp_per_min": fp_min},
                      fh, indent=2, ensure_ascii=False)
        print(f"\nsauvegardé : {a.out}")


if __name__ == "__main__":
    main()
