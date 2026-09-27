from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.capture import FrameGrabber  # noqa: E402
from src.config import Config, DetectorConfig, FastVehicleConfig, GhostConfig, \
    MotorcycleConfig, OutputConfig, TrackerConfig, ZoneConfig  # noqa: E402
from src.geometry import iou_matrix  # noqa: E402
from src.pipeline import VehicleCountingPipeline  # noqa: E402


def build_cfg(a) -> Config:
    return Config(
        source=a.source,
        detector=DetectorConfig(weights=a.weights, imgsz=a.imgsz, conf=a.conf,
                                tiled=a.tiled, adaptive=False, min_fps=0.0),
        tracker=TrackerConfig(name=a.tracker, with_reid=a.with_reid,
                              gmc_method=a.gmc),
        zone=ZoneConfig(height=200, y_top=a.zone_top, y_bottom=a.zone_bottom,
                        ratio=a.zone_ratio),
        motorcycle=MotorcycleConfig(),
        fast=FastVehicleConfig(),
        ghost=GhostConfig(enable=False),   # hors sujet pour le tracking pur
        output=OutputConfig(display=False, show_tracks=False),
    )


def collect_tracks(a):
    """Joue la vidéo et renvoie {frame: [(id, box), ...]} pour les pistes YOLO."""
    cfg = build_cfg(a)
    grabber = FrameGrabber(cfg.source)
    grabber.start()
    ok, frame = grabber.read(timeout=10.0)
    if not ok:
        grabber.stop()
        raise SystemExit("source illisible")
    pipe = VehicleCountingPipeline(cfg)
    pipe.build(frame)

    out = {}
    fid = 0
    while True:
        ok, frame = grabber.read(timeout=10.0)
        if not ok or frame is None:
            break
        fid += 1
        tracks = pipe.process(frame)
        # les pistes "tuiles" portent un ID synthétique incrémental : elles
        # ne doivent pas entrer dans le calcul (ce sont des doublons).
        out[fid] = [(int(t["id"]), np.asarray(t["box"], dtype=float))
                    for t in tracks if not t.get("refined")]
        if a.max_frames and fid >= a.max_frames:
            break
    grabber.stop()
    return out, pipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/eval.mp4")
    ap.add_argument("--gt", default=None, help="JSON vérité terrain (défaut: <source>_gt.json)")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--tiled", default="off", choices=["off", "band", "full"])
    ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    ap.add_argument("--with-reid", action="store_true")
    ap.add_argument("--gmc", default="sparseOptFlow")
    ap.add_argument("--zone-top", type=int, default=380)
    ap.add_argument("--zone-bottom", type=int, default=580)
    ap.add_argument("--zone-ratio", type=float, default=0.6667)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU mini pour un appariement")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--from-cache", action="store_true", help="relit les prédictions en cache")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    gt_path = a.gt or (os.path.splitext(a.source)[0] + "_gt.json")
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)
    tracks_gt = gt.get("tracks")
    if not tracks_gt:
        raise SystemExit(
            f"{gt_path} ne contient pas de trajectoires ('tracks').\n"
            "Régénérez la vidéo : python tools/make_eval_video.py --out "
            f"{a.source} --frames 200 --seed 11")

    tag = f"{a.tracker}{'_reid' if a.with_reid else ''}_{a.tiled}_{a.imgsz}"
    cache = os.path.join(ROOT, "out", f"mot_preds_{tag}.json")

    print(f"MOT · {os.path.basename(a.source)} · tracker={a.tracker}"
          f"{'+reid' if a.with_reid else ''} · tiled={a.tiled} · imgsz={a.imgsz}")
    if a.from_cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            raw = json.load(fh)
        preds = {int(k): [(i, np.asarray(b, dtype=float)) for i, b in v]
                 for k, v in raw.items()}
        pipe = None
        print(f"  (prédictions relues depuis {os.path.basename(cache)})")
    else:
        preds, pipe = collect_tracks(a)
        # cache : recalculer les métriques ne doit pas rejouer la vidéo
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump({k: [[i, [float(x) for x in b]] for i, b in v]
                       for k, v in preds.items()}, fh)

    try:
        import motmetrics as mm
    except ImportError:
        raise SystemExit("pip install motmetrics")

    acc = mm.MOTAccumulator(auto_id=True)
    n_gt_total = 0
    for f in sorted(preds, key=int):
        g = tracks_gt.get(str(f), tracks_gt.get(f, []))
        if not g:
            continue
        gt_ids = [int(x["id"]) for x in g]
        gt_boxes = np.array([x["box"] for x in g], dtype=float)
        n_gt_total += len(gt_ids)

        items = preds[f]
        pr_ids = [t[0] for t in items]
        pr_boxes = np.array([t[1] for t in items], dtype=float) if items \
            else np.zeros((0, 4), dtype=float)

        if len(gt_boxes) and len(pr_boxes):
            iou = iou_matrix(gt_boxes, pr_boxes)
            dist = 1.0 - iou
            dist[iou < a.iou] = np.nan      # non appariable
        else:
            dist = np.full((len(gt_boxes), max(1, len(pr_boxes))), np.nan)
            pr_ids = pr_ids or []
        acc.update(gt_ids, pr_ids, dist if len(pr_boxes) else
                   np.zeros((len(gt_boxes), 0)))

    mh = mm.metrics.create()
    # NB : dans motmetrics >= 1.2 il n'y a ni `num_frames` ni `num_objects`
    # (ils renvoient NaN) — on les compte soi-même et on utilise
    # `num_unique_objects` pour le nombre d'identités vérité terrain.
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="MOT")
    # `compute` rend un DataFrame (1 ligne) : sans `.iloc[0]`, `summary['mota']`
    # renvoie une Series et `float()` lève — d'où des métriques silencieusement
    # à zéro si on avale l'exception. On normalise tout de suite.
    if hasattr(summary, "iloc"):
        summary = summary.iloc[0]
    n_frames = len(preds)
    n_gt_ids = len({int(x["id"]) for v in tracks_gt.values() for x in v})

    n_events = int(len(acc.mot_events)) if hasattr(acc, "mot_events") else 0
    print(f"  appariements enregistrés : {n_events}")

    def g(k, scale=100.0):
        """Lecture NaN-safe : une métrique non calculable vaut 0, pas NaN."""
        try:
            v = float(summary[k])
        except Exception:
            return 0.0
        return 0.0 if v != v else v * scale    # v != v  <=>  NaN

    res = {
        "source": a.source, "tracker": a.tracker, "with_reid": a.with_reid,
        "tiled": a.tiled, "imgsz": a.imgsz, "conf": a.conf,
        "frames": n_frames, "gt_objects": n_gt_ids,
        "mota": round(g("mota"), 2), "motp": round(g("motp"), 2),
        "idf1": round(g("idf1"), 2),
        "idp": round(g("idp"), 2), "idr": round(g("idr"), 2),
        "precision": round(g("precision"), 2),
        "recall": round(g("recall"), 2), "idsw": int(g("num_switches", 1)),
        "fragments": int(g("num_fragmentations", 1)),
        "misses": int(g("num_misses", 1)), "false_positives": int(g("num_false_positives", 1)),
        "mostly_tracked": int(g("mostly_tracked", 1)),
        "mostly_lost": int(g("mostly_lost", 1)),
        "counted": pipe.counter.summary()["grand_total"] if pipe is not None else None,
    }

    print("\n  métrique                 valeur")
    print("  -------------------------------")
    print(f"  MOTA  (précision suivi)  {res['mota']:>7.2f} %")
    # MOTP tel que calculé ici = distance moyenne (1 - IoU) : plus c'est BAS,
    # meilleur c'est. On affiche donc aussi l'IoU moyen, qui parle davantage.
    print(f"  MOTP  (1 - IoU moyen)    {res['motp']:>7.2f} %   -> IoU moyen "
          f"{100.0 - res['motp']:.2f} %")
    print(f"  IDF1  (identité)         {res['idf1']:>7.2f} %   <-- « un ID par véhicule »")
    print(f"     IDP / IDR             {res['idp']:>7.2f} % / {res['idr']:.2f} %")
    print(f"  précision                {res['precision']:>7.2f} %")
    print(f"  rappel                   {res['recall']:>7.2f} %")
    print(f"  ruptures d'identité      {res['idsw']:>7d}")
    print(f"  fragmentations           {res['fragments']:>7d}")
    print(f"  objets manqués           {res['misses']:>7d}")
    print(f"  fausses détections       {res['false_positives']:>7d}")
    print(f"  pistes >80% suivies      {res['mostly_tracked']:>7d}")
    print(f"  pistes <20% suivies      {res['mostly_lost']:>7d}")
    print(f"  vérité terrain           {res['gt_objects']} identités / {res['frames']} frames")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"\nécrit : {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
