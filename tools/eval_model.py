from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRACKED = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "visdrone", "dataset.yaml"))
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None, help="écrit le résultat dans ce fichier")
    a = ap.parse_args()

    from ultralytics import YOLO

    r = YOLO(a.weights).val(data=a.data, imgsz=a.imgsz, batch=a.batch,
                            device=a.device, workers=2, verbose=False, plots=False)
    # PIÈGE : `r.box.ap50` ne contient QUE les classes présentes dans le split
    # (ici 4 entrées), et l'ordre n'est pas l'ID de classe. Les vrais IDs sont
    # dans `r.box.ap_class_index` (ici [2, 3, 5, 7]). Indexer `ap50` par
    # position faisait passer bus/truck pour car/motorcycle.
    import numpy as np

    ap = np.asarray(r.box.ap50).ravel()
    ids = [int(c) for c in np.asarray(r.box.ap_class_index).ravel()]
    per = {}
    for cid, v in zip(ids, ap):
        if cid in TRACKED:
            per[TRACKED[cid]] = round(float(v), 4)

    out = {
        "weights": a.weights,
        "imgsz": a.imgsz,
        "map50": round(float(r.box.map50), 4),
        "map": round(float(r.box.map), 4),
        "ap50": per,
        "ap_class_index": ids,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nécrit : {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
