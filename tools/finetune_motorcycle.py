from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def write_yaml(root: str, names: dict) -> str:
    """Écrit le fichier dataset YAML attendu par Ultralytics.

    IMPORTANT : train et val doivent pointer vers des dossiers **disjoints**.
    Une version antérieure pointait les deux vers `images/`, ce qui faisait
    valider sur les images d'entraînement et gonflait tous les scores.
    """
    root = os.path.abspath(root)
    path = os.path.join(root, "dataset.yaml")
    content = (
        f"path: {root}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"names:\n"
        + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items()))
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def split_val(root: str, ratio: float = 0.2) -> tuple[str, str]:
    """Sépare images/ en train/ et val/ (Ultralytics attend deux dossiers)."""
    import random

    img_dir = os.path.join(root, "images")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    if not files:
        raise SystemExit(f"Aucune image dans {img_dir}")
    random.Random(0).shuffle(files)
    n_val = max(1, int(len(files) * ratio))
    val = set(files[:n_val])

    for split, names in (("train", [f for f in files if f not in val]), ("val", list(val))):
        d = os.path.join(root, split, "images")
        os.makedirs(d, exist_ok=True)
        for f in names:
            dst = os.path.join(d, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(img_dir, f), dst)

    lbl_dir = os.path.join(root, "labels")
    for split in ("train", "val"):
        d = os.path.join(root, split, "labels")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(os.path.join(root, split, "images")):
            stem = os.path.splitext(f)[0]
            src = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(d, stem + ".txt"))
    return os.path.join(root, "train"), os.path.join(root, "val")


def pseudo_label_video(video: str, out_root: str, weights: str, conf: float,
                       max_frames: int, every: int) -> int:
    """Self-training : extrait des crops annotés depuis une vidéo."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "labels"), exist_ok=True)
    model = YOLO(weights)
    cap = cv2.VideoCapture(video)
    n = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        idx += 1
        if idx % every:
            continue
        r = model.predict(frame, imgsz=640, conf=conf, iou=0.65,
                          classes=[2, 3, 5, 7], device="cpu", verbose=False)
        boxes = r[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        h, w = frame.shape[:2]
        rows = []
        for b in boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            c = int(b.cls.item())
            rows.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                        f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        name = f"f{idx:06d}.jpg"
        cv2.imwrite(os.path.join(out_root, "images", name), frame)
        with open(os.path.join(out_root, "labels", name[:-4] + ".txt"), "w") as fh:
            fh.write("\n".join(rows) + "\n")
        n += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dossier YOLO (images/ + labels/)")
    ap.add_argument("--from-video", default=None, help="pseudo-étiquetage d'une vidéo")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freeze", type=int, default=10, help="couches gelées (backbone)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--project", default="runs/motorcycle")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--eval-video", default=None, help="vidéo CCTV de contrôle")
    ap.add_argument("--eval-gt", default=None)
    ap.add_argument("--skip-train", action="store_true", help="évalue seulement")
    a = ap.parse_args()

    # Ultralytics exige des indices de classes contigus (0..N-1). On garde donc
    # l'indexation COCO complète 0..7 : seules 4 classes ont des instances, mais
    # les IDs restent identiques à ceux du modèle pré-entraîné -> aucun remapping
    # côté pipeline.
    names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
             4: "airplane", 5: "bus", 6: "train", 7: "truck"}
    tracked = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # --- données ----------------------------------------------------------
    if a.from_video:
        root = a.data or "data/selftrain"
        print(f"[data] pseudo-étiquetage de {a.from_video} ...")
        n = pseudo_label_video(a.from_video, root, a.weights, 0.35, 0, 3)
        print(f"[data] {n} images générées dans {root}")
        a.data = root
    if not a.data:
        raise SystemExit("Il faut --data ou --from-video")

    print(f"[data] split train/val ...")
    split_val(a.data, a.val_ratio)
    yaml_path = write_yaml(a.data, names)
    n_train = len(os.listdir(os.path.join(a.data, "train", "images")))
    n_val = len(os.listdir(os.path.join(a.data, "val", "images")))
    print(f"[data] {n_train} train / {n_val} val -> {yaml_path}")

    from ultralytics import YOLO

    def metrics(model, tag):
        r = model.val(data=yaml_path, imgsz=a.imgsz, batch=a.batch, device=a.device,
                      workers=a.workers, verbose=False, plots=False)
        try:
            m = {"map50": float(r.box.map50), "map": float(r.box.map)}
            # `ap50` ne contient que les classes présentes ; les vrais IDs sont
            # dans `ap_class_index`. Indexer par position renvoyait les
            # mauvaises classes (bus/truck à la place de car/motorcycle).
            import numpy as np

            ap = np.asarray(r.box.ap50).ravel()
            ids = [int(c) for c in np.asarray(r.box.ap_class_index).ravel()]
            m["ap50"] = {tracked[c]: float(v) for c, v in zip(ids, ap) if c in tracked}
        except Exception as e:
            m = {"error": str(e)}
        print(f"[{tag}] mAP50={m.get('map50', 0):.3f}  mAP={m.get('map', 0):.3f}  {m.get('ap50', {})}")
        return m

    # --- baseline ---------------------------------------------------------
    print("\n=== MODÈLE DE BASE ===")
    base = YOLO(a.weights)
    m_before = metrics(base, "avant")

    if a.skip_train:
        return

    # --- entraînement -----------------------------------------------------
    print(f"\n=== FINE-TUNING ({a.epochs} epochs, imgsz={a.imgsz}, freeze={a.freeze}) ===")
    model = YOLO(a.weights)
    model.train(
        data=yaml_path,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        exist_ok=True,
        # --- réglages "petits objets / motos" ---
        mosaic=1.0,        # crée des objets plus petits -> meilleur rappel loin
        copy_paste=0.3,    # sur-représente les classes rares (motos)
        mixup=0.1,
        scale=0.5,         # jitter d'échelle
        fliplr=0.5,
        close_mosaic=max(1, a.epochs // 5),
        # --- anti-oubli ---
        lr0=a.lr,
        lrf=a.lr / 10,
        warmup_epochs=2.0,
        freeze=a.freeze if a.epochs > 3 else 0,
        patience=max(5, a.epochs // 3),
        cos_lr=True,
        verbose=True,
    )

    # Ultralytics n'écrit PAS forcément dans `project/name` : si `project` est
    # relatif il est imbriqué sous le dossier de runs de la tâche (on obtient
    # par ex. runs/detect/runs/motorcycle/ft416). Le seul endroit fiable pour
    # connaître le dossier réel est `model.trainer.save_dir`.
    save_dir = str(getattr(getattr(model, "trainer", None), "save_dir", "") or "")
    best = os.path.join(save_dir, "weights", "best.pt") if save_dir \
        else os.path.join(a.project, a.name, "weights", "best.pt")
    trained = YOLO(best) if os.path.exists(best) else model
    print(f"\n=== MODÈLE FINE-TUNÉ ({best if os.path.exists(best) else 'last'}) ===")
    m_after = metrics(trained, "après")

    # --- contrôle sur vidéo CCTV -------------------------------------------
    report = {"before": m_before, "after": m_after, "weights": str(best)}
    if a.eval_video:
        from src.config import (Config, DetectorConfig, FastVehicleConfig, GhostConfig,
                                MotorcycleConfig, OutputConfig, TrackerConfig, ZoneConfig)
        from src.pipeline import VehicleCountingPipeline

        print(f"\n=== COMPTAGE SUR {a.eval_video} ===")
        res = {}
        for tag, w in (("avant", a.weights), ("après", str(best))):
            cfg = Config(
                source=a.eval_video,
                detector=DetectorConfig(weights=w, tiled="off", conf=0.10, iou=0.65, imgsz=640),
                tracker=TrackerConfig(),
                zone=ZoneConfig(height=200, ratio=0.6667),
                motorcycle=MotorcycleConfig(),
                fast=FastVehicleConfig(),
                ghost=GhostConfig(enable=False),
                output=OutputConfig(display=False, window_scale=1.0),
            )
            s = VehicleCountingPipeline(cfg).run()
            res[tag] = {"total": s["grand_total"], "by_class": s["by_class"]}
            print(f"  {tag:<6} total={s['grand_total']}  {s['by_class']}")
        report["counting"] = res
        if a.eval_gt:
            with open(a.eval_gt, encoding="utf-8") as fh:
                gt = json.load(fh)
            report["counting"]["gt"] = gt["total"]

    out = os.path.join(a.project, a.name, "finetune_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nrapport : {out}")


if __name__ == "__main__":
    main()