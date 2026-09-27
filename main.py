from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import COCO_IDS, Config, DetectorConfig, FastVehicleConfig, GhostConfig, \
    MotorcycleConfig, OutputConfig, TrackerConfig, ZoneConfig
from src.pipeline import VehicleCountingPipeline


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comptage de véhicules YOLOv8 + ByteTrack")
    src = p.add_argument_group("source")
    src.add_argument("--source", default="0", help="fichier vidéo, flux RTSP/HTTP ou index webcam")
    src.add_argument("--scale", type=float, default=1.0, help="redimensionnement de l'image")
    src.add_argument("--max-frames", type=int, default=0, help="limite le nombre de frames (tests)")

    det = p.add_argument_group("détecteur")
    det.add_argument("--weights", default="yolov8s.pt")
    det.add_argument("--imgsz", type=int, default=640)
    det.add_argument("--conf", type=float, default=0.10)
    det.add_argument("--iou", type=float, default=0.65, help="IoU NMS (élevé = petites boîtes conservées)")
    det.add_argument("--max-det", type=int, default=300)
    det.add_argument("--device", default="auto", help="auto | cpu | 0 | 1 ...")
    det.add_argument("--no-half", action="store_true")
    det.add_argument("--classes", nargs="+", default=["car", "motorcycle", "truck", "bus"])
    det.add_argument("--tiled", choices=["off", "band", "full"], default="band",
                     help="inférence en tuiles : rattrape les petites motos")
    det.add_argument("--tile-size", type=int, default=640)
    det.add_argument("--refine-interval", type=int, default=2, help="tuiles 1 frame sur N")
    det.add_argument("--no-adaptive", action="store_true")
    det.add_argument("--min-fps", type=float, default=12.0,
                     help="FPS en dessous duquel le tiling est auto-désactivé "
                          "(0 = jamais désactivé). Le tiling double le rappel moto "
                          "mais coûte ~3x plus cher : sur CPU il faut arbitrer.")

    tr = p.add_argument_group("tracker")
    tr.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack",
                    help="backend de suivi : botsort = IDs plus stables par véhicule")
    tr.add_argument("--with-reid", action="store_true",
                    help="BoT-SORT : ajoute le ReID par apparence (IDs encore plus stables, plus lent)")
    tr.add_argument("--gmc", default="sparseOptFlow",
                    choices=["none", "sparseOptFlow", "orb", "sift", "ecc"],
                    help="compensation du mouvement de caméra (BoT-SORT)")
    tr.add_argument("--track-buffer", type=int, default=60)
    tr.add_argument("--min-hits", type=int, default=2)
    tr.add_argument("--match-thresh", type=float, default=0.85)
    tr.add_argument("--fps", type=int, default=25)

    z = p.add_argument_group("counting zone")
    z.add_argument("--zone-height", type=int, default=200)
    z.add_argument("--zone-ratio", type=float, default=0.62)
    z.add_argument("--zone-orientation", choices=["horizontal", "vertical"], default="horizontal")
    z.add_argument("--zone-top", type=int, default=None)
    z.add_argument("--zone-bottom", type=int, default=None)

    m = p.add_argument_group("motos / véhicules rapides")
    m.add_argument("--no-moto-boost", action="store_true")
    m.add_argument("--no-fast-fix", action="store_true")
    m.add_argument("--no-ghost", action="store_true", help="désactive le filet mouvement")
    m.add_argument("--ghost-lax", action="store_true",
                   help="désactive les filtres anti-bruit du filet (mode permissif)")
    m.add_argument("--ghost-min-frames", type=int, default=3,
                   help="frames observées avant de compter un blob (anti-bruit)")
    m.add_argument("--ghost-min-speed", type=float, default=0.0,
                   help="vitesse moyenne mini en px/frame pour un blob")
    m.add_argument("--ghost-min-dir", type=float, default=0.90,
                   help="cohérence de direction mini d'un blob (le seul filtre discriminant)")

    o = p.add_argument_group("sorties")
    o.add_argument("--no-display", action="store_true")
    o.add_argument("--save-video", default=None)
    o.add_argument("--csv", default=None)
    o.add_argument("--json", default=None)
    o.add_argument("--no-trails", action="store_true")
    o.add_argument("--benchmark", action="store_true", help="mesure les temps d'inférence puis quitte")
    return p.parse_args(argv)


def build_config(a: argparse.Namespace) -> Config:
    classes = tuple(COCO_IDS[c] for c in a.classes if c in COCO_IDS) or (2, 3, 5, 7)
    return Config(
        source=a.source,
        detector=DetectorConfig(
            weights=a.weights, imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=a.max_det,
            device=a.device, half=not a.no_half, classes=classes, tiled=a.tiled,
            tile_size=a.tile_size, refine_interval=a.refine_interval, adaptive=not a.no_adaptive,
            min_fps=a.min_fps,
        ),
        tracker=TrackerConfig(
            name=a.tracker, with_reid=a.with_reid, gmc_method=a.gmc,
            track_buffer=a.track_buffer, min_hits=a.min_hits,
            match_thresh=a.match_thresh, frame_rate=a.fps,
        ),
        zone=ZoneConfig(
            height=a.zone_height, ratio=a.zone_ratio, orientation=a.zone_orientation,
            y_top=a.zone_top, y_bottom=a.zone_bottom,
        ),
        motorcycle=MotorcycleConfig(boost=not a.no_moto_boost),
        fast=FastVehicleConfig(enable=not a.no_fast_fix),
        ghost=GhostConfig(
            enable=not a.no_ghost,
            strict=not a.ghost_lax,
            min_frames=a.ghost_min_frames,
            min_speed=a.ghost_min_speed,
            min_dir_consistency=a.ghost_min_dir,
        ),
        output=OutputConfig(
            display=not a.no_display, save_video=a.save_video, csv=a.csv, json=a.json,
            show_tracks=not a.no_trails, window_scale=a.scale,
        ),
    )


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = build_config(a)

    if a.benchmark:
        import cv2
        from src.capture import FrameGrabber
        cap = FrameGrabber(cfg.source)
        ok, frame = cap.read(timeout=5.0)
        cap.stop()
        if not ok:
            print("source illisible")
            return 1
        from src.detector import VehicleDetector
        d = VehicleDetector(cfg.detector)
        print(json.dumps(d.benchmark(frame), indent=2))
        return 0

    pipe = VehicleCountingPipeline(cfg)
    summary = pipe.run(max_frames=a.max_frames)

    print("\n=== RÉSULTAT ===")
    print(f"frames          : {summary['frames']}   fps moyen : {summary['fps_mean']}")
    print(f"zone            : {summary['zone']}")
    for cls, per in sorted(summary["by_class"].items()):
        print(f"  {cls:<11}: {per}")
    print(f"TOTAL           : {summary['total']}  => {summary['grand_total']} véhicules")
    print(f"IDs uniques     : {summary['unique_ids']}   (récup. mouvement : {summary['ghosts']})")
    print(f"IDs créés       : {summary['ids_created']} par le suiveur · "
          f"ruptures d'identité : {summary['id_switches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
