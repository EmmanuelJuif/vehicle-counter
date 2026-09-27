from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, DetectorConfig, FastVehicleConfig, GhostConfig, \
    MotorcycleConfig, OutputConfig, TrackerConfig, ZoneConfig
from src.pipeline import VehicleCountingPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--tiled", choices=["off", "band", "full"], default="band")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    cfg = Config(
        source=a.source,
        detector=DetectorConfig(tiled=a.tiled, refine_interval=2, classes=(2, 3, 5, 7),
                                conf=0.10, iou=0.65, imgsz=640),
        tracker=TrackerConfig(),
        zone=ZoneConfig(height=200, ratio=0.62),
        motorcycle=MotorcycleConfig(),
        fast=FastVehicleConfig(),
        ghost=GhostConfig(),
        output=OutputConfig(display=False, save_video=None, csv=None, json=None,
                            window_scale=a.scale),
    )
    pipe = VehicleCountingPipeline(cfg)
    t0 = time.perf_counter()
    summary = pipe.run(max_frames=a.max_frames)
    dt = time.perf_counter() - t0
    print(f"\n{summary['frames']} frames en {dt:.1f}s ({summary['frames'] / dt:.2f} FPS)")
    print(summary["by_class"], summary["total"], "ghosts:", summary["ghosts"])


if __name__ == "__main__":
    main()