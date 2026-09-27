from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, FastVehicleConfig, MotorcycleConfig, OutputConfig, TrackerConfig
from src.counting_zone import CountingZone, ZoneCounter
from src.pipeline import VehicleCountingPipeline


class FakeDetector:
    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def detect_and_track(self, frame: np.ndarray):
        if self._i >= len(self._frames):
            return []
        d = self._frames[self._i]
        self._i += 1
        out = []
        for row in d:
            x1, y1, x2, y2, conf, cls, tid = row
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            out.append({"id": int(tid), "box": np.array([x1, y1, x2, y2], dtype=float),
                        "cls": "car" if int(cls) == 2 else "motorcycle",
                        "conf": float(conf), "center": (cx, cy)})
        return out


def main():
    seq = []
    for i in range(10):
        y1 = 150 + i * 60
        y2 = 700 - i * 60
        seq.append(np.array([
            [360, y1 - 25, 440, y1 + 25, 0.85, 2, 1],
            [782, y2 - 22, 818, y2 + 22, 0.55, 3, 2],
        ], dtype=float))

    pipe = VehicleCountingPipeline.__new__(VehicleCountingPipeline)
    pipe.cfg = Config(output=OutputConfig(display=False))
    pipe.detector = FakeDetector(seq)
    pipe.counter = ZoneCounter(CountingZone(280, 480, "y"),
                                TrackerConfig(min_hits=2, track_buffer=60),
                                MotorcycleConfig(), FastVehicleConfig(), 1280 * 720)
    pipe.zone = pipe.counter.zone
    pipe.frame_id = 0
    pipe.trails = {}
    pipe.avg_fps = 0
    pipe.ghost = type("G", (), {"update": lambda *a, **k: []})()
    pipe.vis = type("V", (), {"draw": lambda *a, **k: None})()
    pipe._tile_refine = lambda *a, **k: []

    frame_blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(len(seq)):
        pipe.process(frame_blank)
    s = pipe.counter.summary()
    assert s["grand_total"] == 2, f"attendu 2, obtenu {s['grand_total']}"
    assert sorted(s["by_class"]) == ["car", "motorcycle"]
    print(f"  ok  2 véhicules Track IDs {sorted(pipe.counter.counted_ids)}")
    print(f"\nIntégration pipeline : OK")


if __name__ == "__main__":
    main()