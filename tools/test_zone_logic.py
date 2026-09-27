from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import FastVehicleConfig, MotorcycleConfig, TrackerConfig
from src.counting_zone import CountingZone, ZoneCounter


def make_counter(top=300, bottom=500, axis="y"):
    zone = CountingZone(top, bottom, axis)
    return ZoneCounter(
        zone,
        TrackerConfig(min_hits=2, track_buffer=60),
        MotorcycleConfig(),
        FastVehicleConfig(),
        1280 * 720,
    )


def track(tid, cx, cy, w=80, h=60, cls="car", conf=0.6):
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)
    return {"id": tid, "box": box, "cls": cls, "conf": conf, "center": (cx, cy)}


def feed(counter, tid, ys, cls="car", start_frame=0, x=400, w=80, h=60):
    """Pousse une trajectoire verticale, retourne les événements."""
    evs = []
    for i, y in enumerate(ys):
        evs += counter.update([track(tid, x, y, w, h, cls)], start_frame + i)
    return evs


def test_slow_vehicle():
    c = make_counter()
    evs = feed(c, 1, [200, 260, 320, 380, 440, 520, 580])
    assert len(evs) == 1, f"attendu 1 comptage, obtenu {len(evs)}"
    assert evs[0].direction == "down"
    assert evs[0].cls == "car"
    assert 1 in c.counted_ids
    print("  ok  véhicule lent")


def test_no_duplicate():
    c = make_counter()
    evs = feed(c, 7, [200, 300, 400, 500, 600, 700, 800, 900])
    assert len(evs) == 1, "un même Track ID ne doit être compté qu'une fois"
    assert len(c.counted_ids) == 1
    assert isinstance(c.counted_ids, set), "les IDs comptés doivent être stockés dans un set()"
    print("  ok  pas de doublon (set de Track IDs)")


def test_fast_vehicle_jumps_over_zone():
    """Le véhicule passe de y=150 à y=650 en une frame : aucune frame observée
    dans la bande 300-500, mais le segment de mouvement la traverse."""
    c = make_counter()
    evs = feed(c, 2, [100, 150, 650])
    assert len(evs) == 1, f"franchissement rapide manqué : {len(evs)}"
    assert evs[0].direction == "down"
    assert evs[0].method == "yolo"
    print("  ok  véhicule rapide sautant la zone")


def test_fast_vehicle_upward():
    c = make_counter()
    evs = feed(c, 3, [900, 700, 200])
    assert len(evs) == 1
    assert evs[0].direction == "up"
    print("  ok  véhicule rapide sens inverse")


def test_lost_before_zone_extrapolation():
    """Vu à 100 puis 350 (dans la bande, avant la médiane) puis perdu."""
    c = make_counter()
    evs = feed(c, 4, [100, 350])
    assert not evs, "ne doit pas compter avant certitude"
    evs = c.update([], 10)  # la piste disparaît
    assert len(evs) == 1, f"extrapolation attendue, obtenu {len(evs)}"
    assert evs[0].method == "extrapolated"
    assert evs[0].direction == "down"
    print("  ok  extrapolation à la perte de piste")


def test_enters_then_stops_not_counted():
    c = make_counter()
    evs = feed(c, 5, [200, 320, 350, 352, 351, 353, 352])
    assert not evs, "un véhicule qui s'arrête dans la zone ne franchit pas"
    print("  ok  arrêt dans la zone -> non compté")


def test_motorcycle_class_stickiness():
    c = make_counter()
    evs = feed(c, 6, [180, 240, 330, 430, 520, 610], cls="motorcycle", w=26, h=34)
    assert len(evs) == 1
    assert evs[0].cls == "motorcycle"
    print("  ok  moto comptée avec sa classe")


def test_motorcycle_confused_with_car():
    """Le détecteur hésite : car / motorcycle / car / motorcycle."""
    c = make_counter()
    evs = []
    seq = [("car", 0.30), ("motorcycle", 0.35), ("car", 0.28), ("motorcycle", 0.31)]
    ys = [200, 300, 420, 560]
    for (cls, conf), y in zip(seq, ys):
        t = track(9, 400, y, 30, 38, cls, conf)
        evs += c.update([t], len(evs))
    assert len(evs) == 1
    assert evs[0].cls == "motorcycle", f"classe retenue {evs[0].cls}"
    print("  ok  classe stabilisée par vote temporel")


def test_vertical_zone():
    c = make_counter(top=500, bottom=700, axis="x")
    evs = []
    for i, x in enumerate([300, 420, 560, 680, 820]):
        box = np.array([x - 40, 200, x + 40, 280], dtype=float)
        evs += c.update([{"id": 11, "box": box, "cls": "car", "conf": .8, "center": (x, 240)}], i)
    assert len(evs) == 1 and evs[0].direction == "right"
    print("  ok  zone verticale")


def test_two_vehicles_same_frame():
    c = make_counter()
    evs = []
    for i, (y1, y2) in enumerate([(200, 800), (300, 700), (420, 560), (520, 420), (640, 300)]):
        evs += c.update([track(20, 300, y1), track(21, 900, y2)], i)
    assert len(evs) == 2
    dirs = {e.direction for e in evs}
    assert dirs == {"down", "up"}, dirs
    print("  ok  deux véhicules, deux directions")


def main():
    print("Tests de la logique de comptage")
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ECHEC {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERREUR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
