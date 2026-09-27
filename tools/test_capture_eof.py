from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from src.capture import FrameGrabber  # noqa: E402


def true_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def bench_file(path: str, sleep: float = 0.004) -> tuple[int, int, int]:
    """Consomme lentement un fichier ; renvoie (attendu, lues, drops)."""
    g = FrameGrabber(path)
    g.start()
    got = 0
    while True:
        ok, _ = g.read(timeout=10.0)
        if not ok:
            break
        got += 1
        time.sleep(sleep)  # consumer plus lent que le producteur -> file pleine
    g.stop()
    return true_frame_count(path), got, g.drops


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "eval.mp4")
    if not os.path.exists(path):
        print(f"vidéo introuvable : {path}")
        return 1

    total = true_frame_count(path)
    print(f"fichier            : {os.path.basename(path)}  ({total} frames)")

    ok_all = True
    for sleep in (0.0, 0.004):
        label = "consumer rapide" if sleep == 0.0 else "consumer lent (file sature)"
        expected, got, drops = bench_file(path, sleep)
        ok = got == expected and drops == 0
        ok_all &= ok
        print(f"  {label:<32} attendu={expected:<5} lues={got:<5} "
              f"drops={drops:<3} -> {'OK' if ok else 'ECHEC'}")

    print("\nRÉSULTAT : " + ("OK — aucune frame perdue en fin de fichier"
                             if ok_all else "ECHEC — des frames sont encore perdues"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
