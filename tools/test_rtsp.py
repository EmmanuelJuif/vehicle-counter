from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.capture import FrameGrabber


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".ffmpeg_path")
        if os.path.exists(p):
            return open(p).read().strip()
    raise SystemExit("ffmpeg introuvable (pip install imageio-ffmpeg)")


def start_publisher(ff: str, source: str, url: str) -> subprocess.Popen:
    return subprocess.Popen(
        [ff, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
         "-i", source, "-an", "-c:v", "libx264", "-preset", "ultrafast",
         "-tune", "zerolatency", "-g", "25", "-f", "rtsp", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_stream(url: str, timeout: float = 15.0) -> bool:
    """Attend que le flux soit réellement disponible."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return True
        time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtsp://127.0.0.1:8554/cam")
    ap.add_argument("--source", default="data/eval.mp4", help="vidéo poussée dans le flux")
    ap.add_argument("--seconds", type=float, default=8.0, help="durée du test de débit")
    ap.add_argument("--slow", type=float, default=0.45,
                    help="temps de traitement simulé par frame (s)")
    ap.add_argument("--no-publish", action="store_true",
                    help="ne lance pas de publisher (flux déjà publié)")
    a = ap.parse_args()

    ff = ffmpeg_exe()
    pub = None
    if not a.no_publish:
        print(f"[0/4] publication de {a.source} vers {a.url} ...")
        pub = start_publisher(ff, a.source, a.url)
        if not wait_stream(a.url):
            pub.terminate()
            raise SystemExit(f"flux {a.url} indisponible — serveur RTSP lancé ?\n"
                             f"  python tools/rtsp_server.py --source data/eval.mp4")
        print("      flux disponible")

    results = {}

    # --- 1. détection du mode live -------------------------------------
    print("\n[1/4] détection du mode live ...")
    g = FrameGrabber(a.url)
    ok_live = g.is_live and g.maxlen == 1
    results["live"] = ok_live
    print(f"      is_live={g.is_live}  taille de file={g.maxlen}  "
          f"résolution={g.width}x{g.height}  -> {'OK' if ok_live else 'ÉCHEC'}")
    g.stop()

    # --- 2. débit -------------------------------------------------------
    print(f"\n[2/4] débit pendant {a.seconds:.0f}s (consommateur rapide) ...")
    g = FrameGrabber(a.url).start()
    n = 0
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        ok, frame = g.read(timeout=2.0)
        if not ok:
            break
        n += 1
    dur = time.time() - t0
    fps = n / max(1e-9, dur)
    results["fps_recv"] = fps
    results["frames"] = n
    print(f"      {n} frames en {dur:.1f}s -> {fps:.1f} FPS reçus")
    g.stop()

    # --- 3. consommateur lent : la latence doit rester bornée -----------
    print(f"\n[3/4] consommateur lent ({a.slow:.2f}s/frame) : la latence doit rester bornée ...")
    g = FrameGrabber(a.url).start()
    samples = []
    n_slow = 0
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        t_read = time.time()
        ok, frame = g.read(timeout=5.0)
        if not ok:
            break
        samples.append(time.time() - t_read)   # attente pour obtenir une frame
        time.sleep(a.slow)                      # inférence simulée
        n_slow += 1
    g.stop()
    if samples:
        arr = np.array(samples)
        results["latence_max"] = float(arr.max())
        results["latence_med"] = float(np.median(arr))
        print(f"      {n_slow} frames traitées en {time.time() - t0:.1f}s")
        print(f"      attente par frame : médiane {np.median(arr) * 1000:.0f} ms, "
              f"max {arr.max() * 1000:.0f} ms")
        print(f"      file maximale observée : {g.q.maxsize} (mode live -> 1)")
        bounded = arr.max() < 1.5
        results["latence_bornee"] = bool(bounded)
        print(f"      -> latence {'BORNÉE (OK)' if bounded else 'NON BORNÉE (ÉCHEC)'}")
    else:
        results["latence_bornee"] = False
        print("      aucune frame reçue")

    # --- 4. reconnexion automatique -------------------------------------
    print("\n[4/4] coupure du flux puis reprise (reconnexion automatique) ...")
    if a.no_publish:
        print("      (--no-publish : épreuve ignorée, pas de contrôle du publisher)")
        results["reconnexion"] = None
    else:
        g = FrameGrabber(a.url).start()
        for _ in range(5):
            ok, _ = g.read(timeout=2.0)
        before = g.read_index

        pub.terminate()
        try:
            pub.wait(timeout=5)
        except Exception:
            pub.kill()
        print("      publisher coupé — attente de 6s ...")
        time.sleep(6.0)
        mid_reads = 0
        for _ in range(5):
            ok, _ = g.read(timeout=1.0)
            mid_reads += 1 if ok else 0
        print(f"      pendant la coupure : {mid_reads}/5 lectures réussies (attendu 0)")

        print("      republication ...")
        pub = start_publisher(ff, a.source, a.url)
        wait_stream(a.url, timeout=20.0)
        recovered = 0
        t0 = time.time()
        while time.time() - t0 < 25.0:
            ok, frame = g.read(timeout=3.0)
            if ok and frame is not None:
                recovered += 1
                if recovered >= 5:
                    break
        after = g.read_index
        g.stop()
        results["reconnexion"] = recovered >= 5
        print(f"      {recovered} frames récupérées après republication "
              f"(index {before} -> {after}) -> "
              f"{'OK' if recovered >= 5 else 'ÉCHEC'}")
        pub.terminate()
        try:
            pub.wait(timeout=5)
        except Exception:
            pub.kill()

    # --- bilan -----------------------------------------------------------
    print("\n=== BILAN FLUX RTSP ===")
    for k, v in results.items():
        if v is None:
            print(f"  {k:<16} : non testé")
        elif isinstance(v, bool):
            print(f"  {k:<16} : {'OK' if v else 'ÉCHEC'}")
        elif isinstance(v, float):
            print(f"  {k:<16} : {v:.2f}")
        else:
            print(f"  {k:<16} : {v}")
    sys.exit(0 if all(v is not False for v in results.values()) else 1)


if __name__ == "__main__":
    main()
