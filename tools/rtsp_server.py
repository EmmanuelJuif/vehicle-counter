from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(ROOT, "tools", "rtsp")
MEDIAMTX = os.path.join(BIN_DIR, "mediamtx.exe")
VERSION = "v1.11.3"
ZIP_URL = ("https://github.com/bluenviron/mediamtx/releases/download/"
           f"{VERSION}/mediamtx_{VERSION}_windows_amd64.zip")


def install() -> str:
    import urllib.request

    os.makedirs(BIN_DIR, exist_ok=True)
    zpath = os.path.join(BIN_DIR, "mediamtx.zip")
    print(f"téléchargement de MediaMTX {VERSION} ...")
    urllib.request.urlretrieve(ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(BIN_DIR)
    os.remove(zpath)
    print(f"installé : {MEDIAMTX}")
    return MEDIAMTX


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        p = os.path.join(ROOT, ".ffmpeg_path")
        if os.path.exists(p):
            return open(p).read().strip()
    raise SystemExit("ffmpeg introuvable : pip install imageio-ffmpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="télécharge MediaMTX")
    ap.add_argument("--source", default=None, help="vidéo à publier en boucle")
    ap.add_argument("--url", default="rtsp://127.0.0.1:8554/cam")
    ap.add_argument("--no-publish", action="store_true", help="serveur seulement")
    a = ap.parse_args()

    if a.install or not os.path.exists(MEDIAMTX):
        install()
        if a.install and not a.source:
            return

    print(f"serveur RTSP : {a.url}  (Ctrl-C pour arrêter)")
    srv = subprocess.Popen([MEDIAMTX], cwd=BIN_DIR)
    pub = None
    if a.source and not a.no_publish:
        ff = ffmpeg_exe()
        src = a.source if os.path.isabs(a.source) else os.path.join(ROOT, a.source)
        pub = subprocess.Popen(
            [ff, "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
             "-i", src, "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-tune", "zerolatency", "-g", "25", "-f", "rtsp", a.url])
        print(f"publication de {src} ...")
    try:
        srv.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if pub is not None:
            pub.terminate()
        srv.terminate()


if __name__ == "__main__":
    main()
