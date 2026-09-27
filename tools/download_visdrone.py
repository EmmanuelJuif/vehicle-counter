from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

REPO = "banu4prasad/VisDrone-Dataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"

# VisDrone -> COCO
MAP = {3: 2, 4: 2, 5: 7, 8: 5, 9: 3}
MOTOR = 9


def _get(url: str, timeout: int = 30) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _fetch_many(urls: list[tuple[str, str]], workers: int = 16) -> int:
    """(url, dest) -> nombre de téléchargements réussis."""
    ok = 0

    def one(item):
        url, dest = item
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return True
        data = _get(url)
        if data is None:
            return False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        return True

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, urls):
            ok += 1 if r else 0
    return ok


def list_files(prefix: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{REPO}?full=true"
    data = _get(url, timeout=60)
    if data is None:
        raise SystemExit("Impossible de lister le dépôt HuggingFace.")
    d = json.loads(data)
    return [s["rfilename"] for s in d.get("siblings", []) if s["rfilename"].startswith(prefix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/visdrone")
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--label-scan", type=int, default=900,
                    help="nb de fichiers d'annotations inspectés pour trouver des motos")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    t0 = time.time()
    print("[1/4] listing des annotations VisDrone train ...")
    all_lab = sorted(list_files("VisDrone2019-DET-train/labels/"))
    print(f"   {len(all_lab)} annotations disponibles")
    sample = all_lab[: a.label_scan]

    lab_dir = os.path.join(a.out, "_raw_labels")
    jobs = [(f"{BASE}/{p}", os.path.join(lab_dir, os.path.basename(p))) for p in sample]
    print(f"[2/4] téléchargement de {len(jobs)} annotations ...")
    n = _fetch_many(jobs, a.workers)
    print(f"   {n} récupérées ({time.time() - t0:.0f}s)")

    # images contenant au moins une moto
    # NB : ce dépôt HuggingFace fournit déjà les annotations au format YOLO
    #      (classe cx cy w h normalisé) avec l'indexation VisDrone d'origine.
    wanted = []
    for p in sample:
        fp = os.path.join(lab_dir, os.path.basename(p))
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, encoding="utf-8", errors="ignore") as fh:
                rows = [l.strip() for l in fh if l.strip()]
        except Exception:
            continue
        has_motor = False
        nveh = 0
        for r in rows:
            parts = r.replace(",", " ").split()
            if len(parts) < 5:
                continue
            try:
                c = int(float(parts[0]))
            except ValueError:
                continue
            if c == MOTOR:
                has_motor = True
            if c in MAP:
                nveh += 1
        if has_motor and nveh >= 2:
            wanted.append(os.path.basename(p)[:-4] + ".jpg")

    wanted = wanted[: a.max_images]
    print(f"[3/4] {len(wanted)} images contenant des motos -> téléchargement ...")
    img_dir = os.path.join(a.out, "images")
    jobs = [(f"{BASE}/VisDrone2019-DET-train/images/{n}", os.path.join(img_dir, n))
            for n in wanted]
    n = _fetch_many(jobs, a.workers)
    print(f"   {n} images récupérées ({time.time() - t0:.0f}s)")

    print("[4/4] conversion des annotations au format YOLO-COCO ...")
    lbl_out = os.path.join(a.out, "labels")
    os.makedirs(lbl_out, exist_ok=True)

    kept, nmotor = 0, 0
    for name in wanted:
        stem = name[:-4]
        src = os.path.join(lab_dir, stem + ".txt")
        img = os.path.join(img_dir, name)
        if not (os.path.exists(src) and os.path.exists(img)):
            continue
        # les annotations sont déjà normalisées : on remappe seulement les classes
        out_rows = []
        with open(src, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.replace(",", " ").split()
                if len(parts) < 5:
                    continue
                try:
                    c = int(float(parts[0]))
                    floats = [float(v) for v in parts[1:5]]
                except ValueError:
                    continue
                if c not in MAP:
                    continue
                cx, cy, w, h = floats
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and w > 0 and h > 0):
                    continue
                out_rows.append(f"{MAP[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                if c == MOTOR:
                    nmotor += 1
        if not out_rows:
            continue
        with open(os.path.join(lbl_out, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_rows) + "\n")
        kept += 1

    print(f"\n{kept} images, {nmotor} motos annotées -> {a.out}")
    print(f"durée totale {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()