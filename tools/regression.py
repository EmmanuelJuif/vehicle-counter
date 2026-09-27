from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Console Windows par défaut en cp1252 : sans ça, un simple caractère accentué
# capturé depuis un sous-processus fait planter les print() (UnicodeEncodeError).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
BASELINE = os.path.join(ROOT, "out", "regression_baseline.json")

# métrique -> tolérance de baisse autorisée (en points)
TOLERANCE = {"moto_recall": 2.0, "mota": 2.0, "idf1": 3.0, "grand_total": 2.0}


def run(cmd, timeout=900):
    """Exécute une commande ; renvoie (code_retour, stdout+stderr)."""
    try:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace",
                           env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def check_tests(name, script):
    """Un test unitaire : on n'accepte que le code de sortie 0."""
    t = time.time()
    code, out = run([PY, os.path.join("tools", script)])
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
    print(f"  {name:<26} {'OK' if code == 0 else 'ECHEC':<7} "
          f"{time.time() - t:5.1f}s  {tail[0][:44]}")
    return code == 0, None


def check_metric(name, cmd, json_path, key, scale=100.0):
    """Un banc : lit la métrique dans le JSON produit par l'outil."""
    t = time.time()
    code, out = run(cmd)
    if code != 0:
        print(f"  {name:<26} {'ECHEC':<7} {time.time() - t:5.1f}s  (outil en erreur)")
        return False, None
    try:
        with open(os.path.join(ROOT, json_path), encoding="utf-8") as fh:
            val = float(json.load(fh)[key])
    except Exception as exc:
        print(f"  {name:<26} {'ECHEC':<7} {time.time() - t:5.1f}s  ({exc})")
        return False, None
    if scale != 1.0:
        val *= scale
    print(f"  {name:<26} {val:7.2f}   {time.time() - t:5.1f}s")
    return True, round(val, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="enregistre les valeurs courantes comme référence")
    ap.add_argument("--quick", action="store_true", help="tests unitaires seulement")
    ap.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args()

    print("=== TESTS UNITAIRES (mécanique) ===")
    results = {}
    ok_all = True
    for name, script in (("logique de zone", "test_zone_logic.py"),
                         ("intégration tracking", "test_tracking_integration.py"),
                         ("fin de fichier vidéo", "test_capture_eof.py")):
        ok, _ = check_tests(name, script)
        ok_all &= ok

    if not a.quick:
        print("\n=== BANCS DE PERFORMANCE ===")
        ok, v = check_metric(
            "rappel moto (%)",
            [PY, "tools/bench_motorcycle.py", "--json", "out/_ci_moto.json"],
            "out/_ci_moto.json", "recall")
        ok_all &= ok
        if ok:
            results["moto_recall"] = v

        # NB : `eval_mot.py` écrit déjà MOTA/IDF1 en pourcentage (24.23), alors
        # que `bench_motorcycle.py` écrit un ratio (0.237). D'où scale=1 ici et
        # scale=100 pour le rappel moto.
        ok, v = check_metric(
            "MOTA (%)",
            [PY, "tools/eval_mot.py", "--tracker", "bytetrack",
             "--json", "out/_ci_mot.json"],
            "out/_ci_mot.json", "mota", scale=1.0)
        ok_all &= ok
        if ok:
            results["mota"] = v

        ok, v = check_metric(
            "IDF1 (%)",
            [PY, "tools/eval_mot.py", "--tracker", "bytetrack", "--from-cache",
             "--json", "out/_ci_mot.json"],
            "out/_ci_mot.json", "idf1", scale=1.0)
        ok_all &= ok
        if ok:
            results["idf1"] = v

        ok, v = check_metric(
            "comptage eval.mp4",
            [PY, "main.py", "--source", "data/eval.mp4", "--no-display",
             "--tiled", "off", "--zone-top", "380", "--zone-bottom", "580",
             "--json", "out/_ci_count.json"],
            "out/_ci_count.json", "grand_total", scale=1.0)
        ok_all &= ok
        if ok:
            results["grand_total"] = v

    # --- baseline -------------------------------------------------------
    if a.update_baseline:
        os.makedirs(os.path.dirname(os.path.abspath(a.baseline)), exist_ok=True)
        with open(a.baseline, "w", encoding="utf-8") as fh:
            json.dump({"metrics": results}, fh, indent=2, ensure_ascii=False)
        print(f"\nbaseline enregistrée : {a.baseline}")
        for k, v in results.items():
            print(f"   {k:<14} {v}")
        return 0 if ok_all else 1

    if not os.path.exists(a.baseline):
        print(f"\nAucune baseline ({a.baseline}). Lancez --update-baseline.")
        return 1

    with open(a.baseline, encoding="utf-8") as fh:
        base = json.load(fh).get("metrics", {})

    print("\n=== COMPARAISON À LA BASELINE ===")
    print(f"  {'métrique':<14} {'base':>8} {'actuel':>8} {'tolér.':>7}  verdict")
    regressions = []
    for k, cur in results.items():
        if k not in base:
            print(f"  {k:<14} {'-':>8} {cur:>8} {'-':>7}  nouvelle métrique")
            continue
        b, tol = float(base[k]), TOLERANCE.get(k, 2.0)
        delta = cur - b
        bad = delta < -tol
        if bad:
            regressions.append((k, b, cur, tol))
        print(f"  {k:<14} {b:>8.2f} {cur:>8.2f} ±{tol:<6.1f} "
              f"{'REGRESSION' if bad else 'OK'}  ({delta:+.2f})")

    if regressions:
        print("\n*** RÉGRESSION DÉTECTÉE ***")
        for k, b, cur, tol in regressions:
            print(f"  {k} : {b} -> {cur} (tolérance {tol})")
        return 1
    if not ok_all:
        print("\n*** DES TESTS ÉCHOUENT ***")
        return 1
    print("\nAucune régression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
