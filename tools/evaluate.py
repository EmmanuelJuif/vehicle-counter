from __future__ import annotations

import argparse
import json


def match(gt_items, pred_items, tol=12):
    """Appariement glouton : même sens + écart de frame minimal sous `tol`."""
    used_p = set()
    pairs = []
    fps_gt = []
    for gi, g in enumerate(gt_items):
        best, bd = None, tol + 1
        for pi, p in enumerate(pred_items):
            if pi in used_p:
                continue
            if p["direction"] != g["dir"]:
                continue
            d = abs(p["frame"] - g["frame"])
            if d < bd:
                bd = d
                best = pi
        if best is not None:
            used_p.add(best)
            pairs.append((gi, best, bd))
        else:
            fps_gt.append(gi)
    fps = [pi for pi in range(len(pred_items)) if pi not in used_p]
    return pairs, fps_gt, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=int, default=12, help="tolérance en frames")
    a = ap.parse_args()

    with open(a.pred, encoding="utf-8") as fh:
        pred = json.load(fh)
    with open(a.gt, encoding="utf-8") as fh:
        gt = json.load(fh)

    gt_items = gt.get("items", [])
    pred_items = pred.get("events", [])

    print(f"GT  : {len(gt_items)} franchissements   "
          f"PRÉD: {len(pred_items)}   (vidéo {pred.get('frames')} frames, "
          f"{pred.get('fps_mean')} FPS)")

    # --- A) appariement global (sans classe) ------------------------------
    pairs, missed, extra = match(gt_items, pred_items, a.tol)
    tp, fn, fp = len(pairs), len(missed), len(extra)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    print("\n--- DÉTECTION DE FRANCHISSEMENT (toute classe) ---")
    print(f"  vrais positifs   : {tp}")
    print(f"  faux négatifs    : {fn}   (véhicules réels non comptés)")
    print(f"  faux positifs    : {fp}   (comptages surnuméraires)")
    print(f"  précision : {prec * 100:.1f}%    rappel : {rec * 100:.1f}%    F1 : {f1 * 100:.1f}%")

    # --- B) justesse de la classe sur les paires appariées ----------------
    ok_cls = sum(1 for gi, pi, _ in pairs if gt_items[gi]["cls"] == pred_items[pi]["class"])
    if pairs:
        print(f"\n--- CLASSE (sur {len(pairs)} paires appariées) ---")
        print(f"  correctes : {ok_cls}/{len(pairs)}  ({ok_cls / len(pairs) * 100:.1f}%)")
        confusions = {}
        for gi, pi, _ in pairs:
            g, p = gt_items[gi]["cls"], pred_items[pi]["class"]
            if g != p:
                confusions[(g, p)] = confusions.get((g, p), 0) + 1
        for (g, p), n in sorted(confusions.items(), key=lambda kv: -kv[1]):
            print(f"    {g} -> {p} : {n}")

    # --- C) par classe (sommes) -------------------------------------------
    print("\n--- TOTAUX PAR CLASSE ---")
    print(f"{'classe':<12}{'GT':>5}{'pred':>6}{'delta':>7}")
    print("-" * 30)
    classes = sorted(set(list(gt.get("counts", {})) + list(pred.get("by_class", {}))))
    tg = tp_ = 0
    for c in classes:
        g = sum(gt.get("counts", {}).get(c, {}).values())
        p = sum(pred.get("by_class", {}).get(c, {}).values())
        tg += g
        tp_ += p
        print(f"{c:<12}{g:>5}{p:>6}{p - g:>+7}")
    print("-" * 30)
    print(f"{'TOTAL':<12}{tg:>5}{tp_:>6}{tp_ - tg:>+7}")

    # --- D) méthodes -------------------------------------------------------
    methods = {}
    for e in pred_items:
        methods[e["method"]] = methods.get(e["method"], 0) + 1
    if methods:
        print("\nméthode de comptage :", methods)
        print("  yolo         = véhicule vu plusieurs frames dans la zone")
        print("  extrapolated = trajectoire prolongée (perte de piste / très rapide)")
        print("  ghost        = récupéré par la soustraction de fond")

    if extra:
        print(f"\nfaux positifs aux frames : {sorted(pred_items[i]['frame'] for i in extra)}")
    if missed:
        print(f"véhicules manqués aux frames : {sorted(gt_items[i]['frame'] for i in missed)}")


if __name__ == "__main__":
    main()