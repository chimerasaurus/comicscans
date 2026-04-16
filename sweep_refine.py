#!/usr/bin/env python3
"""Sweep refinement parameters against holdout.

Caches CNN predictions once, then re-runs refine_corners across a grid of
(min_confidence, search_in, strip_in) values. Reports the best by mean error.
"""
import cv2
import numpy as np
import torch
from itertools import product

from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, refine_corners, MODEL_FILE,
)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = _load_model(MODEL_FILE, device)
    holdout_dirs = ckpt.get("holdout_dirs", [])
    entries = _load_entries()
    _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []), holdout_dirs)
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Holdout: {holdout_dirs}  {len(eval_entries)} pages")

    # Cache: image, gt, cnn_pred
    cache = []
    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None:
            continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cnn_pred = predict_corners(model, device, img, rotate180=False)
        cache.append({
            "img": img,
            "gt": entry["gt_corners"],
            "cnn": cnn_pred,
            "dpi": entry.get("dpi", 600),
        })
    print(f"Cached CNN predictions for {len(cache)} pages.\n")

    # Baseline CNN-only
    cnn_dists = [
        np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(c["cnn"], c["gt"])])
        for c in cache
    ]
    print(f"CNN-only baseline:  mean={np.mean(cnn_dists):.2f}  median={np.median(cnn_dists):.2f}\n")

    # Grid
    min_conf_vals = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    search_in_vals = [0.10, 0.125, 0.15, 0.20]
    strip_in_vals = [0.15]

    results = []
    for mc, si, st in product(min_conf_vals, search_in_vals, strip_in_vals):
        dists = []
        improved = 0
        worsened = 0
        for c, cnn_d in zip(cache, cnn_dists):
            hyb = refine_corners(c["img"], c["cnn"], dpi=c["dpi"],
                                 search_in=si, strip_in=st, min_confidence=mc)
            d = np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(hyb, c["gt"])])
            dists.append(d)
            if d < cnn_d: improved += 1
            elif d > cnn_d: worsened += 1
        results.append({
            "min_conf": mc, "search_in": si, "strip_in": st,
            "mean": float(np.mean(dists)),
            "median": float(np.median(dists)),
            "p95": float(np.percentile(dists, 95)),
            "improved": improved, "worsened": worsened,
        })

    results.sort(key=lambda r: r["mean"])
    print(f"{'min_conf':>8s} {'search':>7s} {'strip':>6s}   {'mean':>7s} {'median':>7s} {'p95':>7s}  {'±':>9s}")
    for r in results:
        print(f"{r['min_conf']:>8.2f} {r['search_in']:>7.3f} {r['strip_in']:>6.3f}   "
              f"{r['mean']:>7.2f} {r['median']:>7.2f} {r['p95']:>7.2f}  "
              f"{r['improved']:>3d}/{r['worsened']:<3d}")

    best = results[0]
    print(f"\nBest: min_conf={best['min_conf']}  search_in={best['search_in']}  "
          f"strip_in={best['strip_in']}  mean={best['mean']:.2f}  "
          f"({best['improved']} improved, {best['worsened']} worsened)")


if __name__ == "__main__":
    main()
