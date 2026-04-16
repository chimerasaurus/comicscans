#!/usr/bin/env python3
"""Bleed-aware refinement: adds guards so the edge-snap step doesn't chase
gradients inside the printed bleed.

Failure mode being addressed: the current `refine_corners` finds the strongest
smoothed gradient in the search window. When a page has print bleed reaching
its edge, a strong interior gradient (panel border, word balloon, a line of
dark art) can beat the true paper→bed transition — especially when the page
corner lies near that content. The snap then drags the corner inward by
50-140 px (see diagnose_failures output; max_snap=139.9).

Guards:
  A) max_snap_in (hard limit): if the proposed snap moves the coord by more
     than this many inches, reject and keep the CNN prior.
  B) outside-bright-polarity: for each corner/axis, we know which side of
     the peak is "outside" the page (scanner bed, expected ~230 gray). If
     the mean brightness on the outside half is below a threshold, the peak
     is probably an internal edge — reject.
  C) confidence floor: raise the existing min_confidence slightly so
     low-snr peaks fall back to CNN instead of random noise.

Sweeps over the guards on the 70-page holdout and reports the best.
"""
import cv2, numpy as np, torch
from itertools import product

from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, _refine_coord, MODEL_FILE,
)

# Corner order [TL, TR, BR, BL]
# For each corner, which side is "outside" along each axis:
#   y-axis (top/bottom): TL/TR outside = smaller y (above peak),
#                        BR/BL outside = larger y  (below peak)
#   x-axis (left/right): TL/BL outside = smaller x (left of peak),
#                        TR/BR outside = larger x  (right of peak)
OUTSIDE_Y = [-1, -1, +1, +1]   # sign: -1 = outside is index<peak, +1 = index>peak
OUTSIDE_X = [-1, +1, +1, -1]


def refine_corners_bleed(image_bgr, cnn_corners, dpi=600,
                         search_in=0.20, strip_in=0.15,
                         min_confidence=1.75,
                         max_snap_in=None,
                         outside_min_brightness=None):
    """Same shape as refine_corners, but with optional bleed-aware guards."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    search_r = int(dpi * search_in)
    strip_w = int(dpi * strip_in)
    max_snap_px = None if max_snap_in is None else int(dpi * max_snap_in)

    refined = []
    for corner_i, (cx, cy) in enumerate(cnn_corners):
        cx_i, cy_i = int(cx), int(cy)

        # ---- y refinement ----
        y0 = max(0, cy_i - search_r)
        y1 = min(H, cy_i + search_r + 1)
        x0 = max(0, cx_i - strip_w)
        x1 = min(W, cx_i + strip_w + 1)
        strip = gray[y0:y1, x0:x1]
        refined_y = cy
        if strip.size > 0 and strip.shape[0] >= 3:
            vprofile = strip.mean(axis=1)
            peak, conf = _refine_coord(vprofile, cy_i - y0)
            cand_y = y0 + peak
            ok = conf >= min_confidence
            if ok and max_snap_px is not None and abs(cand_y - cy) > max_snap_px:
                ok = False
            if ok and outside_min_brightness is not None:
                side = OUTSIDE_Y[corner_i]  # -1 or +1
                if side < 0:
                    outside_slice = vprofile[:peak]
                else:
                    outside_slice = vprofile[peak+1:]
                if outside_slice.size >= 3 and outside_slice.mean() < outside_min_brightness:
                    ok = False
            if ok:
                refined_y = cand_y

        # ---- x refinement ----
        y0 = max(0, cy_i - strip_w)
        y1 = min(H, cy_i + strip_w + 1)
        x0 = max(0, cx_i - search_r)
        x1 = min(W, cx_i + search_r + 1)
        strip = gray[y0:y1, x0:x1]
        refined_x = cx
        if strip.size > 0 and strip.shape[1] >= 3:
            hprofile = strip.mean(axis=0)
            peak, conf = _refine_coord(hprofile, cx_i - x0)
            cand_x = x0 + peak
            ok = conf >= min_confidence
            if ok and max_snap_px is not None and abs(cand_x - cx) > max_snap_px:
                ok = False
            if ok and outside_min_brightness is not None:
                side = OUTSIDE_X[corner_i]
                if side < 0:
                    outside_slice = hprofile[:peak]
                else:
                    outside_slice = hprofile[peak+1:]
                if outside_slice.size >= 3 and outside_slice.mean() < outside_min_brightness:
                    ok = False
            if ok:
                refined_x = cand_x

        refined.append([float(refined_x), float(refined_y)])

    return refined


def mean_err(pred, gt):
    return float(np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(pred, gt)]))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = _load_model(MODEL_FILE, device)
    entries = _load_entries()
    _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []), ckpt.get("holdout_dirs", []))
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Holdout: {len(eval_entries)} pages. Baseline reg model val={ckpt['val_px']:.2f} px")

    cache = []
    for e in eval_entries:
        img = cv2.imread(e["filepath"])
        if img is None: continue
        if e["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cnn = predict_corners(model, device, img, tta=True)
        cache.append({"img": img, "gt": e["gt_corners"], "cnn": cnn,
                      "dpi": e.get("dpi", 600)})
    print(f"Cached CNN for {len(cache)} pages.\n")

    # Baselines
    cnn_errs = [mean_err(c["cnn"], c["gt"]) for c in cache]
    from comicml import refine_corners as base_refine
    base_errs = [mean_err(base_refine(c["img"], c["cnn"], dpi=c["dpi"]), c["gt"])
                 for c in cache]
    print(f"CNN-only:       mean={np.mean(cnn_errs):6.2f}  median={np.median(cnn_errs):6.2f}")
    print(f"current refine: mean={np.mean(base_errs):6.2f}  median={np.median(base_errs):6.2f}  (baseline to beat)\n")

    # Grid
    # max_snap_in None means disabled; values in inches (0.10 = ~60 px @ 600 DPI)
    snap_vals = [None, 0.20, 0.15, 0.125, 0.10, 0.08]
    bright_vals = [None, 160, 180, 200, 210, 220]
    min_conf_vals = [1.75, 2.0]

    results = []
    for ms, br, mc in product(snap_vals, bright_vals, min_conf_vals):
        errs = []
        improved = worsened = 0
        for c, ce in zip(cache, cnn_errs):
            hyb = refine_corners_bleed(c["img"], c["cnn"], dpi=c["dpi"],
                                       min_confidence=mc,
                                       max_snap_in=ms,
                                       outside_min_brightness=br)
            d = mean_err(hyb, c["gt"])
            errs.append(d)
            if d < ce: improved += 1
            elif d > ce: worsened += 1
        results.append({"max_snap": ms, "bright": br, "min_conf": mc,
                        "mean": float(np.mean(errs)),
                        "median": float(np.median(errs)),
                        "p95": float(np.percentile(errs, 95)),
                        "max": float(np.max(errs)),
                        "imp": improved, "wor": worsened})

    results.sort(key=lambda r: r["mean"])
    print(f"{'max_snap':>9s} {'bright':>7s} {'minconf':>8s}   {'mean':>7s} {'median':>7s} {'p95':>7s} {'max':>7s}   imp/wor")
    for r in results[:15]:
        ms = "-" if r["max_snap"] is None else f"{r['max_snap']:.3f}"
        br = "-" if r["bright"] is None else f"{r['bright']}"
        print(f"{ms:>9s} {br:>7s} {r['min_conf']:>8.2f}   "
              f"{r['mean']:>7.2f} {r['median']:>7.2f} {r['p95']:>7.2f} {r['max']:>7.2f}   {r['imp']}/{r['wor']}")

    best = results[0]
    print(f"\nBest: max_snap={best['max_snap']}  bright={best['bright']}  min_conf={best['min_conf']}")
    print(f"  mean {np.mean(base_errs):.2f} → {best['mean']:.2f}  (Δ {np.mean(base_errs)-best['mean']:+.2f})")
    print(f"  P95  {np.percentile(base_errs,95):.2f} → {best['p95']:.2f}")


if __name__ == "__main__":
    main()
