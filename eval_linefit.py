#!/usr/bin/env python3
"""Line-fit refinement: detect 4 edges, intersect for corners.

Instead of snapping each corner's x/y independently, this:
  1. Samples many 1D gradient profiles along each edge (top/bottom/left/right)
  2. Finds edge points via _refine_coord (same proven 1D peak detector)
  3. RANSAC-fits a line through confident detections per edge
  4. Intersects adjacent lines → corner coordinates

Advantages over per-corner snap:
  - Uses all pixels along each edge (not just a small window at each corner)
  - Naturally enforces quadrilateral geometry
  - RANSAC rejects outlier detections (bleed, panel borders, etc.)
  - Handles skew correctly (lines can be at slight angles)
"""
import cv2
import numpy as np
import torch
from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, refine_corners, _refine_coord, MODEL_FILE,
)


def _fit_line_ransac(points, n_iter=100, inlier_thresh=8.0):
    """Fit a line to 2D points via RANSAC. Returns (a, b, c) for ax+by+c=0,
    normalized so sqrt(a²+b²)=1, or None if too few points.

    `points` is (N, 2) array of (x, y).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return None
    if n == 2:
        # Exact line through 2 points
        d = pts[1] - pts[0]
        a, b = -d[1], d[0]
        c = -(a * pts[0, 0] + b * pts[0, 1])
        norm = np.hypot(a, b)
        return (a / norm, b / norm, c / norm) if norm > 1e-12 else None

    best_line = None
    best_inliers = 0
    for _ in range(n_iter):
        i, j = np.random.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        a, b = -d[1], d[0]
        c = -(a * pts[i, 0] + b * pts[i, 1])
        norm = np.hypot(a, b)
        if norm < 1e-12:
            continue
        a, b, c = a / norm, b / norm, c / norm
        dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = np.sum(dists < inlier_thresh)
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = (a, b, c)

    if best_line is None or best_inliers < 2:
        return None

    # Refit with all inliers (least-squares)
    a, b, c = best_line
    dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
    mask = dists < inlier_thresh
    inlier_pts = pts[mask]
    if len(inlier_pts) < 2:
        return best_line

    # SVD fit
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]  # last row = smallest singular value direction
    a, b = normal
    c = -(a * centroid[0] + b * centroid[1])
    norm = np.hypot(a, b)
    return (a / norm, b / norm, c / norm) if norm > 1e-12 else best_line


def _intersect_lines(l1, l2):
    """Intersect two lines (a,b,c) → (x, y) or None if parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return [float(x), float(y)]


def _sample_edge_points(gray, p0, p1, n_samples, search_r, strip_w,
                        axis, min_confidence=1.75):
    """Sample edge points along the line from p0 to p1.

    axis='y' means we're detecting a horizontal edge (top/bottom): each sample
    takes a vertical 1D profile and finds the edge y-coord.
    axis='x' means we're detecting a vertical edge (left/right): each sample
    takes a horizontal 1D profile and finds the edge x-coord.

    Returns list of (x, y) detected edge points.
    """
    H, W = gray.shape
    points = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples  # avoid endpoints (near corners)
        cx = p0[0] + t * (p1[0] - p0[0])
        cy = p0[1] + t * (p1[1] - p0[1])
        cx_i, cy_i = int(cx), int(cy)

        if axis == 'y':
            # Vertical profile to find horizontal edge
            y0 = max(0, cy_i - search_r)
            y1 = min(H, cy_i + search_r + 1)
            x0 = max(0, cx_i - strip_w)
            x1 = min(W, cx_i + strip_w + 1)
            strip = gray[y0:y1, x0:x1]
            if strip.size == 0 or strip.shape[0] < 3:
                continue
            profile = strip.mean(axis=1)
            peak, conf = _refine_coord(profile, cy_i - y0)
            if conf >= min_confidence:
                points.append([cx, float(y0 + peak)])
        else:  # axis == 'x'
            # Horizontal profile to find vertical edge
            y0 = max(0, cy_i - strip_w)
            y1 = min(H, cy_i + strip_w + 1)
            x0 = max(0, cx_i - search_r)
            x1 = min(W, cx_i + search_r + 1)
            strip = gray[y0:y1, x0:x1]
            if strip.size == 0 or strip.shape[1] < 3:
                continue
            profile = strip.mean(axis=0)
            peak, conf = _refine_coord(profile, cx_i - x0)
            if conf >= min_confidence:
                points.append([float(x0 + peak), cy])

    return points


def refine_corners_linefit(image_bgr, cnn_corners, dpi=600,
                           search_in=0.20, strip_in=0.15,
                           min_confidence=1.75,
                           n_samples=20, inlier_thresh=8.0,
                           min_edge_points=4, agree_px=None):
    """Refine corners by fitting lines to detected edge points, then
    intersecting. Falls back to per-corner snap for edges with too few
    detections.

    Parameters:
      n_samples: number of 1D profiles sampled along each edge
      inlier_thresh: RANSAC inlier distance in pixels
      min_edge_points: minimum confident detections to attempt line fit
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    search_r = int(dpi * search_in)
    strip_w = int(dpi * strip_in)

    tl, tr, br, bl = cnn_corners

    # Detect edge points along each of the 4 edges
    # top edge (TL→TR): horizontal edge, detect y
    top_pts = _sample_edge_points(gray, tl, tr, n_samples, search_r, strip_w,
                                  axis='y', min_confidence=min_confidence)
    # bottom edge (BL→BR): horizontal edge, detect y
    bot_pts = _sample_edge_points(gray, bl, br, n_samples, search_r, strip_w,
                                  axis='y', min_confidence=min_confidence)
    # left edge (TL→BL): vertical edge, detect x
    left_pts = _sample_edge_points(gray, tl, bl, n_samples, search_r, strip_w,
                                   axis='x', min_confidence=min_confidence)
    # right edge (TR→BR): vertical edge, detect x
    right_pts = _sample_edge_points(gray, tr, br, n_samples, search_r, strip_w,
                                    axis='x', min_confidence=min_confidence)

    # Fit lines
    top_line = _fit_line_ransac(top_pts, inlier_thresh=inlier_thresh) if len(top_pts) >= min_edge_points else None
    bot_line = _fit_line_ransac(bot_pts, inlier_thresh=inlier_thresh) if len(bot_pts) >= min_edge_points else None
    left_line = _fit_line_ransac(left_pts, inlier_thresh=inlier_thresh) if len(left_pts) >= min_edge_points else None
    right_line = _fit_line_ransac(right_pts, inlier_thresh=inlier_thresh) if len(right_pts) >= min_edge_points else None

    # Intersect: TL = top∩left, TR = top∩right, BR = bot∩right, BL = bot∩left
    lines_for_corner = [
        (top_line, left_line),   # TL
        (top_line, right_line),  # TR
        (bot_line, right_line),  # BR
        (bot_line, left_line),   # BL
    ]

    # Fall back to the old per-corner snap for any corner where line-fit failed
    # or disagrees with the per-corner snap (agreement = mutual validation)
    fallback = refine_corners(image_bgr, cnn_corners, dpi=dpi,
                              search_in=search_in, strip_in=strip_in,
                              min_confidence=min_confidence)

    # Agreement threshold: if line-fit and per-corner snap are within this
    # many pixels, trust line-fit (it uses more information). Otherwise
    # fall back to the conservative per-corner snap.
    agree_thresh = agree_px if agree_px is not None else search_r * 0.5

    refined = []
    for i, (la, lb) in enumerate(lines_for_corner):
        if la is not None and lb is not None:
            pt = _intersect_lines(la, lb)
            if pt is not None:
                # Check agreement with per-corner snap
                dist_from_snap = np.hypot(pt[0] - fallback[i][0],
                                          pt[1] - fallback[i][1])
                if dist_from_snap < agree_thresh:
                    refined.append(pt)
                    continue
        # Fallback
        refined.append(fallback[i])

    return refined


def mean_err(pred, gt):
    return float(np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(pred, gt)]))


def per_corner_dists(pred, gt):
    return [float(np.hypot(p[0]-g[0], p[1]-g[1])) for p, g in zip(pred, gt)]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = _load_model(MODEL_FILE, device)
    entries = _load_entries()
    _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []),
                                     ckpt.get("holdout_dirs", []))
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Holdout: {len(eval_entries)} pages\n")

    # Cache CNN predictions
    cache = []
    for e in eval_entries:
        img = cv2.imread(e["filepath"])
        if img is None: continue
        if e["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cnn = predict_corners(model, device, img, tta=True)
        cache.append({"img": img, "gt": e["gt_corners"], "cnn": cnn,
                      "dpi": e.get("dpi", 600),
                      "name": e["scan_dir"].rsplit("/",1)[-1] + "/" +
                              e["filepath"].rsplit("/",1)[-1]})

    print(f"Cached CNN for {len(cache)} pages.\n")

    # Baseline
    old_errs = []
    for c in cache:
        old_hyb = refine_corners(c["img"], c["cnn"], dpi=c["dpi"])
        old_errs.append(mean_err(old_hyb, c["gt"]))
    old_a = np.array(old_errs)
    cnn_errs = np.array([mean_err(c["cnn"], c["gt"]) for c in cache])

    print(f"  {'method':<35s}  {'mean':>6s} {'median':>6s} {'P95':>6s} {'max':>7s}  {'better':>6s} {'worse':>5s}")
    print("-" * 90)
    print(f"  {'CNN-only':<35s}  {cnn_errs.mean():6.2f} {np.median(cnn_errs):6.2f} "
          f"{np.percentile(cnn_errs,95):6.2f} {cnn_errs.max():7.2f}")
    print(f"  {'old (per-corner snap)':<35s}  {old_a.mean():6.2f} {np.median(old_a):6.2f} "
          f"{np.percentile(old_a,95):6.2f} {old_a.max():7.2f}")
    print()

    # Sweep line-fit parameters
    from itertools import product
    agree_vals = [15, 25, 40, 60]
    samples_vals = [12, 20, 30]
    inlier_vals = [5.0, 8.0, 12.0]
    min_pts_vals = [3, 5]

    results = []
    for ag, ns, it, mp in product(agree_vals, samples_vals, inlier_vals, min_pts_vals):
        errs = []
        for c in cache:
            hyb = refine_corners_linefit(c["img"], c["cnn"], dpi=c["dpi"],
                                         agree_px=ag, n_samples=ns,
                                         inlier_thresh=it, min_edge_points=mp)
            errs.append(mean_err(hyb, c["gt"]))
        ea = np.array(errs)
        delta = old_a - ea
        results.append({
            "agree": ag, "samples": ns, "inlier": it, "min_pts": mp,
            "mean": float(ea.mean()), "median": float(np.median(ea)),
            "p95": float(np.percentile(ea, 95)), "max": float(ea.max()),
            "better": int((delta > 0.5).sum()),
            "worse": int((delta < -0.5).sum()),
        })

    results.sort(key=lambda r: r["mean"])
    for r in results[:15]:
        label = f"ag={r['agree']:>2d} ns={r['samples']:>2d} it={r['inlier']:.0f} mp={r['min_pts']}"
        print(f"  {label:<35s}  {r['mean']:6.2f} {r['median']:6.2f} "
              f"{r['p95']:6.2f} {r['max']:7.2f}  {r['better']:>6d} {r['worse']:>5d}")

    best = results[0]
    print(f"\nBest: agree={best['agree']} samples={best['samples']} "
          f"inlier={best['inlier']} min_pts={best['min_pts']}")
    print(f"  old → new:  mean {old_a.mean():.2f} → {best['mean']:.2f}  "
          f"({old_a.mean()-best['mean']:+.2f})")
    print(f"              P95  {np.percentile(old_a,95):.2f} → {best['p95']:.2f}")

    # Detail on the best config
    print(f"\n--- Detail: best config vs old ---")
    CORNER_NAMES = ["TL", "TR", "BR", "BL"]
    details = []
    for c in cache:
        old_hyb = refine_corners(c["img"], c["cnn"], dpi=c["dpi"])
        new_hyb = refine_corners_linefit(c["img"], c["cnn"], dpi=c["dpi"],
                                          agree_px=best['agree'],
                                          n_samples=best['samples'],
                                          inlier_thresh=best['inlier'],
                                          min_edge_points=best['min_pts'])
        oe = mean_err(old_hyb, c["gt"])
        ne = mean_err(new_hyb, c["gt"])
        details.append({"name": c["name"], "old": oe, "new": ne,
                        "old_per": per_corner_dists(old_hyb, c["gt"]),
                        "new_per": per_corner_dists(new_hyb, c["gt"])})

    order = np.argsort([d["old"]-d["new"] for d in details])
    print("Biggest losses:")
    for idx in order[:3]:
        d = details[idx]
        print(f"  {d['name']:<32s}  old={d['old']:6.2f}  new={d['new']:6.2f}  Δ={d['old']-d['new']:+6.2f}")
    print("Biggest wins:")
    for idx in order[-3:][::-1]:
        d = details[idx]
        print(f"  {d['name']:<32s}  old={d['old']:6.2f}  new={d['new']:6.2f}  Δ={d['old']-d['new']:+6.2f}")


if __name__ == "__main__":
    main()
