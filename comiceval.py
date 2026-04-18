#!/usr/bin/env python3
"""
comiceval.py — Ground truth evaluation and parameter tuning for comicscans detection.

Collects manual corrections from .comicscans_session.json files, runs the detector
against those images, and reports accuracy metrics. Optionally tunes detection
parameters to minimize error against the ground truth.

Usage:
    # Collect ground truth from all session files
    python3 comiceval.py collect raw-scans/

    # Evaluate current detection accuracy against ground truth
    python3 comiceval.py eval

    # Tune parameters to minimize error (writes tuned values to comiceval_params.json)
    python3 comiceval.py tune

    # Evaluate with tuned parameters
    python3 comiceval.py eval --params comiceval_params.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from comicscans import (
    detect_page_bounds,
    detect_orientation,
    get_source_dpi,
    load_scans,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"
TUNED_PARAMS_FILE = Path(__file__).parent / "comiceval_params.json"

# ---------------------------------------------------------------------------
# Ground truth collection
# ---------------------------------------------------------------------------

def collect_ground_truth(scan_dirs: list[Path]) -> list[dict]:
    """Walk scan directories for session files and extract ground truth entries.

    A ground truth entry is created for each page that has EITHER:
      - A user override (explicit correction)
      - A detection with no override (user accepted the auto result)

    This gives us both positive examples (accepted detections) and corrections.
    """
    entries = []

    for scan_dir in scan_dirs:
        session_path = scan_dir / ".comicscans_session.json"
        if not session_path.exists():
            continue

        try:
            session_data = json.loads(session_path.read_text())
        except (json.JSONDecodeError, IOError):
            print(f"  Warning: Could not read {session_path}")
            continue

        detections = session_data.get("detections", {})
        overrides = session_data.get("overrides", {})

        # Load scan file list
        scans = load_scans(str(scan_dir))
        scan_map = {i: filepath for i, (_, filepath) in enumerate(scans)}

        # Collect entries for all pages that have data
        all_indices = set(int(k) for k in detections) | set(int(k) for k in overrides)

        for idx in sorted(all_indices):
            filepath = scan_map.get(idx)
            if filepath is None or not filepath.exists():
                continue

            det = detections.get(str(idx), {})
            ovr = overrides.get(str(idx), {})

            # Ground truth = override if present, else detection
            gt = ovr if ovr else det
            if not gt.get("corners"):
                continue

            has_correction = bool(ovr) and ovr.get("corners") != det.get("corners")

            entry = {
                "scan_dir": str(scan_dir),
                "page_index": idx,
                "filepath": str(filepath),
                "dpi": get_source_dpi(filepath),
                "image_width": None,   # filled below
                "image_height": None,
                "gt_corners": gt["corners"],
                "gt_rotation": gt.get("rotation", 0),
                "gt_rotate180": gt.get("rotate180", False),
                "det_corners": det.get("corners"),
                "det_rotation": det.get("rotation", 0),
                "det_rotate180": det.get("rotate180", False),
                "det_bleed_method": det.get("bleed_method"),
                "has_correction": has_correction,
            }

            # Read image dimensions (fast — just header)
            from PIL import Image as PILImage
            with PILImage.open(filepath) as img:
                entry["image_width"], entry["image_height"] = img.size

            entries.append(entry)

        n_corrections = sum(1 for e in entries if e["scan_dir"] == str(scan_dir) and e["has_correction"])
        n_accepted = sum(1 for e in entries if e["scan_dir"] == str(scan_dir) and not e["has_correction"])
        print(f"  {scan_dir.name}: {n_corrections} corrections, {n_accepted} accepted detections")

    return entries


def save_ground_truth(entries: list[dict], path: Path = GROUND_TRUTH_FILE):
    """Save ground truth to JSON."""
    path.write_text(json.dumps(entries, indent=2))
    print(f"\nSaved {len(entries)} ground truth entries to {path}")


def load_ground_truth(path: Path = GROUND_TRUTH_FILE) -> list[dict]:
    """Load ground truth from JSON."""
    if not path.exists():
        print(f"No ground truth file found at {path}")
        print("Run: python3 comiceval.py collect <scan_dirs>")
        sys.exit(1)
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def corner_distance(corners_a, corners_b):
    """Mean Euclidean distance between corresponding corners (px)."""
    a = np.array(corners_a, dtype=np.float64)
    b = np.array(corners_b, dtype=np.float64)
    return np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


def iou_from_corners(corners_a, corners_b, img_w, img_h):
    """Approximate IoU by rasterizing both quadrilaterals on a small canvas."""
    # Scale down for speed
    scale = 500 / max(img_w, img_h)
    sw, sh = int(img_w * scale), int(img_h * scale)

    def draw_quad(corners):
        pts = (np.array(corners, dtype=np.float64) * scale).astype(np.int32)
        mask = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 1)
        return mask

    mask_a = draw_quad(corners_a)
    mask_b = draw_quad(corners_b)

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return intersection / union if union > 0 else 0.0


def crop_dimensions(corners):
    """Return (width, height) of the bounding box of corner points."""
    c = np.array(corners)
    return float(c[:, 0].max() - c[:, 0].min()), float(c[:, 1].max() - c[:, 1].min())


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------

# Default parameters (matching current code)
DEFAULT_PARAMS = {
    # _find_content_bounds / detect_page_bounds
    "bed_mean_offset": 20.0024,  # mean_thresh = bed_mean - this
    "min_mean_thresh": 200,      # floor for mean_thresh
    "std_thresh": 20.6157,       # std deviation threshold for content

    # detect_spine_dark_band
    "spine_mean_max": 60,        # max column mean for spine
    "spine_std_max": 20,         # max column std for spine
    "spine_min_width_in": 0.05,  # min spine width in inches (15px @ 300)

    # detect_bleed_boundary
    "bleed_trigger_ratio": 1.05,   # content must be this * expected_page_px wide
    "bleed_search_pct": 0.08,      # search radius as % of expected page width
    "bleed_edge_margin_in": 0.167, # edge margin in inches (50px @ 300)
    "gutter_peak_offset": 9.8517,  # peak must exceed mean + this
    "trough_depth": 6.0895,        # trough must be mean - this
    "trough_rise": 4,              # both sides must rise this much above trough
    "gradient_min": 2.0298,        # minimum gradient magnitude
    "kernel_size_in": 0.05,        # smoothing kernel in inches (15px @ 300)

    # Trim
    "bleed_inset_in": 0.0497,    # inward shift from bleed boundary (inches)
    "safety_margin_in": 0.1492,  # subtracted from expected width for trim target

    # Edge trim
    "strip_check_in": 0.667,     # max strip check in inches (200px @ 300)
    "band_size_in": 2.0,         # max band size in inches (600px @ 300)

    # Skew
    "skew_canny_lower": 50,
    "skew_canny_upper": 150,
    "skew_hough_threshold": 80,
    "skew_max_angle": 5.0,
}


def run_detection(entry: dict, params: dict = None, preloaded_image=None) -> dict:
    """Run detection on a single ground truth entry, return detected corners.

    If preloaded_image is provided, skip file I/O (used during tuning).
    """
    if params is None:
        params = DEFAULT_PARAMS

    if preloaded_image is not None:
        image = preloaded_image
    else:
        filepath = entry["filepath"]
        image = cv2.imread(filepath)
        if image is None:
            return {"error": f"Could not read {filepath}"}
        if entry["gt_rotate180"]:
            image = cv2.rotate(image, cv2.ROTATE_180)

    dpi = entry["dpi"]
    bounds = detect_page_bounds(image, dpi, params=params)

    # Convert bounds to corners (same as server.py)
    angle = bounds["angle"]
    top, bottom = bounds["top"], bounds["bottom"]
    left, right = bounds["left"], bounds["right"]
    orig_h, orig_w = image.shape[:2]

    deskewed_corners = np.array([
        [left, top], [right, top], [right, bottom], [left, bottom]
    ], dtype=np.float64)

    if abs(angle) <= 0.1:
        corners = deskewed_corners.tolist()
    else:
        rad = np.deg2rad(abs(angle))
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        new_w = int(orig_h * sin_a + orig_w * cos_a)
        new_h = int(orig_h * cos_a + orig_w * sin_a)
        cx_desk, cy_desk = new_w / 2.0, new_h / 2.0
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        corners = []
        for px, py in deskewed_corners:
            dx, dy = px - cx_desk, py - cy_desk
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            corners.append([round(rx + orig_w / 2.0, 1), round(ry + orig_h / 2.0, 1)])

    return {
        "corners": corners,
        "rotation": bounds["angle"],
        "bleed_method": bounds.get("bleed_method"),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(entries: list[dict], params: dict = None, verbose: bool = True) -> dict:
    """Run detection on all ground truth entries and compute metrics."""
    results = []
    errors = []

    for i, entry in enumerate(entries):
        det = run_detection(entry, params)
        if "error" in det:
            errors.append(det["error"])
            continue

        gt_corners = entry["gt_corners"]
        det_corners = det["corners"]
        img_w = entry["image_width"]
        img_h = entry["image_height"]
        dpi = entry["dpi"]

        dist = corner_distance(gt_corners, det_corners)
        iou = iou_from_corners(gt_corners, det_corners, img_w, img_h)

        gt_w, gt_h = crop_dimensions(gt_corners)
        det_w, det_h = crop_dimensions(det_corners)

        result = {
            "page": f"{Path(entry['scan_dir']).name}/{Path(entry['filepath']).name}",
            "corrected": entry["has_correction"],
            "corner_dist_px": round(dist, 1),
            "corner_dist_in": round(dist / dpi, 3),
            "iou": round(iou, 4),
            "width_error_px": round(det_w - gt_w, 1),
            "height_error_px": round(det_h - gt_h, 1),
            "bleed_method": det.get("bleed_method"),
        }
        results.append(result)

    if not results:
        print("No results to evaluate.")
        return {}

    # Aggregate metrics
    all_dist = [r["corner_dist_px"] for r in results]
    all_iou = [r["iou"] for r in results]
    corrected = [r for r in results if r["corrected"]]
    accepted = [r for r in results if not r["corrected"]]

    metrics = {
        "total_pages": len(results),
        "corrected_pages": len(corrected),
        "accepted_pages": len(accepted),
        "mean_corner_dist_px": round(np.mean(all_dist), 1),
        "median_corner_dist_px": round(np.median(all_dist), 1),
        "p95_corner_dist_px": round(np.percentile(all_dist, 95), 1),
        "max_corner_dist_px": round(max(all_dist), 1),
        "mean_iou": round(np.mean(all_iou), 4),
        "min_iou": round(min(all_iou), 4),
    }

    if corrected:
        metrics["corrected_mean_dist_px"] = round(np.mean([r["corner_dist_px"] for r in corrected]), 1)
        metrics["corrected_mean_iou"] = round(np.mean([r["iou"] for r in corrected]), 4)

    if verbose:
        print("\n=== Detection Accuracy Report ===\n")
        print(f"Pages evaluated:    {metrics['total_pages']}")
        print(f"  User-corrected:   {metrics['corrected_pages']}")
        print(f"  Auto-accepted:    {metrics['accepted_pages']}")
        print()
        print(f"Corner distance (mean):   {metrics['mean_corner_dist_px']:>7.1f} px")
        print(f"Corner distance (median): {metrics['median_corner_dist_px']:>7.1f} px")
        print(f"Corner distance (p95):    {metrics['p95_corner_dist_px']:>7.1f} px")
        print(f"Corner distance (max):    {metrics['max_corner_dist_px']:>7.1f} px")
        print(f"IoU (mean):               {metrics['mean_iou']:>7.4f}")
        print(f"IoU (min):                {metrics['min_iou']:>7.4f}")

        if corrected:
            print(f"\n--- Corrected pages only ---")
            print(f"Corner distance (mean):   {metrics['corrected_mean_dist_px']:>7.1f} px")
            print(f"Corner distance (mean):   {round(metrics['corrected_mean_dist_px'] / entries[0]['dpi'], 3):>7.3f} in")
            print(f"IoU (mean):               {metrics['corrected_mean_iou']:>7.4f}")

        # Show worst pages
        worst = sorted(results, key=lambda r: r["corner_dist_px"], reverse=True)[:10]
        print(f"\n--- Worst 10 pages ---")
        print(f"{'Page':50s} {'Dist(px)':>9s} {'IoU':>7s} {'Corrected':>10s} {'Bleed':>20s}")
        for r in worst:
            tag = "YES" if r["corrected"] else ""
            print(f"{r['page']:50s} {r['corner_dist_px']:>9.1f} {r['iou']:>7.4f} {tag:>10s} {r['bleed_method'] or '':>20s}")

        if errors:
            print(f"\n{len(errors)} errors during evaluation:")
            for e in errors[:5]:
                print(f"  {e}")

    return metrics


# ---------------------------------------------------------------------------
# Parameter tuning
# ---------------------------------------------------------------------------

def _save_tuned_params(param_names, x_values, bounds):
    """Save current best parameters to disk (called on every improvement)."""
    tuned = dict(DEFAULT_PARAMS)
    for name, val, (lo, hi) in zip(param_names, x_values, bounds):
        tuned[name] = round(max(lo, min(hi, val)), 4)
    TUNED_PARAMS_FILE.write_text(json.dumps(tuned, indent=2))


# Persistent worker loop. Each worker holds ONLY its own chunk of images
# (loading all 212 × ~30MB 600DPI images in every worker crashed a 128GB box).
# Communication is via Pipe: parent sends params, worker sends back partial sum.
def _worker_loop(conn, entries_chunk):
    images = {}
    for entry in entries_chunk:
        img = cv2.imread(entry["filepath"])
        if img is not None and entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        images[entry["filepath"]] = img
    # Signal readiness
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        params = msg
        total = 0.0
        for entry in entries_chunk:
            img = images.get(entry["filepath"])
            det = run_detection(entry, params, preloaded_image=img)
            if "error" in det:
                total += 1000
                continue
            total += corner_distance(entry["gt_corners"], det["corners"])
        conn.send(total)


def tune_parameters(entries: list[dict]):
    """Tune detection parameters to minimize corner distance on ground truth.

    Preloads all images into memory and uses Nelder-Mead for fast convergence.
    Only tunes against pages with user corrections (those are the labeled errors).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("scipy is required for tuning: pip install scipy")
        sys.exit(1)

    import multiprocessing as mp

    # Only tune against corrected pages (those are the known errors)
    corrected = [e for e in entries if e["has_correction"]]
    if len(corrected) < 3:
        corrected = entries

    # Cap workers conservatively: each worker holds its chunk of decoded 600-DPI
    # images (~30MB each). 8 workers × ~26 pages × 30MB ≈ 6GB, plus ~200MB
    # Python overhead per worker = ~8GB peak. Higher worker counts give
    # diminishing returns (chunks get small, overhead dominates).
    n_workers = 8
    print(f"Tuning against {len(corrected)} corrected pages using {n_workers} workers...")

    # Parameters to tune and their bounds
    # (name, lower, upper, default)
    tune_spec = [
        ("bed_mean_offset", 10, 40, 20),
        ("std_thresh", 10, 40, 20),
        ("bleed_inset_in", 0.01, 0.15, 0.05),
        ("safety_margin_in", 0.05, 0.30, 0.15),
        ("gutter_peak_offset", 5, 25, 10),
        ("trough_depth", 3, 12, 6),
        ("gradient_min", 1.0, 5.0, 2.0),
    ]

    param_names = [s[0] for s in tune_spec]
    bounds = [(s[1], s[2]) for s in tune_spec]
    x0 = [s[3] for s in tune_spec]

    eval_count = [0]
    best_score = [float("inf")]
    best_x = [list(x0)]

    # Split entries across workers — each gets a roughly equal contiguous slice.
    n_entries = len(corrected)
    chunk_size = (n_entries + n_workers - 1) // n_workers
    chunks = [corrected[i:i + chunk_size] for i in range(0, n_entries, chunk_size)]
    n_workers = len(chunks)  # in case rounding gave us fewer chunks

    # Spawn workers and wait for each to finish loading its chunk.
    print(f"Starting {n_workers} workers (each preloads ~{chunk_size} images)...")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    workers = []
    for chunk in chunks:
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker_loop, args=(child_conn, chunk))
        proc.start()
        workers.append((proc, parent_conn))
    # Wait for ready signal from each
    for _, conn in workers:
        msg = conn.recv()
        assert msg == "ready"
    print(f"  Workers ready in {time.time() - t0:.1f}s")

    def objective(x):
        # Clip to bounds
        params = dict(DEFAULT_PARAMS)
        for name, val, (lo, hi) in zip(param_names, x, bounds):
            params[name] = max(lo, min(hi, val))

        # Fan out: send params to every worker, then collect partial sums.
        for _, conn in workers:
            conn.send(params)
        total_dist = sum(conn.recv() for _, conn in workers)

        mean_dist = total_dist / n_entries
        eval_count[0] += 1
        if mean_dist < best_score[0]:
            best_score[0] = mean_dist
            best_x[0] = list(x)
            _save_tuned_params(param_names, best_x[0], bounds)
        if eval_count[0] % 5 == 0:
            print(f"  eval {eval_count[0]:>4d}: current = {mean_dist:.1f} px, best = {best_score[0]:.1f} px",
                  flush=True)
        return mean_dist

    try:
        # Run baseline
        baseline = objective(x0)
        print(f"\nBaseline mean corner distance: {baseline:.1f} px")
        print(f"\nRunning Nelder-Mead optimization (maxiter=120)...\n")

        t0 = time.time()

        result = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            options={
                "maxiter": 120,
                "xatol": 0.05,
                "fatol": 0.3,
                "adaptive": True,
                "disp": True,
            },
        )
    finally:
        # Shut down workers cleanly.
        for _, conn in workers:
            try:
                conn.send(None)
            except Exception:
                pass
        for proc, _ in workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.1f}s ({eval_count[0]} evaluations)")
    print(f"Baseline: {baseline:.1f} px → Tuned: {result.fun:.1f} px "
          f"({baseline - result.fun:+.1f} px improvement)\n")

    tuned = dict(DEFAULT_PARAMS)
    print(f"{'Parameter':<25s} {'Default':>10s} {'Tuned':>10s} {'Delta':>10s}")
    print("-" * 57)
    for name, val, (lo, hi) in zip(param_names, result.x, bounds):
        clamped = max(lo, min(hi, val))
        tuned[name] = round(clamped, 4)
        default = DEFAULT_PARAMS[name]
        delta = clamped - default
        print(f"{name:<25s} {default:>10.4f} {clamped:>10.4f} {delta:>+10.4f}")

    # Save tuned parameters
    TUNED_PARAMS_FILE.write_text(json.dumps(tuned, indent=2))
    print(f"\nTuned parameters saved to {TUNED_PARAMS_FILE}")

    return tuned


# ---------------------------------------------------------------------------
# Add ground truth to server endpoint
# ---------------------------------------------------------------------------

def export_for_webapp(entries: list[dict]):
    """Print a summary suitable for pasting into documentation."""
    print(f"\nGround truth summary:")
    print(f"  Total pages: {len(entries)}")
    dirs = set(e["scan_dir"] for e in entries)
    print(f"  Scan directories: {len(dirs)}")
    for d in sorted(dirs):
        n = sum(1 for e in entries if e["scan_dir"] == d)
        nc = sum(1 for e in entries if e["scan_dir"] == d and e["has_correction"])
        print(f"    {Path(d).name}: {n} pages ({nc} corrected)")
    dpis = set(e["dpi"] for e in entries)
    print(f"  DPI values: {sorted(dpis)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Audit: flag likely-miscllicked GT corners
# ---------------------------------------------------------------------------

def audit_ground_truth(top_n: int = 20,
                       bug_displacement_thresh: float = 40.0,
                       bug_model_err_thresh: float = 60.0):
    """Run the production hybrid detector on every corrected GT entry, rank
    pages by mean corner error, and flag pages where the worst corner looks
    like a mis-click rather than a real model failure.

    Heuristic: for the single worst corner on each page, compare the GT
    position to where a perfect rectangle (built from the other three GT
    corners) would place it. If:
      - the model's error on that corner is large (>bug_model_err_thresh), AND
      - the GT corner is geometrically inconsistent with the other three
        (>bug_displacement_thresh from the rectangle-reconstructed point),
    then the GT is more likely wrong than the model. Flag for human review.

    Real page-corner damage (torn, curled) can trigger false positives. The
    flagged list is advisory, not automatic — always eyeball each one in
    the webapp before editing.
    """
    # Defer these imports: torch/cv2 and the ensemble loader are only needed
    # for audit, not for collect/eval/tune.
    import cv2 as _cv2
    import numpy as _np
    try:
        import comicml as _cm
    except ImportError:
        print("audit requires comicml.py (CNN/ensemble detector) — not found.")
        sys.exit(1)
    import torch as _torch

    entries = [e for e in load_ground_truth() if e["has_correction"]]
    if not entries:
        print("No corrected GT entries found. Run `collect` first.")
        sys.exit(1)

    ensemble, device = _cm._get_cached_ensemble()
    if not ensemble:
        print("No ensemble models available. Check comicml.ENSEMBLE_MODELS.")
        sys.exit(1)
    print(f"Loaded ensemble: {len(ensemble)} models. Auditing {len(entries)} pages…")

    _IX = [+1, -1, -1, +1]
    _IY = [+1, +1, -1, -1]
    SHIFT_X, SHIFT_Y = 13.0, 11.0

    rows = []
    for i, e in enumerate(entries):
        img = _cv2.imread(e["filepath"])
        if img is None:
            continue
        if e["gt_rotate180"]:
            img = _cv2.rotate(img, _cv2.ROTATE_180)
        cnn, dis = _cm.predict_corners_ensemble(ensemble, device, img)
        hyb = _cm.refine_corners_linefit(img, cnn, dpi=e.get("dpi", 600),
                                         tta_disagreements=dis)
        hyb = [[p[0] + SHIFT_X * _IX[j], p[1] + SHIFT_Y * _IY[j]]
               for j, p in enumerate(hyb)]
        gt = e["gt_corners"]
        per_corner = [float(_np.hypot(p[0]-g[0], p[1]-g[1]))
                      for p, g in zip(hyb, gt)]
        rows.append({
            "scan": e["scan_dir"].rsplit("/", 1)[-1],
            "page": e["page_index"],
            "file": Path(e["filepath"]).name,
            "filepath": e["filepath"],
            "mean": float(_np.mean(per_corner)),
            "max": float(max(per_corner)),
            "per_corner": per_corner,
            "gt": gt,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(entries)}")

    # Reconstruction heuristic: worst-corner position is (other_side_x,
    # other_side_y). TL=0, TR=1, BR=2, BL=3.
    RECON = {0: (3, 1), 1: (2, 0), 2: (1, 3), 3: (0, 2)}
    NAMES = ["TL", "TR", "BR", "BL"]

    flagged = []
    for r in rows:
        bi = int(_np.argmax(r["per_corner"]))
        model_err = r["per_corner"][bi]
        xi, yi = RECON[bi]
        gt = r["gt"]
        recon = (gt[xi][0], gt[yi][1])
        recon_err = float(_np.hypot(gt[bi][0] - recon[0], gt[bi][1] - recon[1]))
        if recon_err > bug_displacement_thresh and model_err > bug_model_err_thresh:
            flagged.append({
                "scan": r["scan"], "page": r["page"], "file": r["file"],
                "filepath": r["filepath"], "corner": NAMES[bi],
                "model_err": model_err, "displacement": recon_err,
            })

    flagged.sort(key=lambda r: -r["model_err"])

    # Overall stats (mirrors the shape of `evaluate`)
    means = _np.array([r["mean"] for r in rows])
    print(f"\n=== Hybrid detector on {len(rows)} pages ===")
    print(f"  Mean   corner err: {means.mean():7.2f} px")
    print(f"  Median corner err: {_np.median(means):7.2f} px")
    print(f"  P95    corner err: {_np.percentile(means, 95):7.2f} px")
    print(f"  Max    corner err: {means.max():7.2f} px")

    rows_sorted = sorted(rows, key=lambda r: -r["mean"])
    print(f"\n=== Top {top_n} worst pages by mean corner error ===")
    print(f"{'#':<3} {'scan':<20} {'pg':<4} {'file':<18} {'mean':>7} {'max':>7}")
    print("-" * 70)
    for k, r in enumerate(rows_sorted[:top_n]):
        print(f"{k+1:<3} {r['scan']:<20} {r['page']:<4} {r['file']:<18} "
              f"{r['mean']:7.2f} {r['max']:7.2f}")

    print(f"\n=== {len(flagged)} pages flagged as likely GT bugs ===")
    print(f"   (worst corner >{bug_model_err_thresh:.0f}px from model AND "
          f">{bug_displacement_thresh:.0f}px from rectangle-reconstructed position)\n")
    if flagged:
        print(f"{'#':<3} {'scan':<20} {'pg':<4} {'file':<18} {'corner':<7} {'disp':>7}")
        print("-" * 70)
        for k, r in enumerate(flagged):
            print(f"{k+1:<3} {r['scan']:<20} {r['page']:<4} {r['file']:<18} "
                  f"{r['corner']:<7} {r['displacement']:7.1f}")
        print("\nReview each in the webapp. If the flagged corner marker is clearly "
              "off the real page corner, drag it to the right spot and save.")
        print("Then re-run: comiceval.py collect <scans> && comiceval.py audit")
    else:
        print("No likely-GT-bug pages found.")


def main():
    parser = argparse.ArgumentParser(
        description="Ground truth evaluation and parameter tuning for comicscans detection."
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Collect ground truth from session files")
    p_collect.add_argument("dirs", nargs="+", help="Scan directories (or parent directory)")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate detection accuracy against ground truth")
    p_eval.add_argument("--params", help="JSON file with tuned parameters")

    # tune
    p_tune = sub.add_parser("tune", help="Tune detection parameters against ground truth")

    # summary
    sub.add_parser("summary", help="Print ground truth summary")

    # audit
    p_audit = sub.add_parser("audit",
        help="Run hybrid detector and flag pages where ground truth likely has a mis-clicked corner")
    p_audit.add_argument("--top", type=int, default=20,
                         help="Show top-N pages by mean corner error (default 20)")
    p_audit.add_argument("--bug-displacement", type=float, default=40.0,
                         help="Min geometric inconsistency (px) to flag as likely GT bug (default 40)")
    p_audit.add_argument("--bug-model-err", type=float, default=60.0,
                         help="Min model-vs-GT error (px) required to flag as GT bug (default 60)")

    args = parser.parse_args()

    if args.command == "collect":
        # Find scan directories (walk one level for parent dirs)
        scan_dirs = []
        for d in args.dirs:
            p = Path(d)
            if (p / ".comicscans_session.json").exists():
                scan_dirs.append(p)
            else:
                # Check subdirectories
                for child in sorted(p.iterdir()):
                    if child.is_dir() and (child / ".comicscans_session.json").exists():
                        scan_dirs.append(child)

        if not scan_dirs:
            print(f"No .comicscans_session.json files found in {args.dirs}")
            sys.exit(1)

        print(f"Found {len(scan_dirs)} session(s):\n")
        entries = collect_ground_truth(scan_dirs)
        save_ground_truth(entries)
        export_for_webapp(entries)

    elif args.command == "eval":
        entries = load_ground_truth()
        params = None
        if args.params:
            params = json.loads(Path(args.params).read_text())
            print(f"Using parameters from {args.params}\n")
        evaluate(entries, params)

    elif args.command == "tune":
        entries = load_ground_truth()
        tuned = tune_parameters(entries)
        print("\n--- Re-evaluating with tuned parameters ---")
        evaluate(entries, tuned)

    elif args.command == "summary":
        entries = load_ground_truth()
        export_for_webapp(entries)

    elif args.command == "audit":
        audit_ground_truth(
            top_n=args.top,
            bug_displacement_thresh=args.bug_displacement,
            bug_model_err_thresh=args.bug_model_err,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
