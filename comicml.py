#!/usr/bin/env python3
"""
comicml.py — CNN inference for comic page corner detection.

Runtime inference only. Training, evaluation, and the training dashboard
live in the separate comicml project at /Users/james/Documents/dev/comicml.

Usage:
    # Predict corners for a single image (for spot-checking)
    python3 comicml.py predict path/to/Scan.jpeg --model comicml_model_reg_768_1420pg_e280.pt
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

MODEL_FILE = Path(__file__).parent / "comicml_model_reg_768_1420pg_e280.pt"

# Production ensemble: hybrid eval on DS9E20+E23+DS9_1996_5 holdout. If an
# ensemble_config.json exists alongside this file it is read first and its
# "models" list replaces this default — use the training dashboard "Add to
# ensemble" button to update it without touching this file.
# Set ENSEMBLE_MODELS = [] (or leave files missing) to fall back to single-model.
ENSEMBLE_MODELS = [
    "comicml_model_reg_768_956pg.pt",        # seed 137, 956 pages
    "comicml_model_reg_768_1000pg.pt",       # seed 137, 1000 pages
    "comicml_model_reg_768_1420pg_e280.pt",  # seed 137, 1420 pages, 280 epochs (champion)
]

# Default input resolution for the CNN. 512 gives each feature cell ~10 orig
# pixels at 600 DPI; 768 cuts that to ~7 px and improves localization at
# ~2.25× per-epoch training cost. Stored per-checkpoint so different model
# files can use different resolutions.
INPUT_SIZE = 512

# ImageNet normalization (ResNet-18 is pretrained on it)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def _load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "regression")
    if model_type == "heatmap":
        model = CornerHeatmapRegressor(pretrained=False).to(device)
    else:
        model = CornerRegressor(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Tag the model with its training input size + type so predict_corners picks it up
    model._input_size = ckpt.get("input_size", INPUT_SIZE)
    model._model_type = model_type
    return model, ckpt


def predict_corners(model, device, image_bgr, rotate180=False, input_size=None,
                    tta=True):
    """Given a BGR image (already loaded, possibly 180-rotated), return corners
    in original image pixel space as [TL, TR, BR, BL] list of [x, y].

    With tta=True (default), runs inference on both the image and its
    horizontal mirror, un-mirrors the second result, and averages. Free
    variance reduction — typically shaves 5–10% off mean corner error.
    """
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    size = input_size if input_size is not None else getattr(model, "_input_size", INPUT_SIZE)
    pred_a = _predict_single(model, device, image_bgr, size)
    if not tta:
        return pred_a
    # Horizontal flip → predict → un-flip x and swap corner roles
    W = image_bgr.shape[1]
    flipped = cv2.flip(image_bgr, 1)
    pred_b = _predict_single(model, device, flipped, size)
    pred_b = [[W - x, y] for x, y in pred_b]
    # After mirror, [TL,TR,BR,BL] corresponds to original [TR,TL,BL,BR]
    pred_b = [pred_b[1], pred_b[0], pred_b[3], pred_b[2]]
    return [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
            for a, b in zip(pred_a, pred_b)]


def predict_corners_with_disagreement(model, device, image_bgr, rotate180=False,
                                      input_size=None):
    """Like predict_corners(tta=True), but also returns per-corner TTA
    disagreement (distance between original and mirrored predictions before
    averaging). High disagreement = low CNN confidence on that corner.

    Returns (corners, disagreements) where both are [[x,y],...] × 4 and
    [float,...] × 4 respectively.
    """
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    size = input_size if input_size is not None else getattr(model, "_input_size", INPUT_SIZE)
    pred_a = _predict_single(model, device, image_bgr, size)
    W = image_bgr.shape[1]
    flipped = cv2.flip(image_bgr, 1)
    pred_b = _predict_single(model, device, flipped, size)
    pred_b = [[W - x, y] for x, y in pred_b]
    pred_b = [pred_b[1], pred_b[0], pred_b[3], pred_b[2]]
    corners = [[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
               for a, b in zip(pred_a, pred_b)]
    disagreements = [float(np.hypot(a[0] - b[0], a[1] - b[1]))
                     for a, b in zip(pred_a, pred_b)]
    return corners, disagreements


def predict_corners_ensemble(models, device, image_bgr, rotate180=False):
    """Average predictions from multiple models (multi-seed ensemble).

    Each model runs with TTA independently, results are averaged per-corner.
    Also returns per-corner TTA disagreement (averaged across models).
    """
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    all_corners = []
    all_disagree = []
    for model in models:
        c, d = predict_corners_with_disagreement(model, device, image_bgr)
        all_corners.append(c)
        all_disagree.append(d)
    avg_corners = [[float(np.mean([c[i][0] for c in all_corners])),
                    float(np.mean([c[i][1] for c in all_corners]))]
                   for i in range(4)]
    avg_disagree = [float(np.mean([d[i] for d in all_disagree]))
                    for i in range(4)]
    return avg_corners, avg_disagree


def _predict_single(model, device, image_bgr, size):
    H, W = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t).unsqueeze(0).to(device)
    model_type = getattr(model, "_model_type", "regression")
    with torch.no_grad():
        out = model(t)
        if model_type == "heatmap":
            coords = _soft_argmax_2d(out).view(1, 4, 2)  # [1, 4, 2] normalized
            pred = coords.cpu().numpy().reshape(4, 2)
        else:
            pred = out.cpu().numpy().reshape(4, 2)
    pred[:, 0] *= W
    pred[:, 1] *= H
    return pred.tolist()


# ---------------------------------------------------------------------------
# Hybrid: CNN prior + classical edge-snap refinement
# ---------------------------------------------------------------------------

def _refine_coord(profile, center_idx, smoothing=9, min_peak_ratio=2.0):
    """Find the strongest 1D gradient peak in `profile`, preferring positions
    near `center_idx`. Returns (refined_idx, confidence).

    confidence: ratio of peak gradient to median gradient in the window.
    Values near 1.0 mean no clear edge (flat region); high values mean a
    strong, well-defined edge. Callers can use this to skip refinement on
    low-confidence windows (the CNN's prior is likely better than noise).
    """
    if len(profile) < 3:
        return center_idx, 0.0
    grad = np.abs(np.gradient(profile.astype(np.float32)))
    # Smooth to reject isolated noise peaks
    k = min(smoothing, max(3, len(grad) // 10) | 1)  # odd
    kernel = np.ones(k, dtype=np.float32) / k
    smoothed = np.convolve(grad, kernel, mode="same")
    # Down-weight positions far from the CNN prior (gaussian around center)
    sigma = max(len(profile) / 3.0, 10.0)
    dist = np.arange(len(profile), dtype=np.float32) - center_idx
    weight = np.exp(-0.5 * (dist / sigma) ** 2)
    scored = smoothed * weight
    peak_idx = int(np.argmax(scored))
    peak_val = float(smoothed[peak_idx])
    median = float(np.median(smoothed)) + 1e-6
    confidence = peak_val / median
    return peak_idx, confidence


def refine_corners(image_bgr, cnn_corners, dpi=600,
                   search_in=0.20, strip_in=0.15,
                   min_confidence=1.75):
    """Refine each CNN corner by snapping x and y independently to the nearest
    strong edge. Returns refined corners as [[x,y], ...] × 4.

    The CNN lands within ~80 px of truth; classical edge detection in a small
    window around each corner is accurate to ~1 px on clean edges. If no clear
    edge is found (confidence < min_confidence), the CNN value is preserved —
    better than snapping to noise.

    Parameters (inches — scaled by DPI):
      search_in: search radius along the axis being refined (default 0.125" →
                 ~75 px at 600 DPI). Matches the CNN's typical error band.
      strip_in:  half-width of the perpendicular band averaged to form the 1D
                 profile. Larger = more noise suppression, but risks including
                 irrelevant content.
      min_confidence: peak-gradient / median-gradient ratio below which the
                      edge is considered ambiguous and we fall back to CNN.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    search_r = int(dpi * search_in)
    strip_w = int(dpi * strip_in)

    refined = []
    for cx, cy in cnn_corners:
        cx_i, cy_i = int(cx), int(cy)

        # --- Refine y (top/bottom edge): horizontal band → vertical profile ---
        y0 = max(0, cy_i - search_r)
        y1 = min(H, cy_i + search_r + 1)
        x0 = max(0, cx_i - strip_w)
        x1 = min(W, cx_i + strip_w + 1)
        strip = gray[y0:y1, x0:x1]
        if strip.size > 0 and strip.shape[0] >= 3:
            vprofile = strip.mean(axis=1)
            peak, conf = _refine_coord(vprofile, cy_i - y0)
            refined_y = (y0 + peak) if conf >= min_confidence else cy
        else:
            refined_y = cy

        # --- Refine x (left/right edge): vertical band → horizontal profile ---
        y0 = max(0, cy_i - strip_w)
        y1 = min(H, cy_i + strip_w + 1)
        x0 = max(0, cx_i - search_r)
        x1 = min(W, cx_i + search_r + 1)
        strip = gray[y0:y1, x0:x1]
        if strip.size > 0 and strip.shape[1] >= 3:
            hprofile = strip.mean(axis=0)
            peak, conf = _refine_coord(hprofile, cx_i - x0)
            refined_x = (x0 + peak) if conf >= min_confidence else cx
        else:
            refined_x = cx

        refined.append([float(refined_x), float(refined_y)])

    return refined


def _fit_line_ransac(points, n_iter=100, inlier_thresh=8.0):
    """Fit a line to 2D points via RANSAC. Returns (a, b, c) for ax+by+c=0,
    normalized so sqrt(a²+b²)=1, or None if too few points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return None
    if n == 2:
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

    # Refit with all inliers (SVD least-squares)
    a, b, c = best_line
    dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
    inlier_pts = pts[dists < inlier_thresh]
    if len(inlier_pts) < 2:
        return best_line
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1]
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

    axis='y': detecting a horizontal edge (top/bottom) — vertical profiles.
    axis='x': detecting a vertical edge (left/right) — horizontal profiles.

    Returns list of (x, y) detected edge points.
    """
    H, W = gray.shape
    points = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        cx = p0[0] + t * (p1[0] - p0[0])
        cy = p0[1] + t * (p1[1] - p0[1])
        cx_i, cy_i = int(cx), int(cy)

        if axis == 'y':
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
        else:
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
                           search_in=0.15, strip_in=0.15,
                           min_confidence=1.75,
                           n_samples=30, inlier_thresh=8.0,
                           min_edge_points=5, agree_px=40,
                           tta_disagreements=None, skip_refine_thresh=0.0):
    """Refine corners by fitting lines to detected edge points along each of
    the 4 page edges, then intersecting adjacent lines.

    Falls back to per-corner snap for any corner where:
      - An adjacent edge has too few confident detections to fit a line
      - The line-fit intersection disagrees with per-corner snap by >agree_px

    Adaptive skip (#4): if tta_disagreements is provided (per-corner TTA
    disagreement in pixels), corners with disagreement < skip_refine_thresh
    are kept as-is from the CNN — the model is confident and refinement is
    more likely to hurt than help.

    Parameters:
      n_samples: number of 1D profiles sampled along each edge
      inlier_thresh: RANSAC inlier distance in pixels
      min_edge_points: minimum confident detections to attempt line fit
      agree_px: max allowed distance between line-fit and per-corner-snap
                results; beyond this, per-corner-snap wins (safety net)
      tta_disagreements: per-corner TTA disagreement [float] × 4, or None
      skip_refine_thresh: skip refinement for corners with TTA disagreement
                          below this (pixels). Only used if tta_disagreements
                          is provided.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    search_r = int(dpi * search_in)
    strip_w = int(dpi * strip_in)

    tl, tr, br, bl = cnn_corners

    # Detect edge points along each of the 4 edges
    top_pts = _sample_edge_points(gray, tl, tr, n_samples, search_r, strip_w,
                                  axis='y', min_confidence=min_confidence)
    bot_pts = _sample_edge_points(gray, bl, br, n_samples, search_r, strip_w,
                                  axis='y', min_confidence=min_confidence)
    left_pts = _sample_edge_points(gray, tl, bl, n_samples, search_r, strip_w,
                                   axis='x', min_confidence=min_confidence)
    right_pts = _sample_edge_points(gray, tr, br, n_samples, search_r, strip_w,
                                    axis='x', min_confidence=min_confidence)

    # Fit lines via RANSAC
    top_line = _fit_line_ransac(top_pts, inlier_thresh=inlier_thresh) if len(top_pts) >= min_edge_points else None
    bot_line = _fit_line_ransac(bot_pts, inlier_thresh=inlier_thresh) if len(bot_pts) >= min_edge_points else None
    left_line = _fit_line_ransac(left_pts, inlier_thresh=inlier_thresh) if len(left_pts) >= min_edge_points else None
    right_line = _fit_line_ransac(right_pts, inlier_thresh=inlier_thresh) if len(right_pts) >= min_edge_points else None

    # Intersect: TL = top∩left, TR = top∩right, BR = bot∩right, BL = bot∩left
    lines_for_corner = [
        (top_line, left_line),
        (top_line, right_line),
        (bot_line, right_line),
        (bot_line, left_line),
    ]

    # Per-corner snap as conservative fallback
    fallback = refine_corners(image_bgr, cnn_corners, dpi=dpi,
                              search_in=search_in, strip_in=strip_in,
                              min_confidence=min_confidence)

    refined = []
    for i, (la, lb) in enumerate(lines_for_corner):
        # Adaptive skip: if the CNN is confident (low TTA disagreement),
        # keep the CNN prediction — refinement is more likely to hurt.
        if (tta_disagreements is not None and
                tta_disagreements[i] < skip_refine_thresh):
            refined.append(list(cnn_corners[i]))
            continue

        if la is not None and lb is not None:
            pt = _intersect_lines(la, lb)
            if pt is not None:
                dist_from_snap = np.hypot(pt[0] - fallback[i][0],
                                          pt[1] - fallback[i][1])
                if dist_from_snap < agree_px:
                    refined.append(pt)
                    continue
        refined.append(fallback[i])

    return refined


def predict_corners_hybrid(model, device, image_bgr, rotate180=False, dpi=600):
    """CNN prediction + line-fit edge refinement with adaptive skip."""
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    cnn, disagreements = predict_corners_with_disagreement(model, device, image_bgr)
    return refine_corners_linefit(image_bgr, cnn, dpi=dpi,
                                  tta_disagreements=disagreements)


# ---------------------------------------------------------------------------
# Drop-in replacement for comicscans.detect_page_bounds()
# ---------------------------------------------------------------------------

# Module-level cache so we load the model once per process (not per page)
_MODEL_CACHE = {}


def _get_cached_model(model_path):
    """Lazily load a model, caching by path. Returns (model, device)."""
    key = str(model_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = _load_model(model_path, device)
    _MODEL_CACHE[key] = (model, device)
    return model, device


def _resolve_ensemble_paths():
    """Return list of absolute ensemble model paths that exist on disk.
    Reads ensemble_config.json first if present (written by the comicml
    training dashboard); falls back to the hardcoded ENSEMBLE_MODELS list.
    Silently skips missing files so a partial ensemble still works."""
    base = Path(__file__).parent
    config_path = base / "ensemble_config.json"
    if config_path.exists():
        try:
            import json as _json
            names = _json.loads(config_path.read_text()).get("models", [])
        except Exception:
            names = ENSEMBLE_MODELS
    else:
        names = ENSEMBLE_MODELS
    return [base / p for p in names if (base / p).exists()]


def _get_cached_ensemble():
    """Load and cache the production ensemble models once. Returns (models, device)."""
    paths = _resolve_ensemble_paths()
    if not paths:
        return [], None
    key = ("__ensemble__", tuple(str(p) for p in paths))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models_list = []
    for p in paths:
        m, _ = _load_model(p, device)
        models_list.append(m)
    _MODEL_CACHE[key] = (models_list, device)
    return models_list, device


def detect_page_bounds_hybrid(image, dpi=600, model_path=None,
                              inward_shift_x=13.0, inward_shift_y=11.0):
    """Hybrid CNN+edge-snap detector. Returns the same dict format as
    comicscans.detect_page_bounds() so it can be used as a drop-in replacement.

    The return dict has {top, bottom, left, right, angle, spine_col, bleed_method}
    where top/bottom/left/right are in the DESKEWED canvas coordinate space
    (matching the original detector's convention, so _bounds_to_original_corners
    in the webapp works unchanged).

    Aesthetic-crop post-shift:
      inward_shift_x, inward_shift_y: pixels to trim inward on X and Y axes.
      Applied as a uniform inset of the final bounds. Defaults (13, 11) were
      measured as the median residual between hybrid predictions and manual
      overrides on the DS9E20+E23 holdout. At these defaults, mean holdout
      error drops from 21.4 → 13.2 px. Set to 0 to disable.
    """
    # Prefer the production ensemble if configured and all members exist;
    # otherwise fall back to a single model.
    models_list, device = _get_cached_ensemble() if model_path is None else ([], None)
    if len(models_list) >= 2:
        cnn, disagreements = predict_corners_ensemble(models_list, device, image)
        bleed_method = f"cnn+snap (ensemble×{len(models_list)})"
    else:
        model_path = model_path or MODEL_FILE
        model, device = _get_cached_model(model_path)
        cnn, disagreements = predict_corners_with_disagreement(model, device, image)
        bleed_method = "cnn+snap"

    # CNN + refinement give us 4 corners in original image pixel space,
    # in [TL, TR, BR, BL] order.
    corners = np.array(refine_corners_linefit(image, cnn, dpi=dpi,
                                               tta_disagreements=disagreements),
                       dtype=np.float64)

    # Measure skew from the top edge (TL → TR)
    tl, tr, br, bl = corners
    dy = tr[1] - tl[1]
    dx = tr[0] - tl[0]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    # Clip to the same small-correction range the classical detector uses
    if abs(angle) > 5.0 or abs(angle) < 0.1:
        angle = 0.0

    H, W = image.shape[:2]
    if angle == 0.0:
        # No deskew: bounds = axis-aligned bounding box of the corners in
        # original image space.
        top = float(min(tl[1], tr[1]))
        bottom = float(max(bl[1], br[1]))
        left = float(min(tl[0], bl[0]))
        right = float(max(tr[0], br[0]))
        # Aesthetic inward crop
        top += inward_shift_y
        bottom -= inward_shift_y
        left += inward_shift_x
        right -= inward_shift_x
        return {
            "top": int(round(top)), "bottom": int(round(bottom)),
            "left": int(round(left)), "right": int(round(right)),
            "angle": 0.0, "spine_col": None, "bleed_method": bleed_method,
        }

    # Non-zero deskew: rotate corners by -angle about the original center,
    # then translate to the deskewed canvas (which is larger, matching
    # _deskew_gray's BORDER_CONSTANT expansion).
    rad = np.deg2rad(abs(angle))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    new_w = int(H * sin_a + W * cos_a)
    new_h = int(H * cos_a + W * sin_a)

    # Inverse rotation by angle (not -angle): _bounds_to_original_corners uses
    # +angle to go desk→orig, so we use -angle to go orig→desk.
    theta = np.deg2rad(-angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    desk = []
    for px, py in corners:
        dx0 = px - W / 2.0
        dy0 = py - H / 2.0
        rx = dx0 * cos_t - dy0 * sin_t
        ry = dx0 * sin_t + dy0 * cos_t
        desk.append([rx + new_w / 2.0, ry + new_h / 2.0])
    desk = np.array(desk)

    top = float(min(desk[0, 1], desk[1, 1]))
    bottom = float(max(desk[2, 1], desk[3, 1]))
    left = float(min(desk[0, 0], desk[3, 0]))
    right = float(max(desk[1, 0], desk[2, 0]))
    # Aesthetic inward crop (applied in deskewed coord space, matching
    # the no-deskew branch above)
    top += inward_shift_y
    bottom -= inward_shift_y
    left += inward_shift_x
    right -= inward_shift_x

    return {
        "top": int(round(top)), "bottom": int(round(bottom)),
        "left": int(round(left)), "right": int(round(right)),
        "angle": angle, "spine_col": None, "bleed_method": bleed_method,
    }


def predict_cli(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = _load_model(args.model or MODEL_FILE, device)
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read {args.image}"); sys.exit(1)
    corners = predict_corners(model, device, img)
    print(json.dumps({"corners": corners, "image_size": [img.shape[1], img.shape[0]]}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point — inference only. For training, use the comicml project."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Predict corners for a single image")
    p_pred.add_argument("image")
    p_pred.add_argument("--model", default=None)

    args = parser.parse_args()
    if args.cmd == "predict":
        predict_cli(args)


if __name__ == "__main__":
    main()
