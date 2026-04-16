#!/usr/bin/env python3
"""
comicml.py — CNN-based page corner regression.

Trains a ResNet-18 backbone to predict 4 corners (TL, TR, BR, BL) for each
scan, using the ground truth collected in ground_truth.json. Replaces the
rule-based detector for pages where it hits its accuracy ceiling.

Usage:
    # Train on 4 issues, hold out 2 for eval
    python3 comicml.py train \\
        --train DS9E18,DS9E19,DS9E21,DS9E22 \\
        --holdout DS9E20,DS9E23 \\
        --epochs 60

    # Evaluate a trained model
    python3 comicml.py eval --model comicml_model.pt

    # Predict corners for a single image (for inspection)
    python3 comicml.py predict path/to/Scan.jpeg --model comicml_model.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"
MODEL_FILE = Path(__file__).parent / "comicml_model.pt"

# Default input resolution for the CNN. 512 gives each feature cell ~10 orig
# pixels at 600 DPI; 768 cuts that to ~7 px and improves localization at
# ~2.25× per-epoch training cost. Stored per-checkpoint so different model
# files can use different resolutions.
INPUT_SIZE = 512

# ImageNet normalization (ResNet-18 is pretrained on it)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_entries():
    if not GROUND_TRUTH_FILE.exists():
        print(f"Missing {GROUND_TRUTH_FILE}. Run: python3 comiceval.py collect raw-scans/")
        sys.exit(1)
    return json.loads(GROUND_TRUTH_FILE.read_text())


class PageCornerDataset(Dataset):
    """Loads (image, 8-dim normalized corner target) pairs.

    Image is resized to INPUT_SIZE × INPUT_SIZE (ignoring aspect). Corners are
    normalized to [0, 1] using original image dims — so the network learns
    position as a fraction of each axis, independent of the aspect-distort
    introduced by resizing.

    Augmentation at train time: horizontal flip (corners swap accordingly),
    small brightness/contrast jitter. No rotations — the scanner pipeline
    already deskews before the network would see the image in production.
    """

    def __init__(self, entries, augment=False, input_size=INPUT_SIZE):
        self.entries = entries
        self.augment = augment
        self.input_size = input_size
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = cv2.imread(entry["filepath"])
        if img is None:
            raise RuntimeError(f"Could not read {entry['filepath']}")
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)

        H, W = img.shape[:2]
        corners = np.array(entry["gt_corners"], dtype=np.float32)  # [4, 2] = TL, TR, BR, BL

        # Normalize corners to [0, 1] in original image coords
        norm = corners.copy()
        norm[:, 0] /= W
        norm[:, 1] /= H

        # Resize image (aspect-distorting) and convert BGR → RGB
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            # Horizontal flip: mirror image and remap corners (TL<->TR, BL<->BR)
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                flipped = norm.copy()
                flipped[:, 0] = 1.0 - flipped[:, 0]
                # Reorder: original [TL, TR, BR, BL] → after mirror x, roles
                # swap horizontally: [TR', TL', BL', BR']. Re-index to keep
                # the TL/TR/BR/BL convention:
                norm = np.array([flipped[1], flipped[0], flipped[3], flipped[2]])

            # Brightness/contrast jitter (small)
            if random.random() < 0.5:
                alpha = 1.0 + (random.random() - 0.5) * 0.2  # 0.9–1.1
                beta = (random.random() - 0.5) * 20          # -10..+10
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # To float tensor [C, H, W] in [0, 1], then ImageNet-normalize
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = self.normalize(img_t)

        target = torch.from_numpy(norm.flatten()).float()  # [8]
        meta = {
            "filepath": entry["filepath"],
            "orig_w": W,
            "orig_h": H,
            "scan_dir": entry["scan_dir"],
            "page_index": entry["page_index"],
        }
        return img_t, target, meta


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CornerRegressor(nn.Module):
    """ResNet-18 backbone + linear head predicting 8 normalized corner coords."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Replace classification head with 8-dim regression head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 8)
        self.net = backbone

    def forward(self, x):
        # Output is in [0, 1]-ish range after sigmoid to keep predictions bounded.
        return torch.sigmoid(self.net(x))


class CornerHeatmapRegressor(nn.Module):
    """ResNet-18 encoder + deconv decoder predicting a 4-channel corner heatmap.

    At inference, coordinates are extracted via soft-argmax for sub-pixel
    accuracy. The deconv decoder upsamples by 8× (stride 32 → stride 4),
    producing a heatmap at input_size / 4 resolution.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        # Keep everything up to layer4 (output: [B, 512, H/32, W/32])
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        # Three transposed-conv blocks upsample ×8 (stride 32 → stride 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        # 1×1 conv to 4 channels (one heatmap per corner)
        self.final = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)
        up = self.decoder(feats)
        return self.final(up)  # [B, 4, H/4, W/4] — raw logits


def _make_heatmap_targets(corners_norm, hmap_size, sigma=2.0, device=None):
    """Build Gaussian heatmap targets from normalized corners.

    corners_norm: [B, 8] in [0, 1]. Returns [B, 4, H, W] float tensor.
    sigma is in heatmap-pixels; 2.0 at 192×192 covers ~5 px FWHM ≈ 20 orig-px.
    """
    B = corners_norm.shape[0]
    H = W = hmap_size
    coords = corners_norm.view(B, 4, 2)
    px = coords[..., 0] * (W - 1)  # [B, 4]
    py = coords[..., 1] * (H - 1)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    # Broadcast to [B, 4, H, W]
    px = px.view(B, 4, 1, 1)
    py = py.view(B, 4, 1, 1)
    return torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))


def _soft_argmax_2d(heatmap, temperature=1.0):
    """Differentiable sub-pixel argmax.

    heatmap: [B, C, H, W] logits. Returns [B, C, 2] normalized coords in [0, 1]
    as (x, y).
    """
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, C, -1) / temperature
    probs = torch.softmax(flat, dim=-1).view(B, C, H, W)
    xs = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, W) / (W - 1)
    ys = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, H, 1) / (H - 1)
    x = (probs * xs).sum(dim=(2, 3))
    y = (probs * ys).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)  # [B, C, 2]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _split_entries(entries, train_dirs, holdout_dirs):
    """Split ground truth entries into train and holdout sets by scan_dir."""
    train, holdout = [], []
    train_set = set(train_dirs)
    holdout_set = set(holdout_dirs)
    for e in entries:
        name = e["scan_dir"].rsplit("/", 1)[-1]
        if name in train_set:
            train.append(e)
        elif name in holdout_set:
            holdout.append(e)
    return train, holdout


def _corner_px_error(pred_norm, target_norm, orig_w, orig_h):
    """Mean corner distance in original-image pixels for a batch.

    pred_norm, target_norm: [B, 8] normalized. orig_w, orig_h: [B] ints.
    """
    B = pred_norm.shape[0]
    pred = pred_norm.view(B, 4, 2)
    tgt = target_norm.view(B, 4, 2)
    scale = torch.stack([orig_w.float(), orig_h.float()], dim=1).view(B, 1, 2)
    pred_px = pred * scale
    tgt_px = tgt * scale
    d = torch.sqrt(((pred_px - tgt_px) ** 2).sum(dim=2))  # [B, 4]
    return d.mean().item(), d  # scalar mean + per-corner distances


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    entries = _load_entries()
    train_dirs = args.train.split(",")
    holdout_dirs = args.holdout.split(",")
    train_entries, holdout_entries = _split_entries(entries, train_dirs, holdout_dirs)

    # Drop anything without a correction so we train only on labeled data
    train_entries = [e for e in train_entries if e["has_correction"]]
    holdout_entries = [e for e in holdout_entries if e["has_correction"]]

    print(f"Train: {len(train_entries)} pages from {train_dirs}")
    print(f"Holdout: {len(holdout_entries)} pages from {holdout_dirs}")

    input_size = args.input_size
    print(f"Input resolution: {input_size}×{input_size}")
    train_ds = PageCornerDataset(train_entries, augment=True, input_size=input_size)
    val_ds = PageCornerDataset(holdout_entries, augment=False, input_size=input_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)

    model_type = "heatmap" if args.heatmap else "regression"
    print(f"Model type: {model_type}")
    if model_type == "heatmap":
        model = CornerHeatmapRegressor(pretrained=True).to(device)
        hmap_size = input_size // 4  # stride-4 heatmap
        print(f"Heatmap resolution: {hmap_size}×{hmap_size}  σ={args.hmap_sigma}")
    else:
        model = CornerRegressor(pretrained=True).to(device)
        hmap_size = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    l1_loss = nn.SmoothL1Loss()
    mse_loss = nn.MSELoss()

    def forward_loss(imgs, targets):
        """Returns (loss, pred_coords_norm [B,8]).

        Heatmap path uses DSNT-style supervision: coord L1 loss on soft-argmax +
        a small MSE heatmap regularizer toward a Gaussian target, which keeps
        the predicted distribution peaked rather than diffuse. Without the
        regularizer the network can satisfy the coord loss with multi-modal
        distributions that centroid to the right point.
        """
        out = model(imgs)
        if model_type == "heatmap":
            coords = _soft_argmax_2d(out).view(imgs.size(0), 8)
            coord_loss = l1_loss(coords, targets)
            target_hmap = _make_heatmap_targets(
                targets, hmap_size, sigma=args.hmap_sigma, device=device)
            # Normalize logits to a probability distribution for the reg term
            probs = torch.softmax(out.view(out.size(0), 4, -1), dim=-1).view_as(out)
            # Scale target so it sums to 1 per channel (JS-flavored regularizer)
            tgt_sum = target_hmap.sum(dim=(2, 3), keepdim=True).clamp(min=1e-8)
            target_probs = target_hmap / tgt_sum
            reg = mse_loss(probs, target_probs)
            loss = coord_loss + args.hmap_reg * reg
        else:
            loss = l1_loss(out, targets)
            coords = out
        return loss, coords

    best_val_px = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_px_sum = 0.0
        train_n = 0
        for imgs, targets, meta in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            loss, coords = forward_loss(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            orig_w = meta["orig_w"].to(device)
            orig_h = meta["orig_h"].to(device)
            px, _ = _corner_px_error(coords.detach(), targets, orig_w, orig_h)
            train_px_sum += px * imgs.size(0)
            train_n += imgs.size(0)
        scheduler.step()

        # Validation
        model.eval()
        val_px_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for imgs, targets, meta in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                _, coords = forward_loss(imgs, targets)
                orig_w = torch.tensor(meta["orig_w"]).to(device)
                orig_h = torch.tensor(meta["orig_h"]).to(device)
                px, _ = _corner_px_error(coords, targets, orig_w, orig_h)
                val_px_sum += px * imgs.size(0)
                val_n += imgs.size(0)

        train_px = train_px_sum / max(train_n, 1)
        val_px = val_px_sum / max(val_n, 1)
        elapsed = time.time() - t0
        marker = ""
        if val_px < best_val_px:
            best_val_px = val_px
            marker = " *"
            torch.save({
                "model_state": model.state_dict(),
                "input_size": input_size,
                "model_type": model_type,
                "hmap_sigma": args.hmap_sigma,
                "epoch": epoch,
                "val_px": val_px,
                "train_dirs": train_dirs,
                "holdout_dirs": holdout_dirs,
            }, MODEL_FILE)

        print(f"epoch {epoch+1:>3d}/{args.epochs}  "
              f"train_loss={train_loss/max(train_n,1):.5f}  "
              f"train_px={train_px:7.2f}  val_px={val_px:7.2f}  "
              f"({elapsed:.1f}s){marker}", flush=True)
        history.append((epoch, train_px, val_px))

    print(f"\nBest holdout mean corner error: {best_val_px:.2f} px")
    print(f"Model saved to {MODEL_FILE}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
                           min_edge_points=5, agree_px=40):
    """Refine corners by fitting lines to detected edge points along each of
    the 4 page edges, then intersecting adjacent lines.

    Falls back to per-corner snap for any corner where:
      - An adjacent edge has too few confident detections to fit a line
      - The line-fit intersection disagrees with per-corner snap by >agree_px

    Parameters:
      n_samples: number of 1D profiles sampled along each edge
      inlier_thresh: RANSAC inlier distance in pixels
      min_edge_points: minimum confident detections to attempt line fit
      agree_px: max allowed distance between line-fit and per-corner-snap
                results; beyond this, per-corner-snap wins (safety net)
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
    """CNN prediction + classical edge-snap refinement."""
    if rotate180:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_180)
    cnn = predict_corners(model, device, image_bgr, rotate180=False)
    return refine_corners_linefit(image_bgr, cnn, dpi=dpi)


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


def detect_page_bounds_hybrid(image, dpi=600, model_path=None):
    """Hybrid CNN+edge-snap detector. Returns the same dict format as
    comicscans.detect_page_bounds() so it can be used as a drop-in replacement.

    The return dict has {top, bottom, left, right, angle, spine_col, bleed_method}
    where top/bottom/left/right are in the DESKEWED canvas coordinate space
    (matching the original detector's convention, so _bounds_to_original_corners
    in the webapp works unchanged).
    """
    model_path = model_path or MODEL_FILE
    model, device = _get_cached_model(model_path)

    # CNN + refinement give us 4 corners in original image pixel space,
    # in [TL, TR, BR, BL] order.
    cnn = predict_corners(model, device, image, rotate180=False)
    corners = np.array(refine_corners_linefit(image, cnn, dpi=dpi), dtype=np.float64)

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
        return {
            "top": int(round(top)), "bottom": int(round(bottom)),
            "left": int(round(left)), "right": int(round(right)),
            "angle": 0.0, "spine_col": None, "bleed_method": "cnn+snap",
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

    return {
        "top": int(round(top)), "bottom": int(round(bottom)),
        "left": int(round(left)), "right": int(round(right)),
        "angle": angle, "spine_col": None, "bleed_method": "cnn+snap",
    }


def evaluate(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = args.model or MODEL_FILE
    model, ckpt = _load_model(model_path, device)
    print(f"Loaded model from {model_path} (trained {ckpt.get('epoch', '?')} epochs, "
          f"best val {ckpt.get('val_px', '?'):.2f} px)")

    entries = _load_entries()
    holdout_dirs = ckpt.get("holdout_dirs", [])
    if args.all:
        eval_entries = [e for e in entries if e["has_correction"]]
        print(f"Evaluating on ALL {len(eval_entries)} corrected pages")
    else:
        _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []), holdout_dirs)
        eval_entries = [e for e in eval_entries if e["has_correction"]]
        print(f"Evaluating on {len(eval_entries)} holdout pages ({holdout_dirs})")

    cnn_dists = []
    hybrid_dists = []
    per_dir_cnn = {}
    per_dir_hybrid = {}
    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None:
            continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cnn_pred = predict_corners(model, device, img, rotate180=False)
        gt = entry["gt_corners"]

        cnn_d = np.mean([np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(cnn_pred, gt)])
        cnn_dists.append(cnn_d)

        if args.hybrid:
            dpi = entry.get("dpi", 600)
            hyb_pred = refine_corners_linefit(img, cnn_pred, dpi=dpi)
            hyb_d = np.mean([np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(hyb_pred, gt)])
            hybrid_dists.append(hyb_d)

        name = entry["scan_dir"].rsplit("/", 1)[-1]
        per_dir_cnn.setdefault(name, []).append(cnn_d)
        if args.hybrid:
            per_dir_hybrid.setdefault(name, []).append(hyb_d)

    def _report(title, distances, per_dir):
        d = np.array(distances)
        print(f"\n=== {title} ===")
        print(f"Pages evaluated:  {len(d)}")
        print(f"Mean corner err:  {d.mean():7.2f} px")
        print(f"Median corner err:{np.median(d):7.2f} px")
        print(f"P95 corner err:   {np.percentile(d, 95):7.2f} px")
        print(f"Max corner err:   {d.max():7.2f} px")
        print(f"\nPer-directory:")
        for name in sorted(per_dir):
            v = np.array(per_dir[name])
            print(f"  {name:<12s}  n={len(v):<3d}  mean={v.mean():7.2f} px  "
                  f"median={np.median(v):7.2f} px  max={v.max():7.2f} px")

    _report("CNN Corner-Regressor", cnn_dists, per_dir_cnn)
    if args.hybrid:
        _report("Hybrid (CNN + edge snap)", hybrid_dists, per_dir_hybrid)
        cnn_arr = np.array(cnn_dists)
        hyb_arr = np.array(hybrid_dists)
        delta = cnn_arr - hyb_arr
        print(f"\n--- Hybrid vs CNN improvement ---")
        print(f"  Mean:   {cnn_arr.mean():6.2f} → {hyb_arr.mean():6.2f} px  ({delta.mean():+.2f})")
        print(f"  Median: {np.median(cnn_arr):6.2f} → {np.median(hyb_arr):6.2f} px")
        print(f"  P95:    {np.percentile(cnn_arr, 95):6.2f} → {np.percentile(hyb_arr, 95):6.2f} px")
        improved = (delta > 0).sum()
        worsened = (delta < 0).sum()
        print(f"  {improved}/{len(delta)} pages improved, {worsened} worsened")


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
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--train", default="DS9E18,DS9E19,DS9E21,DS9E22",
                         help="Comma-separated scan dir names for training")
    p_train.add_argument("--holdout", default="DS9E20,DS9E23",
                         help="Comma-separated scan dir names held out for validation")
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--input-size", type=int, default=INPUT_SIZE,
                         help="CNN input resolution (default 512; 768 trades 2.25x time for ~30%% better localization)")
    p_train.add_argument("--heatmap", action="store_true",
                         help="Use heatmap regression head (sub-pixel soft-argmax) instead of direct coord regression")
    p_train.add_argument("--hmap-sigma", type=float, default=2.0,
                         help="Gaussian sigma for heatmap regularizer (in heatmap-pixels)")
    p_train.add_argument("--hmap-reg", type=float, default=1.0,
                         help="Weight of the heatmap-shape regularizer (DSNT-style). 0 = coord loss only.")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--model", default=None)
    p_eval.add_argument("--all", action="store_true",
                        help="Evaluate on all corrected pages (not just holdout)")
    p_eval.add_argument("--hybrid", action="store_true",
                        help="Also evaluate the CNN+edge-snap hybrid detector")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("image")
    p_pred.add_argument("--model", default=None)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "predict":
        predict_cli(args)


if __name__ == "__main__":
    main()
