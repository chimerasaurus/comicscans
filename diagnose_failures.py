#!/usr/bin/env python3
"""Diagnose where the hybrid detector fails on the holdout.

For each holdout page, computes:
  - CNN-only mean corner error
  - Hybrid mean corner error
  - Per-corner snap distances (how far refinement moved each corner)
  - Per-corner residual error after refinement

Prints a sorted table (worst hybrid first) + flags worsened pages, and saves
side-by-side overlay PNGs for the top-N worst + all worsened pages into
diagnose_out/.
"""
import os
import cv2
import numpy as np
import torch

from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, refine_corners, MODEL_FILE,
)

OUT_DIR = "diagnose_out"
TOP_N_WORST = 10
CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def per_corner_dists(pred, gt):
    return [float(np.hypot(p[0]-g[0], p[1]-g[1])) for p, g in zip(pred, gt)]


def draw_overlay(img, cnn, hyb, gt, label, save_path):
    """Save a downscaled overlay with CNN(red), hybrid(green), GT(yellow)."""
    h, w = img.shape[:2]
    scale = 1200 / max(h, w)
    disp = cv2.resize(img, (int(w*scale), int(h*scale)))

    def s(pt): return (int(pt[0]*scale), int(pt[1]*scale))

    def poly(pts, color, thick):
        pts = [s(p) for p in pts]
        for i in range(4):
            cv2.line(disp, pts[i], pts[(i+1) % 4], color, thick)
        for p in pts:
            cv2.circle(disp, p, 8, color, -1)

    poly(cnn, (0, 0, 255), 2)        # red = CNN prior
    poly(hyb, (0, 255, 0), 2)        # green = refined
    poly(gt,  (0, 255, 255), 2)      # yellow = ground truth

    # legend
    cv2.rectangle(disp, (10, 10), (260, 90), (0, 0, 0), -1)
    cv2.putText(disp, "CNN",   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(disp, "Hybrid",(20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(disp, "GT",    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(disp, label, (10, disp.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imwrite(save_path, disp)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = _load_model(MODEL_FILE, device)
    print(f"Model: {MODEL_FILE} (best val {ckpt.get('val_px','?'):.2f} px)")

    entries = _load_entries()
    _, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []),
                                     ckpt.get("holdout_dirs", []))
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Holdout: {len(eval_entries)} pages\n")

    rows = []
    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None:
            continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cnn = predict_corners(model, device, img, rotate180=False)
        hyb = refine_corners(img, cnn, dpi=entry.get("dpi", 600))
        gt = entry["gt_corners"]

        cnn_per = per_corner_dists(cnn, gt)
        hyb_per = per_corner_dists(hyb, gt)
        snap_per = per_corner_dists(hyb, cnn)  # how far refinement moved each corner

        rows.append({
            "name": f"{entry['scan_dir'].rsplit('/',1)[-1]}/{os.path.basename(entry['filepath'])}",
            "img": img, "cnn": cnn, "hyb": hyb, "gt": gt,
            "cnn_mean": float(np.mean(cnn_per)),
            "hyb_mean": float(np.mean(hyb_per)),
            "delta": float(np.mean(cnn_per) - np.mean(hyb_per)),  # +ve = improved
            "cnn_per": cnn_per, "hyb_per": hyb_per, "snap_per": snap_per,
            "max_snap": max(snap_per),
            "worst_corner": CORNER_NAMES[int(np.argmax(hyb_per))],
        })

    # Sort by hybrid error (worst first)
    rows.sort(key=lambda r: -r["hyb_mean"])
    worsened = [r for r in rows if r["delta"] < 0]

    print(f"{'rank':>4s}  {'page':<32s}  {'CNN':>7s}  {'HYB':>7s}  {'Δ':>7s}  "
          f"{'maxsnap':>8s}  per-corner CNN→HYB (worst={'corner':>3s})")
    print("-" * 130)
    for i, r in enumerate(rows[:TOP_N_WORST]):
        per = "  ".join(
            f"{n}:{c:5.1f}→{h:5.1f}" for n, c, h in zip(CORNER_NAMES, r["cnn_per"], r["hyb_per"])
        )
        print(f"{i+1:>4d}  {r['name']:<32s}  {r['cnn_mean']:>7.1f}  {r['hyb_mean']:>7.1f}  "
              f"{r['delta']:>+7.1f}  {r['max_snap']:>8.1f}  {per}  ({r['worst_corner']})")

    if worsened:
        print(f"\n--- {len(worsened)} pages worsened by refinement ---")
        for r in worsened:
            per = "  ".join(
                f"{n}:{c:5.1f}→{h:5.1f}" for n, c, h in zip(CORNER_NAMES, r["cnn_per"], r["hyb_per"])
            )
            print(f"      {r['name']:<32s}  CNN={r['cnn_mean']:6.1f}  HYB={r['hyb_mean']:6.1f}  "
                  f"Δ={r['delta']:+6.1f}  maxsnap={r['max_snap']:6.1f}  {per}")

    # Aggregates
    print("\n--- Aggregates ---")
    cnn_arr = np.array([r["cnn_mean"] for r in rows])
    hyb_arr = np.array([r["hyb_mean"] for r in rows])
    snap_all = np.array([s for r in rows for s in r["snap_per"]])
    print(f"  CNN  mean={cnn_arr.mean():.2f}  median={np.median(cnn_arr):.2f}  P95={np.percentile(cnn_arr,95):.2f}")
    print(f"  HYB  mean={hyb_arr.mean():.2f}  median={np.median(hyb_arr):.2f}  P95={np.percentile(hyb_arr,95):.2f}")
    print(f"  Snap distance: median={np.median(snap_all):.1f}  P95={np.percentile(snap_all,95):.1f}  max={snap_all.max():.1f}")
    pct_no_snap = (snap_all < 1).mean() * 100
    print(f"  Corners not snapped (<1 px move): {pct_no_snap:.1f}%")

    # Save overlays
    save_set = set(id(r) for r in rows[:TOP_N_WORST]) | set(id(r) for r in worsened)
    print(f"\nSaving {len(save_set)} overlays to {OUT_DIR}/...")
    for r in rows:
        if id(r) not in save_set: continue
        tag = "WORSE" if r in worsened else "WORST"
        safe_name = r["name"].replace("/", "_").replace(" ", "_")
        path = f"{OUT_DIR}/{tag}_{r['hyb_mean']:06.1f}_{safe_name}.png"
        label = f"{r['name']}  CNN={r['cnn_mean']:.1f}  HYB={r['hyb_mean']:.1f}  worst={r['worst_corner']}"
        draw_overlay(r["img"], r["cnn"], r["hyb"], r["gt"], label, path)
    print("Done.")


if __name__ == "__main__":
    main()
