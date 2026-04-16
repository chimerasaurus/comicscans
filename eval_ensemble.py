#!/usr/bin/env python3
"""Per-corner ensemble of regression + heatmap models.

Loads both checkpoints, predicts with each, and tries several combiners:
  - regression alone (baseline)
  - heatmap alone
  - mean of both (per-corner x,y average)
  - per-corner-error-weighted mean (oracle upper bound; ignores unknowns)
  - snap-distance-weighted: when refined, take the prediction whose snap was
    smaller (proxy for confidence — small snap means the CNN was already close
    to a strong edge).

All tested both CNN-only and after hybrid refine.
"""
import cv2, numpy as np, torch
from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, refine_corners,
)

REG_PATH  = "comicml_model_reg_768_tta.pt"
HMAP_PATH = "comicml_model_hmap_768.pt"


def mean_err(pred, gt):
    return float(np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(pred, gt)]))


def avg_corners(a, b):
    return [[(x1+x2)/2, (y1+y2)/2] for (x1,y1),(x2,y2) in zip(a, b)]


def pick_min_snap(cnn_a, hyb_a, cnn_b, hyb_b):
    """Per corner: pick the one whose snap distance (|hyb - cnn|) is smaller."""
    out = []
    for ca, ha, cb, hb in zip(cnn_a, hyb_a, cnn_b, hyb_b):
        sa = np.hypot(ha[0]-ca[0], ha[1]-ca[1])
        sb = np.hypot(hb[0]-cb[0], hb[1]-cb[1])
        out.append(ha if sa <= sb else hb)
    return out


def oracle_best(a, b, gt):
    out = []
    for pa, pb, pg in zip(a, b, gt):
        ea = np.hypot(pa[0]-pg[0], pa[1]-pg[1])
        eb = np.hypot(pb[0]-pg[0], pb[1]-pg[1])
        out.append(pa if ea <= eb else pb)
    return out


def report(name, errs):
    a = np.array(errs)
    print(f"  {name:<28s}  mean={a.mean():6.2f}  median={np.median(a):6.2f}  P95={np.percentile(a,95):6.2f}  max={a.max():6.2f}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    reg_model,  reg_ck  = _load_model(REG_PATH, device)
    hmap_model, hmap_ck = _load_model(HMAP_PATH, device)
    print(f"Regression: val={reg_ck['val_px']:.2f} px   Heatmap: val={hmap_ck['val_px']:.2f} px")

    entries = _load_entries()
    _, eval_entries = _split_entries(entries, reg_ck.get("train_dirs", []), reg_ck.get("holdout_dirs", []))
    eval_entries = [e for e in eval_entries if e["has_correction"]]
    print(f"Holdout: {len(eval_entries)} pages\n")

    cnn_reg_errs, cnn_hmap_errs, cnn_avg_errs, cnn_oracle_errs = [], [], [], []
    hyb_reg_errs, hyb_hmap_errs, hyb_avg_errs, hyb_pick_errs, hyb_oracle_errs = [], [], [], [], []

    # Track per-corner-index errors so we can see if failures are uncorrelated
    per_corner_reg  = [[] for _ in range(4)]
    per_corner_hmap = [[] for _ in range(4)]

    for entry in eval_entries:
        img = cv2.imread(entry["filepath"])
        if img is None: continue
        if entry["gt_rotate180"]:
            img = cv2.rotate(img, cv2.ROTATE_180)
        gt = entry["gt_corners"]
        dpi = entry.get("dpi", 600)

        cnn_r = predict_corners(reg_model,  device, img, tta=True)
        cnn_h = predict_corners(hmap_model, device, img, tta=True)
        cnn_m = avg_corners(cnn_r, cnn_h)

        hyb_r = refine_corners(img, cnn_r, dpi=dpi)
        hyb_h = refine_corners(img, cnn_h, dpi=dpi)
        hyb_m = avg_corners(hyb_r, hyb_h)

        cnn_reg_errs.append(mean_err(cnn_r, gt))
        cnn_hmap_errs.append(mean_err(cnn_h, gt))
        cnn_avg_errs.append(mean_err(cnn_m, gt))
        cnn_oracle_errs.append(mean_err(oracle_best(cnn_r, cnn_h, gt), gt))

        hyb_reg_errs.append(mean_err(hyb_r, gt))
        hyb_hmap_errs.append(mean_err(hyb_h, gt))
        hyb_avg_errs.append(mean_err(hyb_m, gt))
        hyb_pick_errs.append(mean_err(pick_min_snap(cnn_r, hyb_r, cnn_h, hyb_h), gt))
        hyb_oracle_errs.append(mean_err(oracle_best(hyb_r, hyb_h, gt), gt))

        for i in range(4):
            per_corner_reg[i].append(np.hypot(hyb_r[i][0]-gt[i][0], hyb_r[i][1]-gt[i][1]))
            per_corner_hmap[i].append(np.hypot(hyb_h[i][0]-gt[i][0], hyb_h[i][1]-gt[i][1]))

    print("=== CNN-only ===")
    report("regression (TTA)", cnn_reg_errs)
    report("heatmap (TTA)",    cnn_hmap_errs)
    report("mean(reg, hmap)",  cnn_avg_errs)
    report("oracle pick",      cnn_oracle_errs)

    print("\n=== Hybrid (refined) ===")
    report("regression",       hyb_reg_errs)
    report("heatmap",          hyb_hmap_errs)
    report("mean(reg, hmap)",  hyb_avg_errs)
    report("per-corner min-snap", hyb_pick_errs)
    report("oracle pick",      hyb_oracle_errs)

    # Correlation check: are their refined errors correlated?
    r = np.corrcoef(hyb_reg_errs, hyb_hmap_errs)[0, 1]
    print(f"\nPearson corr (hyb reg vs hyb hmap per-page): {r:+.3f}")

    print("\nPer-corner hybrid mean (reg | hmap):")
    for i, name in enumerate(["TL","TR","BR","BL"]):
        rr = np.mean(per_corner_reg[i])
        hh = np.mean(per_corner_hmap[i])
        print(f"  {name}:  reg={rr:6.2f}   hmap={hh:6.2f}")


if __name__ == "__main__":
    main()
