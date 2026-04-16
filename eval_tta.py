#!/usr/bin/env python3
"""Compare CNN accuracy with and without test-time augmentation."""
import cv2, numpy as np, torch
from comicml import (
    _load_model, _load_entries, _split_entries,
    predict_corners, refine_corners, MODEL_FILE,
)


def mean_err(pred, gt):
    return float(np.mean([np.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(pred, gt)]))


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, ckpt = _load_model(MODEL_FILE, device)
print(f"Model: input_size={ckpt.get('input_size')} best_val={ckpt.get('val_px'):.2f}")

entries = _load_entries()
_, eval_entries = _split_entries(entries, ckpt.get("train_dirs", []), ckpt.get("holdout_dirs", []))
eval_entries = [e for e in eval_entries if e["has_correction"]]
print(f"Holdout: {len(eval_entries)} pages\n")

no_tta, with_tta = [], []
hyb_no, hyb_with = [], []
for entry in eval_entries:
    img = cv2.imread(entry["filepath"])
    if img is None: continue
    if entry["gt_rotate180"]:
        img = cv2.rotate(img, cv2.ROTATE_180)
    gt = entry["gt_corners"]
    dpi = entry.get("dpi", 600)

    a = predict_corners(model, device, img, tta=False)
    b = predict_corners(model, device, img, tta=True)
    no_tta.append(mean_err(a, gt))
    with_tta.append(mean_err(b, gt))

    hyb_no.append(mean_err(refine_corners(img, a, dpi=dpi), gt))
    hyb_with.append(mean_err(refine_corners(img, b, dpi=dpi), gt))

def rep(name, arr):
    a = np.array(arr)
    print(f"  {name:<18s}  mean={a.mean():6.2f}  median={np.median(a):6.2f}  P95={np.percentile(a,95):6.2f}")

print("=== CNN-only ===")
rep("no TTA", no_tta); rep("with TTA", with_tta)
delta = np.array(no_tta) - np.array(with_tta)
print(f"  TTA improved {(delta>0).sum()}/{len(delta)} pages, mean delta {delta.mean():+.2f} px")
print("\n=== Hybrid (current defaults) ===")
rep("no TTA", hyb_no); rep("with TTA", hyb_with)
delta = np.array(hyb_no) - np.array(hyb_with)
print(f"  TTA improved {(delta>0).sum()}/{len(delta)} pages, mean delta {delta.mean():+.2f} px")
