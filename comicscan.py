#!/usr/bin/env python3
"""
comicscan.py — Process raw comic book scans into clean, aligned page images.

Usage:
    python3 comicscan.py <input_dir> [--output <output_dir>] [--quality 93] [--preview]

Example:
    python3 comicscan.py raw-scans/DS9E17/
    python3 comicscan.py raw-scans/DS9E17/ --output output/DS9E17 --quality 95 --preview
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_scan_filename(filename):
    """Parse scan filenames and return the page index.

    'Scan.jpeg' -> 0, 'Scan 1.jpeg' -> 1, 'Scan 35.jpeg' -> 35
    """
    stem = Path(filename).stem
    match = re.match(r'^Scan(?:\s+(\d+))?$', stem)
    if not match:
        return None
    return int(match.group(1)) if match.group(1) else 0


def load_scans(input_dir):
    """Load and sort scan files from the input directory."""
    input_path = Path(input_dir)
    scans = []

    for f in input_path.iterdir():
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'):
            idx = parse_scan_filename(f.name)
            if idx is not None:
                scans.append((idx, f))

    if not scans:
        print(f"Error: No scan files found in {input_dir}")
        sys.exit(1)

    scans.sort(key=lambda x: x[0])
    print(f"Found {len(scans)} scan files (pages 0-{scans[-1][0]})")

    # Check for gaps
    indices = [s[0] for s in scans]
    expected = list(range(max(indices) + 1))
    missing = set(expected) - set(indices)
    if missing:
        print(f"Warning: Missing page indices: {sorted(missing)}")

    return scans


def detect_page_bounds(image):
    """Detect the comic page rectangle within a scan.

    Scans inward from each edge to find where page content begins,
    based on row/column brightness and variance. The scanner bed appears
    as uniform light gray (~220-240), while page content has lower mean
    brightness and/or higher variance.

    Returns (top, bottom, right, left) crop coordinates and detected skew angle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Determine scanner bed color from the bottom-right corner (most likely empty)
    corner = gray[h - 50:h, w - 50:w]
    bed_mean = corner.mean()
    # Content threshold: a row/column is "content" if mean < bed_mean - 30
    # or standard deviation > 25 (meaning varied content, not uniform bed)
    mean_thresh = max(bed_mean - 30, 180)
    std_thresh = 25

    def is_content_row(y):
        row = gray[y, :]
        return row.mean() < mean_thresh or row.std() > std_thresh

    def is_content_col(x):
        col = gray[:, x]
        return col.mean() < mean_thresh or col.std() > std_thresh

    # Scan from top
    top = 0
    for y in range(h):
        if is_content_row(y):
            top = y
            break

    # Scan from bottom
    bottom = h
    for y in range(h - 1, -1, -1):
        if is_content_row(y):
            bottom = y + 1
            break

    # Scan from left
    left = 0
    for x in range(w):
        if is_content_col(x):
            left = x
            break

    # Scan from right
    right = w
    for x in range(w - 1, -1, -1):
        if is_content_col(x):
            right = x + 1
            break

    # Detect skew angle using edge detection on the cropped region
    cropped_gray = gray[top:bottom, left:right]
    angle = detect_skew(cropped_gray)

    return top, bottom, left, right, angle


def detect_skew(gray_image):
    """Detect skew angle of a page using Hough line detection on edges.

    Only detects small skew (< 3 degrees). Scans placed on a flatbed
    scanner will never have large rotation — any large angle detected
    is a false positive from comic art content.
    """
    h, w = gray_image.shape

    # Focus on the edges of the page (outer 10%) where panel borders
    # and page edges provide the best skew signal
    mask = np.zeros_like(gray_image)
    border = min(h, w) // 10
    mask[:border, :] = 255   # top strip
    mask[-border:, :] = 255  # bottom strip
    mask[:, :border] = 255   # left strip
    mask[:, -border:] = 255  # right strip

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, mask)

    # Detect long lines in the border regions
    min_line_len = min(h, w) // 4
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min_line_len, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    # Only collect angles very close to horizontal or vertical (within 3°)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Near-horizontal lines: angle close to 0° or ±180°
        if abs(angle) < 3:
            angles.append(angle)
        elif abs(angle) > 177:
            angles.append(angle - 180 if angle > 0 else angle + 180)
        # Near-vertical lines: angle close to ±90°
        elif abs(abs(angle) - 90) < 3:
            angles.append(angle - 90 if angle > 0 else angle + 90)

    if not angles:
        return 0.0

    median_angle = np.median(angles)

    # Only correct meaningful skew, and cap at 3 degrees as a safety limit
    if abs(median_angle) < 0.1 or abs(median_angle) > 3.0:
        return 0.0

    return median_angle


def deskew_and_crop(image, bounds):
    """Deskew the image and crop to page bounds."""
    top, bottom, left, right, angle = bounds

    if abs(angle) > 0.05:
        # Rotate to correct skew
        img_h, img_w = image.shape[:2]
        center = (img_w / 2, img_h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(img_h * sin + img_w * cos)
        new_h = int(img_h * cos + img_w * sin)
        rotation_matrix[0, 2] += (new_w - img_w) / 2
        rotation_matrix[1, 2] += (new_h - img_h) / 2

        image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Adjust crop bounds for the rotation offset
        dx = (new_w - img_w) / 2
        dy = (new_h - img_h) / 2
        left = int(left + dx)
        right = int(right + dx)
        top = int(top + dy)
        bottom = int(bottom + dy)

    # Crop
    cropped = image[top:bottom, left:right]
    return cropped


def normalize_dimensions(pages, target_w, target_h):
    """Center-composite each page onto a black canvas of uniform dimensions."""
    normalized = []
    for page in pages:
        h, w = page.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate placement (centered)
        paste_w = min(w, target_w)
        paste_h = min(h, target_h)

        # Source crop (center of the page if it's larger than target)
        src_x = max(0, (w - target_w) // 2)
        src_y = max(0, (h - target_h) // 2)

        # Destination offset (center on canvas)
        dst_x = max(0, (target_w - w) // 2)
        dst_y = max(0, (target_h - h) // 2)

        canvas[dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = \
            page[src_y:src_y + paste_h, src_x:src_x + paste_w]

        normalized.append(canvas)
    return normalized


def save_pages(pages, output_dir, quality, dpi):
    """Save processed pages as JPEG files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(pages):
        filename = output_path / f"Scan {i}.jpg"
        # Convert from BGR (OpenCV) to RGB (Pillow)
        img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        img.save(str(filename), 'JPEG', quality=quality, dpi=(dpi, dpi))
        size_mb = filename.stat().st_size / (1024 * 1024)
        print(f"  Saved {filename.name} ({img.width}x{img.height}, {size_mb:.1f} MB)")

    return output_path


def preview_pages(pages, indices=None):
    """Open a few pages for preview using the system image viewer."""
    if indices is None:
        # Show first, middle, and last
        n = len(pages)
        indices = [0, n // 2, n - 1]

    import tempfile
    preview_files = []
    for i in indices:
        page = pages[i]
        img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        tmp = tempfile.NamedTemporaryFile(suffix=f'_page{i}.jpg', delete=False)
        img.save(tmp.name, 'JPEG', quality=93)
        preview_files.append(tmp.name)
        print(f"  Preview: page {i} -> {tmp.name}")

    # Open with system viewer on macOS
    for f in preview_files:
        subprocess.Popen(['open', f])

    response = input("\nProceed with saving all pages? [Y/n] ").strip().lower()
    return response != 'n'


def get_source_dpi(filepath):
    """Read DPI from the source image."""
    try:
        img = Image.open(filepath)
        dpi = img.info.get('dpi', (300, 300))
        return int(dpi[0])
    except Exception:
        return 300


def process(input_dir, output_dir=None, quality=93, preview=False):
    """Main processing pipeline."""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = Path('output') / input_path.name

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Load and sort scans
    print("Step 1: Loading scans...")
    scans = load_scans(input_dir)
    dpi = get_source_dpi(scans[0][1])
    print(f"  Source DPI: {dpi}")
    print()

    # Step 2-3: Detect page bounds, deskew, and crop each page
    print("Step 2-3: Detecting page bounds, deskewing, and cropping...")
    cropped_pages = []
    for idx, filepath in scans:
        print(f"  Processing page {idx}: {filepath.name}")
        image = cv2.imread(str(filepath))
        if image is None:
            print(f"  Error: Could not read {filepath}")
            continue

        bounds = detect_page_bounds(image)
        top, bottom, left, right, angle = bounds
        det_w = right - left
        det_h = bottom - top
        margins = f"T={top} B={image.shape[0]-bottom} L={left} R={image.shape[1]-right}"
        print(f"    Detected: {det_w}x{det_h}, skew={angle:.2f}°, margins: {margins}")

        cropped = deskew_and_crop(image, bounds)
        ch, cw = cropped.shape[:2]
        print(f"    Cropped:  {cw}x{ch}")
        cropped_pages.append(cropped)
    print()

    # Step 4: Normalize dimensions
    print("Step 4: Normalizing dimensions...")
    widths = [p.shape[1] for p in cropped_pages]
    heights = [p.shape[0] for p in cropped_pages]
    target_w = int(np.median(widths))
    target_h = int(np.median(heights))
    print(f"  Median dimensions: {target_w}x{target_h}")
    print(f"  Width  range: {min(widths)}-{max(widths)} (spread: {max(widths)-min(widths)})")
    print(f"  Height range: {min(heights)}-{max(heights)} (spread: {max(heights)-min(heights)})")

    normalized = normalize_dimensions(cropped_pages, target_w, target_h)
    print(f"  All {len(normalized)} pages normalized to {target_w}x{target_h}")
    print()

    # Preview if requested
    if preview:
        print("Step 4.5: Preview...")
        if not preview_pages(normalized):
            print("Aborted by user.")
            return None
        print()

    # Step 5: Save
    print("Step 5: Saving processed pages...")
    saved_path = save_pages(normalized, output_dir, quality, dpi)
    print()
    print(f"Done! {len(normalized)} pages saved to {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description='Process raw comic book scans into clean, aligned page images.')
    parser.add_argument('input_dir', help='Directory containing raw scan images')
    parser.add_argument('--output', '-o', help='Output directory (default: output/<input_name>)')
    parser.add_argument('--quality', '-q', type=int, default=93,
                        help='JPEG quality 1-100 (default: 93)')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Preview pages before saving')

    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    process(args.input_dir, args.output, args.quality, args.preview)


if __name__ == '__main__':
    main()
