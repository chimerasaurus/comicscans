#!/usr/bin/env python3
"""
comicscans.py — Process raw comic book scans into clean, aligned page images.

Usage:
    python3 comicscans.py <input_dir> [--output <output_dir>] [--format jpg|webp] [--quality 85] [--preview]

Rotation options (for upside-down pages):
    --rotate 2,4,6,8       Rotate specific pages 180°
    --rotate-range 2-14    Rotate a range of pages 180°
    --rotate-even           Rotate all even-numbered pages 180°
    --rotate-odd            Rotate all odd-numbered pages 180°
    --auto-rotate           Auto-detect orientation via Tesseract OCR

Example:
    python3 comicscans.py raw-scans/DS9E17/
    python3 comicscans.py raw-scans/DS9E17/ --output output/DS9E17 --rotate-even
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Orientation detection via Tesseract OCR
# ---------------------------------------------------------------------------

TESSERACT_BIN = "/opt/homebrew/bin/tesseract"

# Common English words used to score OCR output quality
COMMON_WORDS = set("""the a an is it in on to of and for are was not you all can had her his one our
out but has have this that with from they been what when who will more than them
then some just know take come make like back only your here there where we he she
my me no so do if up about into over after look going get think now very how right
want need said could would should been much well also way even because these those
its too any only own good new first last long great little just such been before
know most find here between does each those over own same tell us give day
yes sir captain commander lieutenant wait stop go let please don't can't won't
what's that's he's she's they're we're you're i'm i'll i've it's there's
really want need call security enough still must quite thought station major
doctor sir hold help kill knew something someone nothing everything""".split())


def _count_real_words(text):
    """Count words that appear in the common English word list."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for w in words if w in COMMON_WORDS or
               (len(w) >= 4 and w.rstrip("'s") in COMMON_WORDS))


def _tesseract_ocr(image_path):
    """Run Tesseract OCR on an image and return recognised text."""
    result = subprocess.run(
        [TESSERACT_BIN, str(image_path), "stdout", "--psm", "3", "-l", "eng"],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout


def detect_orientation(cv_image):
    """Detect whether a page image is upside-down using Tesseract OCR.

    Runs OCR on both the normal and 180°-rotated image, counts common
    English words in each, and returns True if the page should be rotated.

    Returns (should_rotate, normal_words, rotated_words).
    """
    # Downscale large images for faster OCR (target ~2000px wide)
    h, w = cv_image.shape[:2]
    ocr_image = cv_image
    if w > 2500:
        scale = 2000 / w
        ocr_image = cv2.resize(cv_image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

    # Write normal image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        normal_path = tmp.name
        cv2.imwrite(normal_path, ocr_image)

    # Write 180°-rotated image to temp file
    rotated_img = cv2.rotate(ocr_image, cv2.ROTATE_180)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        rotated_path = tmp.name
        cv2.imwrite(rotated_path, rotated_img)

    try:
        normal_text = _tesseract_ocr(normal_path)
        rotated_text = _tesseract_ocr(rotated_path)

        normal_words = _count_real_words(normal_text)
        rotated_words = _count_real_words(rotated_text)

        # 15% margin threshold: only rotate if the rotated version is
        # clearly better to avoid false positives
        if normal_words == 0 and rotated_words == 0:
            should_rotate = False
        elif normal_words == 0:
            should_rotate = True
        else:
            ratio = rotated_words / normal_words
            should_rotate = ratio > 1.15
    finally:
        os.unlink(normal_path)
        os.unlink(rotated_path)

    return should_rotate, normal_words, rotated_words


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

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


def get_source_dpi(filepath):
    """Read DPI from the source image."""
    try:
        img = Image.open(filepath)
        dpi = img.info.get('dpi', (300, 300))
        return int(dpi[0])
    except Exception:
        return 300


def parse_rotate_pages(args, total_pages):
    """Build a set of page indices to rotate 180° from CLI arguments."""
    pages_to_rotate = set()

    if args.rotate:
        for part in args.rotate.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                pages_to_rotate.update(range(int(start), int(end) + 1))
            else:
                pages_to_rotate.add(int(part))

    if args.rotate_range:
        start, end = args.rotate_range.split('-', 1)
        pages_to_rotate.update(range(int(start), int(end) + 1))

    if args.rotate_even:
        pages_to_rotate.update(range(0, total_pages, 2))

    if args.rotate_odd:
        pages_to_rotate.update(range(1, total_pages, 2))

    return pages_to_rotate


# ---------------------------------------------------------------------------
# Two-page bleed / spine detection
# ---------------------------------------------------------------------------

# Standard US comic page width in inches (trim size)
COMIC_PAGE_WIDTH_INCHES = 6.625


def detect_spine_dark_band(gray, content_bottom, search_start, search_end,
                           dpi=300):
    """Detect a dark spine/binding shadow (original method).

    Returns (spine_center, spine_width) or None.
    """
    h = min(content_bottom, gray.shape[0])
    dpi_scale = dpi / 300.0
    min_spine_width = int(15 * dpi_scale)

    best_spine = None
    best_spine_width = 0

    col = search_start
    while col < search_end:
        strip = gray[:h, col]
        col_mean = strip.mean()
        col_std = strip.std()

        if col_mean < 60 and col_std < 20:
            spine_start = col
            while col < search_end:
                s = gray[:h, col]
                if s.mean() >= 60 or s.std() >= 20:
                    break
                col += 1
            spine_end = col
            spine_width = spine_end - spine_start

            if spine_width >= min_spine_width and spine_width > best_spine_width:
                best_spine = (spine_start + spine_end) // 2
                best_spine_width = spine_width
        else:
            col += 1

    if best_spine is not None:
        return best_spine, best_spine_width
    return None


def detect_bleed_boundary(gray, content_top, content_bottom,
                          content_left, content_right, dpi, params=None):
    """Detect two-page bleed using expected page width from DPI.

    Pages are always placed in the top-left corner of the scanner. When a
    comic is opened flat, the adjacent page may be partially visible on the
    right side. After 180° rotation, the bleed appears on the left side.

    We determine which side has bleed by checking the scanner bed margins:
      - Bottom margin > top margin → page is non-rotated → bleed on RIGHT
      - Top margin > bottom margin → page was rotated 180° → bleed on LEFT

    Returns (crop_col, method_str) or None.
    """
    bp = params or {}
    bleed_trigger_ratio = bp.get("bleed_trigger_ratio", 1.05)
    bleed_search_pct = bp.get("bleed_search_pct", 0.08)
    gutter_peak_offset = bp.get("gutter_peak_offset", 9.8517)
    trough_depth = bp.get("trough_depth", 6.0895)
    trough_rise = bp.get("trough_rise", 4)
    gradient_min = bp.get("gradient_min", 2.0298)

    content_width = content_right - content_left
    expected_page_px = int(dpi * COMIC_PAGE_WIDTH_INCHES)

    # Only trigger if content is wider than expected single page
    if content_width < expected_page_px * bleed_trigger_ratio:
        return None

    h = min(content_bottom, gray.shape[0])
    img_h = gray.shape[0]

    # Determine bleed side from scanner bed margin position.
    # Page is placed top-left on the scanner bed:
    #   - Non-rotated scan: large bottom margin (scanner bed), bleed on RIGHT
    #   - Rotated 180° scan: large top margin (scanner bed), bleed on LEFT
    top_margin = content_top
    bottom_margin = img_h - content_bottom

    if bottom_margin > top_margin:
        # Non-rotated: bleed is on the right
        bleed_side = 'right'
        expected_boundary = content_left + expected_page_px
    else:
        # Rotated: bleed is on the left
        bleed_side = 'left'
        expected_boundary = content_right - expected_page_px

    # Search in a window around the expected boundary for the best cut point
    dpi_scale = dpi / 300.0
    search_radius = int(expected_page_px * bleed_search_pct)
    edge_margin = int(50 * dpi_scale)
    win_start = max(content_left + edge_margin, expected_boundary - search_radius)
    win_end = min(content_right - edge_margin, expected_boundary + search_radius)

    if win_start >= win_end:
        return None

    # Compute column means across the search window
    col_means = np.array([gray[:h, x].mean() for x in range(win_start, win_end)])

    # Method 1: Look for a dark spine band in this narrower region
    dark_result = detect_spine_dark_band(gray, content_bottom, win_start, win_end, dpi)
    if dark_result is not None:
        spine_center, spine_width = dark_result
        if bleed_side == 'right':
            return spine_center - spine_width // 2, 'dark_spine'
        else:
            return spine_center + spine_width // 2, 'dark_spine'

    # Method 2: Look for a bright gutter (local brightness peak — white
    # page margin between the two pages). Cut at the gutter EDGE toward the
    # main page, not at the peak, so the gutter strip is fully removed.
    min_cols = int(20 * dpi_scale)
    kernel_size = int(15 * dpi_scale) | 1  # ensure odd
    if len(col_means) > min_cols:
        smoothed = np.convolve(col_means, np.ones(kernel_size) / kernel_size, mode='same')
        # Exclude edge artifacts from convolution
        margin = kernel_size // 2
        valid_start = margin
        valid_end = len(smoothed) - margin
        valid_region = smoothed[valid_start:valid_end]
        overall_mean = valid_region.mean()

        peak_idx_in_valid = np.argmax(valid_region)
        peak_val = valid_region[peak_idx_in_valid]
        peak_idx = valid_start + peak_idx_in_valid
        if peak_val > overall_mean + gutter_peak_offset:
            # Walk from the peak toward the main page side until brightness
            # drops below the midpoint between peak and the page content level.
            threshold = (peak_val + overall_mean) / 2
            if bleed_side == 'right':
                # Main page is to the LEFT of the gutter — walk left from peak
                edge_idx = peak_idx
                for i in range(peak_idx, valid_start - 1, -1):
                    if smoothed[i] < threshold:
                        edge_idx = i
                        break
                cut_col = win_start + edge_idx
            else:
                # Main page is to the RIGHT of the gutter — walk right from peak
                edge_idx = peak_idx
                for i in range(peak_idx, valid_end):
                    if smoothed[i] < threshold:
                        edge_idx = i
                        break
                cut_col = win_start + edge_idx
            return cut_col, 'bright_gutter'

    # Method 3: Local brightness minimum (subtle spine shadow that's darker
    # than surroundings but not dark enough for the strict dark-band detector).
    # Must be a true V-shaped local minimum — significantly darker than the
    # surrounding region on BOTH sides — not just the global minimum of a
    # gradually decreasing brightness slope.
    if len(col_means) > min_cols:
        smoothed = np.convolve(col_means, np.ones(kernel_size) / kernel_size, mode='same')
        margin = kernel_size // 2
        valid_start = margin
        valid_end = len(smoothed) - margin
        valid_region = smoothed[valid_start:valid_end]
        overall_mean = valid_region.mean()

        trough_idx_in_valid = np.argmin(valid_region)
        trough_val = valid_region[trough_idx_in_valid]
        if trough_val < overall_mean - trough_depth:
            # Verify it's a true local minimum (V-shape), not just the low
            # end of a brightness gradient. Check that the brightness rises
            # on both sides of the trough within a local neighborhood.
            neighborhood = max(int(20 * dpi_scale), len(valid_region) // 10)
            left_region = valid_region[max(0, trough_idx_in_valid - neighborhood):trough_idx_in_valid]
            right_region = valid_region[trough_idx_in_valid + 1:min(len(valid_region), trough_idx_in_valid + neighborhood + 1)]
            if len(left_region) > 0 and len(right_region) > 0:
                left_max = left_region.max()
                right_max = right_region.max()
                # Both sides must be at least 4 brightness points above the trough
                if left_max > trough_val + trough_rise and right_max > trough_val + trough_rise:
                    cut_col = win_start + valid_start + trough_idx_in_valid
                    return cut_col, 'dark_trough'

    # Method 4: Sharpest brightness gradient, weighted by proximity to
    # expected boundary. Panel borders can produce strong gradients
    # anywhere, so we prefer gradients near where we expect the spine.
    grad_kernel = int(5 * dpi_scale) | 1  # ensure odd
    if len(col_means) > int(10 * dpi_scale):
        gradient = np.abs(np.diff(np.convolve(col_means,
                          np.ones(grad_kernel) / grad_kernel, mode='same')))
        # Gaussian proximity weight centered on expected boundary
        center_offset = expected_boundary - win_start
        positions = np.arange(len(gradient))
        sigma = len(gradient) / 4
        proximity = np.exp(-0.5 * ((positions - center_offset) / sigma) ** 2)
        weighted_gradient = gradient * proximity

        grad_idx = np.argmax(weighted_gradient)
        grad_val = gradient[grad_idx]
        if grad_val > gradient_min:
            cut_col = win_start + grad_idx
            return cut_col, 'gradient'

    # Method 5: Fall back to the expected boundary position
    return expected_boundary, 'expected_width'


# ---------------------------------------------------------------------------
# Skew detection
# ---------------------------------------------------------------------------

def detect_skew(gray_image):
    """Detect skew angle using Hough lines on panel borders and page edges.

    Focuses on the outer border region of the page. Only returns small
    corrections (0.1° - 5°). Scans on a flatbed never need large rotation.
    """
    h, w = gray_image.shape

    # Focus on the outer 15% border strips where panel edges and page
    # edges give the best skew signal
    mask = np.zeros_like(gray_image)
    border_w = max(w // 7, 50)
    border_h = max(h // 7, 50)
    mask[:border_h, :] = 255       # top strip
    mask[-border_h:, :] = 255      # bottom strip
    mask[:, :border_w] = 255       # left strip
    mask[:, -border_w:] = 255      # right strip

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, mask)

    # Require long lines for reliable angle detection
    min_line_len = min(h, w) // 5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_len, maxLineGap=15)

    if lines is None or len(lines) == 0:
        return 0.0

    # Collect angles from near-horizontal and near-vertical lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Weight longer lines more heavily
        weight = length / min_line_len

        # Near-horizontal lines (within 5° of horizontal)
        if abs(angle) < 5:
            angles.extend([angle] * int(weight))
        elif abs(angle) > 175:
            corrected = angle - 180 if angle > 0 else angle + 180
            angles.extend([corrected] * int(weight))
        # Near-vertical lines (within 5° of vertical)
        elif abs(abs(angle) - 90) < 5:
            corrected = angle - 90 if angle > 0 else angle + 90
            angles.extend([corrected] * int(weight))

    if len(angles) < 3:
        return 0.0

    median_angle = np.median(angles)

    # Only correct if skew is noticeable but not extreme
    if abs(median_angle) < 0.1 or abs(median_angle) > 5.0:
        return 0.0

    return float(median_angle)


# ---------------------------------------------------------------------------
# Page boundary detection
# ---------------------------------------------------------------------------

def _find_content_bounds(gray, mean_thresh, std_thresh):
    """Scan inward from each edge to find where content begins."""
    h, w = gray.shape

    top = 0
    for y in range(h):
        row = gray[y, :]
        if row.mean() < mean_thresh or row.std() > std_thresh:
            top = y
            break

    bottom = h
    for y in range(h - 1, -1, -1):
        row = gray[y, :]
        if row.mean() < mean_thresh or row.std() > std_thresh:
            bottom = y + 1
            break

    left = 0
    for x in range(w):
        col = gray[:, x]
        if col.mean() < mean_thresh or col.std() > std_thresh:
            left = x
            break

    right = w
    for x in range(w - 1, -1, -1):
        col = gray[:, x]
        if col.mean() < mean_thresh or col.std() > std_thresh:
            right = x + 1
            break

    return top, bottom, left, right


def _deskew_gray(gray, angle, fill_value):
    """Rotate a grayscale image to correct skew, expanding canvas."""
    h, w = gray.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(gray, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=int(fill_value))
    return rotated


def detect_page_bounds(image, dpi=300, params=None):
    """Detect the comic page boundaries within a scanner image.

    Two-pass approach:
      Pass 1 — Rough content detection + skew angle measurement.
      Deskew  — Straighten the grayscale image so the spine is vertical.
      Pass 2 — Re-detect content, then run bleed / trim / edge-trim on
               the deskewed image where column-based cuts are accurate.

    Returns dict with keys: top, bottom, left, right, angle, spine_col, bleed_method
    """
    # Tunable parameters with defaults
    p = params or {}
    bed_mean_offset = p.get("bed_mean_offset", 20.0024)
    min_mean_thresh = p.get("min_mean_thresh", 200)
    p_std_thresh = p.get("std_thresh", 20.6157)
    p_bleed_inset_in = p.get("bleed_inset_in", 0.0497)
    p_safety_margin_in = p.get("safety_margin_in", 0.1492)
    p_strip_check_in = p.get("strip_check_in", 0.667)
    p_band_size_in = p.get("band_size_in", 2.0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Scale factor for DPI-dependent pixel constants (calibrated at 300 DPI)
    dpi_scale = dpi / 300.0

    # Determine scanner bed color from the brightest corner.
    cs = int(50 * dpi_scale)  # corner sample size
    corners = [
        gray[:cs, :cs],            # top-left
        gray[:cs, w - cs:w],       # top-right
        gray[h - cs:h, :cs],       # bottom-left
        gray[h - cs:h, w - cs:w],  # bottom-right
    ]
    bed_mean = max(c.mean() for c in corners)
    mean_thresh = max(bed_mean - bed_mean_offset, min_mean_thresh)
    std_thresh = p_std_thresh

    # --- Pass 1: rough content bounds + skew detection ---
    top, bottom, left, right = _find_content_bounds(gray, mean_thresh, std_thresh)
    cropped_gray = gray[top:bottom, left:right]
    angle = detect_skew(cropped_gray)

    # --- Deskew: straighten the image so spine/edges are vertical/horizontal ---
    # This is critical for accurate column-based bleed detection: at 1° of
    # skew over 3000 px the spine drifts ~52 px — no single vertical cut
    # can cleanly separate two pages without deskewing first.
    if abs(angle) > 0.1:
        gray = _deskew_gray(gray, angle, bed_mean)
        h, w = gray.shape

    # --- Pass 2: re-detect content on the deskewed image ---
    top, bottom, left, right = _find_content_bounds(gray, mean_thresh, std_thresh)

    spine_col = None
    bleed_method = None

    # Strategy 1: Dark spine shadow (works when binding shadow is very pronounced)
    search_start = int(w * 0.1)
    search_end = int(w * 0.9)
    spine_result = detect_spine_dark_band(gray, bottom, search_start, search_end, dpi)

    if spine_result is not None:
        spine_center, spine_width = spine_result
        left_area = (spine_center - spine_width // 2) - left
        right_area = right - (spine_center + spine_width // 2)

        if left_area > right_area:
            right = spine_center - spine_width // 2
        else:
            left = spine_center + spine_width // 2
        spine_col = spine_center
        bleed_method = 'dark_spine'

    # Strategy 2: DPI-based expected width + boundary detection
    if spine_col is None:
        bleed_result = detect_bleed_boundary(gray, top, bottom, left, right, dpi, params)
        if bleed_result is not None:
            cut_col, method = bleed_result

            # Small inward shift on the bleed side to remove adjacent-page
            # content that sneaks past the detected gutter/trough boundary.
            bleed_inset = int(dpi * p_bleed_inset_in)

            mid = (left + right) / 2
            if cut_col > mid:
                right = cut_col - bleed_inset
            else:
                left = cut_col + bleed_inset

            spine_col = cut_col
            bleed_method = method

    # Strategy 3: Secondary trim — trim opposite edge to expected page width
    expected_page_px = int(dpi * COMIC_PAGE_WIDTH_INCHES)
    safety_margin = int(dpi * p_safety_margin_in)
    trim_target = expected_page_px - safety_margin
    content_width = right - left
    excess = content_width - trim_target
    top_margin = top
    bottom_margin = h - bottom
    if spine_col is not None and excess > 0:
        if bottom_margin > top_margin:
            left = left + excess
        else:
            right = right - excess
        if bleed_method:
            bleed_method += '+trim'

    # Strategy 5: Edge trim — scan inward from each edge looking for
    # uniform bright strips. Uses a narrow center band to avoid artifacts.
    max_strip = int(dpi * p_strip_check_in)
    strip_check = min(max_strip, (bottom - top) // 20, (right - left) // 20)
    max_band = int(dpi * p_band_size_in)
    if strip_check > int(20 * dpi_scale):
        mid_y = (top + bottom) // 2
        band_h = min(max_band, (bottom - top) // 3)
        band_top = mid_y - band_h // 2
        band_bot = mid_y + band_h // 2
        mid_x = (left + right) // 2
        band_w = min(max_band, (right - left) // 3)
        band_left = mid_x - band_w // 2
        band_right = mid_x + band_w // 2

        for y in range(top, min(top + strip_check, bottom)):
            row = gray[y, band_left:band_right]
            if row.mean() < mean_thresh or row.std() > std_thresh:
                top = y
                break
        for y in range(bottom - 1, max(bottom - strip_check, top), -1):
            row = gray[y, band_left:band_right]
            if row.mean() < mean_thresh or row.std() > std_thresh:
                bottom = y + 1
                break
        for x in range(left, min(left + strip_check, right)):
            col = gray[band_top:band_bot, x]
            if col.mean() < mean_thresh or col.std() > std_thresh:
                left = x
                break
        for x in range(right - 1, max(right - strip_check, left), -1):
            col = gray[band_top:band_bot, x]
            if col.mean() < mean_thresh or col.std() > std_thresh:
                right = x + 1
                break

    return {
        'top': top, 'bottom': bottom, 'left': left, 'right': right,
        'angle': angle, 'spine_col': spine_col, 'bleed_method': bleed_method,
    }


# ---------------------------------------------------------------------------
# Deskew and crop
# ---------------------------------------------------------------------------

def deskew_and_crop(image, bounds):
    """Apply deskew rotation and crop to detected page bounds."""
    top = bounds['top']
    bottom = bounds['bottom']
    left = bounds['left']
    right = bounds['right']
    angle = bounds['angle']

    if abs(angle) > 0.05:
        img_h, img_w = image.shape[:2]
        center = (img_w / 2, img_h / 2)
        # Negate the angle: if the page is skewed by -1°, we rotate +1° to correct
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(img_h * sin + img_w * cos)
        new_h = int(img_h * cos + img_w * sin)
        rotation_matrix[0, 2] += (new_w - img_w) / 2
        rotation_matrix[1, 2] += (new_h - img_h) / 2

        image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        # Adjust crop bounds for rotation offset
        dx = (new_w - img_w) / 2
        dy = (new_h - img_h) / 2
        left = int(left + dx)
        right = int(right + dx)
        top = int(top + dy)
        bottom = int(bottom + dy)

    cropped = image[top:bottom, left:right]
    return cropped


# ---------------------------------------------------------------------------
# Dimension normalization
# ---------------------------------------------------------------------------

def normalize_dimensions(pages, target_w, target_h):
    """Center-composite each page onto a black canvas of uniform dimensions."""
    normalized = []
    for page in pages:
        h, w = page.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        paste_w = min(w, target_w)
        paste_h = min(h, target_h)

        src_x = max(0, (w - target_w) // 2)
        src_y = max(0, (h - target_h) // 2)
        dst_x = max(0, (target_w - w) // 2)
        dst_y = max(0, (target_h - h) // 2)

        canvas[dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = \
            page[src_y:src_y + paste_h, src_x:src_x + paste_w]

        normalized.append(canvas)
    return normalized


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_pages(pages, output_dir, quality, dpi, fmt='jpg', lossless=False):
    """Save processed pages as JPEG or WebP files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ext = 'webp' if fmt == 'webp' else 'jpg'

    for i, page in enumerate(pages):
        filename = output_path / f"Scan {i}.{ext}"
        img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))
        if fmt == 'webp':
            if lossless:
                img.save(str(filename), 'WEBP', lossless=True, method=4)
            else:
                img.save(str(filename), 'WEBP', quality=quality, method=4)
        else:
            img.save(str(filename), 'JPEG', quality=quality, dpi=(dpi, dpi))
        size_mb = filename.stat().st_size / (1024 * 1024)
        print(f"  Saved {filename.name} ({img.width}x{img.height}, {size_mb:.1f} MB)")

    return output_path


def preview_pages(pages, indices=None):
    """Open pages for preview using the system image viewer."""
    if indices is None:
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

    for f in preview_files:
        subprocess.Popen(['open', f])

    response = input("\nProceed with saving all pages? [Y/n] ").strip().lower()
    return response != 'n'


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process(input_dir, output_dir=None, quality=85, preview=False,
            pages_to_rotate=None, auto_rotate=False, fmt='jpg', lossless=False):
    """Main processing pipeline."""
    input_path = Path(input_dir)

    if output_dir is None:
        output_dir = Path('output') / input_path.name

    if pages_to_rotate is None:
        pages_to_rotate = set()

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if pages_to_rotate:
        print(f"Rotate 180°: pages {sorted(pages_to_rotate)}")
    if auto_rotate:
        if not os.path.isfile(TESSERACT_BIN):
            print(f"Error: Tesseract not found at {TESSERACT_BIN}")
            print("Install with: brew install tesseract")
            sys.exit(1)
        print(f"Auto-rotate: enabled (Tesseract OCR)")
    print()

    # Step 1: Load and sort scans
    print("Step 1: Loading scans...")
    scans = load_scans(input_dir)

    # Read DPI for every scan (may vary between pages)
    scan_dpis = []
    for _, fp in scans:
        scan_dpis.append(get_source_dpi(fp))
    unique_dpis = sorted(set(scan_dpis))
    if len(unique_dpis) == 1:
        print(f"  Source DPI: {unique_dpis[0]}")
    else:
        dpi_counts = Counter(scan_dpis)
        parts = ", ".join(f"{d} DPI ({n} pages)" for d, n in dpi_counts.most_common())
        print(f"  Mixed DPI: {parts}")
    # Use the most common DPI as the "output" DPI
    output_dpi = Counter(scan_dpis).most_common(1)[0][0]
    print()

    # Step 2: Detect, deskew, crop (and rotate if needed)
    print("Step 2: Processing pages...")
    cropped_pages = []
    for (idx, filepath), page_dpi in zip(scans, scan_dpis):
        dpi_note = f" [{page_dpi}dpi]" if len(unique_dpis) > 1 else ""
        print(f"  Page {idx}: {filepath.name}{dpi_note}")
        image = cv2.imread(str(filepath))
        if image is None:
            print(f"    Error: Could not read {filepath}")
            continue

        # Rotate 180° if requested manually or detected automatically
        if idx in pages_to_rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)
            print(f"    Rotated 180° (manual)")
        elif auto_rotate:
            should_rotate, nw, rw = detect_orientation(image)
            if should_rotate:
                image = cv2.rotate(image, cv2.ROTATE_180)
                print(f"    Rotated 180° (auto: {nw}w normal, {rw}w rotated)")
            else:
                print(f"    Orientation OK (auto: {nw}w normal, {rw}w rotated)")

        # Detect page bounds using this page's actual DPI
        bounds = detect_page_bounds(image, page_dpi)
        det_w = bounds['right'] - bounds['left']
        det_h = bounds['bottom'] - bounds['top']
        margins = (f"T={bounds['top']} "
                   f"B={image.shape[0] - bounds['bottom']} "
                   f"L={bounds['left']} "
                   f"R={image.shape[1] - bounds['right']}")

        extra = ""
        if bounds['spine_col'] is not None:
            method = bounds.get('bleed_method', 'unknown')
            extra += f" BLEED@{bounds['spine_col']}({method})"
        if abs(bounds['angle']) > 0.05:
            extra += f" skew={bounds['angle']:.2f}°"

        print(f"    Detected: {det_w}x{det_h}, margins: {margins}{extra}")

        # Deskew and crop
        cropped = deskew_and_crop(image, bounds)
        ch, cw = cropped.shape[:2]
        print(f"    Cropped:  {cw}x{ch}")
        cropped_pages.append(cropped)
    print()

    # Step 3: Normalize dimensions
    print("Step 3: Normalizing dimensions...")
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
        print("Step 3.5: Preview...")
        if not preview_pages(normalized):
            print("Aborted by user.")
            return None
        print()

    # Step 4: Save
    print("Step 4: Saving processed pages...")
    saved_path = save_pages(normalized, output_dir, quality, output_dpi, fmt, lossless)
    print()
    print(f"Done! {len(normalized)} pages saved to {saved_path}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description='Process raw comic book scans into clean, aligned page images.')
    parser.add_argument('input_dir',
                        help='Directory containing raw scan images')
    parser.add_argument('--output', '-o',
                        help='Output directory (default: output/<input_name>)')
    parser.add_argument('--format', '-f', choices=['jpg', 'webp'], default='jpg',
                        help='Output image format (default: jpg)')
    parser.add_argument('--quality', '-q', type=int, default=85,
                        help='Image quality 1-100 (default: 85)')
    parser.add_argument('--lossless', action='store_true',
                        help='Use lossless compression (WebP only, overrides --quality)')
    parser.add_argument('--preview', '-p', action='store_true',
                        help='Preview pages before saving')

    # 180° rotation options
    rotate_group = parser.add_argument_group('rotation',
        'Options for correcting upside-down pages')
    rotate_group.add_argument('--rotate',
        help='Comma-separated page indices to rotate 180° (e.g., "2,4,6,8" or "2-8")')
    rotate_group.add_argument('--rotate-range',
        help='Range of pages to rotate 180° (e.g., "2-14")')
    rotate_group.add_argument('--rotate-even', action='store_true',
        help='Rotate all even-numbered pages 180°')
    rotate_group.add_argument('--rotate-odd', action='store_true',
        help='Rotate all odd-numbered pages 180°')
    rotate_group.add_argument('--auto-rotate', action='store_true',
        help='Auto-detect and correct upside-down pages using Tesseract OCR')

    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    # Count pages for rotate calculation
    scans = load_scans(args.input_dir)
    total_pages = max(idx for idx, _ in scans) + 1
    pages_to_rotate = parse_rotate_pages(args, total_pages)

    # Suppress the duplicate "Found N scan files" from process()
    print()
    process(args.input_dir, args.output, args.quality, args.preview,
            pages_to_rotate, auto_rotate=args.auto_rotate, fmt=args.format,
            lossless=args.lossless)


if __name__ == '__main__':
    main()
