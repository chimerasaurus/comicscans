# comicscans

A Python toolkit for converting raw flatbed scanner images of physical comic books into clean, reader-ready CBZ archives. Built to replace manual workflows like ScanTailor Advanced with a fully automated pipeline.

> **Note:** This project was built collaboratively with [Claude](https://claude.ai) (Anthropic's AI assistant), which helped design the architecture, implement the image processing algorithms, and debug the bleed detection system. The commit history reflects this — all commits are co-authored.

---

## How It Works

```
raw-scans/DS9E17/            comicscans.py              comicpackage.py
 Scan.jpeg    ──┐
 Scan 1.jpeg  ──┤         ┌──────────────┐           ┌──────────────┐
 Scan 2.jpeg  ──┼────────>│  Detect      │           │  QC Checks   │
 ...          ──┤         │  Rotate      │  output/  │  ComicInfo   │   .cbz
 Scan 35.jpeg ──┘         │  Deskew      ├──────────>│  Package     ├──────>
                          │  Crop        │  DS9E17/  │              │
                          │  Normalize   │           └──────────────┘
                          └──────────────┘
```

The workflow is split into two scripts so you can inspect the processed pages before packaging, and re-package with different metadata without reprocessing.

| Script | Purpose |
|--------|---------|
| `comicscans.py` | Image processing: page detection, rotation, deskew, bleed removal, normalization |
| `comicpackage.py` | Quality control, ComicInfo.xml metadata, CBZ archive creation |

---

## Installation

```bash
# Python dependencies
pip install -r requirements.txt

# External dependency (required for --auto-rotate)
brew install tesseract
```

### Requirements

- Python 3.8+
- OpenCV (`opencv-python >= 4.8`)
- Pillow (`>= 10.0`)
- NumPy (`>= 1.24`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for automatic orientation detection)

---

## Quick Start

### 1. Process raw scans

```bash
python3 comicscans.py raw-scans/DS9E17/ \
  --output output/DS9E17/ \
  --auto-rotate \
  --rotate 1
```

### 2. Package into CBZ

```bash
python3 comicpackage.py output/DS9E17/ \
  --title "The Secret of the Lost Orb" \
  --series "Star Trek: Deep Space Nine" \
  --number 17 \
  --year 1994 --month 11 \
  --publisher "Malibu Comics" \
  --language en
```

Output: `output/Star Trek: Deep Space Nine-issue_017-(1994).cbz`

---

## Usage: comicscans.py

```
python3 comicscans.py <input_dir> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `--output`, `-o` | Output directory (default: `output/<input_name>`) |
| `--format`, `-f` | Output format: `jpg` or `webp` (default: `jpg`) |
| `--quality`, `-q` | Image quality 1-100 (default: 85) |
| `--preview`, `-p` | Preview first, middle, and last pages before saving |
| `--auto-rotate` | Auto-detect upside-down pages using Tesseract OCR |
| `--rotate` | Rotate specific pages 180° (e.g., `2,4,6` or `2-8`) |
| `--rotate-range` | Rotate a range of pages 180° (e.g., `2-14`) |
| `--rotate-even` | Rotate all even-numbered pages 180° |
| `--rotate-odd` | Rotate all odd-numbered pages 180° |

### Input format

Place raw scanner images in a directory. Files must follow the naming pattern:
- `Scan.jpeg` (page 0 / front cover)
- `Scan 1.jpeg` (page 1)
- `Scan 2.jpeg` (page 2)
- ...and so on

Supported formats: JPEG, PNG, TIFF, BMP.

### Examples

```bash
# Basic processing (no rotation correction)
python3 comicscans.py raw-scans/DS9E17/

# Auto-detect orientation + manually fix one page
python3 comicscans.py raw-scans/DS9E17/ --auto-rotate --rotate 1

# Rotate all even pages (alternating scan pattern)
python3 comicscans.py raw-scans/DS9E17/ --rotate-even

# Preview before saving
python3 comicscans.py raw-scans/DS9E17/ --auto-rotate --preview

# WebP output for smaller file size
python3 comicscans.py raw-scans/DS9E17/ --auto-rotate --format webp

# Lower quality for even smaller files
python3 comicscans.py raw-scans/DS9E17/ --format webp --quality 75
```

---

## Usage: comicpackage.py

```
python3 comicpackage.py <pages_dir> [metadata options]
```

### Modes

| Flag | Description |
|------|-------------|
| `--interactive`, `-i` | Prompt for each metadata field |
| `--qc-only` | Run QC checks only, don't package |

### Metadata options

`--title`, `--series`, `--number`, `--volume`, `--year`, `--month`, `--day`, `--writer`, `--penciller`, `--inker`, `--colorist`, `--letterer`, `--editor`, `--publisher`, `--web`, `--language`, `--characters`, `--teams`, `--locations`

### Examples

```bash
# Full metadata via CLI
python3 comicpackage.py output/DS9E17/ \
  --title "The Secret of the Lost Orb" \
  --series "Star Trek: Deep Space Nine" \
  --number 17 --year 1994 --month 11 \
  --writer "Laurie S. Sutton" \
  --penciller "Leonard Kirk" \
  --inker "Jack Snider" \
  --publisher "Malibu Comics" \
  --language en

# Interactive mode
python3 comicpackage.py output/DS9E17/ --interactive

# QC only (no packaging)
python3 comicpackage.py output/DS9E17/ --qc-only
```

---

## Processing Pipeline

### Step 1: Load & Sort

Scans are loaded from the input directory, sorted by page index, and checked for gaps. DPI is read from image metadata (defaults to 300).

### Step 2: Orientation Correction

Pages are rotated 180° as needed. With `--auto-rotate`, Tesseract OCR runs on both the normal and flipped image, counting recognized English words. If the flipped version produces 15%+ more words, the page is upside-down.

Manual `--rotate` flags take precedence and skip auto-detection for those pages.

### Step 3: Two-Page Bleed Detection

When a comic is opened on a flatbed scanner, the adjacent page is often partially visible. The pipeline detects and removes this bleed using a layered strategy:

1. **Dark spine shadow** — Very dark, uniform vertical bands from the binding shadow
2. **Bright gutter** — White page margin between the two visible pages
3. **Dark trough** — Subtle brightness dip (lighter than a full spine shadow)
4. **Gradient detection** — Sharpest brightness transition, weighted toward the expected position
5. **Expected width fallback** — Uses standard comic page width (6.625") at the scan DPI
6. **Secondary trim** — If the page is still too wide after the primary cut, trims the opposite edge

The bleed side (left vs. right) is determined by scanner bed margin position: bottom margin indicates a non-rotated page (bleed on right), top margin indicates a rotated page (bleed on left).

### Step 4: Deskew

Hough line detection on the outer border regions finds near-horizontal and near-vertical edges. Lines are length-weighted and the median angle is used. Only small corrections (0.1°–5°) are applied — flatbed scans never need large rotation.

### Step 5: Crop & Normalize

Pages are cropped to their detected boundaries, then center-composited onto a uniform canvas (median width x median height across all pages). No rescaling — native resolution is preserved.

### Step 6: Save

Pages are saved as JPEG or WebP (selected via `--format`) with the specified quality. JPEG preserves DPI metadata; WebP typically produces 30–50% smaller files at equivalent visual quality.

---

## Quality Control Checks

The packager runs 8 automated checks before creating the CBZ:

| Check | What it detects |
|-------|-----------------|
| Page count | Warns if outside 24–48 range |
| Dimensions | Flags inconsistent page sizes |
| File size outliers | Z-score > 2.5 sigma from mean |
| Blank pages | Low pixel standard deviation |
| Duplicates | Perceptual hash comparison (hamming distance < 5) |
| JPEG integrity | Corrupt or unreadable files |
| Spine remnants | Dark vertical bands from incomplete bleed removal |
| Orientation suspects | Dark top edge + light bottom (possible upside-down) |

---

## CBZ Output Format

The CBZ archive is a ZIP file (using `ZIP_STORED` — no compression, since JPEGs are already compressed) containing:

```
Star Trek: Deep Space Nine-issue_017-(1994).cbz
├── ComicInfo.xml
└── DS9E17/
    ├── Scan 0.jpg
    ├── Scan 1.jpg
    ├── ...
    └── Scan 35.jpg
```

`ComicInfo.xml` follows the [ComicRack schema](https://anansi-project.github.io/docs/comicinfo/intro) with auto-generated page metadata (dimensions, file size, front cover designation).

---

## Scanner Setup

Designed for the Canon LiDE 300 flatbed scanner (and similar models):

- Place the comic page in the **top-left corner** of the scanner bed
- Scan at **300 DPI** or **600 DPI** (both supported)
- The pipeline automatically handles:
  - Scanner bed margins (bottom and right edges)
  - Scanner bed color (~228–243 grayscale, not pure white)
  - Two-page bleeds from open comic bindings
  - Slight skew from page placement

---

## License

This project is personal-use tooling for digitizing a physical comic book collection. No comic book content is included in this repository.
