#!/usr/bin/env python3
"""
comicpackage.py — Package processed comic pages into a CBZ archive with metadata.

Usage:
    python3 comicpackage.py <pages_dir> --title "Title" --series "Series Name" --number 17
    python3 comicpackage.py <pages_dir> --interactive
    python3 comicpackage.py <pages_dir> --qc-only

Example:
    python3 comicpackage.py output/DS9E17/ --title "The Secret of Kling" \\
        --series "Star Trek: Deep Space Nine" --number 17 --year 1994 --month 11
"""

import argparse
import hashlib
import os
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


# --- ComicInfo.xml Generation ---

METADATA_FIELDS = [
    ('title', 'Title', 'Issue title'),
    ('series', 'Series', 'Series name'),
    ('number', 'Number', 'Issue number'),
    ('volume', 'Volume', 'Volume number'),
    ('year', 'Year', 'Publication year'),
    ('month', 'Month', 'Publication month'),
    ('day', 'Day', 'Publication day'),
    ('writer', 'Writer', 'Writer(s)'),
    ('penciller', 'Penciller', 'Penciller(s)'),
    ('inker', 'Inker', 'Inker(s)'),
    ('colorist', 'Colorist', 'Colorist(s)'),
    ('letterer', 'Letterer', 'Letterer(s)'),
    ('editor', 'Editor', 'Editor(s)'),
    ('publisher', 'Publisher', 'Publisher'),
    ('web', 'Web', 'Web URL'),
    ('language', 'LanguageISO', 'Language code (e.g., en)'),
    ('characters', 'Characters', 'Characters (comma-separated)'),
    ('teams', 'Teams', 'Teams (comma-separated)'),
    ('locations', 'Locations', 'Locations (comma-separated)'),
]


def collect_metadata_interactive():
    """Prompt user for each metadata field."""
    print("\nEnter comic metadata (press Enter to skip a field):\n")
    metadata = {}
    for key, xml_name, description in METADATA_FIELDS:
        value = input(f"  {description} [{xml_name}]: ").strip()
        if value:
            metadata[key] = value
    return metadata


def collect_metadata_from_args(args):
    """Extract metadata from CLI arguments."""
    metadata = {}
    for key, _, _ in METADATA_FIELDS:
        value = getattr(args, key, None)
        if value is not None:
            metadata[key] = str(value)
    return metadata


def generate_comicinfo_xml(pages_dir, metadata):
    """Generate ComicInfo.xml content from page images and metadata."""
    pages_path = Path(pages_dir)
    page_files = sorted(pages_path.glob('Scan *.jpg'),
                        key=lambda f: int(f.stem.split(' ')[1]))

    if not page_files:
        # Try without space
        page_files = sorted(pages_path.glob('Scan*.jpg'),
                            key=lambda f: int(f.stem.replace('Scan', '') or '0'))

    root = ET.Element('ComicInfo')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xmlns:xsd', 'http://www.w3.org/2001/XMLSchema')

    # Add metadata fields
    for key, xml_name, _ in METADATA_FIELDS:
        if key in metadata:
            elem = ET.SubElement(root, xml_name)
            elem.text = metadata[key]

    # Page count
    page_count = ET.SubElement(root, 'PageCount')
    page_count.text = str(len(page_files))

    # Pages section with per-page metadata
    pages_elem = ET.SubElement(root, 'Pages')
    for i, page_file in enumerate(page_files):
        img = Image.open(page_file)
        width, height = img.size
        file_size = page_file.stat().st_size

        page_attrs = {
            'Image': str(i),
            'ImageHeight': str(height),
            'ImageWidth': str(width),
            'ImageSize': str(file_size),
        }
        if i == 0:
            page_attrs['Type'] = 'FrontCover'

        ET.SubElement(pages_elem, 'Page', page_attrs)

    # Pretty-print
    ET.indent(root, space='  ')
    xml_str = ET.tostring(root, encoding='unicode', xml_declaration=True)
    return xml_str


# --- CBZ Packaging ---

def create_cbz(pages_dir, output_cbz, metadata):
    """Create a CBZ archive from processed pages and metadata."""
    pages_path = Path(pages_dir)
    output_path = Path(output_cbz)

    # Determine the image subfolder name (use the directory name)
    folder_name = pages_path.name

    # Generate ComicInfo.xml
    xml_content = generate_comicinfo_xml(pages_dir, metadata)

    # Write ComicInfo.xml to the pages directory temporarily
    xml_path = pages_path.parent / 'ComicInfo.xml'
    xml_path.write_text(xml_content, encoding='utf-8')
    print(f"  Generated ComicInfo.xml")

    # Collect page files
    page_files = sorted(pages_path.glob('Scan *.jpg'),
                        key=lambda f: int(f.stem.split(' ')[1]))

    if not page_files:
        print("Error: No processed page files found")
        sys.exit(1)

    # Create CBZ (ZIP with stored compression for JPEGs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_STORED) as zf:
        # Add ComicInfo.xml at the archive root
        zf.write(str(xml_path), 'ComicInfo.xml')

        # Add page images in subfolder
        for page_file in page_files:
            arcname = f"{folder_name}/{page_file.name}"
            zf.write(str(page_file), arcname)

    cbz_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Created {output_path} ({cbz_size_mb:.1f} MB, {len(page_files)} pages)")
    return output_path


# --- QC Checks ---

def compute_phash(image_path, hash_size=8):
    """Compute a simple perceptual hash for duplicate detection."""
    img = Image.open(image_path).convert('L').resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.array(img)
    mean = pixels.mean()
    return (pixels > mean).flatten()


def hamming_distance(hash1, hash2):
    """Compute hamming distance between two perceptual hashes."""
    return np.sum(hash1 != hash2)


def run_qc(pages_dir):
    """Run quality control checks on processed pages."""
    pages_path = Path(pages_dir)
    page_files = sorted(pages_path.glob('Scan *.jpg'),
                        key=lambda f: int(f.stem.split(' ')[1]))

    if not page_files:
        print("Error: No page files found for QC")
        return False

    print(f"\nQC: Checking {len(page_files)} pages in {pages_path}")
    issues = []

    # 1. Page count
    n = len(page_files)
    if n < 24 or n > 48:
        issues.append(f"Unusual page count: {n} (expected 24-48)")
    print(f"  Page count: {n}", "⚠" if n < 24 or n > 48 else "OK")

    # 2. Dimension consistency
    dimensions = []
    file_sizes = []
    for f in page_files:
        img = Image.open(f)
        dimensions.append(img.size)
        file_sizes.append(f.stat().st_size)

    unique_dims = set(dimensions)
    if len(unique_dims) > 1:
        issues.append(f"Inconsistent dimensions: {unique_dims}")
        print(f"  Dimensions: INCONSISTENT - {unique_dims}")
    else:
        w, h = dimensions[0]
        print(f"  Dimensions: {w}x{h} (all consistent) OK")

    # 3. File size outliers
    if len(file_sizes) > 2:
        mean_size = statistics.mean(file_sizes)
        stdev_size = statistics.stdev(file_sizes)
        if stdev_size > 0:
            for i, (f, size) in enumerate(zip(page_files, file_sizes)):
                z_score = (size - mean_size) / stdev_size
                if abs(z_score) > 2.5:
                    size_mb = size / (1024 * 1024)
                    issues.append(f"Page {i} ({f.name}) file size outlier: {size_mb:.1f} MB (z={z_score:.1f})")
        mean_mb = mean_size / (1024 * 1024)
        stdev_mb = stdev_size / (1024 * 1024)
        print(f"  File sizes: mean={mean_mb:.1f} MB, stdev={stdev_mb:.2f} MB")
    else:
        print(f"  File sizes: too few pages for statistical analysis")

    # 4. Blank page detection
    for i, f in enumerate(page_files):
        img = Image.open(f).convert('L')
        pixels = np.array(img)
        std = pixels.std()
        if std < 10:
            issues.append(f"Page {i} ({f.name}) appears blank (std={std:.1f})")

    blank_count = sum(1 for issue in issues if 'appears blank' in issue)
    print(f"  Blank pages: {blank_count} detected", "⚠" if blank_count else "OK")

    # 5. Duplicate detection
    hashes = []
    for f in page_files:
        hashes.append(compute_phash(f))

    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming_distance(hashes[i], hashes[j])
            if dist < 5:  # Very similar
                duplicates.append((i, j, dist))
                issues.append(f"Possible duplicate: page {i} and page {j} (distance={dist})")

    print(f"  Duplicates: {len(duplicates)} pairs detected", "⚠" if duplicates else "OK")

    # 6. JPEG integrity
    corrupt = 0
    for i, f in enumerate(page_files):
        try:
            img = Image.open(f)
            img.verify()
        except Exception as e:
            corrupt += 1
            issues.append(f"Page {i} ({f.name}) corrupt: {e}")

    print(f"  JPEG integrity: {corrupt} corrupt", "⚠" if corrupt else "OK")

    # 7. Two-page bleed remnant detection
    # Look for pages with a dark vertical band (spine shadow) still present
    spine_pages = 0
    for i, f in enumerate(page_files):
        img_arr = np.array(Image.open(f).convert('L'))
        ph, pw = img_arr.shape
        # Check for dark, low-variance vertical bands in the middle 80%
        search_start = int(pw * 0.1)
        search_end = int(pw * 0.9)
        col = search_start
        found_spine = False
        while col < search_end and not found_spine:
            strip = img_arr[:, col]
            if strip.mean() < 50 and strip.std() < 15:
                band_width = 0
                while col < search_end:
                    s = img_arr[:, col]
                    if s.mean() >= 50 or s.std() >= 15:
                        break
                    band_width += 1
                    col += 1
                if band_width >= 10:
                    found_spine = True
                    spine_pages += 1
                    issues.append(f"Page {i} ({f.name}) may have spine remnant (dark band {band_width}px wide)")
            else:
                col += 1

    print(f"  Spine remnants: {spine_pages} detected", "⚠" if spine_pages else "OK")

    # 8. Potential upside-down page detection
    # Heuristic: pages with large black borders on unusual sides
    # (e.g., black top edge but not bottom) may indicate incorrect rotation
    orientation_suspect = 0
    for i, f in enumerate(page_files):
        img_arr = np.array(Image.open(f).convert('L'))
        ph, pw = img_arr.shape
        top_strip = img_arr[:20, :].mean()
        bottom_strip = img_arr[-20:, :].mean()
        # If top is very dark (< 30) but bottom is light (> 100), may be upside-down
        # (scanner bed margin would appear as black band after normalization)
        if top_strip < 30 and bottom_strip > 100:
            orientation_suspect += 1
            issues.append(f"Page {i} ({f.name}) may be upside-down (dark top edge, light bottom)")

    print(f"  Orientation suspects: {orientation_suspect} detected",
          "⚠" if orientation_suspect else "OK")

    # Summary
    print()
    if issues:
        print(f"QC WARNINGS ({len(issues)}):")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("QC PASSED: No issues found.")

    return len(issues) == 0


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Package processed comic pages into a CBZ archive.')
    parser.add_argument('pages_dir', help='Directory containing processed page images')
    parser.add_argument('--output', '-o', help='Output CBZ file path')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactively prompt for metadata')
    parser.add_argument('--qc-only', action='store_true',
                        help='Only run QC checks, do not package')
    parser.add_argument('--folder-name',
                        help='Name for the image subfolder in the CBZ (default: pages dir name)')

    # Metadata arguments
    for key, xml_name, description in METADATA_FIELDS:
        parser.add_argument(f'--{key}', help=description)

    args = parser.parse_args()

    pages_dir = Path(args.pages_dir)
    if not pages_dir.is_dir():
        print(f"Error: {pages_dir} is not a directory")
        sys.exit(1)

    # QC-only mode
    if args.qc_only:
        run_qc(pages_dir)
        return

    # Collect metadata
    if args.interactive:
        metadata = collect_metadata_interactive()
    else:
        metadata = collect_metadata_from_args(args)

    if not metadata.get('language'):
        metadata['language'] = 'en'

    # Determine output path
    if args.output:
        output_cbz = args.output
    else:
        # Build name from metadata or directory name
        if 'series' in metadata and 'number' in metadata:
            name = f"{metadata['series']}-issue_{metadata['number'].zfill(3)}"
            if 'year' in metadata:
                name += f"-({metadata['year']})"
            output_cbz = f"output/{name}.cbz"
        else:
            output_cbz = f"output/{pages_dir.name}.cbz"

    # Run QC first
    qc_passed = run_qc(pages_dir)

    if not qc_passed:
        response = input("\nQC warnings found. Continue with packaging? [y/N] ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return

    # Create CBZ
    print(f"\nPackaging CBZ...")
    create_cbz(pages_dir, output_cbz, metadata)
    print(f"\nDone! CBZ archive: {output_cbz}")


if __name__ == '__main__':
    main()
