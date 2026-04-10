#!/usr/bin/env python3
"""
FastAPI web application for interactive comic scan processing.

Provides a REST API for loading scans, running page-boundary detection,
previewing results with user-adjustable crop corners, and batch processing
to final output files.

Usage:
    python webapp/server.py
    # Then open http://127.0.0.1:8000
"""

import io
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Import existing detection logic from the parent package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comicscans import (
    load_scans,
    get_source_dpi,
    detect_page_bounds,
    detect_orientation,
    normalize_dimensions,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="ComicScan Web", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    """Serve the main application page."""
    html = (STATIC_DIR / "index.html").read_text()
    return Response(content=html, media_type="text/html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# sessions[session_id] = {
#     "input_dir": str,
#     "scans": [(idx, Path), ...],          -- from load_scans()
#     "pages": [{index, filename, dpi, width, height}, ...],
#     "thumbnails": {page_index: bytes},     -- JPEG thumbnail cache
#     "detection": {page_index: {...}},      -- auto-detected results
#     "overrides": {page_index: {...}},      -- user overrides
# }
sessions: dict = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    input_dir: str


class UpdatePageRequest(BaseModel):
    corners: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    rotation: float
    rotate180: bool


class ProcessRequest(BaseModel):
    output_dir: str
    format: str = "jpg"
    quality: int = 85


# ---------------------------------------------------------------------------
# Helper: load a page image from disk
# ---------------------------------------------------------------------------
def _load_page_image(session: dict, page_index: int) -> np.ndarray:
    """Load the raw scan image for a page by its list index."""
    if page_index < 0 or page_index >= len(session["scans"]):
        raise HTTPException(status_code=404, detail=f"Page index {page_index} out of range")
    _, filepath = session["scans"][page_index]
    image = cv2.imread(str(filepath))
    if image is None:
        raise HTTPException(status_code=500, detail=f"Could not read image: {filepath}")
    return image


def _encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a cv2 image as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode JPEG")
    return buf.tobytes()


def _get_session(sid: str) -> dict:
    """Look up a session or raise 404."""
    if sid not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {sid} not found")
    return sessions[sid]


# ---------------------------------------------------------------------------
# Coordinate conversion: deskewed bounds -> original image corners
# ---------------------------------------------------------------------------
def _bounds_to_original_corners(bounds: dict, orig_w: int, orig_h: int) -> list:
    """Convert detect_page_bounds() results to 4 corner points in original
    image coordinates.

    detect_page_bounds internally deskews the grayscale image before its
    Pass 2 detection.  The returned {top, bottom, left, right} are in that
    deskewed coordinate space.  To map back:

    1. Build the rectangle corners in deskewed space.
    2. Compute the deskewed canvas size (same math as _deskew_gray).
    3. Rotate each corner by +angle around the deskewed canvas center
       (inverse of the -angle rotation used for deskewing).
    4. Offset by the canvas expansion to land in original image coords.
    """
    angle = bounds["angle"]
    top, bottom = bounds["top"], bounds["bottom"]
    left, right = bounds["left"], bounds["right"]

    # Corners in deskewed space: TL, TR, BR, BL
    deskewed_corners = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
    ], dtype=np.float64)

    if abs(angle) <= 0.1:
        # No deskew was applied; bounds are already in original coords
        return deskewed_corners.tolist()

    # Compute deskewed canvas dimensions (mirrors _deskew_gray)
    rad = np.deg2rad(abs(angle))
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    new_w = int(orig_h * sin_a + orig_w * cos_a)
    new_h = int(orig_h * cos_a + orig_w * sin_a)

    # Center of the deskewed canvas
    cx_desk = new_w / 2.0
    cy_desk = new_h / 2.0

    # Inverse rotation: rotate by +angle (undo the -angle deskew)
    theta = np.deg2rad(angle)  # positive angle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    original_corners = []
    for px, py in deskewed_corners:
        # Translate to deskewed center
        dx = px - cx_desk
        dy = py - cy_desk
        # Rotate by +angle
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        # Translate to original image center
        ox = rx + orig_w / 2.0
        oy = ry + orig_h / 2.0
        original_corners.append([round(ox, 1), round(oy, 1)])

    return original_corners


# ---------------------------------------------------------------------------
# Perspective crop
# ---------------------------------------------------------------------------
def perspective_crop(image: np.ndarray, corners: list) -> np.ndarray:
    """Crop a quadrilateral region and warp it to a rectangle.

    corners: 4 points [[x,y], ...] ordered TL, TR, BR, BL.
    """
    src = np.float32(corners)
    w_top = np.linalg.norm(src[1] - src[0])
    w_bot = np.linalg.norm(src[2] - src[3])
    h_left = np.linalg.norm(src[3] - src[0])
    h_right = np.linalg.norm(src[2] - src[1])
    w = int(max(w_top, w_bot))
    h = int(max(h_left, h_right))
    if w < 1 or h < 1:
        raise HTTPException(status_code=400, detail="Degenerate crop region")
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/session/create")
def create_session(req: CreateSessionRequest):
    """Create a new processing session from a scan directory."""
    input_dir = req.input_dir
    if not Path(input_dir).is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {input_dir}")

    scans = load_scans(input_dir)

    pages = []
    for i, (idx, filepath) in enumerate(scans):
        dpi = get_source_dpi(filepath)
        from PIL import Image as PILImage
        with PILImage.open(filepath) as img:
            width, height = img.size
        pages.append({
            "index": i,
            "scan_index": idx,
            "filename": filepath.name,
            "dpi": dpi,
            "width": width,
            "height": height,
        })

    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "input_dir": input_dir,
        "scans": scans,
        "pages": pages,
        "thumbnails": {},
        "detection": {},
        "overrides": {},
    }

    return {"session_id": sid, "pages": pages}


@app.get("/api/session/{sid}/thumbnail/{page_index}")
def get_thumbnail(sid: str, page_index: int):
    """Return a JPEG thumbnail (max 400px wide), cached in memory."""
    session = _get_session(sid)

    if page_index in session["thumbnails"]:
        return Response(content=session["thumbnails"][page_index],
                        media_type="image/jpeg")

    image = _load_page_image(session, page_index)
    h, w = image.shape[:2]
    max_w = 400
    if w > max_w:
        scale = max_w / w
        image = cv2.resize(image, (max_w, int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    jpeg_bytes = _encode_jpeg(image, quality=70)
    session["thumbnails"][page_index] = jpeg_bytes
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/session/{sid}/image/{page_index}")
def get_image(
    sid: str,
    page_index: int,
    max_size: int = Query(default=2000),
    rotate180: bool = Query(default=False),
):
    """Return a display-resolution JPEG (max_size on longest edge).

    If rotate180=true, the image is rotated 180° before serving. This lets
    the frontend show the correctly-oriented image that matches the corner
    coordinates from detection.
    """
    session = _get_session(sid)
    image = _load_page_image(session, page_index)
    if rotate180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_size:
        scale = max_size / longest
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    jpeg_bytes = _encode_jpeg(image, quality=85)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/api/session/{sid}/detect/{page_index}")
def detect_page(sid: str, page_index: int):
    """Run auto-detection on a single page.

    The detection pipeline mirrors the CLI:
      1. Detect orientation (should the image be flipped 180°?)
      2. Apply 180° rotation if needed
      3. Run detect_page_bounds on the correctly-oriented image
      4. Return corners in the oriented-image coordinate space

    The corners are always relative to the "display" image — i.e. the image
    after any 180° rotation has been applied. The frontend shows the image
    in this orientation and overlays the corners directly.
    """
    session = _get_session(sid)
    image = _load_page_image(session, page_index)
    page_info = session["pages"][page_index]
    dpi = page_info["dpi"]

    # Step 1: Detect orientation
    try:
        rotate180, normal_words, rotated_words = detect_orientation(image)
    except Exception:
        rotate180 = False

    # Step 2: Apply 180° rotation before detection (matches CLI pipeline)
    oriented_image = image
    if rotate180:
        oriented_image = cv2.rotate(image, cv2.ROTATE_180)

    orig_h, orig_w = oriented_image.shape[:2]

    # Step 3: Detect page boundaries on the correctly-oriented image
    bounds = detect_page_bounds(oriented_image, dpi)

    # Step 4: Convert deskewed bounds to oriented-image corner coordinates
    corners = _bounds_to_original_corners(bounds, orig_w, orig_h)

    result = {
        "corners": corners,
        "rotation": bounds["angle"],
        "rotate180": rotate180,
        "bleed_method": bounds.get("bleed_method"),
        "dpi": dpi,
        "original_bounds": {
            "top": bounds["top"],
            "bottom": bounds["bottom"],
            "left": bounds["left"],
            "right": bounds["right"],
            "angle": bounds["angle"],
        },
    }

    session["detection"][page_index] = result
    return result


@app.post("/api/session/{sid}/detect-all")
def detect_all_pages(sid: str):
    """Run detection on every page in the session sequentially."""
    session = _get_session(sid)
    results = []
    for i in range(len(session["scans"])):
        result = detect_page(sid, i)
        results.append(result)
    return results


@app.post("/api/session/{sid}/update/{page_index}")
def update_page(sid: str, page_index: int, req: UpdatePageRequest):
    """Store user overrides for a page's crop corners and rotation."""
    session = _get_session(sid)
    if page_index < 0 or page_index >= len(session["scans"]):
        raise HTTPException(status_code=404, detail=f"Page index {page_index} out of range")

    session["overrides"][page_index] = {
        "corners": req.corners,
        "rotation": req.rotation,
        "rotate180": req.rotate180,
    }
    return {"status": "ok", "page_index": page_index}


def _get_effective_settings(session: dict, page_index: int) -> dict:
    """Return the merged detection + override settings for a page."""
    detection = session["detection"].get(page_index)
    override = session["overrides"].get(page_index)
    if override is not None:
        return override
    if detection is not None:
        return {
            "corners": detection["corners"],
            "rotation": detection["rotation"],
            "rotate180": detection["rotate180"],
        }
    raise HTTPException(
        status_code=400,
        detail=f"No detection or override data for page {page_index}. Run detect first.",
    )


@app.post("/api/session/{sid}/preview/{page_index}")
def preview_page(sid: str, page_index: int):
    """Generate a cropped preview using current corners (auto or override).

    Corners are in oriented-image space (after any 180° rotation), so we
    rotate the raw image first and then apply corners directly.
    """
    session = _get_session(sid)
    settings = _get_effective_settings(session, page_index)
    image = _load_page_image(session, page_index)

    # Rotate 180 if flagged — corners are already in rotated-image coords
    if settings.get("rotate180"):
        image = cv2.rotate(image, cv2.ROTATE_180)

    corners = settings["corners"]
    cropped = perspective_crop(image, corners)

    # Resize for display if very large
    ch, cw = cropped.shape[:2]
    max_dim = 2000
    if max(ch, cw) > max_dim:
        scale = max_dim / max(ch, cw)
        cropped = cv2.resize(cropped, (int(cw * scale), int(ch * scale)),
                             interpolation=cv2.INTER_AREA)

    jpeg_bytes = _encode_jpeg(cropped, quality=88)
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.post("/api/session/{sid}/process")
def process_all(sid: str, req: ProcessRequest):
    """Process all pages and save to output directory."""
    session = _get_session(sid)
    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = req.format if req.format in ("jpg", "webp") else "jpg"
    quality = max(1, min(100, req.quality))
    ext = "webp" if fmt == "webp" else "jpg"

    cropped_pages = []
    page_results = []

    for i in range(len(session["scans"])):
        settings = _get_effective_settings(session, i)
        image = _load_page_image(session, i)

        # Rotate 180 if flagged — corners already in rotated-image coords
        if settings.get("rotate180"):
            image = cv2.rotate(image, cv2.ROTATE_180)

        corners = settings["corners"]
        cropped = perspective_crop(image, corners)
        cropped_pages.append(cropped)

    # Normalize dimensions to median
    if cropped_pages:
        widths = [p.shape[1] for p in cropped_pages]
        heights = [p.shape[0] for p in cropped_pages]
        target_w = int(np.median(widths))
        target_h = int(np.median(heights))
        normalized = normalize_dimensions(cropped_pages, target_w, target_h)
    else:
        normalized = []

    # Determine output DPI (most common among pages)
    from collections import Counter
    dpis = [p["dpi"] for p in session["pages"]]
    output_dpi = Counter(dpis).most_common(1)[0][0] if dpis else 300

    # Save each page
    from PIL import Image as PILImage
    for i, page in enumerate(normalized):
        filename = output_dir / f"Scan {i}.{ext}"
        img = PILImage.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGB))

        if fmt == "webp":
            img.save(str(filename), "WEBP", quality=quality, method=4)
        else:
            img.save(str(filename), "JPEG", quality=quality,
                     dpi=(output_dpi, output_dpi))

        size_mb = filename.stat().st_size / (1024 * 1024)
        page_results.append({
            "index": i,
            "filename": filename.name,
            "width": img.width,
            "height": img.height,
            "size_mb": round(size_mb, 2),
        })

    return {
        "output_dir": str(output_dir),
        "num_pages": len(page_results),
        "pages": page_results,
    }


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
