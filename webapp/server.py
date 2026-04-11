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
import json
import sys
import time
import uuid
import urllib.request
import urllib.parse
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
app = FastAPI(title="ComicScans Web", version="0.1.0")

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


class CVSearchRequest(BaseModel):
    query: str


class CVIssuesRequest(BaseModel):
    volume_id: int
    issue_number: Optional[str] = None


class CVIssueDetailRequest(BaseModel):
    issue_id: int


class CreateCBZRequest(BaseModel):
    output_dir: str  # where processed pages are
    metadata: dict   # ComicInfo metadata fields
    cbz_output: Optional[str] = None  # custom output path


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


SAVE_FILENAME = ".comicscans_session.json"


def _save_session_file(session: dict):
    """Save detection and override data to a JSON file in the input directory."""
    input_dir = Path(session["input_dir"])
    save_path = input_dir / SAVE_FILENAME

    save_data = {
        "version": 1,
        "detections": {},
        "overrides": {},
    }

    for idx, det in session["detection"].items():
        save_data["detections"][str(idx)] = {
            "corners": det.get("corners"),
            "rotation": det.get("rotation"),
            "rotate180": det.get("rotate180"),
            "bleed_method": det.get("bleed_method"),
        }

    for idx, ovr in session["overrides"].items():
        save_data["overrides"][str(idx)] = {
            "corners": ovr.get("corners"),
            "rotation": ovr.get("rotation"),
            "rotate180": ovr.get("rotate180"),
        }

    save_path.write_text(json.dumps(save_data, indent=2))


def _load_session_file(session: dict):
    """Load saved detection and override data from the input directory if it exists."""
    input_dir = Path(session["input_dir"])
    save_path = input_dir / SAVE_FILENAME

    if not save_path.exists():
        return

    try:
        save_data = json.loads(save_path.read_text())
    except (json.JSONDecodeError, IOError):
        return

    for idx_str, det in save_data.get("detections", {}).items():
        idx = int(idx_str)
        session["detection"][idx] = det

    for idx_str, ovr in save_data.get("overrides", {}).items():
        idx = int(idx_str)
        session["overrides"][idx] = ovr


def _clear_session_file(session: dict) -> bool:
    """Delete the session JSON file from the input directory. Returns True if deleted."""
    input_dir = Path(session["input_dir"])
    save_path = input_dir / SAVE_FILENAME
    if save_path.exists():
        save_path.unlink()
        return True
    return False


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

    _load_session_file(sessions[sid])

    # Include saved detections/overrides so the frontend can restore state
    saved_detections = {str(k): v for k, v in sessions[sid]["detection"].items()}
    saved_overrides = {str(k): v for k, v in sessions[sid]["overrides"].items()}

    return {
        "session_id": sid,
        "pages": pages,
        "has_saved_session": bool(saved_detections),
        "detections": saved_detections,
        "overrides": saved_overrides,
    }


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
    _save_session_file(session)
    return result


@app.post("/api/session/{sid}/clear-cache")
def clear_session_cache(sid: str):
    """Clear saved session data (detections + overrides) from disk and memory."""
    session = _get_session(sid)
    deleted = _clear_session_file(session)
    session["detection"].clear()
    session["overrides"].clear()
    session["thumbnails"].clear()
    return {"status": "ok", "file_deleted": deleted}


@app.post("/api/session/{sid}/detect-all")
def detect_all_pages(sid: str):
    """Run detection on every page in the session sequentially."""
    session = _get_session(sid)
    results = []
    for i in range(len(session["scans"])):
        result = detect_page(sid, i)
        results.append(result)
    _save_session_file(session)
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
    _save_session_file(session)
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
# Config helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = Path.home() / ".comicscans_config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _save_config(config: dict):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# API key endpoints
# ---------------------------------------------------------------------------
@app.get("/api/config/api-key")
def get_api_key():
    config = _load_config()
    key = config.get("comicvine_api_key", "")
    # Mask the key for display
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else ("*" * len(key))
    return {"has_key": bool(key), "masked": masked}


@app.post("/api/config/api-key")
def set_api_key(req: dict):
    key = req.get("api_key", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key required")
    config = _load_config()
    config["comicvine_api_key"] = key
    _save_config(config)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# ComicVine proxy helpers
# ---------------------------------------------------------------------------
_last_cv_request = 0.0


def _cv_request(endpoint: str, params: dict) -> dict:
    """Make a rate-limited request to the ComicVine API."""
    global _last_cv_request
    config = _load_config()
    api_key = config.get("comicvine_api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="ComicVine API key not configured")

    # Rate limit: 1 request per second
    now = time.time()
    wait = 1.0 - (now - _last_cv_request)
    if wait > 0:
        time.sleep(wait)
    _last_cv_request = time.time()

    params["api_key"] = api_key
    params["format"] = "json"

    url = f"https://comicvine.gamespot.com/api/{endpoint}/"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={
        "User-Agent": "comicscans/1.0 (comic scan processor)",
    })

    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    if data.get("error") != "OK" and data.get("status_code") != 1:
        raise HTTPException(status_code=502, detail=f"ComicVine error: {data.get('error', 'unknown')}")

    return data


# ---------------------------------------------------------------------------
# ComicVine proxy endpoints
# ---------------------------------------------------------------------------
@app.post("/api/comicvine/search")
def cv_search_volumes(req: CVSearchRequest):
    """Search ComicVine for volumes (series) by name."""
    data = _cv_request("search", {
        "query": req.query,
        "resources": "volume",
        "field_list": "id,name,start_year,publisher,count_of_issues,image,description",
        "limit": 10,
    })
    results = []
    for vol in data.get("results", []):
        publisher_name = ""
        if vol.get("publisher"):
            publisher_name = vol["publisher"].get("name", "")
        image_url = ""
        if vol.get("image"):
            image_url = vol["image"].get("thumb_url", "") or vol["image"].get("small_url", "")
        results.append({
            "id": vol["id"],
            "name": vol["name"],
            "start_year": vol.get("start_year"),
            "publisher": publisher_name,
            "count_of_issues": vol.get("count_of_issues"),
            "image_url": image_url,
        })
    return {"results": results}


@app.post("/api/comicvine/issues")
def cv_get_issues(req: CVIssuesRequest):
    """Get issues for a volume, optionally filtered by issue number."""
    params = {
        "filter": f"volume:{req.volume_id}",
        "field_list": "id,name,issue_number,cover_date,image,volume",
        "sort": "issue_number:asc",
        "limit": 100,
    }
    if req.issue_number:
        # Normalize: strip leading zeros
        try:
            normalized = str(int(float(req.issue_number)))
        except (ValueError, TypeError):
            normalized = req.issue_number
        params["filter"] += f",issue_number:{normalized}"

    data = _cv_request("issues", params)
    results = []
    for issue in data.get("results", []):
        image_url = ""
        if issue.get("image"):
            image_url = issue["image"].get("thumb_url", "") or issue["image"].get("small_url", "")
        results.append({
            "id": issue["id"],
            "name": issue.get("name"),
            "issue_number": issue.get("issue_number"),
            "cover_date": issue.get("cover_date"),
            "image_url": image_url,
        })
    return {"results": results}


@app.post("/api/comicvine/issue-detail")
def cv_get_issue_detail(req: CVIssueDetailRequest):
    """Get full metadata for a specific issue."""
    data = _cv_request(f"issue/4000-{req.issue_id}", {
        "field_list": "id,name,issue_number,cover_date,description,deck,"
                      "person_credits,character_credits,story_arc_credits,volume",
    })
    issue = data.get("results", {})

    # Extract credits
    writers, pencillers, inkers, colorists, cover_artists, editors = [], [], [], [], [], []
    for person in issue.get("person_credits", []) or []:
        name = person.get("name", "")
        role = person.get("role", "").lower()
        if "writer" in role:
            writers.append(name)
        if "pencil" in role:
            pencillers.append(name)
        if "ink" in role and "drink" not in role:
            inkers.append(name)
        if "color" in role:
            colorists.append(name)
        if "cover" in role:
            cover_artists.append(name)
        if "edit" in role:
            editors.append(name)

    # Extract characters
    characters = [c["name"] for c in (issue.get("character_credits") or [])[:20]]

    # Story arcs
    arcs = [a["name"] for a in (issue.get("story_arc_credits") or [])]

    # Volume info
    volume = issue.get("volume", {}) or {}

    # Clean HTML from description
    import re
    description = issue.get("description") or issue.get("deck") or ""
    # Strip HTML tags
    description = re.sub(r'<table[^>]*>.*?</table>', '', description, flags=re.DOTALL)
    description = re.sub(r'<br\s*/?>', '\n', description)
    description = re.sub(r'<[^>]+>', '', description)
    # Unescape HTML entities
    import html as html_mod
    description = html_mod.unescape(description).strip()
    if len(description) > 2000:
        description = description[:2000] + "..."

    # Parse cover date
    cover_date = issue.get("cover_date") or ""
    year, month = "", ""
    if cover_date:
        parts = cover_date.split("-")
        year = parts[0] if len(parts) >= 1 else ""
        month = str(int(parts[1])) if len(parts) >= 2 else ""

    return {
        "id": issue.get("id"),
        "name": issue.get("name"),
        "issue_number": issue.get("issue_number"),
        "cover_date": cover_date,
        "year": year,
        "month": month,
        "series": volume.get("name", ""),
        "description": description,
        "writer": ", ".join(writers),
        "penciller": ", ".join(pencillers),
        "inker": ", ".join(inkers),
        "colorist": ", ".join(colorists),
        "cover_artist": ", ".join(cover_artists),
        "editor": ", ".join(editors),
        "characters": ", ".join(characters),
        "story_arcs": ", ".join(arcs),
    }


# ---------------------------------------------------------------------------
# CBZ creation endpoint
# ---------------------------------------------------------------------------
@app.post("/api/session/{sid}/create-cbz")
def create_cbz_endpoint(sid: str, req: CreateCBZRequest):
    """Create a CBZ archive from processed pages with metadata."""
    session = _get_session(sid)

    pages_dir = Path(req.output_dir)
    if not pages_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Output directory not found: {req.output_dir}")

    # Import from comicpackage
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from comicpackage import create_cbz, find_page_files

    page_files = find_page_files(pages_dir)
    if not page_files:
        raise HTTPException(status_code=400, detail="No processed page files found in output directory")

    # Build metadata dict
    metadata = {}
    field_map = {
        "title": "title", "series": "series", "number": "number",
        "volume": "volume", "year": "year", "month": "month",
        "writer": "writer", "penciller": "penciller", "inker": "inker",
        "colorist": "colorist", "editor": "editor", "publisher": "publisher",
        "characters": "characters", "web": "web",
        "cover_artist": "cover_artist", "story_arcs": "story_arcs",
    }
    for src_key, dst_key in field_map.items():
        val = req.metadata.get(src_key)
        if val:
            metadata[dst_key] = str(val)

    if not metadata.get("language"):
        metadata["language"] = "en"

    # Determine CBZ output path
    if req.cbz_output:
        cbz_path = req.cbz_output
    else:
        series = metadata.get("series", pages_dir.name)
        number = metadata.get("number", "0").zfill(3)
        year = metadata.get("year", "")
        name = f"{series}-issue_{number}"
        if year:
            name += f"-({year})"
        cbz_path = str(pages_dir.parent / f"{name}.cbz")

    result_path = create_cbz(str(pages_dir), cbz_path, metadata)
    size_mb = Path(result_path).stat().st_size / (1024 * 1024)

    return {
        "status": "ok",
        "cbz_path": str(result_path),
        "size_mb": round(size_mb, 2),
        "pages": len(page_files),
    }


@app.post("/api/browse")
def browse_directory(req: dict):
    """List directory contents for the file picker."""
    path = req.get("path", "")
    if not path:
        p = Path.home()
    else:
        p = Path(path).resolve()

    if not p.exists():
        p = p.parent.resolve()
    if not p.is_dir():
        p = Path.home()

    entries = []
    try:
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith('.'):
                continue
            entries.append({
                "name": item.name,
                "path": str(item.resolve()),
                "is_dir": item.is_dir(),
            })
    except PermissionError:
        pass

    return {
        "current": str(p.resolve()),
        "parent": str(p.parent.resolve()) if p.parent != p else None,
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Run with uvicorn
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
