// =============================================================================
// ComicScans — Frontend Application
// =============================================================================

// ===== State Management =====

const state = {
    sessionId: null,
    pages: [],            // [{index, filename, dpi, width, height}]
    detections: {},       // {pageIndex: {corners, rotation, rotate180, bleed_method, ...}}
    overrides: {},        // {pageIndex: {corners, rotation, rotate180}} — user edits
    currentPage: null,    // index of page being edited
    editorImage: null,    // Image object for the editor canvas
    editorScale: 1,       // display_size / original_size ratio
    draggingCorner: null, // 0-3 or null during drag
};

// Display-space corner coordinates for the overlay canvas
let displayCorners = [[0, 0], [0, 0], [0, 0], [0, 0]];

// ===== DOM References =====

const dom = {
    inputDir:       document.getElementById('input-dir'),
    outputDir:      document.getElementById('output-dir'),
    formatSelect:   document.getElementById('format-select'),
    qualityInput:   document.getElementById('quality-input'),
    btnLoad:        document.getElementById('btn-load'),
    btnDetectAll:   document.getElementById('btn-detect-all'),
    btnClearSession:document.getElementById('btn-clear-session'),
    btnProcessAll:  document.getElementById('btn-process-all'),
    gridView:       document.getElementById('grid-view'),
    gridContainer:  document.getElementById('grid-container'),
    editorView:     document.getElementById('editor-view'),
    bgCanvas:       document.getElementById('bg-canvas'),
    overlayCanvas:  document.getElementById('overlay-canvas'),
    canvasContainer:document.getElementById('canvas-container'),
    editorTitle:    document.getElementById('editor-title'),
    rotationSlider: document.getElementById('rotation-slider'),
    rotationValue:  document.getElementById('rotation-value'),
    btnRotate180:   document.getElementById('btn-rotate180'),
    rotate180Status:document.getElementById('rotate180-status'),
    btnReset:       document.getElementById('btn-reset'),
    btnPreview:     document.getElementById('btn-preview'),
    btnApply:       document.getElementById('btn-apply'),
    btnPrev:        document.getElementById('btn-prev'),
    btnNext:        document.getElementById('btn-next'),
    btnCloseEditor: document.getElementById('btn-close-editor'),
    progressModal:  document.getElementById('progress-modal'),
    progressTitle:  document.getElementById('progress-title'),
    progressBar:    document.getElementById('progress-bar'),
    progressText:   document.getElementById('progress-text'),
    btnCloseModal:  document.getElementById('btn-close-modal'),
    btnCreateCBZ:   document.getElementById('btn-create-cbz'),
};

// Canvas contexts
const bgCtx = dom.bgCanvas.getContext('2d');
const overlayCtx = dom.overlayCanvas.getContext('2d');

// Zoom lens elements
const zoomLens = document.getElementById('zoom-lens');
const zoomCanvas = document.getElementById('zoom-canvas');
const zoomCtx = zoomCanvas.getContext('2d');
const ZOOM_SIZE = 150;   // lens diameter in px
const ZOOM_LEVEL = 4;    // magnification factor

// ===== API Functions =====

async function apiPost(path, body = {}) {
    const resp = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
    }
    return resp.json();
}

async function apiGet(path) {
    const resp = await fetch(path);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
    }
    return resp.json();
}

/** Create a new session from an input directory.
 *  If a .comicscans_session.json exists in the directory, the server loads
 *  its detections/overrides and returns them inline for immediate restore. */
async function createSession(inputDir) {
    const data = await apiPost('/api/session/create', { input_dir: inputDir });
    state.sessionId = data.session_id;
    state.pages = data.pages || [];
    state.detections = {};
    state.overrides = {};
    state.currentPage = null;

    // Restore saved detections and overrides from the server response
    if (data.detections) {
        for (const [idx, det] of Object.entries(data.detections)) {
            state.detections[parseInt(idx)] = det;
        }
    }
    if (data.overrides) {
        for (const [idx, ovr] of Object.entries(data.overrides)) {
            state.overrides[parseInt(idx)] = ovr;
        }
    }

    return data;
}

/** Run detection on a single page. */
async function detectPage(pageIndex) {
    const sid = state.sessionId;
    const data = await apiPost(`/api/session/${sid}/detect/${pageIndex}`);
    state.detections[pageIndex] = data;
    return data;
}

/** Run detection on all pages one-by-one, updating the grid after each. */
async function detectAll() {
    const sid = state.sessionId;
    const total = state.pages.length;
    for (let i = 0; i < total; i++) {
        dom.btnDetectAll.textContent = `Detecting ${i + 1}/${total}...`;
        try {
            const result = await apiPost(`/api/session/${sid}/detect/${i}`);
            state.detections[i] = result;
            // Update just this card's status badge
            const card = document.querySelector(`.grid-card[data-index="${i}"]`);
            if (card) {
                const dot = card.querySelector('.status-dot');
                if (dot) {
                    dot.classList.remove('pending');
                    dot.classList.add('detected');
                }
                const label = card.querySelector('.card-status');
                if (label) {
                    label.textContent = result.bleed_method || 'detected';
                }
            }
        } catch (err) {
            console.error(`Detection failed for page ${i}:`, err);
        }
    }
}

/** Send user overrides for a page to the server. */
async function updatePage(pageIndex, overrideData) {
    const sid = state.sessionId;
    const data = await apiPost(`/api/session/${sid}/update/${pageIndex}`, overrideData);
    return data;
}

/** Get a preview of the processed page. Returns image data. */
async function getPreview(pageIndex) {
    const sid = state.sessionId;
    const resp = await fetch(`/api/session/${sid}/preview/${pageIndex}`, { method: 'POST' });
    if (!resp.ok) throw new Error(`Preview error ${resp.status}`);
    const blob = await resp.blob();
    return URL.createObjectURL(blob);
}

/** Process all pages to the output directory. */
async function processAll(outputDir, format, quality) {
    const sid = state.sessionId;
    const data = await apiPost(`/api/session/${sid}/process`, {
        output_dir: outputDir,
        format: format,
        quality: parseInt(quality),
    });
    return data;
}

// ===== Grid View =====

/** Render the grid of page thumbnails. */
function renderGrid() {
    dom.gridContainer.innerHTML = '';

    state.pages.forEach((page, i) => {
        const card = document.createElement('div');
        card.className = 'grid-card';
        card.dataset.index = i;
        if (state.currentPage === i) card.classList.add('active');

        // Thumbnail image
        const img = document.createElement('img');
        img.src = `/api/session/${state.sessionId}/thumbnail/${i}`;
        img.alt = page.filename || `Page ${i}`;
        img.loading = 'lazy';
        card.appendChild(img);

        // Footer with page number and status
        const footer = document.createElement('div');
        footer.className = 'card-footer';

        const label = document.createElement('span');
        label.className = 'page-label';
        label.textContent = page.filename || `Page ${i}`;
        footer.appendChild(label);

        const badge = document.createElement('span');
        badge.className = 'status-badge';

        const dot = document.createElement('span');
        dot.className = 'status-dot';
        let statusText = 'Pending';

        if (state.overrides[i]) {
            dot.classList.add('adjusted');
            statusText = 'Adjusted';
        } else if (state.detections[i]) {
            dot.classList.add('detected');
            statusText = 'Detected';
        } else {
            dot.classList.add('pending');
        }

        badge.appendChild(dot);
        badge.appendChild(document.createTextNode(statusText));
        footer.appendChild(badge);

        card.appendChild(footer);

        // Click to open editor
        card.addEventListener('click', () => openEditor(i));

        dom.gridContainer.appendChild(card);
    });
}

/** Update just the status badge for a single card without re-rendering everything. */
function updateCardStatus(pageIndex) {
    const card = dom.gridContainer.querySelector(`.grid-card[data-index="${pageIndex}"]`);
    if (!card) return;

    const dot = card.querySelector('.status-dot');
    const badge = card.querySelector('.status-badge');
    if (!dot || !badge) return;

    dot.className = 'status-dot';
    let statusText = 'Pending';

    if (state.overrides[pageIndex]) {
        dot.classList.add('adjusted');
        statusText = 'Adjusted';
    } else if (state.detections[pageIndex]) {
        dot.classList.add('detected');
        statusText = 'Detected';
    } else {
        dot.classList.add('pending');
    }

    badge.innerHTML = '';
    badge.appendChild(dot);
    badge.appendChild(document.createTextNode(statusText));
}

// ===== Editor View =====

/** Open the editor for a given page index. */
function openEditor(pageIndex) {
    // Auto-sync the previous page's overrides to the server before switching
    if (state.currentPage !== null && state.overrides[state.currentPage]) {
        syncPageToServer(state.currentPage);
    }

    state.currentPage = pageIndex;
    dom.gridView.classList.add('hidden');
    dom.editorView.classList.remove('hidden');

    const page = state.pages[pageIndex];
    dom.editorTitle.textContent = page.filename || `Page ${pageIndex}`;

    // Update nav label
    const navLabel = document.getElementById('nav-page-label');
    if (navLabel) navLabel.textContent = `${pageIndex + 1} / ${state.pages.length}`;

    loadEditorImage(pageIndex);
}

/** Close the editor and return to grid view. */
function closeEditor() {
    // Auto-sync current page's overrides to the server
    if (state.currentPage !== null && state.overrides[state.currentPage]) {
        syncPageToServer(state.currentPage);
    }

    dom.editorView.classList.add('hidden');
    dom.gridView.classList.remove('hidden');
    state.currentPage = null;
    state.draggingCorner = null;
    renderGrid();
}

/** Load the display-resolution image for the editor.
 *  If detection data exists and rotate180 is set, request the rotated image
 *  from the server so the canvas coordinates match the corner coordinates. */
function loadEditorImage(pageIndex) {
    const sid = state.sessionId;
    const data = getPageData(pageIndex);
    const rotate180 = data ? data.rotate180 : false;

    // Reset any CSS rotation from the previous page
    dom.canvasContainer.style.transform = '';

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        state.editorImage = img;
        setupCanvas(img);
        drawBackgroundImage();
        loadDetectionOverlay(pageIndex);
    };
    img.onerror = () => {
        console.error('Failed to load editor image for page', pageIndex);
    };
    img.src = `/api/session/${sid}/image/${pageIndex}?max_size=2000&rotate180=${rotate180}`;
}

/** Set canvas dimensions to fit the image within the editor panel. */
function setupCanvas(img) {
    const panel = dom.canvasContainer.parentElement; // #editor-canvas-panel
    const panelStyle = getComputedStyle(panel);
    const padTop = parseFloat(panelStyle.paddingTop) || 0;
    const padBot = parseFloat(panelStyle.paddingBottom) || 0;
    const padLeft = parseFloat(panelStyle.paddingLeft) || 0;
    const padRight = parseFloat(panelStyle.paddingRight) || 0;
    // Available content area minus extra margin for corner handles (8px radius + breathing room)
    const maxW = panel.clientWidth - padLeft - padRight - 24;
    const maxH = panel.clientHeight - padTop - padBot - 24;

    // Scale to fit
    const scaleW = maxW / img.naturalWidth;
    const scaleH = maxH / img.naturalHeight;
    const scale = Math.min(scaleW, scaleH, 1); // never upscale

    const w = Math.round(img.naturalWidth * scale);
    const h = Math.round(img.naturalHeight * scale);

    dom.bgCanvas.width = w;
    dom.bgCanvas.height = h;
    dom.overlayCanvas.width = w;
    dom.overlayCanvas.height = h;

    // Calculate the mapping scale between display and original image coordinates.
    // The image served at ?max_size=2000 may already be downscaled from the true
    // original scan dimensions. We keep track of *two* relationships:
    //   editorScale = canvas pixels / original scan pixels
    // Original scan dimensions come from state.pages[].width/height.
    const page = state.pages[state.currentPage];
    if (page && page.width) {
        state.editorScale = w / page.width;
    } else {
        // Fallback: assume the served image IS the original
        state.editorScale = w / img.naturalWidth;
    }
}

/** Draw the scan image on the background canvas.
 *  The server already applies any 180° rotation, so we draw directly. */
function drawBackgroundImage() {
    const img = state.editorImage;
    if (!img) return;

    const w = dom.bgCanvas.width;
    const h = dom.bgCanvas.height;

    bgCtx.clearRect(0, 0, w, h);

    // The image is already correctly oriented from the server
    if (false) {
        // placeholder for future canvas-side rotation (unused)
    } else {
        bgCtx.drawImage(img, 0, 0, w, h);
    }
}

/** Get the effective page data (override if exists, else detection). */
function getPageData(pageIndex) {
    return state.overrides[pageIndex] || state.detections[pageIndex] || null;
}

/** Load detection data into the overlay. */
function loadDetectionOverlay(pageIndex) {
    const data = getPageData(pageIndex);
    if (!data || !data.corners) {
        // No detection yet — clear overlay and set defaults
        overlayCtx.clearRect(0, 0, dom.overlayCanvas.width, dom.overlayCanvas.height);
        dom.rotationSlider.value = 0;
        dom.rotationValue.textContent = '0.00';
        dom.rotate180Status.textContent = 'Off';
        displayCorners = [
            [0, 0],
            [dom.overlayCanvas.width, 0],
            [dom.overlayCanvas.width, dom.overlayCanvas.height],
            [0, dom.overlayCanvas.height],
        ];
        updateCornerDisplay();
        drawOverlay();
        return;
    }

    // Set rotation controls
    const rotation = data.rotation || 0;
    dom.rotationSlider.value = rotation;
    dom.rotationValue.textContent = rotation.toFixed(2);
    dom.rotate180Status.textContent = data.rotate180 ? 'On' : 'Off';

    // Convert original-pixel corners to display-pixel corners
    // corners format: [[x,y], [x,y], [x,y], [x,y]] = TL, TR, BR, BL
    const scale = state.editorScale;
    displayCorners = data.corners.map(([x, y]) => [x * scale, y * scale]);

    updateCornerDisplay();
    drawOverlay();
}

/** Draw the overlay: mask, quadrilateral edges, corner handles. */
function drawOverlay() {
    const ctx = overlayCtx;
    const w = dom.overlayCanvas.width;
    const h = dom.overlayCanvas.height;
    ctx.clearRect(0, 0, w, h);

    if (displayCorners.length < 4) return;

    // Draw dark mask outside the crop quadrilateral
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, w, h);
    // Cut out the quadrilateral
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath();
    ctx.moveTo(displayCorners[0][0], displayCorners[0][1]);
    for (let i = 1; i < 4; i++) {
        ctx.lineTo(displayCorners[i][0], displayCorners[i][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Draw edges
    ctx.strokeStyle = '#e94560';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(displayCorners[0][0], displayCorners[0][1]);
    for (let i = 1; i < 4; i++) {
        ctx.lineTo(displayCorners[i][0], displayCorners[i][1]);
    }
    ctx.closePath();
    ctx.stroke();

    // Draw corner handles
    for (let i = 0; i < 4; i++) {
        const [x, y] = displayCorners[i];
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

/** Convert display corners back to original image coordinates and store as override. */
function saveCorners() {
    const scale = state.editorScale;
    const origCorners = displayCorners.map(([x, y]) => [
        Math.round(x / scale),
        Math.round(y / scale),
    ]);

    const pageIndex = state.currentPage;
    if (!state.overrides[pageIndex]) {
        // Start from detection data
        const detection = state.detections[pageIndex];
        state.overrides[pageIndex] = detection ? { ...detection } : {};
    }
    state.overrides[pageIndex].corners = origCorners;

    updateCornerDisplay();
}

/** Extract only the fields the server expects for an update. */
function extractUpdatePayload(data) {
    return {
        corners: data.corners,
        rotation: data.rotation || 0,
        rotate180: data.rotate180 || false,
    };
}

/** Sync a single page's overrides to the server (fire-and-forget). */
async function syncPageToServer(pageIndex) {
    const data = getPageData(pageIndex);
    if (!data || !data.corners) return;
    try {
        await updatePage(pageIndex, extractUpdatePayload(data));
    } catch (err) {
        console.error(`Failed to sync page ${pageIndex} to server:`, err);
    }
}

/** Sync ALL pages that have client-side overrides or detections to the server. */
async function syncAllOverridesToServer() {
    const promises = [];
    // Sync overrides (user-adjusted pages)
    for (const [pageIndex, data] of Object.entries(state.overrides)) {
        if (data && data.corners) {
            promises.push(updatePage(parseInt(pageIndex), extractUpdatePayload(data)));
        }
    }
    // Also sync detection-only pages so the server has corners for all detected pages
    for (const [pageIndex, data] of Object.entries(state.detections)) {
        if (!state.overrides[pageIndex] && data && data.corners) {
            promises.push(updatePage(parseInt(pageIndex), extractUpdatePayload(data)));
        }
    }
    if (promises.length > 0) {
        await Promise.all(promises);
    }
}

/** Update the corner coordinate readout in the controls panel. */
function updateCornerDisplay() {
    const scale = state.editorScale;
    for (let i = 0; i < 4; i++) {
        const el = document.getElementById(`corner-${i}`);
        if (el && displayCorners[i]) {
            const ox = Math.round(displayCorners[i][0] / scale);
            const oy = Math.round(displayCorners[i][1] / scale);
            el.textContent = `(${ox}, ${oy})`;
        }
    }
}

// ===== Zoom Lens =====

/** Show the zoom lens at a given canvas position during corner drag. */
function showZoomLens(canvasX, canvasY) {
    if (!state.editorImage) return;

    // Position the lens offset from the drag point so it doesn't obscure
    // the corner being dragged. Place it above-right by default, but flip
    // if near the edges.
    const containerRect = dom.canvasContainer.getBoundingClientRect();
    const canvasRect = dom.overlayCanvas.getBoundingClientRect();
    const offsetX = canvasX < dom.overlayCanvas.width - 180 ? 40 : -ZOOM_SIZE - 40;
    const offsetY = canvasY > 100 ? -ZOOM_SIZE - 20 : 40;

    zoomLens.style.left = (canvasX + offsetX) + 'px';
    zoomLens.style.top  = (canvasY + offsetY) + 'px';
    zoomLens.style.display = 'block';

    // Draw magnified region from the BACKGROUND canvas (the scan image).
    // Sample a region centered on (canvasX, canvasY) at 1/ZOOM_LEVEL size.
    const srcSize = ZOOM_SIZE / ZOOM_LEVEL;
    const sx = canvasX - srcSize / 2;
    const sy = canvasY - srcSize / 2;

    zoomCtx.clearRect(0, 0, ZOOM_SIZE, ZOOM_SIZE);

    // Draw the background image portion (zoomed)
    zoomCtx.drawImage(
        dom.bgCanvas,
        sx, sy, srcSize, srcSize,
        0, 0, ZOOM_SIZE, ZOOM_SIZE
    );

    // Draw the overlay edges on the zoom lens too for context
    zoomCtx.save();
    zoomCtx.scale(ZOOM_LEVEL, ZOOM_LEVEL);
    zoomCtx.translate(-sx, -sy);
    // Draw edges
    zoomCtx.strokeStyle = '#e94560';
    zoomCtx.lineWidth = 2 / ZOOM_LEVEL;
    zoomCtx.beginPath();
    zoomCtx.moveTo(displayCorners[0][0], displayCorners[0][1]);
    for (let i = 1; i < 4; i++) {
        zoomCtx.lineTo(displayCorners[i][0], displayCorners[i][1]);
    }
    zoomCtx.closePath();
    zoomCtx.stroke();
    zoomCtx.restore();
}

function hideZoomLens() {
    zoomLens.style.display = 'none';
}

// ===== Canvas Mouse Interaction =====

function handleMouseDown(e) {
    if (state.currentPage === null) return;
    const rect = dom.overlayCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Hit test: find closest corner within 20px
    for (let i = 0; i < 4; i++) {
        const [cx, cy] = displayCorners[i];
        if (Math.hypot(x - cx, y - cy) < 20) {
            state.draggingCorner = i;
            dom.overlayCanvas.style.cursor = 'none'; // hide cursor, lens replaces it
            showZoomLens(cx, cy);
            e.preventDefault();
            return;
        }
    }
}

function handleMouseMove(e) {
    const rect = dom.overlayCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (state.draggingCorner !== null) {
        // Clamp to canvas bounds — allows dragging right to the edge (0 or max)
        const clampedX = Math.max(0, Math.min(x, dom.overlayCanvas.width));
        const clampedY = Math.max(0, Math.min(y, dom.overlayCanvas.height));
        displayCorners[state.draggingCorner] = [clampedX, clampedY];
        drawOverlay();
        updateCornerDisplay();
        showZoomLens(clampedX, clampedY);
        return;
    }

    // Hover cursor: show grab cursor when near a corner
    let nearCorner = false;
    for (let i = 0; i < 4; i++) {
        const [cx, cy] = displayCorners[i];
        if (Math.hypot(x - cx, y - cy) < 20) {
            nearCorner = true;
            break;
        }
    }
    dom.overlayCanvas.style.cursor = nearCorner ? 'grab' : 'default';
}

function handleMouseUp(e) {
    if (state.draggingCorner !== null) {
        state.draggingCorner = null;
        dom.overlayCanvas.style.cursor = 'default';
        hideZoomLens();
        saveCorners();
        // Remove document-level listeners that were added during drag
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
    }
}

// Attach canvas events — mousedown on canvas starts drag,
// then mousemove/mouseup are tracked on document so dragging
// to the very edge of the canvas (or slightly beyond) works.
dom.overlayCanvas.addEventListener('mousedown', (e) => {
    handleMouseDown(e);
    if (state.draggingCorner !== null) {
        // Track globally so cursor leaving the canvas doesn't drop the corner
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }
});
dom.overlayCanvas.addEventListener('mousemove', handleMouseMove);

// ===== Editor Controls =====

// Rotation slider — real-time visual rotation preview via CSS transform
dom.rotationSlider.addEventListener('input', () => {
    const val = parseFloat(dom.rotationSlider.value);
    dom.rotationValue.textContent = val.toFixed(2);

    const pageIndex = state.currentPage;
    if (pageIndex === null) return;

    if (!state.overrides[pageIndex]) {
        const detection = state.detections[pageIndex];
        state.overrides[pageIndex] = detection ? { ...detection } : {};
    }
    state.overrides[pageIndex].rotation = val;

    // Apply the fine-rotation visually to the canvas container.
    // The detection's base rotation is already baked into the corner coordinates,
    // so we show the *delta* from the detected angle as a CSS transform.
    const detected = state.detections[pageIndex];
    const baseRotation = detected ? (detected.rotation || 0) : 0;
    const delta = val - baseRotation;
    dom.canvasContainer.style.transform = `rotate(${delta}deg)`;
});

// 180 degree toggle
dom.btnRotate180.addEventListener('click', () => {
    const pageIndex = state.currentPage;
    if (pageIndex === null) return;

    if (!state.overrides[pageIndex]) {
        const detection = state.detections[pageIndex];
        state.overrides[pageIndex] = detection ? { ...detection } : {};
    }

    const current = state.overrides[pageIndex].rotate180 || false;
    state.overrides[pageIndex].rotate180 = !current;
    dom.rotate180Status.textContent = !current ? 'On' : 'Off';

    // Transform existing corners to match the new orientation.
    // Flipping 180° maps (x, y) → (W - x, H - y), and the corner roles
    // swap: TL↔BR, TR↔BL.
    const page = state.pages[pageIndex];
    if (page && state.overrides[pageIndex].corners) {
        const W = page.width;
        const H = page.height;
        const c = state.overrides[pageIndex].corners;
        // c = [TL, TR, BR, BL] → after 180° each point mirrors and roles swap
        state.overrides[pageIndex].corners = [
            [W - c[2][0], H - c[2][1]],  // new TL = old BR mirrored
            [W - c[3][0], H - c[3][1]],  // new TR = old BL mirrored
            [W - c[0][0], H - c[0][1]],  // new BR = old TL mirrored
            [W - c[1][0], H - c[1][1]],  // new BL = old TR mirrored
        ];
    }

    // Reload the image with the new rotation so the display matches
    loadEditorImage(pageIndex);
});

// Reset to auto-detected values
dom.btnReset.addEventListener('click', () => {
    const pageIndex = state.currentPage;
    if (pageIndex === null) return;

    delete state.overrides[pageIndex];
    dom.canvasContainer.style.transform = '';
    loadDetectionOverlay(pageIndex);
    drawBackgroundImage();
    updateCardStatus(pageIndex);
});

// Preview button — request a server-rendered preview
dom.btnPreview.addEventListener('click', async () => {
    const pageIndex = state.currentPage;
    if (pageIndex === null) return;

    dom.btnPreview.disabled = true;
    dom.btnPreview.textContent = 'Loading...';

    try {
        // Send current overrides to the server first
        const data = getPageData(pageIndex);
        if (data) {
            await updatePage(pageIndex, extractUpdatePayload(data));
        }
        const previewUrl = await getPreview(pageIndex);
        // Show preview in a new window/tab
        window.open(previewUrl, '_blank');
    } catch (err) {
        console.error('Preview failed:', err);
        alert('Preview failed: ' + err.message);
    } finally {
        dom.btnPreview.disabled = false;
        dom.btnPreview.textContent = 'Preview';
    }
});

// Apply button — save overrides to server
dom.btnApply.addEventListener('click', async () => {
    const pageIndex = state.currentPage;
    if (pageIndex === null) return;

    const data = getPageData(pageIndex);
    if (!data) return;

    dom.btnApply.disabled = true;
    dom.btnApply.textContent = 'Saving...';

    try {
        await updatePage(pageIndex, extractUpdatePayload(data));
        updateCardStatus(pageIndex);
    } catch (err) {
        console.error('Apply failed:', err);
        alert('Failed to save: ' + err.message);
    } finally {
        dom.btnApply.disabled = false;
        dom.btnApply.textContent = 'Apply';
    }
});

// Navigation: Prev / Next
dom.btnPrev.addEventListener('click', () => navigateEditor(-1));
dom.btnNext.addEventListener('click', () => navigateEditor(1));

function navigateEditor(delta) {
    if (state.currentPage === null) return;
    const newIndex = state.currentPage + delta;
    if (newIndex >= 0 && newIndex < state.pages.length) {
        openEditor(newIndex);
    }
}

// Close editor
dom.btnCloseEditor.addEventListener('click', closeEditor);

// ===== Keyboard Shortcuts =====

document.addEventListener('keydown', (e) => {
    // Only handle shortcuts when editor is open
    if (state.currentPage === null) return;

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigateEditor(-1);
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        navigateEditor(1);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        closeEditor();
    }
});

// ===== Top Bar Actions =====

// Load button
dom.btnLoad.addEventListener('click', async () => {
    const inputDir = dom.inputDir.value.trim();
    if (!inputDir) {
        alert('Please enter a directory path.');
        return;
    }

    dom.btnLoad.disabled = true;
    dom.btnLoad.textContent = 'Loading...';

    try {
        await createSession(inputDir);
        renderGrid();
    } catch (err) {
        console.error('Failed to create session:', err);
        alert('Failed to load directory: ' + err.message);
    } finally {
        dom.btnLoad.disabled = false;
        dom.btnLoad.textContent = 'Load';
    }
});

// Detect All button
dom.btnDetectAll.addEventListener('click', async () => {
    if (!state.sessionId) {
        alert('Load a directory first.');
        return;
    }

    dom.btnDetectAll.disabled = true;
    dom.btnDetectAll.textContent = 'Detecting...';

    try {
        await detectAll();
        renderGrid();
    } catch (err) {
        console.error('Detect all failed:', err);
        alert('Detection failed: ' + err.message);
    } finally {
        dom.btnDetectAll.disabled = false;
        dom.btnDetectAll.textContent = 'Detect All';
    }
});

// Clear Cache button
dom.btnClearSession.addEventListener('click', async () => {
    if (!state.sessionId) {
        alert('No session loaded.');
        return;
    }

    if (!confirm('Clear all saved detections and overrides? You will need to re-detect all pages.')) {
        return;
    }

    try {
        await apiPost(`/api/session/${state.sessionId}/clear-cache`);
        state.detections = {};
        state.overrides = {};
        // Reset grid badges
        renderGrid();
        // If editor is open, reset the overlay
        if (state.currentPage !== null) {
            loadDetectionOverlay(state.currentPage);
        }
    } catch (err) {
        console.error('Clear cache failed:', err);
        alert('Failed to clear cache: ' + err.message);
    }
});

// Process All button
dom.btnProcessAll.addEventListener('click', async () => {
    if (!state.sessionId) {
        alert('Load a directory first.');
        return;
    }

    const outputDir = dom.outputDir.value.trim();
    if (!outputDir) {
        alert('Please enter an output directory.');
        return;
    }

    const format = dom.formatSelect.value;
    const quality = dom.qualityInput.value;

    // Show progress modal
    showProgress('Processing All Pages...', 'Syncing overrides...');

    try {
        // Sync all pending client-side overrides to the server first
        await syncAllOverridesToServer();

        dom.progressText.textContent = 'Processing pages...';
        const result = await processAll(outputDir, format, quality);

        // Stop the indeterminate animation and jump to 100%
        if (dom.progressModal._interval) {
            clearInterval(dom.progressModal._interval);
            dom.progressModal._interval = null;
        }
        dom.progressBar.style.width = '100%';
        const count = result.num_pages || result.processed;
        if (count !== undefined) {
            dom.progressText.textContent = `Done! ${count} pages processed.`;
        } else {
            dom.progressText.textContent = 'Processing complete!';
        }
        dom.progressTitle.textContent = 'Complete';
        dom.btnCloseModal.classList.remove('hidden');
    } catch (err) {
        console.error('Process all failed:', err);
        if (dom.progressModal._interval) {
            clearInterval(dom.progressModal._interval);
            dom.progressModal._interval = null;
        }
        dom.progressTitle.textContent = 'Error';
        dom.progressText.textContent = 'Processing failed: ' + err.message;
        dom.btnCloseModal.classList.remove('hidden');
    }
});

// ===== Progress Modal =====

function showProgress(title, text) {
    dom.progressTitle.textContent = title;
    dom.progressText.textContent = text;
    dom.progressBar.style.width = '0%';
    dom.btnCloseModal.classList.add('hidden');
    dom.progressModal.classList.remove('hidden');

    // Animate an indeterminate-ish progress bar
    let progress = 0;
    const interval = setInterval(() => {
        if (progress >= 90) {
            clearInterval(interval);
            return;
        }
        // Slow down as it approaches 90%
        progress += (90 - progress) * 0.05;
        dom.progressBar.style.width = progress + '%';
    }, 200);

    // Store interval ID so we can clear it
    dom.progressModal._interval = interval;
}

dom.btnCloseModal.addEventListener('click', () => {
    if (dom.progressModal._interval) {
        clearInterval(dom.progressModal._interval);
    }
    dom.progressModal.classList.add('hidden');
});

// ===== Window Resize Handling =====

window.addEventListener('resize', () => {
    if (state.currentPage !== null && state.editorImage) {
        setupCanvas(state.editorImage);
        drawBackgroundImage();
        // Recalculate display corners from original coordinates
        loadDetectionOverlay(state.currentPage);
    }
});

// ===== CBZ Creation =====

const cbzDom = {
    modal: document.getElementById('cbz-modal'),
    apiKeyInput: document.getElementById('cv-api-key'),
    btnSaveKey: document.getElementById('btn-save-api-key'),
    keyStatus: document.getElementById('cv-key-status'),
    searchInput: document.getElementById('cv-search-input'),
    btnSearch: document.getElementById('btn-cv-search'),
    searchResults: document.getElementById('cv-search-results'),
    stepSearch: document.getElementById('cbz-step-search'),
    stepIssue: document.getElementById('cbz-step-issue'),
    stepMetadata: document.getElementById('cbz-step-metadata'),
    selectedSeriesName: document.getElementById('cv-selected-series-name'),
    btnBackSearch: document.getElementById('btn-cv-back-search'),
    issueNumberInput: document.getElementById('cv-issue-number'),
    btnFindIssue: document.getElementById('btn-cv-find-issue'),
    issueUrlInput: document.getElementById('cv-issue-url'),
    btnUseUrl: document.getElementById('btn-cv-use-url'),
    issueResults: document.getElementById('cv-issue-results'),
    btnBackIssue: document.getElementById('btn-cv-back-issue'),
    btnCreateGo: document.getElementById('btn-create-cbz-go'),
    btnCancel: document.getElementById('btn-cbz-cancel'),
    btnClose: document.getElementById('btn-cbz-close'),
};

let cbzState = {
    selectedVolume: null,  // {id, name, publisher, start_year}
    selectedIssue: null,   // issue metadata from CV
};

// Open CBZ modal
dom.btnCreateCBZ.addEventListener('click', async () => {
    if (!state.sessionId) {
        alert('Load a directory first.');
        return;
    }
    const outputDir = dom.outputDir.value.trim();
    if (!outputDir) {
        alert('Please enter an output directory (pages must be processed first).');
        return;
    }

    cbzState = { selectedVolume: null, selectedIssue: null };
    cbzDom.stepIssue.classList.add('hidden');
    cbzDom.stepMetadata.classList.add('hidden');
    cbzDom.stepSearch.classList.remove('hidden');
    cbzDom.searchResults.innerHTML = '';
    cbzDom.issueResults.innerHTML = '';
    cbzDom.modal.classList.remove('hidden');

    // Check API key status
    try {
        const resp = await apiGet('/api/config/api-key');
        if (resp.has_key) {
            cbzDom.keyStatus.textContent = `API key saved (${resp.masked})`;
            cbzDom.apiKeyInput.placeholder = 'Key saved — enter new key to replace';
        } else {
            cbzDom.keyStatus.innerHTML = 'No API key saved. <a href="https://comicvine.gamespot.com/api/" target="_blank">Get one here</a>';
        }
    } catch (e) {
        console.error('Failed to check API key:', e);
    }
});

// Save API key
cbzDom.btnSaveKey.addEventListener('click', async () => {
    const key = cbzDom.apiKeyInput.value.trim();
    if (!key) return;
    try {
        await apiPost('/api/config/api-key', { api_key: key });
        cbzDom.keyStatus.textContent = 'API key saved!';
        cbzDom.apiKeyInput.value = '';
        cbzDom.apiKeyInput.placeholder = 'Key saved — enter new key to replace';
    } catch (e) {
        cbzDom.keyStatus.textContent = 'Failed to save key: ' + e.message;
    }
});

// Search volumes
cbzDom.btnSearch.addEventListener('click', async () => {
    const query = cbzDom.searchInput.value.trim();
    if (!query) return;
    cbzDom.btnSearch.disabled = true;
    cbzDom.btnSearch.textContent = 'Searching...';
    cbzDom.searchResults.innerHTML = '';

    try {
        const data = await apiPost('/api/comicvine/search', { query });
        if (data.results.length === 0) {
            cbzDom.searchResults.innerHTML = '<p class="cbz-hint">No results found.</p>';
        } else {
            data.results.forEach(vol => {
                const item = document.createElement('div');
                item.className = 'cbz-result-item';
                item.innerHTML = `
                    <img src="${vol.image_url || ''}" alt="" onerror="this.style.display='none'">
                    <div class="cbz-result-info">
                        <div class="name">${escapeHtml(vol.name)}</div>
                        <div class="detail">${escapeHtml(vol.publisher || 'Unknown')} · ${vol.start_year || '?'} · ${vol.count_of_issues || '?'} issues</div>
                    </div>
                `;
                item.addEventListener('click', () => selectVolume(vol));
                cbzDom.searchResults.appendChild(item);
            });
        }
    } catch (e) {
        cbzDom.searchResults.innerHTML = `<p class="cbz-hint" style="color:#e94560">${e.message}</p>`;
    } finally {
        cbzDom.btnSearch.disabled = false;
        cbzDom.btnSearch.textContent = 'Search';
    }
});

// Allow Enter key in search
cbzDom.searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') cbzDom.btnSearch.click();
});

cbzDom.issueNumberInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') cbzDom.btnFindIssue.click();
});

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
}

// Select a volume/series
function selectVolume(vol) {
    cbzState.selectedVolume = vol;
    cbzDom.selectedSeriesName.textContent = `${vol.name} (${vol.publisher || 'Unknown'}, ${vol.start_year || '?'})`;
    cbzDom.stepSearch.classList.add('hidden');
    cbzDom.stepIssue.classList.remove('hidden');
    cbzDom.issueResults.innerHTML = '';
}

// Back to search
cbzDom.btnBackSearch.addEventListener('click', () => {
    cbzDom.stepIssue.classList.add('hidden');
    cbzDom.stepMetadata.classList.add('hidden');
    cbzDom.stepSearch.classList.remove('hidden');
});

// Find issue by number
cbzDom.btnFindIssue.addEventListener('click', async () => {
    const num = cbzDom.issueNumberInput.value.trim();
    if (!num || !cbzState.selectedVolume) return;
    cbzDom.btnFindIssue.disabled = true;
    cbzDom.btnFindIssue.textContent = 'Finding...';
    cbzDom.issueResults.innerHTML = '';

    try {
        const data = await apiPost('/api/comicvine/issues', {
            volume_id: cbzState.selectedVolume.id,
            issue_number: num,
        });
        if (data.results.length === 0) {
            cbzDom.issueResults.innerHTML = '<p class="cbz-hint">No matching issue found.</p>';
        } else {
            data.results.forEach(issue => {
                const item = document.createElement('div');
                item.className = 'cbz-result-item';
                item.innerHTML = `
                    <img src="${issue.image_url || ''}" alt="" onerror="this.style.display='none'">
                    <div class="cbz-result-info">
                        <div class="name">#${escapeHtml(issue.issue_number || '?')}${issue.name ? ' — ' + escapeHtml(issue.name) : ''}</div>
                        <div class="detail">${issue.cover_date || 'No date'}</div>
                    </div>
                `;
                item.addEventListener('click', () => selectIssue(issue.id));
                cbzDom.issueResults.appendChild(item);
            });
        }
    } catch (e) {
        cbzDom.issueResults.innerHTML = `<p class="cbz-hint" style="color:#e94560">${e.message}</p>`;
    } finally {
        cbzDom.btnFindIssue.disabled = false;
        cbzDom.btnFindIssue.textContent = 'Find';
    }
});

// Use ComicVine URL to extract issue ID
cbzDom.btnUseUrl.addEventListener('click', async () => {
    const url = cbzDom.issueUrlInput.value.trim();
    if (!url) return;
    // Extract issue ID from URL like https://comicvine.gamespot.com/star-trek.../4000-12345/
    const match = url.match(/4000-(\d+)/);
    if (match) {
        await selectIssue(parseInt(match[1]));
    } else {
        alert('Could not extract issue ID from URL. Expected format: .../4000-XXXXX/');
    }
});

// Select an issue and fetch full metadata
async function selectIssue(issueId) {
    try {
        const detail = await apiPost('/api/comicvine/issue-detail', { issue_id: issueId });
        cbzState.selectedIssue = detail;

        // Populate metadata fields
        document.getElementById('meta-series').value = detail.series || (cbzState.selectedVolume ? cbzState.selectedVolume.name : '');
        document.getElementById('meta-title').value = detail.name || '';
        document.getElementById('meta-number').value = detail.issue_number || '';
        document.getElementById('meta-year').value = detail.year || '';
        document.getElementById('meta-month').value = detail.month || '';
        document.getElementById('meta-publisher').value = cbzState.selectedVolume ? (cbzState.selectedVolume.publisher || '') : '';
        document.getElementById('meta-writer').value = detail.writer || '';
        document.getElementById('meta-penciller').value = detail.penciller || '';
        document.getElementById('meta-inker').value = detail.inker || '';
        document.getElementById('meta-colorist').value = detail.colorist || '';
        document.getElementById('meta-editor').value = detail.editor || '';
        document.getElementById('meta-characters').value = detail.characters || '';
        document.getElementById('meta-summary').value = detail.description || '';

        cbzDom.stepIssue.classList.add('hidden');
        cbzDom.stepMetadata.classList.remove('hidden');
    } catch (e) {
        alert('Failed to fetch issue details: ' + e.message);
    }
}

// Back to issue selection
cbzDom.btnBackIssue.addEventListener('click', () => {
    cbzDom.stepMetadata.classList.add('hidden');
    cbzDom.stepIssue.classList.remove('hidden');
});

// Create CBZ
cbzDom.btnCreateGo.addEventListener('click', async () => {
    const outputDir = dom.outputDir.value.trim();
    if (!outputDir) {
        alert('Please enter an output directory.');
        return;
    }

    const metadata = {
        series: document.getElementById('meta-series').value,
        title: document.getElementById('meta-title').value,
        number: document.getElementById('meta-number').value,
        year: document.getElementById('meta-year').value,
        month: document.getElementById('meta-month').value,
        publisher: document.getElementById('meta-publisher').value,
        writer: document.getElementById('meta-writer').value,
        penciller: document.getElementById('meta-penciller').value,
        inker: document.getElementById('meta-inker').value,
        colorist: document.getElementById('meta-colorist').value,
        editor: document.getElementById('meta-editor').value,
        characters: document.getElementById('meta-characters').value,
        summary: document.getElementById('meta-summary').value,
    };

    cbzDom.btnCreateGo.disabled = true;
    cbzDom.btnCreateGo.textContent = 'Creating...';

    try {
        const result = await apiPost(`/api/session/${state.sessionId}/create-cbz`, {
            output_dir: outputDir,
            metadata: metadata,
        });
        alert(`CBZ created!\n${result.cbz_path}\n${result.size_mb} MB, ${result.pages} pages`);
        cbzDom.modal.classList.add('hidden');
    } catch (e) {
        alert('Failed to create CBZ: ' + e.message);
    } finally {
        cbzDom.btnCreateGo.disabled = false;
        cbzDom.btnCreateGo.textContent = 'Create CBZ';
    }
});

// Cancel / Close
cbzDom.btnCancel.addEventListener('click', () => {
    cbzDom.modal.classList.add('hidden');
});
cbzDom.btnClose.addEventListener('click', () => {
    cbzDom.modal.classList.add('hidden');
});

// ===== Theme Switching =====

const themeSelect = document.getElementById('theme-select');

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('comicscans-theme', theme);
    themeSelect.value = theme;
}

// Restore saved theme
const savedTheme = localStorage.getItem('comicscans-theme') || 'dark';
setTheme(savedTheme);

themeSelect.addEventListener('change', () => {
    setTheme(themeSelect.value);
});

// ===== File Picker =====

const fpDom = {
    modal: document.getElementById('filepicker-modal'),
    title: document.getElementById('fp-title'),
    pathInput: document.getElementById('fp-path-input'),
    entries: document.getElementById('fp-entries'),
    btnUp: document.getElementById('btn-fp-up'),
    btnGo: document.getElementById('btn-fp-go'),
    btnSelect: document.getElementById('btn-fp-select'),
    btnClose: document.getElementById('btn-fp-close'),
};

let fpState = {
    currentPath: '',
    targetInput: null,  // which input element to set the result on
};

/** Open the file picker for a given input element. */
function openFilePicker(targetInput, title) {
    fpState.targetInput = targetInput;
    fpDom.title.textContent = title || 'Select Directory';
    const startPath = targetInput.value.trim() || '';
    fpDom.modal.classList.remove('hidden');
    browseDir(startPath);
}

/** Browse a directory and render entries. */
async function browseDir(path) {
    try {
        const data = await apiPost('/api/browse', { path: path || '' });
        fpState.currentPath = data.current;
        fpDom.pathInput.value = data.current;
        fpDom.entries.innerHTML = '';

        // Parent directory entry
        if (data.parent) {
            const el = document.createElement('div');
            el.className = 'fp-entry is-dir';
            el.innerHTML = '<span class="fp-entry-icon">\u2191</span><span class="fp-entry-name">..</span>';
            el.addEventListener('click', () => browseDir(data.parent));
            fpDom.entries.appendChild(el);
        }

        for (const entry of data.entries) {
            const el = document.createElement('div');
            el.className = 'fp-entry' + (entry.is_dir ? ' is-dir' : '');
            const icon = entry.is_dir ? '\uD83D\uDCC1' : '\uD83D\uDCC4';
            el.innerHTML = `<span class="fp-entry-icon">${icon}</span><span class="fp-entry-name">${escapeHtml(entry.name)}</span>`;
            if (entry.is_dir) {
                el.addEventListener('click', () => browseDir(entry.path));
            }
            fpDom.entries.appendChild(el);
        }
    } catch (e) {
        fpDom.entries.innerHTML = `<p class="hint" style="padding:12px;color:var(--accent)">${e.message}</p>`;
    }
}

// File picker events
document.getElementById('btn-browse-input').addEventListener('click', () => {
    openFilePicker(dom.inputDir, 'Select Scan Directory');
});

document.getElementById('btn-browse-output').addEventListener('click', () => {
    openFilePicker(dom.outputDir, 'Select Output Directory');
});

fpDom.btnUp.addEventListener('click', () => {
    const parts = fpState.currentPath.split('/');
    if (parts.length > 1) {
        parts.pop();
        browseDir(parts.join('/') || '/');
    }
});

fpDom.btnGo.addEventListener('click', () => {
    browseDir(fpDom.pathInput.value.trim());
});

fpDom.pathInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') fpDom.btnGo.click();
});

fpDom.btnSelect.addEventListener('click', () => {
    if (fpState.targetInput) {
        fpState.targetInput.value = fpState.currentPath;
    }
    fpDom.modal.classList.add('hidden');
});

fpDom.btnClose.addEventListener('click', () => {
    fpDom.modal.classList.add('hidden');
});

// ===== Initialization =====

console.log('ComicScans frontend loaded.');
