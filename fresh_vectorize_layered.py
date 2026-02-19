#!/usr/bin/env python3
"""
Multi-layer floor plan PNG → SVG using Gemini AI image generation.

Pipeline:
  1. Vision pre-flight: ask Gemini which element types exist in the plan.
  2. For each present layer, send original image to Gemini and ask it to
     generate a new image showing ONLY that layer's elements.
  3. Clean each Gemini layer image (bilateral filter → threshold → morph
     close → noise removal) — same preprocessing as fresh_vectorize.py.
  4. Vectorize each cleaned layer via potrace with layer-specific params.
  5. Compose single layered SVG with <g> groups per layer.
  6. Save debug overlay + individual layer images.
"""

import subprocess
import sys
import os
import re
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_PNG = "BIMIT Plan _ Model X.png"
BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / INPUT_PNG
OUTPUT_DIR = BASE_DIR / "fresh_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load API key
ENV_PATH = BASE_DIR / ".env"
for line in ENV_PATH.read_text().strip().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    sys.exit("GEMINI_API_KEY not set in .env")

GEMINI_MODEL = "nano-banana-pro-preview"

# ──────────────────────────────────────────────
# Layer definitions
# ──────────────────────────────────────────────
LAYERS = {
    "walls": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the walls and structural elements — the thick black lines "
            "that form room boundaries, exterior walls, and interior partitions. "
            "Remove ALL doors, door swings, fixtures, appliances, furniture, "
            "stairs, windows, and any other non-structural elements. "
            "Keep the exact same image dimensions, scale, and positions. "
            "Output as clean black lines on a pure white background."
        ),
        "alphamax": 0.3,
        "turdsize": 10,
        "smooth_kernel": 5,
        "smooth_sigma": 1.0,
        "color": "#1a1a1a",
        "debug_bgr": (50, 50, 50),
    },
    "doors": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the doors — the door swings (quarter-circle arcs), door panels "
            "(straight lines), and sliding door markers. "
            "Keep the exact same image dimensions, scale, and positions. "
            "Output as clean black lines on a pure white background."
        ),
        "alphamax": 1.2,
        "turdsize": 2,
        "smooth_kernel": 3,
        "smooth_sigma": 0.5,
        "color": "#2563eb",
        "debug_bgr": (235, 99, 37),
    },
    "windows": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the windows — the parallel lines within walls that indicate "
        ),
        "alphamax": 0.5,
        "turdsize": 5,
        "smooth_kernel": 7,
        "smooth_sigma": 1.5,
        "color": "#06b6d4",
        "debug_bgr": (212, 182, 6),
    },
    "stairs": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the stairs it exists"
        ),
        "alphamax": 0.5,
        "turdsize": 5,
        "smooth_kernel": 5,
        "smooth_sigma": 1.0,
        "color": "#7c3aed",
        "debug_bgr": (237, 58, 124),
    },
    "furniture": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the furniture on a pure white background, if any exists."
        ),
        "alphamax": 0.8,
        "turdsize": 3,
        "smooth_kernel": 5,
        "smooth_sigma": 1.0,
        "color": "#84cc16",
        "debug_bgr": (22, 204, 132),
    },
}

LAYER_ORDER = ["walls", "doors", "windows", "stairs", "furniture"]

# Hardcoded hinge positions from wall structure analysis.
# Gemini's hinge detection is non-deterministic, so we use known values.
# Key = (x1, y1, x2, y2) bbox tuple, Value = hinge corner (TL/TR/BL/BR)
DOOR_HINGES = {
    (1650, 626, 1722, 698): "TL",  # Door into top-right bathroom
    (1538, 862, 1630, 954): "TL",  # Door into central area
    (814, 982, 906, 1054):  "TR",  # Door into bottom-left room
    (674, 950, 738, 1014):  "TR",  # Door into small room (bottom)
    (674, 862, 738, 930):   "TR",  # Door into small room (middle)
    (674, 790, 738, 862):   "TR",  # Door into small room (top)
    (1086, 514, 1186, 586): "TL",  # Door into main living area
}
# D2 bbox (1650,862,1742,954) is a false positive — wall opening, no swing door


# ──────────────────────────────────────────────
# Gemini vision pre-flight: detect which elements exist
# ──────────────────────────────────────────────
def detect_present_elements(img_bytes: bytes) -> Dict[str, bool]:
    """
    Ask Gemini's vision model to analyze the floor plan and report
    which architectural element types are actually present.
    Returns dict of layer_name → True/False.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    layer_names = ", ".join(LAYER_ORDER)
    prompt = (
        f"Analyze this architectural floor plan image carefully. "
        f"For each of the following element types, answer YES or NO — "
        f"is this type of element present in the drawing?\n\n"
        f"Element types: {layer_names}\n\n"
        f"Reply ONLY with a JSON object mapping each element type to true/false. "
        f"Example: {{\"walls\": true, \"doors\": true, \"windows\": false, "
        f"\"stairs\": false, \"furniture\": true}}\n"
        f"No explanation, just the JSON."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    raw = response.text.strip()
    # Parse JSON (handle markdown fences)
    json_text = raw
    if "```" in json_text:
        match = re.search(r"```(?:json)?\s*(.*?)```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    try:
        result = json.loads(json_text)
        return {k: bool(v) for k, v in result.items()}
    except (json.JSONDecodeError, ValueError):
        print(f"  Warning: Could not parse Gemini response, assuming all present")
        print(f"  Raw: {raw[:200]}")
        return {name: True for name in LAYER_ORDER}


# ──────────────────────────────────────────────
# Gemini image generation per layer
# ──────────────────────────────────────────────
def generate_layer_image(
    img_bytes: bytes,
    layer_name: str,
    prompt: str,
    output_path: Path,
) -> Optional[Path]:
    """
    Send the floor plan to Gemini and ask it to generate a new image
    showing only the specified layer. Saves result and returns path.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            temperature=0.1,
        ),
    )

    # Extract generated image from response
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
            with open(output_path, "wb") as f:
                f.write(part.inline_data.data)
            size_kb = output_path.stat().st_size / 1024
            print(f"  [{layer_name}] Generated image: {output_path} ({size_kb:.0f} KB)")
            return output_path

    print(f"  [{layer_name}] No image in Gemini response")
    return None


# ──────────────────────────────────────────────
# Programmatic door arc generation (CAD-quality SVG arcs)
# ──────────────────────────────────────────────
def detect_door_hinges(img_bytes: bytes, doors: list) -> list:
    """
    Ask Gemini vision to identify the hinge corner for each door bbox.
    Returns list of dicts with hinge_corner (TL/TR/BL/BR) and bbox.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    door_list = "\n".join(
        f"  Door {i+1}: bbox=[{d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]}] — {d['description']}"
        for i, d in enumerate(doors)
    )

    prompt = (
        f"Look at this architectural floor plan. For each door listed below, "
        f"determine which corner of its bounding box is the HINGE (pivot point) "
        f"where the door connects to the wall. The hinge is the corner that does NOT move.\n\n"
        f"{door_list}\n\n"
        f"For each door, reply with the hinge corner as one of: TL (top-left), "
        f"TR (top-right), BL (bottom-left), BR (bottom-right).\n"
        f"Reply ONLY with a JSON array of objects: "
        f'[{{"door": 1, "hinge": "TL"}}, ...]\n'
        f"No explanation, just the JSON."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ]
            )
        ],
        config=types.GenerateContentConfig(temperature=0.0),
    )

    raw = response.text.strip()
    json_text = raw
    if "```" in json_text:
        match = re.search(r"```(?:json)?\s*(.*?)```", json_text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()

    try:
        result = json.loads(json_text)
        # Merge hinge info into door data
        for item in result:
            idx = item["door"] - 1
            if 0 <= idx < len(doors):
                doors[idx]["hinge"] = item.get("hinge", "TL")
    except (json.JSONDecodeError, ValueError, KeyError):
        print(f"  Warning: Could not parse hinge data, defaulting to TL")
        print(f"  Raw: {raw[:200]}")
        for d in doors:
            d.setdefault("hinge", "TL")

    return doors


def generate_door_svg_paths(doors: list, transform: str = "") -> str:
    """
    Generate clean SVG <path> elements for door arcs from bounding box data.
    Each door gets a quarter-circle arc + panel line.

    The hinge corner determines which quadrant the arc sweeps through:
      TL hinge: arc from top-left, sweeps to bottom-right
      TR hinge: arc from top-right, sweeps to bottom-left
      BL hinge: arc from bottom-left, sweeps to top-right
      BR hinge: arc from bottom-right, sweeps to top-left
    """
    paths = []

    for door in doors:
        x1, y1, x2, y2 = door["bbox"]
        hinge = door.get("hinge", "TL")
        w = x2 - x1
        h = y2 - y1

        # Radius = panel length. Use the dimension matching the panel direction
        # so the panel fits exactly to the bbox edge. The arc may extend slightly
        # beyond the bbox in the perpendicular direction — that's acceptable.
        if hinge in ("TL", "BR"):
            r = w   # panel is horizontal
        else:
            r = h   # panel is vertical (TR, BL)

        # Determine hinge point, arc start, arc end, and sweep direction
        if hinge == "TL":
            hx, hy = x1, y1
            sx, sy = x1 + r, y1      # start: right of hinge
            ex, ey = x1, y1 + r      # end: below hinge
            sweep = 1
        elif hinge == "TR":
            hx, hy = x2, y1
            sx, sy = x2, y1 + r      # start: below hinge
            ex, ey = x2 - r, y1      # end: left of hinge
            sweep = 1
        elif hinge == "BL":
            hx, hy = x1, y2
            sx, sy = x1, y2 - r      # start: above hinge
            ex, ey = x1 + r, y2      # end: right of hinge
            sweep = 1
        elif hinge == "BR":
            hx, hy = x2, y2
            sx, sy = x2 - r, y2      # start: left of hinge
            ex, ey = x2, y2 - r      # end: above hinge
            sweep = 1
        else:
            continue

        # SVG arc: M(start) A(rx ry rotation large-arc sweep end)
        arc_path = f"M {sx},{sy} A {r},{r} 0 0 {sweep} {ex},{ey}"
        # Panel line: from hinge to the arc start (the door panel)
        panel_path = f"M {hx},{hy} L {sx},{sy}"

        paths.append(f'      <path d="{panel_path}" />')
        paths.append(f'      <path d="{arc_path}" />')

    return "\n".join(paths)


# ──────────────────────────────────────────────
# Preprocess Gemini layer image into clean binary mask
# ──────────────────────────────────────────────
def preprocess_layer_image(
    img_path: Path,
    target_h: int,
    target_w: int,
    smooth_kernel: int = 5,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Load a Gemini-generated layer image, resize to match original dimensions,
    then clean it up:
      1. Resize to target dimensions (INTER_CUBIC for smooth edges)
      2. Bilateral filter (smooth noise, preserve edges)
      3. Global threshold at 200 (captures anti-aliased edges)
      4. Morphological close (reconnect tiny gaps)
      5. Contour smoothing via Gaussian blur → re-threshold
      6. Remove small noise specks via connected components

    smooth_kernel / smooth_sigma control contour smoothing intensity.
    Larger values = smoother curves (good for doors), smaller = preserve detail.

    Returns ink_mask where 255 = ink pixels, at target dimensions.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    h, w = img.shape

    # 0. Resize FIRST — before any processing, so thin lines survive upscale
    if h != target_h or w != target_w:
        print(f"    Resizing {w}x{h} → {target_w}x{target_h}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    # 1. Bilateral filter — smooth noise but preserve edges
    blurred = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. Global threshold at 200
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 3. Morphological close — reconnect tiny gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 4. Contour smoothing — Gaussian blur then re-threshold.
    #    Rounds out jagged pixel edges from Gemini's generation.
    #    Per-layer kernel/sigma: doors get aggressive smoothing (15, 3.0),
    #    walls get light smoothing (5, 1.0) to keep sharp corners.
    k = smooth_kernel if smooth_kernel % 2 == 1 else smooth_kernel + 1
    smooth = cv2.GaussianBlur(binary, (k, k), smooth_sigma)
    _, binary = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)

    # 5. Remove small noise specks
    inv = cv2.bitwise_not(binary)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    removed = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < 25:
            inv[labels == i] = 0
            removed += 1
    if removed:
        print(f"    Removed {removed} noise specks")

    return inv


# ──────────────────────────────────────────────
# Potrace vectorization
# ──────────────────────────────────────────────
def save_bmp(binary: np.ndarray, out_path: Path):
    """Save as 1-bit BMP for potrace."""
    img = Image.fromarray(binary)
    img = img.convert("1")
    img.save(str(out_path))


def vectorize_layer(
    mask: np.ndarray,
    layer_name: str,
    alphamax: float = 0.55,
    turdsize: int = 10,
    opttolerance: float = 0.4,
) -> Optional[str]:
    """Convert a binary mask (255=ink) to SVG via potrace."""
    if cv2.countNonZero(mask) == 0:
        print(f"  [{layer_name}] Empty mask, skipping")
        return None

    bmp_binary = cv2.bitwise_not(mask)  # potrace: black=foreground

    with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp_bmp:
        bmp_path = Path(tmp_bmp.name)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_svg:
        svg_path = Path(tmp_svg.name)

    try:
        save_bmp(bmp_binary, bmp_path)
        cmd = [
            "potrace", str(bmp_path), "-s", "-o", str(svg_path),
            "-t", str(turdsize), "-a", str(alphamax), "-O", str(opttolerance),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [{layer_name}] potrace error: {result.stderr}")
            return None

        svg_content = svg_path.read_text()
        px = cv2.countNonZero(mask)
        print(f"  [{layer_name}] Vectorized ({px} ink pixels)")
        return svg_content
    finally:
        bmp_path.unlink(missing_ok=True)
        svg_path.unlink(missing_ok=True)


def extract_paths_from_potrace_svg(svg_content: str) -> Tuple[str, str]:
    """Extract transform and path data from potrace SVG output."""
    g_match = re.search(
        r"<g\s+transform=\"([^\"]*)\"\s*[^>]*>(.*?)</g>", svg_content, re.DOTALL
    )
    if g_match:
        transform = g_match.group(1)
        paths = g_match.group(2).strip()
        paths = re.sub(r"<metadata>.*?</metadata>\s*", "", paths, flags=re.DOTALL)
        return transform, paths
    paths = re.findall(r"<path[^/]*/?>", svg_content)
    return "", "\n".join(paths)


# ──────────────────────────────────────────────
# Compose layered SVG
# ──────────────────────────────────────────────
def compose_layered_svg(
    layer_svg_data: Dict[str, Optional[str]],
    width: int,
    height: int,
) -> str:
    """Combine all layer SVGs into a single layered SVG."""
    layer_groups = []
    shared_transform = None

    for layer_name in LAYER_ORDER:
        svg_content = layer_svg_data.get(layer_name)
        if svg_content is None:
            continue

        transform, paths = extract_paths_from_potrace_svg(svg_content)
        if not paths.strip():
            continue

        if shared_transform is None and transform:
            shared_transform = transform

        color = LAYERS[layer_name]["color"]
        layer_groups.append(
            f'    <g id="{layer_name}" class="layer {layer_name}" '
            f'fill="{color}" stroke="none">\n'
            f"      {paths}\n"
            f"    </g>"
        )

    viewbox = f"0 0 {width} {height}"
    for svg_content in layer_svg_data.values():
        if svg_content:
            vb_match = re.search(r'viewBox="([^"]+)"', svg_content)
            if vb_match:
                viewbox = vb_match.group(1)
                break

    transform_open = ""
    transform_close = ""
    if shared_transform:
        transform_open = f'  <g transform="{shared_transform}">\n'
        transform_close = "  </g>\n"

    css_rules = "\n".join(
        f"      .{name} {{ fill: {cfg['color']}; }}"
        for name, cfg in LAYERS.items()
    )
    groups_content = "\n\n".join(layer_groups)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="{viewbox}"
     width="{width}" height="{height}"
     preserveAspectRatio="xMidYMid meet">

  <title>BIMIT Floor Plan - Model X (Layered)</title>
  <desc>Architectural floor plan with AI-generated layers (Gemini {GEMINI_MODEL})</desc>

  <defs>
    <style type="text/css">
      .layer {{ stroke: none; }}
{css_rules}

      /* Add class="monochrome" to SVG root for traditional black view */
      .monochrome .layer {{ fill: #1a1a1a; stroke: none; }}
    </style>
  </defs>

  <!-- Background -->
  <rect class="background" fill="#ffffff" width="100%" height="100%"/>

  <!-- Floor Plan Layers -->
{transform_open}{groups_content}
{transform_close}
</svg>"""


# ──────────────────────────────────────────────
# Debug output
# ──────────────────────────────────────────────
def save_debug_output(
    masks: Dict[str, np.ndarray],
    image_h: int,
    image_w: int,
    output_dir: Path,
):
    """Save color-coded overlay of all layers."""
    overlay = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255

    for layer_name in LAYER_ORDER:
        mask = masks.get(layer_name)
        if mask is None:
            continue
        color = LAYERS[layer_name]["debug_bgr"]
        overlay[mask > 0] = color

    overlay_path = output_dir / "layer_debug_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"  Debug overlay: {overlay_path}")


# ──────────────────────────────────────────────
# Programmatic door enhancements
# ──────────────────────────────────────────────
def _load_door_elements() -> list:
    """Load door bounding boxes from gemini_elements.json."""
    elements_path = OUTPUT_DIR / "gemini_elements.json"
    if not elements_path.exists():
        return []
    with open(elements_path) as f:
        elements = json.load(f)
    return [e for e in elements if e["label"] == "door"]


def _generate_door_enhancements(
    door_elements: list,
    door_mask: Optional[np.ndarray],
) -> str:
    """
    Generate ALL door arcs + panel lines as clean SVG strokes.

    Potrace creates thick filled crescents from raster door pixels, which
    don't match the original's thin arc lines. Instead, we use mathematical
    quarter-circle arcs from known bounding boxes and hinge positions.

    Each door gets:
      - Panel line: straight stroke from hinge to arc start
      - Quarter-circle arc: SVG arc command with correct radius and sweep

    All paths use pixel coordinates (matching viewBox) with stroke rendering.
    """
    stroke_paths = []
    door_color = LAYERS["doors"]["color"]

    for door in door_elements:
        bbox = tuple(door["bbox"])
        hinge = DOOR_HINGES.get(bbox)
        if hinge is None:
            continue

        x1, y1, x2, y2 = bbox
        w_d = x2 - x1
        h_d = y2 - y1

        # Compute hinge point, arc geometry
        if hinge == "TL":
            hx, hy = x1, y1
            r = w_d
            sx, sy = x1 + r, y1       # arc start (right of hinge)
            ex, ey = x1, y1 + r       # arc end (below hinge)
            sweep = 1
        elif hinge == "TR":
            hx, hy = x2, y1
            r = h_d
            sx, sy = x2, y1 + r       # arc start (below hinge)
            ex, ey = x2 - r, y1       # arc end (left of hinge)
            sweep = 1
        elif hinge == "BL":
            hx, hy = x1, y2
            r = h_d
            sx, sy = x1, y2 - r       # arc start (above hinge)
            ex, ey = x1 + r, y2       # arc end (right of hinge)
            sweep = 1
        elif hinge == "BR":
            hx, hy = x2, y2
            r = w_d
            sx, sy = x2 - r, y2       # arc start (left of hinge)
            ex, ey = x2, y2 - r       # arc end (above hinge)
            sweep = 1
        else:
            continue

        # Panel line (hinge → arc start)
        stroke_paths.append(
            f'    <path d="M {hx},{hy} L {sx},{sy}" />'
        )
        # Quarter-circle arc
        stroke_paths.append(
            f'    <path d="M {sx},{sy} A {r},{r} 0 0 {sweep} {ex},{ey}" />'
        )

    if not stroke_paths:
        return ""

    paths_str = "\n".join(stroke_paths)
    return (
        f'  <!-- Programmatic door arcs + panel lines (stroke-based) -->\n'
        f'  <g id="doors-programmatic"\n'
        f'     style="fill:none; stroke:{door_color}; stroke-width:3; stroke-linecap:round">\n'
        f'{paths_str}\n'
        f'  </g>'
    )


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("MULTI-LAYER FLOOR PLAN VECTORIZER")
    print(f"  Gemini {GEMINI_MODEL} image generation")
    print("=" * 60)

    # Load original image just to get dimensions
    orig = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)
    if orig is None:
        sys.exit(f"Cannot read {INPUT_PATH}")
    h, w = orig.shape
    print(f"\n  Input: {w}x{h} — {INPUT_PATH.name}")

    img_bytes = INPUT_PATH.read_bytes()

    # Step 1: Vision pre-flight — detect which elements exist
    print(f"\n[1/5] Asking Gemini what elements are present...")
    present = detect_present_elements(img_bytes)
    for name in LAYER_ORDER:
        status = "YES" if present.get(name, False) else "NO"
        print(f"  {name:12s}: {status}")

    # Step 2: Generate layer images via Gemini
    print(f"\n[2/5] Generating layer images via Gemini {GEMINI_MODEL}...")
    layer_images: Dict[str, Path] = {}

    for layer_name in LAYER_ORDER:
        if not present.get(layer_name, False):
            print(f"\n  --- {layer_name.upper()} --- SKIPPED (not detected)")
            continue

        cfg = LAYERS[layer_name]
        out_img = OUTPUT_DIR / f"gemini_{layer_name}.png"

        # Skip regeneration if cached image exists
        if out_img.exists() and out_img.stat().st_size > 1000:
            print(f"\n  --- {layer_name.upper()} --- (cached)")
            layer_images[layer_name] = out_img
            continue

        print(f"\n  --- {layer_name.upper()} ---")
        result = generate_layer_image(img_bytes, layer_name, cfg["prompt"], out_img)
        if result:
            layer_images[layer_name] = result

    # Step 3: Clean each Gemini layer image
    print(f"\n[3/5] Preprocessing layer images...")
    masks: Dict[str, np.ndarray] = {}

    for layer_name, img_path in layer_images.items():
        cfg = LAYERS[layer_name]
        print(f"  [{layer_name}] Cleaning {img_path.name} (smooth={cfg['smooth_kernel']}, σ={cfg['smooth_sigma']})...")
        ink_mask = preprocess_layer_image(
            img_path, h, w,
            smooth_kernel=cfg["smooth_kernel"],
            smooth_sigma=cfg["smooth_sigma"],
        )
        px = cv2.countNonZero(ink_mask)
        masks[layer_name] = ink_mask
        print(f"    {px} ink pixels")

        # Save cleaned layer mask
        mask_path = OUTPUT_DIR / f"layer_{layer_name}.png"
        cv2.imwrite(str(mask_path), ink_mask)

    # Step 3b: Clean door mask — subtract wall duplicates + remove stair artifacts.
    if "doors" in masks and "walls" in masks:
        print(f"  [doors] Subtracting wall bleed...")
        wall_mask = masks["walls"]
        door_mask = masks["doors"]
        before_px = cv2.countNonZero(door_mask)

        # Dilate walls slightly to cover anti-aliased edges
        wall_dilated = cv2.dilate(
            wall_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        door_clean = cv2.bitwise_and(door_mask, cv2.bitwise_not(wall_dilated))

        after_px = cv2.countNonZero(door_clean)
        removed_pct = (1 - after_px / max(before_px, 1)) * 100
        print(f"    Before: {before_px} → After: {after_px} ({removed_pct:.0f}% wall bleed removed)")

        # Filter out stair zigzag artifacts by shape analysis.
        # Real door arcs: solidity ~0.17-0.20, aspect ~1.0 (roughly square bbox)
        # Stair zigzags: solidity >0.30, aspect >2.0 (tall/narrow, densely filled)
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            door_clean, connectivity=8
        )
        removed_artifacts = 0
        for i in range(1, n_cc):
            area = cc_stats[i, cv2.CC_STAT_AREA]
            w_cc = cc_stats[i, cv2.CC_STAT_WIDTH]
            h_cc = cc_stats[i, cv2.CC_STAT_HEIGHT]
            # Solidity = area / convex hull area
            comp = (cc_labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(cnts[0]))
            solidity = area / max(hull_area, 1)
            aspect = max(w_cc, h_cc) / max(min(w_cc, h_cc), 1)
            if solidity > 0.30 and aspect > 2.0:
                door_clean[cc_labels == i] = 0
                removed_artifacts += 1
                print(f"    Removed stair artifact: CC{i} area={area} solid={solidity:.2f} aspect={aspect:.1f}")
        if removed_artifacts:
            print(f"    Removed {removed_artifacts} stair zigzag artifact(s)")

        final_px = cv2.countNonZero(door_clean)
        print(f"    Final door pixels: {final_px}")

        # Skeletonize to 1px centerlines, then dilate to a visible width.
        # This converts thick Gemini blobs into clean thin arcs that potrace
        # traces as slender filled shapes (matching original line style).
        from skimage.morphology import skeletonize as ski_skeletonize
        skeleton = ski_skeletonize(door_clean > 0).astype(np.uint8) * 255
        skel_px = cv2.countNonZero(skeleton)
        # Dilate with elliptical kernel for smooth, visible arcs
        skel_dilated = cv2.dilate(
            skeleton,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        dilated_px = cv2.countNonZero(skel_dilated)
        print(f"    Skeletonized: {skel_px} → dilated 5×5: {dilated_px} px")
        masks["doors"] = skel_dilated

        mask_path = OUTPUT_DIR / "layer_doors.png"
        cv2.imwrite(str(mask_path), skel_dilated)

    # Step 4: Vectorize all layers with potrace
    print(f"\n[4/5] Vectorizing layers with potrace...")
    layer_svgs: Dict[str, Optional[str]] = {}
    for layer_name in LAYER_ORDER:
        mask = masks.get(layer_name)
        if mask is None or cv2.countNonZero(mask) == 0:
            layer_svgs[layer_name] = None
            continue
        cfg = LAYERS[layer_name]
        layer_svgs[layer_name] = vectorize_layer(
            mask, layer_name,
            alphamax=cfg["alphamax"],
            turdsize=cfg["turdsize"],
        )

    # Step 5: Compose layered SVG
    print(f"\n[5/5] Composing layered SVG...")
    layered_svg = compose_layered_svg(layer_svgs, w, h)

    output_svg = OUTPUT_DIR / "BIMIT_Plan_Model_X_layered.svg"
    with open(output_svg, "w") as f:
        f.write(layered_svg)
    print(f"  Layered SVG: {output_svg} ({output_svg.stat().st_size / 1024:.0f} KB)")

    # Debug overlay
    save_debug_output(masks, h, w, OUTPUT_DIR)

    active = [n for n in LAYER_ORDER if layer_svgs.get(n) is not None]
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {output_svg}")
    print(f"  Layers: {', '.join(active)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
