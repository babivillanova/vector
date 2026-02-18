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
        "color": "#1a1a1a",
        "debug_bgr": (50, 50, 50),
    },
    "doors": {
        "prompt": (
            "Generate a new image from this architectural floor plan showing "
            "ONLY the doors — the ones with arc openings, panels, and slide ones. "
        ),
        "alphamax": 1.0,
        "turdsize": 5,
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
        "color": "#84cc16",
        "debug_bgr": (22, 204, 132),
    },
}

LAYER_ORDER = ["walls", "doors", "windows", "stairs", "furniture"]


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
# Preprocess Gemini layer image into clean binary mask
# ──────────────────────────────────────────────
def preprocess_layer_image(
    img_path: Path,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Load a Gemini-generated layer image, resize to match original dimensions,
    then clean it up using the same preprocessing pipeline as fresh_vectorize.py:
      1. Resize to target dimensions (before any processing, to preserve thin lines)
      2. Bilateral filter (smooth noise, preserve edges)
      3. Global threshold at 200 (captures anti-aliased edges)
      4. Morphological close (reconnect tiny gaps)
      5. Remove small noise specks via connected components

    Returns ink_mask where 255 = ink pixels, at target dimensions.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    h, w = img.shape

    # 0. Resize FIRST — before any processing, so thin lines survive upscale
    if h != target_h or w != target_w:
        print(f"    Resizing {w}x{h} → {target_w}x{target_h}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # 1. Bilateral filter — smooth noise but preserve edges
    blurred = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. Global threshold at 200
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 3. Morphological close — reconnect tiny gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 4. Remove small noise specks
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
    opttolerance: float = 0.2,
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
      .monochrome .layer {{ fill: #1a1a1a; }}
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

    # Step 2: Generate a layer image for each PRESENT element type
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

    # Step 3: Clean each Gemini layer image (bilateral filter → threshold → morph → denoise)
    print(f"\n[3/5] Preprocessing Gemini layer images...")
    masks: Dict[str, np.ndarray] = {}

    for layer_name, img_path in layer_images.items():
        print(f"  [{layer_name}] Cleaning {img_path.name}...")
        ink_mask = preprocess_layer_image(img_path, h, w)
        px = cv2.countNonZero(ink_mask)
        masks[layer_name] = ink_mask
        print(f"    {px} ink pixels")

        # Save cleaned layer mask
        mask_path = OUTPUT_DIR / f"layer_{layer_name}.png"
        cv2.imwrite(str(mask_path), ink_mask)

    # Step 4: Vectorize each layer
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
