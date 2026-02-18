#!/usr/bin/env python3
"""
Fresh approach to architectural floor plan PNG → SVG conversion.

Pipeline:
  1. Load & pre-process (grayscale, threshold, denoise)
  2. Generate clean BMP for potrace
  3. Run potrace with optimized settings for architectural drawings
  4. Post-process SVG (clean viewBox, proper styling)
"""

import subprocess
import sys
import os
from pathlib import Path

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


def preprocess(img_path: Path) -> np.ndarray:
    """
    Load image and produce a clean binary (black lines on white bg).

    Two-pass approach:
      Pass 1: Otsu for strong features (walls, thick lines)
      Pass 2: Higher threshold to catch thin lines (door arcs, fixtures)
      Merge: Union of both passes → preserves everything
    """
    # Load
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Cannot read {img_path}")

    h, w = img.shape
    print(f"  Input: {w}x{h} grayscale")

    # 1. Bilateral filter — smooth JPEG noise but preserve edges
    blurred = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. Global threshold — tuned to preserve thin anti-aliased lines
    #    Otsu finds ~134 for this image, but that kills lighter gray pixels.
    #    Using 200 captures thin fixture lines, door arcs, etc.
    thresh_val = 200
    _, binary_merged = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    print(f"  Threshold: {thresh_val} (captures anti-aliased edges)")

    # 5. Morphological close — reconnect tiny gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_merged = cv2.morphologyEx(binary_merged, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 6. Remove small noise specks via connected components
    inverted = cv2.bitwise_not(binary_merged)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    min_area = 25  # slightly lower to keep small fixture details
    removed = 0
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            inverted[labels == i] = 0
            removed += 1
    binary_clean = cv2.bitwise_not(inverted)
    print(f"  Removed {removed} noise specks (< {min_area}px)")

    return binary_clean


def save_bmp(binary: np.ndarray, out_path: Path):
    """Save as 1-bit BMP for potrace (potrace reads PBM/BMP natively)."""
    # Potrace expects: black = foreground (ink), white = background
    # Our binary: 0=black lines, 255=white bg — perfect for BMP
    img = Image.fromarray(binary)
    img = img.convert("1")  # 1-bit
    img.save(str(out_path))
    print(f"  BMP saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def run_potrace(bmp_path: Path, svg_path: Path, width: int, height: int):
    """
    Run potrace with settings tuned for architectural floor plans.

    Key parameters:
      -s          : SVG output
      -t N        : suppress speckles of up to N pixels (turdsize)
      -a N        : corner threshold (0=sharp, 1.334=default smooth)
      -O N        : optimization tolerance (higher=simpler paths)

    NOTE: We do NOT use --tight or -W/-H here. Potrace uses the BMP page
    dimensions directly, which preserves the original whitespace and
    positioning of the floor plan within the image.
    """
    cmd = [
        "potrace",
        str(bmp_path),
        "-s",                  # SVG output
        "-o", str(svg_path),
        "-t", "10",            # Remove specks < 10px
        "-a", "0.55",          # Moderate alphamax — sharp corners but smooth arcs
        "-O", "0.2",           # Tight optimization tolerance
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  potrace error: {result.stderr}")
        sys.exit(1)
    print(f"  SVG saved: {svg_path} ({svg_path.stat().st_size / 1024:.0f} KB)")


def postprocess_svg(svg_path: Path, final_path: Path, width: int, height: int):
    """
    Post-process the potrace SVG into a clean, architectural-standard SVG:
      - Proper viewBox matching original pixel dimensions
      - Semantic layer grouping
      - Clean fill/stroke styling
      - Metadata and title
    """
    import re

    with open(svg_path, "r") as f:
        svg_content = f.read()

    # Extract the viewBox from potrace output
    vb_match = re.search(r'viewBox="([^"]+)"', svg_content)
    viewbox = vb_match.group(1) if vb_match else f"0 0 {width} {height}"

    # Extract the g transform and path data
    g_match = re.search(r'(<g\s+transform="[^"]*"[^>]*>)(.*?)(</g>)', svg_content, re.DOTALL)
    if g_match:
        g_open = g_match.group(1)
        paths_content = g_match.group(2).strip()
        g_close = g_match.group(3)
    else:
        g_open = f'<g fill="#000000" stroke="none">'
        inner_match = re.search(r'<svg[^>]*>(.*?)</svg>', svg_content, re.DOTALL)
        paths_content = inner_match.group(1).strip() if inner_match else ""
        g_close = '</g>'

    # Remove old metadata from paths
    paths_content = re.sub(r'<metadata>.*?</metadata>\s*', '', paths_content, flags=re.DOTALL)

    # Build clean architectural SVG
    clean_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="{viewbox}"
     width="{width}" height="{height}"
     preserveAspectRatio="xMidYMid meet">

  <title>BIMIT Floor Plan - Model X</title>
  <desc>Architectural floor plan vectorized from raster source</desc>

  <defs>
    <style type="text/css">
      .floor-plan {{ fill: #1a1a1a; stroke: none; }}
      .background {{ fill: #ffffff; }}
    </style>
  </defs>

  <!-- Background -->
  <rect class="background" width="100%" height="100%"/>

  <!-- Floor Plan Geometry -->
  {g_open.replace('fill="#000000"', 'class="floor-plan"')}
{paths_content}
  {g_close}

</svg>'''

    with open(final_path, "w") as f:
        f.write(clean_svg)

    print(f"  Final SVG: {final_path} ({final_path.stat().st_size / 1024:.0f} KB)")


def main():
    print("=" * 60)
    print("ARCHITECTURAL FLOOR PLAN VECTORIZER")
    print("=" * 60)

    # Step 1: Pre-process
    print("\n[1/4] Pre-processing image...")
    binary = preprocess(INPUT_PATH)
    h, w = binary.shape

    # Save intermediate for inspection
    debug_path = OUTPUT_DIR / "01_preprocessed.png"
    cv2.imwrite(str(debug_path), binary)
    print(f"  Debug image: {debug_path}")

    # Step 2: Save as BMP for potrace
    print("\n[2/4] Converting to BMP...")
    bmp_path = OUTPUT_DIR / "02_input.bmp"
    save_bmp(binary, bmp_path)

    # Step 3: Run potrace
    print("\n[3/4] Running potrace vectorization...")
    raw_svg = OUTPUT_DIR / "03_potrace_raw.svg"
    run_potrace(bmp_path, raw_svg, w, h)

    # Step 4: Post-process
    print("\n[4/4] Post-processing SVG...")
    final_svg = OUTPUT_DIR / "BIMIT_Plan_Model_X.svg"
    postprocess_svg(raw_svg, final_svg, w, h)

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Input:  {INPUT_PATH}")
    print(f"  Output: {final_svg}")
    print("=" * 60)


if __name__ == "__main__":
    main()
