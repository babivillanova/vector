import numpy as np
import cv2
from skimage.morphology import skeletonize
from PIL import Image
from pathlib import Path
import argparse

def _get_skeleton_intersections(skel_bool):
    """
    Return a boolean mask of skeleton pixels that have >2 neighbors (8-connectivity).
    This detects branch/junction pixels in the skeleton graph.
    """
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    neighbors = cv2.filter2D(skel_u8, -1, kernel)
    return (skel_bool & (neighbors > 2))


def _fit_arc_path_from_skeleton_points(xs, ys, *, snap_90deg: bool):
    """
    Fit a circle to skeleton point coordinates and return an SVG path string, or None.
    """
    if len(xs) < 3:
        return None

    # Circle Fitting (Least Squares)
    # Solve: (x - cx)^2 + (y - cy)^2 = r^2
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    b = -(xs**2 + ys**2)
    try:
        C, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -C[0] / 2, -C[1] / 2
        r_sq = cx**2 + cy**2 - C[2]
        if r_sq <= 0:
            return None
        r = np.sqrt(r_sq)
    except Exception:
        return None

    # Determine Angles & SVG Path
    angles = np.arctan2(ys - cy, xs - cx)

    # Handle angle wrapping
    base_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    unwrapped = base_angle + ((angles - base_angle + np.pi) % (2 * np.pi) - np.pi)

    start_angle = np.min(unwrapped)
    end_angle = np.max(unwrapped)

    # Enforce 90 degree arcs for precision (Optional but recommended)
    if snap_90deg:
        mid_angle = (start_angle + end_angle) / 2
        start_angle = mid_angle - (np.pi / 4)  # -45 deg
        end_angle = mid_angle + (np.pi / 4)    # +45 deg

    # Calculate start/end points
    x0 = cx + r * np.cos(start_angle)
    y0 = cy + r * np.sin(start_angle)
    x1 = cx + r * np.cos(end_angle)
    y1 = cy + r * np.sin(end_angle)

    # Create SVG Path string
    return f'<path d="M {x0:.3f} {y0:.3f} A {r:.3f} {r:.3f} 0 0 1 {x1:.3f} {y1:.3f}" />'


def analyze_image_and_generate_svg(
    image_path,
    output_svg_path,
    *,
    snap_90deg: bool = True,
    min_component_points: int = 10,
):
    # 1. Load and Preprocess
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    
    # Isolate Red/Dark strokes
    # Rule: Red channel is dominant over Green/Blue, but overall pixel is dark
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    mask = (R > G + 20) & (R > B + 20) & (R < 220)
    
    # 2. Skeletonization (Get the centerlines)
    mask_u8 = (mask * 255).astype(np.uint8)
    # Close small gaps
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    
    # Get connected components
    num_labels, labels = cv2.connectedComponents(mask_u8)
    
    svg_paths = []
    
    for label in range(1, num_labels):
        comp_mask = (labels == label)
        
        # Skeletonize this specific component
        skel = skeletonize(comp_mask)
        branches = _get_skeleton_intersections(skel)

        # If there are branch points, cut them out to split into separate parts
        if np.any(branches):
            cut_mask = cv2.dilate(branches.astype(np.uint8), np.ones((3, 3), np.uint8))
            skel_cut = (skel & ~(cut_mask.astype(bool)))
            num_parts, parts_labels = cv2.connectedComponents(skel_cut.astype(np.uint8))

            for part_id in range(1, num_parts):
                part_mask = (parts_labels == part_id)
                ys, xs = np.where(part_mask)
                if len(xs) < int(min_component_points):
                    continue
                path = _fit_arc_path_from_skeleton_points(xs, ys, snap_90deg=snap_90deg)
                if path is not None:
                    svg_paths.append(path)
        else:
            ys, xs = np.where(skel)
            if len(xs) < int(min_component_points):
                continue
            path = _fit_arc_path_from_skeleton_points(xs, ys, snap_90deg=snap_90deg)
            if path is not None:
                svg_paths.append(path)

    # 6. Generate File
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" fill="none">
    <g stroke="#7e4a45" stroke-width="6" stroke-linecap="round" stroke-linejoin="round">
    {''.join(svg_paths)}
    </g>
    </svg>"""
    
    with open(output_svg_path, "w") as f:
        f.write(svg_content)
    
    return output_svg_path


def _iter_input_images(inputs):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in exts and child.is_file():
                    yield child
        else:
            yield p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Vectorize red/dark quarter-arc strokes into SVG arc paths."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image file(s) and/or directory(ies) containing images.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory where SVGs will be written.",
    )
    parser.add_argument(
        "--no-snap",
        action="store_true",
        help="Do not snap each detected arc to a perfect 90° span.",
    )
    parser.add_argument(
        "--min-component-points",
        type=int,
        default=10,
        help="Minimum skeleton points for a connected component to be considered (noise filter).",
    )

    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote_any = False
    for img_path in _iter_input_images(args.inputs):
        if not img_path.exists():
            raise FileNotFoundError(str(img_path))
        out_svg = out_dir / f"{img_path.stem}.svg"
        analyze_image_and_generate_svg(
            str(img_path),
            str(out_svg),
            snap_90deg=not args.no_snap,
            min_component_points=args.min_component_points,
        )
        wrote_any = True

    if not wrote_any:
        raise RuntimeError("No input images found.")

    return 0

# To use:
# analyze_image_and_generate_svg("your_image.png", "output.svg")


if __name__ == "__main__":
    raise SystemExit(main())