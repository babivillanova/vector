import cv2
import numpy as np
import argparse
from pathlib import Path
import math


# -----------------------------
# Preprocessing
# -----------------------------
def binarize_and_clean(
    gray: np.ndarray,
    blur_ksize: int = 5,
    close_ksize: int = 3,
    close_iter: int = 1,
    open_ksize: int = 3,
    open_iter: int = 0,
    min_component_area: int = 250,
) -> np.ndarray:
    """
    Returns a binary image (uint8 0/255) where walls/ink are WHITE (255) on black background.
    Good for thick-line floorplans with anti-aliasing and small speckle noise.
    """
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Otsu threshold (invert so dark ink becomes white)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close: connect tiny gaps in strokes
    if close_iter > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=close_iter)

    # Optional open: remove tiny protrusions (use carefully)
    if open_iter > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=open_iter)

    # Remove small connected components (speckles)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    cleaned = np.zeros_like(bw)
    # stats[0] is background
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == i] = 255

    return cleaned


# -----------------------------
# Geometry simplification helpers
# -----------------------------
def remove_near_collinear(points: np.ndarray, angle_eps_deg: float = 5.0) -> np.ndarray:
    """
    Removes points that are nearly collinear to reduce tiny zig-zags after contour approximation.
    points: (N,2) float or int. closed polygon expected.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts

    def angle(a, b, c):
        ab = b - a
        cb = b - c
        nab = np.linalg.norm(ab) + 1e-9
        ncb = np.linalg.norm(cb) + 1e-9
        cosang = float(np.dot(ab, cb) / (nab * ncb))
        cosang = np.clip(cosang, -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    keep = [pts[0]]
    for i in range(1, len(pts) - 1):
        a = keep[-1]
        b = pts[i]
        c = pts[i + 1]
        ang = angle(a, b, c)
        # if angle is close to 180, b is redundant
        if abs(180.0 - ang) < angle_eps_deg:
            continue
        keep.append(b)
    keep.append(pts[-1])
    return np.array(keep, dtype=float)


def orthogonal_snap(points: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    """
    Lightly snaps segments to vertical/horizontal if clearly axis-aligned.
    IMPORTANT: does not mutate input and uses snapped chain to avoid drift surprises.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts

    snapped = [pts[0].copy()]
    for i in range(1, len(pts)):
        p1 = snapped[-1]
        p2 = pts[i].copy()
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        # Mostly vertical
        if dx < dy * ratio:
            p2[0] = p1[0]
        # Mostly horizontal
        elif dy < dx * ratio:
            p2[1] = p1[1]

        snapped.append(p2)

    return np.array(snapped, dtype=float)


# -----------------------------
# SVG export
# -----------------------------
def contours_to_svg(
    contours,
    hierarchy,
    width: int,
    height: int,
    fill: str = "black",
    stroke: str = "none",
    stroke_width: float = 1.0,
    evenodd: bool = True,
) -> str:
    """
    Converts contours into SVG paths. If you keep hierarchy and want holes, use evenodd fill-rule.
    """
    # Note: SVG y-axis points down by default, same as image coordinates.
    fill_rule = "evenodd" if evenodd else "nonzero"
    path_elems = []

    for cnt in contours:
        pts = cnt[:, 0, :]
        if len(pts) < 3:
            continue
        d = [f"M {pts[0,0]:.1f} {pts[0,1]:.1f}"]
        for p in pts[1:]:
            d.append(f"L {p[0]:.1f} {p[1]:.1f}")
        d.append("Z")
        path_elems.append(f'<path d="{" ".join(d)}" />')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 {width} {height}"
  width="{width}" height="{height}">
  <g fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" fill-rule="{fill_rule}">
    {''.join(path_elems)}
  </g>
</svg>"""
    return svg


# -----------------------------
# Main vectorization
# -----------------------------
def vectorize_floorplan_contours(
    image_path: str,
    mode: str = "external",
    approx_mode: str = "simple",
    min_contour_area: float = 500.0,
    epsilon_ratio: float = 0.0015,
    snap_ratio: float = 0.08,
    collinear_angle_eps: float = 5.0,
) -> tuple[str, np.ndarray]:
    """
    mode:
      - "external": single outer silhouette(s) (no holes)
      - "ccomp": includes holes/children contours; export with evenodd fill-rule
    approx_mode:
      - "none": CHAIN_APPROX_NONE (big)
      - "simple": CHAIN_APPROX_SIMPLE (recommended)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = gray.shape
    bw = binarize_and_clean(gray)

    retrieval = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_CCOMP
    chain = cv2.CHAIN_APPROX_SIMPLE if approx_mode == "simple" else cv2.CHAIN_APPROX_NONE

    contours, hierarchy = cv2.findContours(bw, retrieval, chain)

    processed = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = epsilon_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        pts = approx[:, 0, :]  # (N,2)
        pts = remove_near_collinear(pts, angle_eps_deg=collinear_angle_eps)
        pts = orthogonal_snap(pts, ratio=snap_ratio)

        processed.append(pts.reshape(-1, 1, 2).astype(np.float32))

    svg = contours_to_svg(
        processed,
        hierarchy,
        width=w,
        height=h,
        fill="black",
        stroke="none",
        stroke_width=1.0,
        evenodd=(mode != "external"),
    )

    return svg, bw


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Contour-based floorplan vectorization to SVG.")
    p.add_argument("image", help="Input image (png/jpg).")
    p.add_argument("--out", default=None, help="Output SVG path.")
    p.add_argument("--mode", choices=["external", "ccomp"], default="external",
                   help="external = outer silhouettes only; ccomp = include holes/children contours.")
    p.add_argument("--min-area", type=float, default=500.0, help="Min contour area to keep.")
    p.add_argument("--epsilon-ratio", type=float, default=0.0015,
                   help="RDP epsilon ratio of perimeter (smaller = more detail).")
    p.add_argument("--snap-ratio", type=float, default=0.08,
                   help="Orthogonal snap aggressiveness (smaller = less snapping).")
    p.add_argument("--collinear-deg", type=float, default=5.0,
                   help="Drop near-collinear points within this angle threshold.")
    p.add_argument("--write-mask", action="store_true",
                   help="Also writes the cleaned binary mask next to SVG.")
    args = p.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    if args.out is None:
        out_path = image_path.with_suffix(".svg")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    svg, mask = vectorize_floorplan_contours(
        str(image_path),
        mode=args.mode,
        min_contour_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
        snap_ratio=args.snap_ratio,
        collinear_angle_eps=args.collinear_deg,
    )

    out_path.write_text(svg, encoding="utf-8")

    if args.write_mask:
        mask_path = out_path.with_suffix(".mask.png")
        cv2.imwrite(str(mask_path), mask)

    print(f"Wrote: {out_path}")
    if args.write_mask:
        print(f"Wrote: {mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
