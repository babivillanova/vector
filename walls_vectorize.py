#!/usr/bin/env python3
"""
Hybrid v2 floorplan vectorizer:
- Smooth long edges (anti-aliased upsample -> re-threshold)
- Crisp corners (vector-space orthogonalization + exact intersections)
- Cleans weird "stub ends" / burrs on small pieces (hi-res rect deburr + vector notch removal)
- Proper holes (compound paths + evenodd)

Typical usage:
  python vectorize_hybrid_v2.py input.png --out out.svg --write-debug --ortho

Good starting params (for your provided walls.png):
  --scale 5 --aa-sigma 0.65 --aa-threshold 0.50 --epsilon-ratio 0.00022 --ortho
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# Raster preprocessing
# -----------------------------
def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Binary where ink/walls are WHITE (255) on black background."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def remove_small_components(bw: np.ndarray, min_area: int = 20) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def rect_morph(
    bw: np.ndarray,
    close_k: int = 3,
    close_iter: int = 1,
    open_k: int = 0,
    open_iter: int = 0,
) -> np.ndarray:
    """
    RECT kernels preserve crisp 90° corners.
    (Ellipse kernels will round corners.)
    """
    out = bw
    if close_k > 0 and close_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iter)
    if open_k > 0 and open_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iter)
    return out


def estimate_stroke_width(bw: np.ndarray) -> float:
    """
    Estimate typical stroke thickness in pixels (at native resolution).
    Uses distance transform median * 2.
    """
    fg = (bw > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 8.0
    return float(np.median(vals) * 2.0)


def aa_upsample_rethreshold(
    bw: np.ndarray,
    scale: int = 4,
    aa_sigma: float = 0.6,
    aa_threshold: float = 0.5,
) -> np.ndarray:
    """
    Smooth edges without rounding corners:
    - upscale with INTER_LINEAR (removes pixel stair-steps)
    - tiny blur at high-res
    - re-threshold back to clean binary
    """
    h, w = bw.shape
    f = bw.astype(np.float32) / 255.0
    f_big = cv2.resize(f, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    if aa_sigma and aa_sigma > 0:
        f_big = cv2.GaussianBlur(f_big, (0, 0), sigmaX=float(aa_sigma), sigmaY=float(aa_sigma))
    bw_big = ((f_big > float(aa_threshold)) * 255).astype(np.uint8)
    return bw_big


# -----------------------------
# Geometry cleanup helpers
# -----------------------------
def remove_near_collinear(points: np.ndarray, eps_deg: float = 1.0) -> np.ndarray:
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
        a, b, c = keep[-1], pts[i], pts[i + 1]
        if abs(180.0 - angle(a, b, c)) < eps_deg:
            continue
        keep.append(b)
    keep.append(pts[-1])
    return np.array(keep, dtype=float)


def prune_spikes(points: np.ndarray, min_len: float, acute_deg: float = 35.0) -> np.ndarray:
    """
    Removes "needle" vertices:
    - adjacent segment too short
    - or very acute angle
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts

    # closed for processing
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    changed = True
    while changed and len(pts) > 4:
        changed = False
        keep = [pts[0]]
        for i in range(1, len(pts) - 1):
            a = keep[-1]
            b = pts[i]
            c = pts[i + 1]

            ab = b - a
            bc = c - b
            lab = np.linalg.norm(ab)
            lbc = np.linalg.norm(bc)

            if lab < min_len or lbc < min_len:
                changed = True
                continue

            cosang = np.dot(ab, -bc) / (lab * lbc + 1e-9)
            cosang = np.clip(cosang, -1.0, 1.0)
            ang = math.degrees(math.acos(cosang))
            if ang < acute_deg:
                changed = True
                continue

            keep.append(b)

        keep.append(pts[-1])
        pts = np.array(keep, dtype=float)

    # unclose
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def orthogonalize_poly(points: np.ndarray, angle_thr_deg: float = 10.0, min_seg: float = 6.0) -> np.ndarray:
    """
    Manhattanize near-axis segments and rebuild corners as exact intersections.
    Free/curvy segments not near axis are kept as-is.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return pts

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    segs = []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        l = math.hypot(dx, dy)
        if l < 1e-6:
            continue

        ang = abs(math.degrees(math.atan2(dy, dx)))  # 0..180
        ang = ang if ang <= 180 else 360 - ang
        to_h = min(ang, abs(180 - ang))
        to_v = abs(90 - ang)

        if to_h <= angle_thr_deg:
            y = (p0[1] + p1[1]) / 2.0
            segs.append(("H", y, p0, p1))
        elif to_v <= angle_thr_deg:
            x = (p0[0] + p1[0]) / 2.0
            segs.append(("V", x, p0, p1))
        else:
            segs.append(("F", None, p0, p1))

    if not segs:
        return pts[:-1]

    # Merge consecutive H/H or V/V
    merged = [segs[0]]
    for s in segs[1:]:
        t, v, p0, p1 = s
        t2, v2, q0, q1 = merged[-1]
        if t in ("H", "V") and t == t2:
            merged[-1] = (t, (v + v2) / 2.0, q0, p1)
        else:
            merged.append(s)

    # Rebuild points
    out = [merged[0][2]]
    for i in range(len(merged) - 1):
        t1, v1, a0, a1 = merged[i]
        t2, v2, b0, b1 = merged[i + 1]

        if t1 == "H" and t2 == "V":
            out.append(np.array([v2, v1]))
        elif t1 == "V" and t2 == "H":
            out.append(np.array([v1, v2]))
        else:
            out.append(merged[i][3])

    out.append(out[0])
    out = np.array(out, dtype=float)

    # Drop tiny segments
    cleaned = [out[0]]
    for p in out[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) >= min_seg:
            cleaned.append(p)

    if len(cleaned) > 2 and np.allclose(cleaned[0], cleaned[-1]):
        cleaned = cleaned[:-1]

    return np.array(cleaned, dtype=float)


def remove_tiny_notches(points: np.ndarray, depth_max: float, span_max: float) -> np.ndarray:
    """
    Removes small rectangular in/out steps ("burrs") that often appear at wall ends.
    We remove a local 4-step detour if its bounding box is small enough.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 6:
        return pts

    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    def is_axis(v):
        return abs(v[0]) < 1e-6 or abs(v[1]) < 1e-6

    changed = True
    while changed and len(pts) > 6:
        changed = False
        new = [pts[0]]
        i = 1
        while i < len(pts) - 1:
            if i + 3 < len(pts):
                p0 = new[-1]
                p1, p2, p3, p4 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]

                v01 = p1 - p0
                v12 = p2 - p1
                v23 = p3 - p2
                v34 = p4 - p3

                if is_axis(v01) and is_axis(v12) and is_axis(v23) and is_axis(v34):
                    xs = [p0[0], p1[0], p2[0], p3[0], p4[0]]
                    ys = [p0[1], p1[1], p2[1], p3[1], p4[1]]
                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    depth = min(dx, dy)
                    span = max(dx, dy)

                    if depth <= depth_max and span <= span_max:
                        new.append(p4)
                        i += 4
                        changed = True
                        continue

            new.append(pts[i])
            i += 1

        pts = np.array(new, dtype=float)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


# -----------------------------
# SVG export (compound paths for holes)
# -----------------------------
def contours_to_svg_compound(
    contours,
    hierarchy,
    width: int,
    height: int,
    scale: float,
    decimals: int = 2,
) -> str:
    if hierarchy is None:
        hierarchy = np.full((len(contours), 4), -1, dtype=int)
    else:
        hierarchy = hierarchy[0]

    fmt = f"{{:.{decimals}f}}"

    def subpath(cnt):
        pts = cnt[:, 0, :].astype(float) / float(scale)
        if len(pts) < 3:
            return ""
        d = [f"M {fmt.format(pts[0,0])} {fmt.format(pts[0,1])}"]
        for p in pts[1:]:
            d.append(f"L {fmt.format(p[0])} {fmt.format(p[1])}")
        d.append("Z")
        return " ".join(d)

    paths = []
    for i, cnt in enumerate(contours):
        if i >= len(hierarchy) or hierarchy[i][3] != -1:
            continue  # not an outer contour

        parts = [subpath(cnt)]
        child = hierarchy[i][2]
        while child != -1 and child < len(contours):
            parts.append(subpath(contours[child]))
            if child >= len(hierarchy):
                break
            child = hierarchy[child][0]  # next sibling

        d = " ".join([p for p in parts if p])
        if d:
            paths.append(f'<path d="{d}" />')

    return f"""<svg xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <g fill="black" stroke="none" fill-rule="evenodd" shape-rendering="geometricPrecision">
    {''.join(paths)}
  </g>
</svg>
"""


# -----------------------------
# Vectorization pipeline
# -----------------------------
def vectorize_hybrid_v2(
    gray: np.ndarray,
    *,
    scale: int = 5,
    aa_sigma: float = 0.65,
    aa_threshold: float = 0.50,
    min_component_area: int = 20,
    pre_close_k: int = 3,
    pre_close_iter: int = 1,
    pre_open_k: int = 0,
    pre_open_iter: int = 0,
    # hi-res "deburr" morphology (relative to stroke*scale)
    hi_close_factor: float = 0.22,
    hi_open_factor: float = 0.18,
    min_contour_area: float = 120.0,      # in native px^2
    epsilon_ratio: float = 0.00022,
    epsilon_min_px: float = 1.0,          # in upsampled px
    epsilon_max_factor: float = 0.35,     # cap epsilon to avoid melting small ends
    collinear_deg: float = 1.0,
    ortho: bool = True,
    ortho_angle_deg: float = 10.0,
    spike_acute_deg: float = 35.0,
    notch_depth_factor: float = 0.40,     # relative to stroke
    notch_span_factor: float = 1.20,      # relative to stroke
    svg_decimals: int = 2,
):
    h, w = gray.shape

    bw = binarize_otsu(gray)
    bw = remove_small_components(bw, min_area=min_component_area)
    bw = rect_morph(bw, close_k=pre_close_k, close_iter=pre_close_iter, open_k=pre_open_k, open_iter=pre_open_iter)

    stroke = estimate_stroke_width(bw)  # native px

    # hybrid smoothing (high-res)
    bw_big = aa_upsample_rethreshold(bw, scale=scale, aa_sigma=aa_sigma, aa_threshold=aa_threshold)

    # high-res deburr with RECT kernels (keeps corners crisp)
    hi_close = max(1, int(round(stroke * scale * hi_close_factor)))
    hi_open = max(0, int(round(stroke * scale * hi_open_factor)))
    bw_big = rect_morph(
        bw_big,
        close_k=hi_close,
        close_iter=1,
        open_k=hi_open,
        open_iter=1 if hi_open > 0 else 0,
    )

    contours, hierarchy = cv2.findContours(bw_big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    processed = []
    min_area_big = float(min_contour_area) * (scale * scale)

    # adaptive thresholds (in upsampled px)
    spike_min_len = max(2.0, stroke * 0.20) * scale
    ortho_min_seg = max(4.0, stroke * 0.35) * scale
    notch_depth_max = (stroke * notch_depth_factor) * scale
    notch_span_max = (stroke * notch_span_factor) * scale

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_big:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = max(float(epsilon_min_px), float(epsilon_ratio) * float(peri))
        eps = min(eps, float(epsilon_max_factor) * stroke * scale)

        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx[:, 0, :].astype(float)

        pts = remove_near_collinear(pts, eps_deg=collinear_deg)
        pts = prune_spikes(pts, min_len=spike_min_len, acute_deg=spike_acute_deg)
        pts = remove_tiny_notches(pts, depth_max=notch_depth_max, span_max=notch_span_max)

        if ortho:
            pts = orthogonalize_poly(pts, angle_thr_deg=ortho_angle_deg, min_seg=ortho_min_seg)
            pts = remove_tiny_notches(pts, depth_max=notch_depth_max, span_max=notch_span_max)

        pts = remove_near_collinear(pts, eps_deg=max(0.5, collinear_deg * 0.8))

        processed.append(pts.reshape(-1, 1, 2).astype(np.float32))

    svg = contours_to_svg_compound(processed, hierarchy, width=w, height=h, scale=float(scale), decimals=svg_decimals)
    debug = {
        "stroke_width_px": stroke,
        "hi_close_k": hi_close,
        "hi_open_k": hi_open,
        "spike_min_len_bigpx": spike_min_len,
        "ortho_min_seg_bigpx": ortho_min_seg,
        "notch_depth_max_bigpx": notch_depth_max,
        "notch_span_max_bigpx": notch_span_max,
        "scale": scale,
        "aa_sigma": aa_sigma,
        "aa_threshold": aa_threshold,
        "epsilon_ratio": epsilon_ratio,
    }
    return svg, bw, bw_big, debug


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Hybrid v2 floorplan vectorization (smooth edges + crisp corners + clean ends).")
    p.add_argument("image", help="Input image (png/jpg).")
    p.add_argument("--out", default=None, help="Output SVG path.")
    p.add_argument("--write-debug", action="store_true", help="Write debug masks + JSON next to SVG.")
    p.add_argument("--ortho", action="store_true", help="Enable Manhattan corner crisping (recommended).")

    p.add_argument("--scale", type=int, default=5)
    p.add_argument("--aa-sigma", type=float, default=0.65)
    p.add_argument("--aa-threshold", type=float, default=0.50)

    p.add_argument("--min-component-area", type=int, default=20)
    p.add_argument("--min-contour-area", type=float, default=120.0)

    p.add_argument("--epsilon-ratio", type=float, default=0.00022)
    p.add_argument("--epsilon-min-px", type=float, default=1.0)
    p.add_argument("--epsilon-max-factor", type=float, default=0.35)

    p.add_argument("--hi-close-factor", type=float, default=0.22)
    p.add_argument("--hi-open-factor", type=float, default=0.18)

    p.add_argument("--collinear-deg", type=float, default=1.0)
    p.add_argument("--ortho-angle-deg", type=float, default=10.0)
    p.add_argument("--spike-acute-deg", type=float, default=35.0)

    p.add_argument("--notch-depth-factor", type=float, default=0.40)
    p.add_argument("--notch-span-factor", type=float, default=1.20)

    p.add_argument("--svg-decimals", type=int, default=2)

    args = p.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    out_path = Path(args.out) if args.out else image_path.with_suffix(".hybrid_v2.svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    svg, bw, bw_big, debug = vectorize_hybrid_v2(
        gray,
        scale=args.scale,
        aa_sigma=args.aa_sigma,
        aa_threshold=args.aa_threshold,
        min_component_area=args.min_component_area,
        min_contour_area=args.min_contour_area,
        epsilon_ratio=args.epsilon_ratio,
        epsilon_min_px=args.epsilon_min_px,
        epsilon_max_factor=args.epsilon_max_factor,
        hi_close_factor=args.hi_close_factor,
        hi_open_factor=args.hi_open_factor,
        collinear_deg=args.collinear_deg,
        ortho=bool(args.ortho),
        ortho_angle_deg=args.ortho_angle_deg,
        spike_acute_deg=args.spike_acute_deg,
        notch_depth_factor=args.notch_depth_factor,
        notch_span_factor=args.notch_span_factor,
        svg_decimals=args.svg_decimals,
    )

    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.write_debug:
        mask_path = out_path.with_suffix(".mask.png")
        mask_big_path = out_path.with_suffix(".mask_big.png")
        json_path = out_path.with_suffix(".debug.json")

        cv2.imwrite(str(mask_path), bw)
        cv2.imwrite(str(mask_big_path), bw_big)
        json_path.write_text(__import__("json").dumps(debug, indent=2), encoding="utf-8")

        # quick preview (downsample big mask to original, black walls on white)
        preview = cv2.resize(bw_big, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
        preview = 255 - preview
        cv2.imwrite(str(out_path.with_suffix(".preview.png")), preview)

        print(f"Wrote: {mask_path}")
        print(f"Wrote: {mask_big_path}")
        print(f"Wrote: {json_path}")
        print(f"Wrote: {out_path.with_suffix('.preview.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
