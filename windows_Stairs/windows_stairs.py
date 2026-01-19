#!/usr/bin/env python3
"""Centerline (stroke) vectorization for CAD-like raster snippets.

Key points for floorplan fragments:
  - Work on *centerlines* (skeleton), not outlines, to avoid double-edge duplicates.
  - Cluster junction pixels into single nodes to avoid "star" artifacts.
  - Trace both node-to-node paths and pure cycles (closed loops).
  - Default output is *straight segments only* (no SVG arc commands), because
    anti-aliased straight strokes can be mistakenly fit as arcs.

If you *really* need arcs, enable --allow_arcs and use conservative arc gates.

Dependencies:
  - numpy
  - opencv-python
  - scikit-image

Usage:
  python vectorize_centerline_v3.py input.png output.svg

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

Point = Tuple[float, float]  # (x, y)


@dataclass
class Edge:
    pts_yx: List[Tuple[int, int]]  # list of (y, x) pixels
    start_node: int  # 0 if none
    end_node: int    # 0 if none


@dataclass
class LineSeg:
    p0: Point
    p1: Point


@dataclass
class PolySeg:
    pts: List[Point]


@dataclass
class ArcSeg:
    p0: Point
    p1: Point
    center: Point
    r: float
    sweep: int
    large_arc: int


Primitive = Tuple[str, object]  # ('line', LineSeg) etc.


# ----------------------------- preprocessing -----------------------------

def preprocess_mask(gray_u8: np.ndarray, sigma: float = 35.0) -> np.ndarray:
    """Binary stroke mask (bool) from grayscale."""
    f = gray_u8.astype(np.float32) / 255.0

    # Normalize illumination with a large blur.
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
    norm = f / (bg + 1e-3)
    norm = np.clip(norm, 0.0, 1.0)

    inv = 1.0 - norm  # dark strokes -> bright
    t = float(threshold_otsu(inv))
    mask = inv > t

    # Light denoise without moving geometry too much.
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.medianBlur(mask_u8, 3)
    mask = mask_u8 > 0

    return mask


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask)


# ----------------------------- polyline utils ----------------------------

def polyline_length(points: List[Point]) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())


def rdp(points: List[Point], eps: float) -> List[Point]:
    """Ramer–Douglas–Peucker simplification."""
    if len(points) <= 2:
        return points

    pts = np.array(points, dtype=np.float32)
    a = pts[0]
    b = pts[-1]
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))

    if ab_len2 == 0.0:
        dists = np.linalg.norm(pts - a, axis=1)
        i = int(np.argmax(dists))
        if float(dists[i]) <= eps:
            return [points[0], points[-1]]
        left = rdp(points[: i + 1], eps)
        right = rdp(points[i:], eps)
        return left[:-1] + right

    ap = pts - a
    t = (ap @ ab) / ab_len2
    t = np.clip(t, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    d = np.linalg.norm(pts - proj, axis=1)

    i = int(np.argmax(d))
    if float(d[i]) <= eps:
        return [points[0], points[-1]]

    left = rdp(points[: i + 1], eps)
    right = rdp(points[i:], eps)
    return left[:-1] + right


def remove_consecutive_duplicates(pts: List[Point], tol: float = 1e-6) -> List[Point]:
    out: List[Point] = []
    for p in pts:
        if not out:
            out.append(p)
            continue
        if abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    return out


def merge_collinear_polyline(pts: List[Point], ang_eps_deg: float = 2.0) -> List[Point]:
    """Remove middle points that keep segments nearly collinear."""
    if len(pts) <= 2:
        return pts

    ang_eps = math.radians(ang_eps_deg)

    def angle(p0: Point, p1: Point) -> float:
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a1 = angle(out[-1], pts[i])
        a2 = angle(pts[i], pts[i + 1])
        d = a2 - a1
        while d <= -math.pi:
            d += 2 * math.pi
        while d > math.pi:
            d -= 2 * math.pi
        if abs(d) < ang_eps:
            # skip pts[i]
            continue
        out.append(pts[i])
    out.append(pts[-1])
    return out


# ----------------------------- fitting ----------------------------

def fit_line(points_xy: np.ndarray, line_eps: float = 0.9) -> Optional[LineSeg]:
    """Fit line via PCA; accept if 95th percentile perpendicular error <= line_eps."""
    if len(points_xy) < 2:
        return None

    mean = points_xy.mean(axis=0)
    X = points_xy - mean
    # SVD for principal direction
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]

    proj = X @ direction
    closest = np.outer(proj, direction)
    dist = np.linalg.norm(X - closest, axis=1)
    if float(np.percentile(dist, 95)) > line_eps:
        return None

    t0, t1 = float(proj.min()), float(proj.max())
    p0 = mean + t0 * direction
    p1 = mean + t1 * direction
    return LineSeg((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))


def arc_candidate(points: List[Point], *, min_turn_deg: float = 35.0, min_len: float = 25.0) -> bool:
    """Heuristic gate to avoid fitting arcs to almost-straight polylines."""
    if len(points) < 6:
        return False
    if polyline_length(points) < min_len:
        return False

    # Compute segment angles
    angs: List[float] = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        if dx == 0.0 and dy == 0.0:
            continue
        angs.append(math.atan2(dy, dx))
    if len(angs) < 5:
        return False

    def wrap(d: float) -> float:
        while d <= -math.pi:
            d += 2 * math.pi
        while d > math.pi:
            d -= 2 * math.pi
        return d

    diffs = [wrap(angs[i + 1] - angs[i]) for i in range(len(angs) - 1)]
    # Ignore tiny diffs
    diffs = [d for d in diffs if abs(d) > math.radians(2.0)]
    if len(diffs) < 3:
        return False

    total_turn = sum(abs(d) for d in diffs)
    if total_turn < math.radians(min_turn_deg):
        return False

    # Turning direction should be mostly consistent for a circular arc
    signs = [1 if d > 0 else -1 for d in diffs]
    consistency = abs(sum(signs)) / len(signs)
    if consistency < 0.7:
        return False

    return True


def fit_circle_arc(points_xy: np.ndarray, arc_eps: float = 0.8, min_radius: float = 12.0) -> Optional[ArcSeg]:
    """Least-squares circle fit; return as SVG arc from first->last."""
    if len(points_xy) < 6:
        return None

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    xc, yc, c0 = c
    r2 = xc * xc + yc * yc + c0
    if r2 <= 0:
        return None
    r = float(math.sqrt(r2))
    if r < min_radius:
        return None

    rr = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    resid = np.abs(rr - r)
    if float(np.percentile(resid, 95)) > arc_eps:
        return None

    p0 = points_xy[0]
    p1 = points_xy[-1]

    a0 = math.atan2(p0[1] - yc, p0[0] - xc)
    a1 = math.atan2(p1[1] - yc, p1[0] - xc)

    def angdiff(a: float, b: float) -> float:
        d = b - a
        while d <= -math.pi:
            d += 2 * math.pi
        while d > math.pi:
            d -= 2 * math.pi
        return d

    d = angdiff(a0, a1)
    sweep = 1 if d > 0 else 0
    large_arc = 1 if abs(d) > math.pi else 0

    return ArcSeg(
        p0=(float(p0[0]), float(p0[1])),
        p1=(float(p1[0]), float(p1[1])),
        center=(float(xc), float(yc)),
        r=r,
        sweep=sweep,
        large_arc=large_arc,
    )


def polyline_to_primitives(
    points: List[Point],
    *,
    line_eps: float,
    allow_arcs: bool,
    arc_eps: float,
    arc_min_turn_deg: float,
    arc_min_len: float,
) -> List[Primitive]:
    """Convert a polyline to a list of primitives.

    Strategy:
      1) Try to fit a single line.
      2) Optionally try a single arc (with strong gates).
      3) Else output as a straight polyline (M/L only).

    (We keep it simple/deterministic; more advanced split&merge can be added later.)
    """

    pts_np = np.array(points, dtype=np.float32)

    line = fit_line(pts_np, line_eps=line_eps)
    if line is not None:
        return [('line', line)]

    if allow_arcs and arc_candidate(points, min_turn_deg=arc_min_turn_deg, min_len=arc_min_len):
        arc = fit_circle_arc(pts_np, arc_eps=arc_eps)
        if arc is not None:
            return [('arc', arc)]

    return [('poly', PolySeg(points))]


# ----------------------------- skeleton -> edges ----------------------------

_OFFS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def skeleton_to_edges(skel: np.ndarray, mask: np.ndarray, node_dilate: int = 1) -> Tuple[List[Edge], Dict[int, Tuple[int, int]]]:
    """Extract edges between node components plus cycles."""
    h, w = skel.shape
    sk_u8 = skel.astype(np.uint8)

    # Degree via convolution
    k = np.ones((3, 3), np.uint8)
    nb_cnt = cv2.filter2D(sk_u8, -1, k, borderType=cv2.BORDER_CONSTANT)
    deg = nb_cnt - sk_u8

    node = skel & (deg != 2)

    # Dilate node mask (restricted to skeleton) to merge junction pixel clouds.
    if node_dilate > 0:
        kk = np.ones((2 * node_dilate + 1, 2 * node_dilate + 1), np.uint8)
        node_reg = cv2.dilate(node.astype(np.uint8), kk, iterations=1).astype(bool)
        node_reg = node_reg & skel
    else:
        node_reg = node

    num_labels, labels = cv2.connectedComponents(node_reg.astype(np.uint8), connectivity=8)

    # Representative point per node: choose deepest (max dist) point inside node region.
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    node_rep: Dict[int, Tuple[int, int]] = {}
    for nid in range(1, num_labels):
        ys, xs = np.where(labels == nid)
        if ys.size == 0:
            continue
        vals = dist[ys, xs]
        j = int(np.argmax(vals))
        node_rep[nid] = (int(ys[j]), int(xs[j]))

    visited = np.zeros((h, w), dtype=bool)  # visited non-node skeleton pixels
    edges: List[Edge] = []

    def trace(start_node: int, start_px: Tuple[int, int], next_px: Tuple[int, int]) -> Edge:
        path = [start_px, next_px]
        prev = start_px
        cur = next_px
        end_node = 0

        if labels[cur] == 0:
            visited[cur] = True

        while True:
            cy, cx = cur
            nbs: List[Tuple[int, int]] = []
            for dy, dx in _OFFS8:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not skel[ny, nx]:
                    continue
                if (ny, nx) == prev:
                    continue
                nbs.append((ny, nx))

            if not nbs:
                break

            # Prefer stepping into a *different* node region
            next_step = None
            for nb in nbs:
                if labels[nb] != 0 and labels[nb] != start_node:
                    next_step = nb
                    break

            if next_step is None:
                # Prefer unvisited non-node
                for nb in nbs:
                    if labels[nb] == 0 and not visited[nb]:
                        next_step = nb
                        break

            if next_step is None:
                next_step = nbs[0]

            path.append(next_step)

            if labels[next_step] != 0 and labels[next_step] != start_node:
                end_node = int(labels[next_step])
                break

            if labels[next_step] == 0:
                if visited[next_step]:
                    break
                visited[next_step] = True

            prev, cur = cur, next_step

        return Edge(path, start_node=start_node, end_node=end_node)

    # Node-to-node edges
    for nid in range(1, num_labels):
        ys, xs = np.where(labels == nid)
        if ys.size == 0:
            continue
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dy, dx in _OFFS8:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not skel[ny, nx]:
                    continue
                if labels[ny, nx] == nid:
                    continue
                if labels[ny, nx] != 0:
                    continue
                if visited[ny, nx]:
                    continue

                edges.append(trace(nid, (y, x), (ny, nx)))

    # Remaining cycles/chains with no nodes
    ys, xs = np.where(skel & (labels == 0) & (~visited))
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0] or labels[y0, x0] != 0 or not skel[y0, x0]:
            continue

        start = (y0, x0)
        visited[y0, x0] = True
        path = [start]

        # pick first neighbor
        first_nbs = []
        for dy, dx in _OFFS8:
            ny, nx = y0 + dy, x0 + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and labels[ny, nx] == 0:
                first_nbs.append((ny, nx))
        if not first_nbs:
            continue

        prev = start
        cur = first_nbs[0]
        path.append(cur)
        visited[cur] = True

        safety = 0
        while safety < h * w:
            safety += 1
            cy, cx = cur
            nbs = []
            for dy, dx in _OFFS8:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not skel[ny, nx] or labels[ny, nx] != 0:
                    continue
                if (ny, nx) == prev:
                    continue
                nbs.append((ny, nx))

            if not nbs:
                break

            nxt = nbs[0]
            path.append(nxt)

            if nxt == start:
                break

            if visited[nxt]:
                break
            visited[nxt] = True
            prev, cur = cur, nxt

        edges.append(Edge(path, start_node=0, end_node=0))

    return edges, node_rep


# ----------------------------- SVG export -----------------------------

def export_svg(
    prims: List[Primitive],
    w: int,
    h: int,
    out_path: str,
    *,
    stroke_width: float = 1.0,
    linecap: str = 'butt',
    linejoin: str = 'miter',
    force_no_arcs: bool = True,
) -> None:
    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append(
        f'<g fill="none" stroke="#000" stroke-width="{stroke_width}" '
        f'stroke-linecap="{linecap}" stroke-linejoin="{linejoin}">' 
    )

    for kind, obj in prims:
        if kind == 'line':
            ln: LineSeg = obj
            x1, y1 = ln.p0
            x2, y2 = ln.p1
            parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" />')
        elif kind == 'arc':
            if force_no_arcs:
                # Fallback: approximate by straight polyline between endpoints.
                ac: ArcSeg = obj
                x1, y1 = ac.p0
                x2, y2 = ac.p1
                parts.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" />')
            else:
                ac: ArcSeg = obj
                x1, y1 = ac.p0
                x2, y2 = ac.p1
                r = ac.r
                parts.append(
                    f'<path d="M {x1:.3f} {y1:.3f} A {r:.3f} {r:.3f} 0 {ac.large_arc} {ac.sweep} {x2:.3f} {y2:.3f}" />'
                )
        else:
            pl: PolySeg = obj
            if len(pl.pts) < 2:
                continue
            d = [f'M {pl.pts[0][0]:.3f} {pl.pts[0][1]:.3f}']
            for x, y in pl.pts[1:]:
                d.append(f'L {x:.3f} {y:.3f}')
            parts.append(f'<path d="{" ".join(d)}" />')

    parts.append('</g></svg>')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ----------------------------- main vectorize -----------------------------

def vectorize(
    img_path: str,
    out_path: str,
    *,
    sigma: float,
    scale: float,
    node_dilate: int,
    min_edge_len: float,
    rdp_eps: float,
    line_eps: float,
    allow_arcs: bool,
    arc_eps: float,
    arc_min_turn_deg: float,
    arc_min_len: float,
    stroke_width: float,
    linecap: str,
    linejoin: str,
) -> None:
    gray0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray0 is None:
        raise FileNotFoundError(img_path)

    orig_h, orig_w = gray0.shape

    if scale != 1.0:
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        gray = cv2.resize(gray0, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        gray = gray0

    mask = preprocess_mask(gray, sigma=sigma)
    skel = skeletonize_mask(mask)

    edges, node_rep = skeleton_to_edges(skel, mask, node_dilate=node_dilate)

    node_xy: Dict[int, Point] = {nid: (float(x), float(y)) for nid, (y, x) in node_rep.items()}

    # Scale tolerances
    line_eps_s = line_eps * scale
    arc_eps_s = arc_eps * scale
    rdp_eps_s = rdp_eps * scale
    min_edge_len_s = min_edge_len * scale

    prims: List[Primitive] = []

    for e in edges:
        pts: List[Point] = [(float(x), float(y)) for (y, x) in e.pts_yx]

        # Snap endpoints to representative node points
        if e.start_node in node_xy:
            pts[0] = node_xy[e.start_node]
        if e.end_node in node_xy:
            pts[-1] = node_xy[e.end_node]

        pts = remove_consecutive_duplicates(pts)
        if len(pts) < 2:
            continue
        if polyline_length(pts) < min_edge_len_s:
            continue

        # Simplify + collinear merge
        if rdp_eps_s > 0:
            pts = rdp(pts, eps=rdp_eps_s)
        pts = merge_collinear_polyline(pts, ang_eps_deg=2.0)

        # Convert to primitives
        prim_list = polyline_to_primitives(
            pts,
            line_eps=line_eps_s,
            allow_arcs=allow_arcs,
            arc_eps=arc_eps_s,
            arc_min_turn_deg=arc_min_turn_deg,
            arc_min_len=arc_min_len * scale,
        )

        # Back to original coord system
        if scale != 1.0:
            fixed: List[Primitive] = []
            for kind, obj in prim_list:
                if kind == 'line':
                    ln: LineSeg = obj
                    fixed.append(('line', LineSeg((ln.p0[0] / scale, ln.p0[1] / scale), (ln.p1[0] / scale, ln.p1[1] / scale))))
                elif kind == 'arc':
                    ac: ArcSeg = obj
                    fixed.append((
                        'arc',
                        ArcSeg(
                            p0=(ac.p0[0] / scale, ac.p0[1] / scale),
                            p1=(ac.p1[0] / scale, ac.p1[1] / scale),
                            center=(ac.center[0] / scale, ac.center[1] / scale),
                            r=ac.r / scale,
                            sweep=ac.sweep,
                            large_arc=ac.large_arc,
                        ),
                    ))
                else:
                    pl: PolySeg = obj
                    fixed.append(('poly', PolySeg([(x / scale, y / scale) for (x, y) in pl.pts])))
            prim_list = fixed

        prims.extend(prim_list)

    export_svg(
        prims,
        orig_w,
        orig_h,
        out_path,
        stroke_width=stroke_width,
        linecap=linecap,
        linejoin=linejoin,
        force_no_arcs=not allow_arcs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='Input raster (png/jpg)')
    ap.add_argument('output', help='Output SVG')

    ap.add_argument('--sigma', type=float, default=35.0, help='Background blur sigma')
    ap.add_argument('--scale', type=float, default=1.0, help='Optional upscale factor before skeletonization')
    ap.add_argument('--node_dilate', type=int, default=1, help='Node dilation radius (pixels) before clustering')

    ap.add_argument('--min_edge_len', type=float, default=2.0, help='Drop edges shorter than this length (pixels)')
    ap.add_argument('--rdp_eps', type=float, default=0.4, help='Polyline simplification epsilon (pixels)')

    ap.add_argument('--line_eps', type=float, default=0.9, help='Line fit tolerance (pixels, 95th pct)')

    ap.add_argument('--allow_arcs', action='store_true', help='Enable SVG arc fitting (off by default)')
    ap.add_argument('--arc_eps', type=float, default=0.8, help='Arc fit tolerance (pixels, 95th pct)')
    ap.add_argument('--arc_min_turn_deg', type=float, default=35.0, help='Min total turning for arc candidate')
    ap.add_argument('--arc_min_len', type=float, default=25.0, help='Min polyline length for arc candidate (pixels)')

    ap.add_argument('--stroke_width', type=float, default=1.0, help='SVG stroke width')
    ap.add_argument('--linecap', type=str, default='butt', choices=['butt', 'round', 'square'])
    ap.add_argument('--linejoin', type=str, default='miter', choices=['miter', 'round', 'bevel'])

    args = ap.parse_args()

    vectorize(
        args.input,
        args.output,
        sigma=args.sigma,
        scale=args.scale,
        node_dilate=args.node_dilate,
        min_edge_len=args.min_edge_len,
        rdp_eps=args.rdp_eps,
        line_eps=args.line_eps,
        allow_arcs=args.allow_arcs,
        arc_eps=args.arc_eps,
        arc_min_turn_deg=args.arc_min_turn_deg,
        arc_min_len=args.arc_min_len,
        stroke_width=args.stroke_width,
        linecap=args.linecap,
        linejoin=args.linejoin,
    )


if __name__ == '__main__':
    main()
