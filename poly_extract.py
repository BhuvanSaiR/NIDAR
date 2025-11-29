import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.ops import split

PLAN_FILE = "rule.plan"   


# 1) Load polygon from QGroundControl .plan file
def load_polygon_from_plan(path):
    with open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    polys = data["geoFence"]["polygons"]
    if not polys:
        raise ValueError("No polygons in geoFence")

    # Each point is [lat, lon]; we convert to (x, y) = (lon, lat)
    raw = polys[0]["polygon"]
    coords = [(p[1], p[0]) for p in raw]   # (lon, lat)

    # Make shapely polygon
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix minor self-intersections if any

    return poly


# 2) Given a polygon, angle, and t, compute area on one side of line n·x = t
def area_on_minus_side(poly, angle, t):
    """
    minus side = { x : n·x <= t } where n = (cos angle, sin angle)
    We approximate by:
      - Construct long segment of that line across polygon's bounding box
      - Split polygon with that line segment
      - Choose pieces whose centroids satisfy n·c <= t
      - Sum their areas
    """
    n = np.array([math.cos(angle), math.sin(angle)])  # normal
    # centroid of polygon
    cx, cy = poly.centroid.x, poly.centroid.y
    c = np.array([cx, cy])

    # Point on the line with normal n and offset t:
    # We want p such that n·p = t. Use projection of centroid onto line.
    # n·c0 = t  =>  c0 = c + (t - n·c) * n
    nc = np.dot(n, c)
    p0 = c + (t - nc) * n

    # Direction vector of the line (perpendicular to n)
    v = np.array([-n[1], n[0]])

    # Make a long segment that surely crosses the polygon
    # Use bounding box size as reference
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = diag * 1.0  # long enough

    p1 = p0 - v * L
    p2 = p0 + v * L

    line_seg = LineString([tuple(p1), tuple(p2)])

    # If line doesn't intersect polygon, area is either 0 or full
    if not line_seg.intersects(poly):
        # Check any polygon point's side
        x0, y0 = poly.exterior.coords[0]
        sign = np.dot(n, np.array([x0, y0])) - t
        if sign <= 0:
            return poly.area
        else:
            return 0.0

    pieces = split(poly, line_seg)

    minus_area = 0.0
    for part in pieces.geoms:
        pcx, pcy = part.centroid.x, part.centroid.y
        sign = np.dot(n, np.array([pcx, pcy])) - t
        if sign <= 0:  # in minus half-space
            minus_area += part.area

    return minus_area


# 3) For a given angle, find t such that area_on_minus_side = total_area / 2
def find_bisector_for_angle(poly, angle, tol=1e-6, max_iter=60):
    n = np.array([math.cos(angle), math.sin(angle)])

    # Project all vertices on the normal to get search range for t
    xs, ys = zip(*poly.exterior.coords)
    points = np.array(list(zip(xs, ys)))
    projs = points @ n  # dot product with n

    t_min = float(np.min(projs))
    t_max = float(np.max(projs))

    total_area = poly.area
    target = total_area / 2.0

    # Binary search
    for _ in range(max_iter):
        t_mid = 0.5 * (t_min + t_max)
        area_minus = area_on_minus_side(poly, angle, t_mid)

        if abs(area_minus - target) < tol:
            return t_mid

        if area_minus < target:
            # need more area on minus side -> increase t (expand half-space)
            t_min = t_mid
        else:
            t_max = t_mid

    # Return best mid even if not perfect
    return 0.5 * (t_min + t_max)


# 4) Compute 2n bisecting lines
def compute_2n_bisectors(poly):
    # number of vertices (last == first in exterior coords)
    # num_vertices = len(poly.exterior.coords) - 1
    num_lines = 14

    bisectors = []

    # Choose 2n different angles in [0, π) (since line with normal θ and θ+π is same line)
    angles = np.linspace(0.0, math.pi, num_lines, endpoint=False)

    for angle in angles:
        t = find_bisector_for_angle(poly, angle)
        bisectors.append((angle, t))

    return bisectors


# 5) Visualize: polygon + all 2n bisecting lines
def plot_polygon_and_bisectors_clipped(poly, bisectors, out_file="bisectors_clipped.png"):
    xs, ys = poly.exterior.xy

    plt.figure(figsize=(7, 7))
    # Draw polygon
    plt.fill(xs, ys, alpha=0.3)
    plt.plot(xs, ys, linewidth=2)

    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = diag * 3.0  # long enough for extended line

    for angle, t in bisectors:
        n = np.array([math.cos(angle), math.sin(angle)])
        v = np.array([-n[1], n[0]])

        # Construct a point p0 on the line: n·p0 = t
        cx, cy = poly.centroid.x, poly.centroid.y
        c = np.array([cx, cy])
        nc = np.dot(n, c)
        p0 = c + (t - nc) * n

        # Extended line segment
        p1 = p0 - v * L
        p2 = p0 + v * L
        extended_line = LineString([tuple(p1), tuple(p2)])

        # Intersect with polygon -> gives the chord(s) inside
        inter = poly.intersection(extended_line)

        if inter.is_empty:
            continue

        # Could be LineString or MultiLineString
        if inter.geom_type == "LineString":
            xs_l, ys_l = inter.xy
            plt.plot(xs_l, ys_l, linewidth=1)
        elif inter.geom_type == "MultiLineString":
            for seg in inter.geoms:
                xs_l, ys_l = seg.xy
                plt.plot(xs_l, ys_l, linewidth=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.title("Polygon and 2n equal-area bisecting chords")
    plt.xlabel("X (lon)")
    plt.ylabel("Y (lat)")

    # plt.savefig(out_file, dpi=3600)
    plt.show()


if __name__ == "__main__":
    poly = load_polygon_from_plan(PLAN_FILE)
    bisectors = compute_2n_bisectors(poly)
    print(f"Found {len(bisectors)} bisecting lines")
    plot_polygon_and_bisectors_clipped(poly, bisectors, "bisectors_clipped.png")
    