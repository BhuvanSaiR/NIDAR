import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely import affinity

PLAN_FILE = "generated_polygons/polygon_5_sides.plan"   # change if needed

FEET_PER_METER = 3.280839895  # approx conversion


# ---------------- 0. Lat/Lon -> local meters ----------------

def latlon_to_local_xy_m(lat, lon, lat0, lon0):
    """
    Convert (lat, lon) in degrees to local (x, y) in meters
    using a simple equirectangular approximation around (lat0, lon0).
    x = East, y = North.
    """
    lat_rad = math.radians(lat)
    lat0_rad = math.radians(lat0)

    # Approximate meters per degree at reference latitude
    m_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat0_rad)
        + 1.175 * math.cos(4 * lat0_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat0_rad)
        - 93.5 * math.cos(3 * lat0_rad)
    )

    dlat_deg = lat - lat0
    dlon_deg = lon - lon0

    x = dlon_deg * m_per_deg_lon  # East
    y = dlat_deg * m_per_deg_lat  # North

    return x, y, m_per_deg_lat, m_per_deg_lon


# ---------------- 1. Load polygon from QGC plan (in meters) ----------------

def load_polygon_from_plan_in_meters(path):
    """
    Load geoFence polygon from QGroundControl .plan file and convert to local meters.
    """
    with open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    polys = data["geoFence"]["polygons"]
    if not polys:
        raise ValueError("No polygons found in geoFence")

    raw = polys[0]["polygon"]  # list of [lat, lon]

    lats = [p[0] for p in raw]
    lons = [p[1] for p in raw]

    # Use centroid in lat/lon as reference origin
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)

    coords_m = []
    m_per_deg_lat = None
    m_per_deg_lon = None

    for lat, lon in zip(lats, lons):
        x, y, m_lat, m_lon = latlon_to_local_xy_m(lat, lon, lat0, lon0)
        coords_m.append((x, y))
        m_per_deg_lat = m_lat
        m_per_deg_lon = m_lon

    poly_m = Polygon(coords_m)
    if not poly_m.is_valid:
        poly_m = poly_m.buffer(0)

    return poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon)


# ---------------- 2. Equal-area bisector (adapted poly_extract) ----------------

def area_on_minus_side(poly, angle, t):
    """
    minus side = { x : n·x <= t } where n = (cos angle, sin angle)
    """
    n = np.array([math.cos(angle), math.sin(angle)])  # normal

    # centroid of polygon
    cx, cy = poly.centroid.x, poly.centroid.y
    c = np.array([cx, cy])

    # point on the line with normal n and offset t
    nc = np.dot(n, c)
    p0 = c + (t - nc) * n

    # direction vector of the line (perpendicular to n)
    v = np.array([-n[1], n[0]])

    # long segment across polygon
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = diag * 1.5

    p1 = p0 - v * L
    p2 = p0 + v * L
    line_seg = LineString([tuple(p1), tuple(p2)])

    if not line_seg.intersects(poly):
        x0, y0 = poly.exterior.coords[0]
        sign = np.dot(n, np.array([x0, y0])) - t
        return poly.area if sign <= 0 else 0.0

    pieces = split(poly, line_seg)

    minus_area = 0.0
    for part in pieces.geoms:
        pcx, pcy = part.centroid.x, part.centroid.y
        sign = np.dot(n, np.array([pcx, pcy])) - t
        if sign <= 0:
            minus_area += part.area

    return minus_area


def find_bisector_for_angle(poly, angle, tol=1e-6, max_iter=60):
    n = np.array([math.cos(angle), math.sin(angle)])

    xs, ys = zip(*poly.exterior.coords)
    points = np.array(list(zip(xs, ys)))
    projs = points @ n

    t_min = float(np.min(projs))
    t_max = float(np.max(projs))

    total_area = poly.area
    target = total_area / 2.0

    for _ in range(max_iter):
        t_mid = 0.5 * (t_min + t_max)
        area_minus = area_on_minus_side(poly, angle, t_mid)

        if abs(area_minus - target) < tol:
            return t_mid

        if area_minus < target:
            t_min = t_mid
        else:
            t_max = t_mid

    return 0.5 * (t_min + t_max)


def compute_equal_area_split(poly, angle_rad=0.0):
    """
    Compute ONE line that splits polygon into 2 equal-area halves.
    angle_rad is the angle of the line's NORMAL.
    """
    t = find_bisector_for_angle(poly, angle_rad)
    n = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    v = np.array([-n[1], n[0]])

    cx, cy = poly.centroid.x, poly.centroid.y
    c = np.array([cx, cy])
    nc = np.dot(n, c)
    p0 = c + (t - nc) * n

    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    L = diag * 2.0

    p1 = p0 - v * L
    p2 = p0 + v * L
    line_seg = LineString([tuple(p1), tuple(p2)])

    pieces = split(poly, line_seg)
    if len(pieces.geoms) < 2:
        raise RuntimeError("Could not split polygon into two parts")

    # pick two largest pieces
    parts_sorted = sorted(pieces.geoms, key=lambda g: g.area, reverse=True)
    poly1, poly2 = parts_sorted[0], parts_sorted[1]
    return poly1, poly2, line_seg


# ---------------- 3. Longest side midpoint (TO/Land point) ----------------

def longest_side_midpoint(poly_m):
    coords = list(poly_m.exterior.coords)
    max_len = -1.0
    best_mid = None

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len > max_len:
            max_len = seg_len
            best_mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    return best_mid[0], best_mid[1], max_len


# ---------------- 4. Survey path generator with exact spacing rules ----------------

def compute_survey_path(poly_m, angle_deg, separation_m):
    """
    Returns a LineString zig-zag survey path inside poly_m.

    Geometry guarantees (using an inset polygon with buffer(-separation_m/2)):
      - distance between ORIGINAL boundary (incl. bisector for halves)
        and nearest survey line >= separation_m / 2  (Euclidean)
      - distance between parallel survey lines = separation_m (normal direction)
    """
    if separation_m <= 0:
        return None

    # Shrink polygon by separation/2 so all lines are at least separation/2
    # away from original boundary.
    inner = poly_m.buffer(-separation_m / 2.0)
    if inner.is_empty:
        return None

    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)

    centroid = inner.centroid

    # Rotate inner polygon so tracks are horizontal
    inner_rot = affinity.rotate(inner, -angle_deg, origin=centroid, use_radians=False)

    minx, miny, maxx, maxy = inner_rot.bounds

    # Normal direction is vertical (y) in rotated frame.
    # We now place lines every 'separation_m' in y so
    # distance between parallel lines = separation_m.
    # Since we've already buffered by separation/2, the distance from
    # original boundary to any point of inner_rot is >= separation/2.
    # So we don't need extra margin here; we just sweep fully across inner_rot.
    first_y = miny
    last_y = maxy

    if last_y < first_y:
        return None

    y_values = np.arange(first_y, last_y + 1e-9, separation_m)
    diag = math.hypot(maxx - minx, maxy - miny)
    segments_rot = []

    for y in y_values:
        p1 = (minx - diag, y)
        p2 = (maxx + diag, y)
        hline = LineString([p1, p2])

        inter = inner_rot.intersection(hline)
        if inter.is_empty:
            continue

        if inter.geom_type == "LineString":
            segments_rot.append(inter)
        elif inter.geom_type == "MultiLineString":
            segments_rot.extend(list(inter.geoms))

    if not segments_rot:
        return None

    # Sort segments bottom -> top
    segments_rot.sort(key=lambda s: s.centroid.y)

    # Build zig-zag path
    path_points = []
    for i, seg in enumerate(segments_rot):
        xs, ys = seg.xy

        # Ensure left->right ordering
        if xs[0] <= xs[-1]:
            pts = list(zip(xs, ys))
        else:
            pts = list(zip(xs[::-1], ys[::-1]))

        if i % 2 == 0:
            path_points.extend(pts)
        else:
            path_points.extend(pts[::-1])

    path_rot = LineString(path_points)
    path_m = affinity.rotate(path_rot, angle_deg, origin=centroid, use_radians=False)
    return path_m


# ---------------- 5. Best angle for ONE region (full or half) ----------------

def find_best_angle_for_region(region_poly, separation_m, takeoff_point, angle_step_deg=1.0):
    """
    For a given polygon region (full polygon or one half),
    and fixed separation, find angle (0..360) minimizing:
        survey_len + |TO-start| + |end-TO|
    """
    tx, ty = takeoff_point

    best_angle = None
    best_path = None
    best_survey_len = None
    best_transit_len = None
    best_total_len = None

    angle = 0.0
    while angle < 360.0:
        path = compute_survey_path(region_poly, angle, separation_m)
        if path is not None and len(path.coords) > 1:
            xs_p, ys_p = path.xy
            x_start, y_start = xs_p[0], ys_p[0]
            x_end, y_end = xs_p[-1], ys_p[-1]

            survey_len = path.length
            d_to_start = math.hypot(x_start - tx, y_start - ty)
            d_end_to_to = math.hypot(x_end - tx, y_end - ty)
            transit_len = d_to_start + d_end_to_to
            total_len = survey_len + transit_len

            if (best_total_len is None) or (total_len < best_total_len):
                best_total_len = total_len
                best_angle = angle
                best_path = path
                best_survey_len = survey_len
                best_transit_len = transit_len

        angle += angle_step_deg

    return best_angle, best_path, best_survey_len, best_transit_len, best_total_len


# ---------------- 6. Plot everything ----------------

def plot_all(poly_m,
             poly1_m,
             poly2_m,
             cut_line,
             to_point,
             separation_m,
             # full survey
             angle_full,
             path_full,
             survey_len_full,
             transit_len_full,
             total_len_full,
             # split surveys
             angle_half1,
             path1,
             survey_len1,
             transit_len1,
             total_len1,
             angle_half2,
             path2,
             survey_len2,
             transit_len2,
             total_len2,
             longest_side_len_m):
    tx, ty = to_point

    fig, ax = plt.subplots(figsize=(8, 8))

    # Full polygon outline
    xs, ys = poly_m.exterior.xy
    ax.fill(xs, ys, alpha=0.1, color="lightgray")
    ax.plot(xs, ys, linewidth=1.5, color="black", label="Full polygon")

    # Two halves
    xs1, ys1 = poly1_m.exterior.xy
    xs2, ys2 = poly2_m.exterior.xy
    ax.fill(xs1, ys1, alpha=0.25, color="lightblue", label="Half 1")
    ax.fill(xs2, ys2, alpha=0.25, color="lightgreen", label="Half 2")

    # Equal-area cut (clipped)
    inter_cut = poly_m.intersection(cut_line)
    if not inter_cut.is_empty:
        if inter_cut.geom_type == "LineString":
            cx_l, cy_l = inter_cut.xy
            ax.plot(cx_l, cy_l, color="purple", linestyle="--", linewidth=2,
                    label="Equal-area cut")
        elif inter_cut.geom_type == "MultiLineString":
            for seg in inter_cut.geoms:
                cx_l, cy_l = seg.xy
                ax.plot(cx_l, cy_l, color="purple", linestyle="--", linewidth=2)

    # Full reference survey
    if path_full is not None:
        xf, yf = path_full.xy
        ax.plot(xf, yf, color="gray", linewidth=1.2,
                label=f"Full survey (angle={angle_full:.1f}°)")
        # mid_full = path_full.interpolate(0.5, normalized=True)
        # ax.scatter(mid_full.x, mid_full.y, color="gray", s=30, zorder=5)
        # ax.text(mid_full.x, mid_full.y, " full_mid", color="gray",
        #         fontsize=7, ha="left", va="bottom")

    # Half 1 survey
    def plot_region_path(path, angle, color, label):
        if path is None:
            return
        xp, yp = path.xy
        ax.plot(xp, yp, color=color, linewidth=1.5,
                label=f"{label} (angle={angle:.1f}°)")
        # mid = path.interpolate(0.5, normalized=True)
        # ax.scatter(mid.x, mid.y, color=color, s=35, zorder=5)
        # ax.text(mid.x, mid.y, f" {label}_mid", color=color,
        #         fontsize=7, ha="left", va="bottom")

    plot_region_path(path1, angle_half1, "red", "Half1")
    plot_region_path(path2, angle_half2, "orange", "Half2")

    # TO/Land point
    ax.scatter(tx, ty, color="magenta", s=60, marker="X", zorder=6)
    ax.text(tx, ty, " TO/Land (mid longest side)", color="magenta",
            fontsize=8, ha="left", va="bottom", zorder=7)

    # TO legs for surveys
    def plot_to_legs(path, color):
        if path is None:
            return
        xp, yp = path.xy
        sx, sy = xp[0], yp[0]
        ex, ey = xp[-1], yp[-1]
        ax.plot([tx, sx], [ty, sy], linestyle="--", color=color, linewidth=1)
        ax.plot([ex, tx], [ey, ty], linestyle="--", color=color, linewidth=1)

    plot_to_legs(path_full, "gray")
    plot_to_legs(path1, "red")
    plot_to_legs(path2, "orange")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Full vs split equal-area surveys (meters)")

    separation_ft = separation_m * FEET_PER_METER
    longest_side_ft = longest_side_len_m * FEET_PER_METER

    full_survey_ft = survey_len_full * FEET_PER_METER if survey_len_full is not None else 0
    full_transit_ft = transit_len_full * FEET_PER_METER if transit_len_full is not None else 0
    full_total_ft = total_len_full * FEET_PER_METER if total_len_full is not None else 0

    split_survey_len = survey_len1 + survey_len2
    split_transit_len = transit_len1 + transit_len2
    split_total_len = total_len1 + total_len2

    split_survey_ft = split_survey_len * FEET_PER_METER
    split_transit_ft = split_transit_len * FEET_PER_METER
    split_total_ft = split_total_len * FEET_PER_METER

    text = (
        f"Separation x: {separation_m:.2f} m ({separation_ft:.1f} ft)\n"
        f"Boundary clearance: x/2 = {separation_m/2:.2f} m\n"
        f"Longest side: {longest_side_len_m:.1f} m ({longest_side_ft:.1f} ft)\n\n"
        f"FULL survey:\n"
        f"  Best angle: {angle_full:.1f}°\n"
        f"  Survey length: {survey_len_full:.1f} m ({full_survey_ft:.1f} ft)\n"
        f"  TO legs: {transit_len_full:.1f} m ({full_transit_ft:.1f} ft)\n"
        f"  TOTAL: {total_len_full:.1f} m ({full_total_ft:.1f} ft)\n\n"
        f"SPLIT (2 halves) survey (independent angles):\n"
        f"  Half1 angle: {angle_half1:.1f}°, Half2 angle: {angle_half2:.1f}°\n"
        f"  Survey length sum: {split_survey_len:.1f} m ({split_survey_ft:.1f} ft)\n"
        f"  TO legs sum: {split_transit_len:.1f} m ({split_transit_ft:.1f} ft)\n"
        f"  TOTAL: {split_total_len:.1f} m ({split_total_ft:.1f} ft)"
    )

    # ---- draw info box OUTSIDE the plot ----
    fig.subplots_adjust(right=0.70)   # shrink plot width to make space on the right

    fig.text(
        0.72, 0.98,
        text,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    ax.legend(loc="lower right", fontsize=8)
    plt.show()


# ---------------- 7. Main ----------------

def main():
    poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon) = load_polygon_from_plan_in_meters(PLAN_FILE)

    minx, miny, maxx, maxy = poly_m.bounds
    diag_m = math.hypot(maxx - minx, maxy - miny)

    # Takeoff / landing = midpoint of longest side
    tx, ty, longest_side_len_m = longest_side_midpoint(poly_m)

    print("Reference origin (approx polygon center):")
    print(f"  lat0 = {lat0:.8f}°, lon0 = {lon0:.8f}°")
    print("Conversion factors around this latitude:")
    print(f"  1° latitude  ≈ {m_per_deg_lat:.3f} m")
    print(f"  1° longitude ≈ {m_per_deg_lon:.3f} m")
    print(f"Polygon diagonal ≈ {diag_m:.3f} m  ({diag_m * FEET_PER_METER:.1f} ft)")
    print(f"Longest side     ≈ {longest_side_len_m:.3f} m  ({longest_side_len_m * FEET_PER_METER:.1f} ft)")
    print(f"TO/Land (meters) = ({tx:.3f}, {ty:.3f})")

    # Equal-area split of the full polygon (normal angle_rad = 0 => vertical-ish cut)
    poly1_m, poly2_m, cut_line = compute_equal_area_split(poly_m, angle_rad=0.0)
    print(f"Total area: {poly_m.area:.3f}, half areas: {poly1_m.area:.3f}, {poly2_m.area:.3f}")

    # Ask user for separation in meters
    while True:
        try:
            separation_m = float(input("Enter separation distance x in meters: "))
            if separation_m <= 0:
                print("Separation must be > 0")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # 1) Best full-polygon survey (reference)
    print("\nSearching best angle for FULL polygon survey...")
    angle_full, path_full, survey_len_full, transit_len_full, total_len_full = find_best_angle_for_region(
        poly_m, separation_m, (tx, ty), angle_step_deg=1.0
    )

    # 2) Best survey for each half, with its own angle
    print("Searching best angle for HALF 1 survey...")
    angle_half1, path1, survey_len1, transit_len1, total_len1 = find_best_angle_for_region(
        poly1_m, separation_m, (tx, ty), angle_step_deg=1.0
    )

    print("Searching best angle for HALF 2 survey...")
    angle_half2, path2, survey_len2, transit_len2, total_len2 = find_best_angle_for_region(
        poly2_m, separation_m, (tx, ty), angle_step_deg=1.0
    )

    print("\nFULL survey results:")
    print(f"  Best angle: {angle_full:.2f}°")
    print(f"  Survey length: {survey_len_full:.2f} m  ({survey_len_full * FEET_PER_METER:.2f} ft)")
    print(f"  TO legs: {transit_len_full:.2f} m  ({transit_len_full * FEET_PER_METER:.2f} ft)")
    print(f"  TOTAL: {total_len_full:.2f} m  ({total_len_full * FEET_PER_METER:.2f} ft)")

    print("\nHALF 1 survey results:")
    print(f"  Best angle: {angle_half1:.2f}°")
    print(f"  Survey length: {survey_len1:.2f} m  ({survey_len1 * FEET_PER_METER:.2f} ft)")
    print(f"  TO legs: {transit_len1:.2f} m  ({transit_len1 * FEET_PER_METER:.2f} ft)")
    print(f"  TOTAL: {total_len1:.2f} m  ({total_len1 * FEET_PER_METER:.2f} ft)")

    print("\nHALF 2 survey results:")
    print(f"  Best angle: {angle_half2:.2f}°")
    print(f"  Survey length: {survey_len2:.2f} m  ({survey_len2 * FEET_PER_METER:.2f} ft)")
    print(f"  TO legs: {transit_len2:.2f} m  ({transit_len2 * FEET_PER_METER:.2f} ft)")
    print(f"  TOTAL: {total_len2:.2f} m  ({total_len2 * FEET_PER_METER:.2f} ft)")

    plot_all(
        poly_m,
        poly1_m,
        poly2_m,
        cut_line,
        (tx, ty),
        separation_m,
        angle_full,
        path_full,
        survey_len_full,
        transit_len_full,
        total_len_full,
        angle_half1,
        path1,
        survey_len1,
        transit_len1,
        total_len1,
        angle_half2,
        path2,
        survey_len2,
        transit_len2,
        total_len2,
        longest_side_len_m,
    )


if __name__ == "__main__":
    main()
