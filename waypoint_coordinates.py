import json
import math
import numpy as np

from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely import affinity

# --------------- CONFIG ----------------

PLAN_FILE = "generated_polygons/polygon_16_sides.plan"  # change as needed
OUTPUT_FILE = "split_waypoints.json"


# --------------- 0. Lat/Lon -> local meters ----------------

def latlon_to_local_xy_m(lat, lon, lat0, lon0):
    """
    Convert (lat, lon) in degrees to local (x, y) in meters
    using a simple equirectangular approximation around (lat0, lon0).
    x = East, y = North.
    """
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


# --------------- 1. Load polygon from QGC plan (in meters) ----------------

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


# --------------- 2. Equal-area bisector and split ----------------

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

    parts_sorted = sorted(pieces.geoms, key=lambda g: g.area, reverse=True)
    poly1, poly2 = parts_sorted[0], parts_sorted[1]
    return poly1, poly2, line_seg


# --------------- 3. Longest side midpoint (TO/Land point) ----------------

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


# --------------- 4. Survey path generator with clearance ----------------

def compute_survey_path(poly_m, angle_deg, separation_m):
    """
    Returns a LineString zig-zag survey path inside poly_m.

    Uses inset polygon with buffer(-separation_m/2):
      - distance between original boundary (incl. bisector for halves)
        and nearest survey line >= separation_m / 2
      - distance between parallel survey lines = separation_m
    """
    if separation_m <= 0:
        return None

    inner = poly_m.buffer(-separation_m / 2.0)
    if inner.is_empty:
        return None

    if inner.geom_type == "MultiPolygon":
        inner = max(inner.geoms, key=lambda g: g.area)

    centroid = inner.centroid
    inner_rot = affinity.rotate(inner, -angle_deg, origin=centroid, use_radians=False)

    minx, miny, maxx, maxy = inner_rot.bounds

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

    segments_rot.sort(key=lambda s: s.centroid.y)

    path_points = []
    for i, seg in enumerate(segments_rot):
        xs, ys = seg.xy

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


# --------------- 5. Best angle for ONE region ----------------

def find_best_angle_for_region(region_poly, separation_m, takeoff_point, angle_step_deg=1.0):
    """
    For a given polygon region (half polygon),
    find angle (0..360) minimizing:
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


# --------------- 6. Extract turning points from whole mission path ----------------

def extract_turn_points(path: LineString, takeoff_xy, eps=1e-6):
    """
    Given a survey path LineString (in XY),
    build a full mission path: TO -> survey -> TO,
    then extract all turning points (corners) from that.
    Returns a list of (x, y) in order:
      start(TO), ..., end(TO).
    """
    tx, ty = takeoff_xy
    survey_coords = list(path.coords)

    # Build full mission path coords: TO -> survey -> TO
    mission_coords = [(tx, ty)] + survey_coords + [(tx, ty)]
    mission_line = LineString(mission_coords)

    coords = list(mission_line.coords)
    if len(coords) <= 2:
        return coords

    turn_points = [coords[0]]  # always include start (TO)

    for i in range(1, len(coords) - 1):
        x_prev, y_prev = coords[i - 1]
        x_cur, y_cur = coords[i]
        x_next, y_next = coords[i + 1]

        v1 = (x_cur - x_prev, y_cur - y_prev)
        v2 = (x_next - x_cur, y_next - y_cur)

        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if abs(cross) > eps:
            turn_points.append((x_cur, y_cur))

    turn_points.append(coords[-1])  # end (TO)
    return turn_points


# --------------- 7. Local XY -> lat, lon using TO as (0,0) reference ----------------

def local_xy_turns_to_latlon(turn_points_xy, takeoff_xy, lat0, lon0, m_per_deg_lat, m_per_deg_lon):
    """
    Convert mission turning points from local XY to lat/lon.
    Takeoff point in XY is treated as (0,0) reference internally.
    """
    tx, ty = takeoff_xy

    # Compute takeoff lat/lon from reference origin
    takeoff_lat = lat0 + (ty / m_per_deg_lat)
    takeoff_lon = lon0 + (tx / m_per_deg_lon)

    latlon_points = []
    for (x, y) in turn_points_xy:
        dx = x - tx
        dy = y - ty

        lat = takeoff_lat + (dy / m_per_deg_lat)
        lon = takeoff_lon + (dx / m_per_deg_lon)
        latlon_points.append((lat, lon))

    return takeoff_lat, takeoff_lon, latlon_points


# --------------- 8. Main ----------------

def main():
    poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon) = load_polygon_from_plan_in_meters(PLAN_FILE)

    # Takeoff / landing = midpoint of longest side of FULL polygon
    tx, ty, longest_side_len_m = longest_side_midpoint(poly_m)
    takeoff_xy = (tx, ty)

    print(f"Loaded polygon from: {PLAN_FILE}")
    print(f"Takeoff/Land local XY (m): ({tx:.3f}, {ty:.3f})")
    print(f"Longest side length: {longest_side_len_m:.3f} m")
    print(f"Total area (m^2): {poly_m.area:.3f}")

    # Equal-area split of the full polygon (normal angle_rad = 0 => vertical-ish cut)
    poly1_m, poly2_m, cut_line = compute_equal_area_split(poly_m, angle_rad=0.0)
    print(f"Half areas: {poly1_m.area:.3f}, {poly2_m.area:.3f}")

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

    # For HALF 1
    print("\nSearching best angle for HALF 1 survey...")
    angle_half1, path1, survey_len1, transit_len1, total_len1 = find_best_angle_for_region(
        poly1_m, separation_m, takeoff_xy, angle_step_deg=1.0
    )

    if angle_half1 is None or path1 is None:
        print(f"No valid survey path for HALF 1 with x = {separation_m} m (too narrow).")
        return

    print(f"HALF 1: angle={angle_half1:.2f}°, survey_len={survey_len1:.2f} m, total≈{total_len1:.2f} m")

    # For HALF 2
    print("\nSearching best angle for HALF 2 survey...")
    angle_half2, path2, survey_len2, transit_len2, total_len2 = find_best_angle_for_region(
        poly2_m, separation_m, takeoff_xy, angle_step_deg=1.0
    )

    if angle_half2 is None or path2 is None:
        print(f"No valid survey path for HALF 2 with x = {separation_m} m (too narrow).")
        return

    print(f"HALF 2: angle={angle_half2:.2f}°, survey_len={survey_len2:.2f} m, total≈{total_len2:.2f} m")

    # Extract turning points (mission path) for each half
    turn_pts_half1_xy = extract_turn_points(path1, takeoff_xy)
    turn_pts_half2_xy = extract_turn_points(path2, takeoff_xy)

    print(f"Half 1 turning points (including TO and LAND): {len(turn_pts_half1_xy)}")
    print(f"Half 2 turning points (including TO and LAND): {len(turn_pts_half2_xy)}")

    # Convert to lat/lon, using TO as reference (0,0 internally)
    takeoff_lat, takeoff_lon, half1_turn_latlon = local_xy_turns_to_latlon(
        turn_pts_half1_xy, takeoff_xy, lat0, lon0, m_per_deg_lat, m_per_deg_lon
    )

    # For half2 we reuse same takeoff reference; so we get same TO lat/lon
    _, _, half2_turn_latlon = local_xy_turns_to_latlon(
        turn_pts_half2_xy, takeoff_xy, lat0, lon0, m_per_deg_lat, m_per_deg_lon
    )

    # Build ordered waypoint lists for each half:
    #   TAKEOFF, all TURN corners, LAND
    # (Note: turn_pts_* already includes TO at index 0 and TO at last index)
    def build_waypoint_list(turn_latlon_list):
        waypoints = []
        # TAKEOFF
        waypoints.append({
            "type": "TAKEOFF",
            "lat": takeoff_lat,
            "lon": takeoff_lon,
        })
        # interior turning points (excluding first and last since they are TO)
        for i in range(1, len(turn_latlon_list) - 1):
            lat, lon = turn_latlon_list[i]
            waypoints.append({
                "type": "TURN",
                "index": i - 1,
                "lat": lat,
                "lon": lon,
            })
        # LAND
        waypoints.append({
            "type": "LAND",
            "lat": takeoff_lat,
            "lon": takeoff_lon,
        })
        return waypoints

    waypoints_half1 = build_waypoint_list(half1_turn_latlon)
    waypoints_half2 = build_waypoint_list(half2_turn_latlon)

    output = {
        "plan_file": PLAN_FILE,
        "separation_m": separation_m,
        "takeoff": {
            "lat": takeoff_lat,
            "lon": takeoff_lon,
        },
        "half1": {
            "best_angle_deg": angle_half1,
            "survey_length_m": survey_len1,
            "total_length_m": total_len1,
            "waypoints_ordered": waypoints_half1
        },
        "half2": {
            "best_angle_deg": angle_half2,
            "survey_length_m": survey_len2,
            "total_length_m": total_len2,
            "waypoints_ordered": waypoints_half2
        }
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\nSplit-survey waypoints saved to: {OUTPUT_FILE}")
    print("Each half has: TAKEOFF -> TURN corners -> LAND, in lat, lon format.")


if __name__ == "__main__":
    main()
