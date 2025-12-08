import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely import affinity




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

    _, _, m_per_deg_lat, m_per_deg_lon = latlon_to_local_xy_m(lat0, lon0, lat0, lon0)

    coords_m = []
    for lat, lon in zip(lats, lons):
        x, y, _, _ = latlon_to_local_xy_m(lat, lon, lat0, lon0)
        coords_m.append((x, y))

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

