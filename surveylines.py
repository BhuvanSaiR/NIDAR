import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely import affinity

PLAN_FILE = "rule.plan"   # <-- change if needed

# ---------------- 0. Unit conversion helpers ----------------

FEET_PER_METER = 3.280839895  # approx conversion


def latlon_to_local_xy_m(lat, lon, lat0, lon0):
    """
    Convert (lat, lon) in degrees to local (x, y) in meters
    using a simple equirectangular approximation around (lat0, lon0).
    x = East, y = North.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

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


# ------------ 1. Load polygon from QGC .plan and convert to meters ------------

def load_polygon_from_plan_in_meters(path):
    """
    Load geoFence polygon from QGroundControl .plan file.
    Original coordinates are [lat, lon] in degrees.
    Convert them to local (x, y) in meters around the polygon centroid.
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


# ------------ 2. Find midpoint of longest side (takeoff/landing point) ------------

def longest_side_midpoint(poly_m):
    """
    Find the longest edge of the polygon and return:
    (midpoint_x, midpoint_y, longest_length_m)
    """
    coords = list(poly_m.exterior.coords)
    # last coord == first, so iterate to len-1
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


# ------------ 3. Compute continuous zig-zag survey path (in meters) ------------

def compute_survey_path(poly_m, angle_deg, spacing_m):
    """
    Returns a single LineString path representing a zig-zag survey pattern
    inside `poly_m`, with track direction = angle_deg and separation = spacing_m.
    poly_m and spacing_m are in meters.
    """
    if spacing_m <= 0:
        return None

    centroid = poly_m.centroid

    # Rotate polygon so tracks become horizontal
    poly_rot = affinity.rotate(poly_m, -angle_deg, origin=centroid, use_radians=False)

    minx, miny, maxx, maxy = poly_rot.bounds
    diag = math.hypot(maxx - minx, maxy - miny)

    # Horizontal lines in rotated space, spaced by 'spacing_m'
    y_start = miny - spacing_m
    y_end = maxy + spacing_m
    y_values = np.arange(y_start, y_end + spacing_m * 0.5, spacing_m)

    segments_rot = []

    for y in y_values:
        p1 = (minx - diag, y)
        p2 = (maxx + diag, y)
        hline = LineString([p1, p2])

        inter = poly_rot.intersection(hline)
        if inter.is_empty:
            continue

        if inter.geom_type == "LineString":
            segments_rot.append(inter)
        elif inter.geom_type == "MultiLineString":
            segments_rot.extend(list(inter.geoms))

    if not segments_rot:
        return None

    # Sort segments bottom -> top (in rotated frame)
    segments_rot.sort(key=lambda s: s.centroid.y)

    # Build zig-zag path in rotated frame
    path_points = []
    for i, seg in enumerate(segments_rot):
        xs, ys = seg.xy

        # Ensure left->right ordering
        if xs[0] <= xs[-1]:
            pts = list(zip(xs, ys))
        else:
            pts = list(zip(xs[::-1], ys[::-1]))

        if i % 2 == 0:
            # even: left -> right
            path_points.extend(pts)
        else:
            # odd: right -> left
            path_points.extend(pts[::-1])

    path_rot = LineString(path_points)
    path_m = affinity.rotate(path_rot, angle_deg, origin=centroid, use_radians=False)
    return path_m


# ------------ 4. Search for best angle (min total mission length) ------------

def find_best_angle(poly_m, spacing_m, takeoff_point, angle_step_deg=1.0):
    """
    For a fixed spacing (meters) and a given takeoff/landing point,
    scan angles in [0, 180) with given step and return:
      (best_angle_deg,
       best_path_m,
       best_survey_len_m,
       best_transit_len_m,
       best_total_len_m)
    where:
      survey_len  = length of zig-zag path
      transit_len = |TO - start| + |end - TO|
      total_len   = survey_len + transit_len
    """
    tx, ty = takeoff_point

    best_angle = None
    best_path = None
    best_survey_len = None
    best_transit_len = None
    best_total_len = None

    angle = 0.0
    while angle < 360.0:
        path = compute_survey_path(poly_m, angle, spacing_m)
        if path is not None and len(path.coords) > 1:
            xs_p, ys_p = path.xy
            x_start, y_start = xs_p[0], ys_p[0]
            x_end, y_end = xs_p[-1], ys_p[-1]

            # survey length
            survey_len = path.length

            # transit lengths: TO -> start, end -> TO
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


# ------------ 5. Plot result with TO point + lengths ------------

def plot_best_path(poly_m,
                   best_angle,
                   best_path_m,
                   survey_len_m,
                   transit_len_m,
                   total_len_m,
                   spacing_m,
                   takeoff_point,
                   longest_side_len_m):
    xs, ys = poly_m.exterior.xy

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.fill(xs, ys, alpha=0.3, color="lightblue")
    ax.plot(xs, ys, linewidth=2, color="blue")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Optimal survey path (meters) with takeoff/landing point")

    tx, ty = takeoff_point

    if best_path_m is not None:
        xs_p, ys_p = best_path_m.xy
        # path
        ax.plot(xs_p, ys_p, linewidth=1.5, color="red")

        # start & end markers
        x_start, y_start = xs_p[0], ys_p[0]
        x_end, y_end = xs_p[-1], ys_p[-1]

        ax.scatter(x_start, y_start, s=40, color="green", zorder=5)
        ax.scatter(x_end, y_end, s=40, color="red", zorder=5)

        ax.text(
            x_start, y_start,
            "  Start",
            color="green",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=6,
        )
        ax.text(
            x_end, y_end,
            "  End",
            color="red",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=6,
        )

        # Takeoff / landing point
        ax.scatter(tx, ty, s=50, color="magenta", marker="X", zorder=6)
        ax.text(
            tx, ty,
            "  TO/Land (mid of longest side)",
            color="magenta",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=7,
        )

        # Transit legs: TO -> Start, End -> TO
        ax.plot([tx, x_start], [ty, y_start], linestyle="--", color="gray", linewidth=1)
        ax.plot([x_end, tx], [y_end, ty], linestyle="--", color="gray", linewidth=1)

        # Lengths in meters and feet
        survey_ft = survey_len_m * FEET_PER_METER
        transit_ft = transit_len_m * FEET_PER_METER
        total_ft = total_len_m * FEET_PER_METER
        spacing_ft = spacing_m * FEET_PER_METER
        longest_side_ft = longest_side_len_m * FEET_PER_METER

        # Info box
        ax.text(
            0.02,
            0.98,
            "Spacing: {:.2f} m ({:.1f} ft)\n"
            "Best angle: {:.1f}°\n"
            "Longest side: {:.2f} m ({:.1f} ft)\n"
            "Survey length: {:.2f} m ({:.1f} ft)\n"
            "Transit length (TO legs): {:.2f} m ({:.1f} ft)\n"
            "Total mission length: {:.2f} m ({:.1f} ft)".format(
                spacing_m, spacing_ft,
                best_angle,
                longest_side_len_m, longest_side_ft,
                survey_len_m, survey_ft,
                transit_len_m, transit_ft,
                total_len_m, total_ft,
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No valid path found.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )

    plt.show()


# ------------ 6. Main ------------

def main():
    poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon) = load_polygon_from_plan_in_meters(PLAN_FILE)

    minx, miny, maxx, maxy = poly_m.bounds
    diag_m = math.hypot(maxx - minx, maxy - miny)

    # Takeoff / landing point = midpoint of longest side
    tx, ty, longest_side_len_m = longest_side_midpoint(poly_m)

    print("Reference origin (approx polygon center):")
    print(f"  lat0 = {lat0:.8f}°, lon0 = {lon0:.8f}°")
    print("Conversion factors around this latitude:")
    print(f"  1° latitude  ≈ {m_per_deg_lat:.3f} m")
    print(f"  1° longitude ≈ {m_per_deg_lon:.3f} m")
    print(f"Polygon diagonal ≈ {diag_m:.3f} m  ({diag_m * FEET_PER_METER:.1f} ft)")
    print(f"Longest side    ≈ {longest_side_len_m:.3f} m  ({longest_side_len_m * FEET_PER_METER:.1f} ft)")
    print(f"Takeoff/Landing point (local meters): ({tx:.3f}, {ty:.3f})")

    # Ask user for spacing in meters
    while True:
        try:
            spacing_m = float(input("Enter track separation in meters: "))
            if spacing_m <= 0:
                print("Spacing must be > 0")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    print("Searching for best angle based on TOTAL mission length (survey + TO legs)...")

    best_angle, best_path_m, survey_len_m, transit_len_m, total_len_m = find_best_angle(
        poly_m,
        spacing_m,
        (tx, ty),
        angle_step_deg=1.0,
    )

    if best_angle is None or best_path_m is None:
        print("No valid survey path found for this spacing.")
        return

    print("\nRESULTS")
    print(f"  Spacing: {spacing_m:.3f} m  ({spacing_m * FEET_PER_METER:.2f} ft)")
    print(f"  Best angle (deg): {best_angle:.2f}")
    print(f"  Survey length: {survey_len_m:.3f} m  ({survey_len_m * FEET_PER_METER:.2f} ft)")
    print(f"  Transit (TO legs): {transit_len_m:.3f} m  ({transit_len_m * FEET_PER_METER:.2f} ft)")
    print(f"  TOTAL mission length: {total_len_m:.3f} m  ({total_len_m * FEET_PER_METER:.2f} ft)")

    plot_best_path(
        poly_m,
        best_angle,
        best_path_m,
        survey_len_m,
        transit_len_m,
        total_len_m,
        spacing_m,
        (tx, ty),
        longest_side_len_m,
    )


if __name__ == "__main__":
    main()
