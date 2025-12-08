import json
import math
import numpy as np

from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely import affinity
import custom_survey as cs
# --------------- CONFIG ----------------

PLAN_FILE = "generated_polygons/polygon_16_sides.plan"  # change as needed
OUTPUT_FILE = "split_waypoints.json"



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
    poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon) = cs.load_polygon_from_plan_in_meters(PLAN_FILE)

    # Takeoff / landing = midpoint of longest side of FULL polygon
    tx, ty, longest_side_len_m = cs.longest_side_midpoint(poly_m)
    takeoff_xy = (tx, ty)

    print(f"Loaded polygon from: {PLAN_FILE}")
    print(f"Takeoff/Land local XY (m): ({tx:.3f}, {ty:.3f})")
    print(f"Longest side length: {longest_side_len_m:.3f} m")
    print(f"Total area (m^2): {poly_m.area:.3f}")

    # Equal-area split of the full polygon (normal angle_rad = 0 => vertical-ish cut)
    poly1_m, poly2_m, cut_line = cs.compute_equal_area_split(poly_m, angle_rad=0.0)
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
    angle_half1, path1, survey_len1, transit_len1, total_len1 = cs.find_best_angle_for_region(
        poly1_m, separation_m, takeoff_xy, angle_step_deg=1.0
    )

    if angle_half1 is None or path1 is None:
        print(f"No valid survey path for HALF 1 with x = {separation_m} m (too narrow).")
        return

    print(f"HALF 1: angle={angle_half1:.2f}°, survey_len={survey_len1:.2f} m, total≈{total_len1:.2f} m")

    # For HALF 2
    print("\nSearching best angle for HALF 2 survey...")
    angle_half2, path2, survey_len2, transit_len2, total_len2 = cs.find_best_angle_for_region(
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
