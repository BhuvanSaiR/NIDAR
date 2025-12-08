import json
import math
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely import affinity
import custom_survey as cs

PLAN_FILE = "generated_polygons/polygon_3_sides.plan"   # change if needed

FEET_PER_METER = 3.280839895  # approx conversion


# ---------------- Ploting everything ----------------

def plot_all(poly_m,
             poly1_m,
             poly2_m,
             cut_line,
             to_point,
             separation_m,
             # full survey
            #  angle_full,
            #  path_full,
            #  survey_len_full,
            #  transit_len_full,
            #  total_len_full,
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
    # if path_full is not None:
    #     xf, yf = path_full.xy
    #     ax.plot(xf, yf, color="gray", linewidth=1.2,
    #             label=f"Full survey (angle={angle_full:.1f}°)")
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

    # plot_to_legs(path_full, "gray")
    plot_to_legs(path1, "red")
    plot_to_legs(path2, "orange")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Full vs split equal-area surveys (meters)")

    separation_ft = separation_m * FEET_PER_METER
    longest_side_ft = longest_side_len_m * FEET_PER_METER

    # full_survey_ft = survey_len_full * FEET_PER_METER if survey_len_full is not None else 0
    # full_transit_ft = transit_len_full * FEET_PER_METER if transit_len_full is not None else 0
    # full_total_ft = total_len_full * FEET_PER_METER if total_len_full is not None else 0

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
        # f"  Best angle: {angle_full:.1f}°\n"
        # f"  Survey length: {survey_len_full:.1f} m ({full_survey_ft:.1f} ft)\n"
        # f"  TO legs: {transit_len_full:.1f} m ({full_transit_ft:.1f} ft)\n"
        # f"  TOTAL: {total_len_full:.1f} m ({full_total_ft:.1f} ft)\n\n"
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
    poly_m, (lat0, lon0, m_per_deg_lat, m_per_deg_lon) = cs.load_polygon_from_plan_in_meters(PLAN_FILE)

    minx, miny, maxx, maxy = poly_m.bounds
    diag_m = math.hypot(maxx - minx, maxy - miny)

    # Takeoff / landing = midpoint of longest side
    tx, ty, longest_side_len_m = cs.longest_side_midpoint(poly_m)

    print("Reference origin (approx polygon center):")
    print(f"  lat0 = {lat0:.8f}°, lon0 = {lon0:.8f}°")
    print("Conversion factors around this latitude:")
    print(f"  1° latitude  ≈ {m_per_deg_lat:.3f} m")
    print(f"  1° longitude ≈ {m_per_deg_lon:.3f} m")
    print(f"Polygon diagonal ≈ {diag_m:.3f} m  ({diag_m * FEET_PER_METER:.1f} ft)")
    print(f"Longest side     ≈ {longest_side_len_m:.3f} m  ({longest_side_len_m * FEET_PER_METER:.1f} ft)")
    print(f"TO/Land (meters) = ({tx:.3f}, {ty:.3f})")

    # Equal-area split of the full polygon (normal angle_rad = 0 => vertical-ish cut)
    poly1_m, poly2_m, cut_line = cs.compute_equal_area_split(poly_m, angle_rad=0.0)
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
    # print("\nSearching best angle for FULL polygon survey...")
    # angle_full, path_full, survey_len_full, transit_len_full, total_len_full = cs.find_best_angle_for_region(
    #     poly_m, separation_m, (tx, ty), angle_step_deg=1.0
    # )

    # 2) Best survey for each half, with its own angle
    print("Searching best angle for HALF 1 survey...")
    angle_half1, path1, survey_len1, transit_len1, total_len1 = cs.find_best_angle_for_region(
        poly1_m, separation_m, (tx, ty), angle_step_deg=1.0
    )

    print("Searching best angle for HALF 2 survey...")
    angle_half2, path2, survey_len2, transit_len2, total_len2 = cs.find_best_angle_for_region(
        poly2_m, separation_m, (tx, ty), angle_step_deg=1.0
    )

    # print("\nFULL survey results:")
    # print(f"  Best angle: {angle_full:.2f}°")
    # print(f"  Survey length: {survey_len_full:.2f} m  ({survey_len_full * FEET_PER_METER:.2f} ft)")
    # print(f"  TO legs: {transit_len_full:.2f} m  ({transit_len_full * FEET_PER_METER:.2f} ft)")
    # print(f"  TOTAL: {total_len_full:.2f} m  ({total_len_full * FEET_PER_METER:.2f} ft)")

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
        # angle_full,
        # path_full,
        # survey_len_full,
        # transit_len_full,
        # total_len_full,
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
