import json
import math
import random
import os


# ----------------- Config -----------------

TARGET_AREA_M2 = 300000.0           # required polygon area in square meters
OUTPUT_FOLDER = "generated_polygons"   # folder to save .plan files


# Base reference position (around which polygon will be created)
BASE_LAT = 28.41341081756747
BASE_LON = 77.522344243784


# ----------------- Helpers -----------------

def meters_per_degree(lat_deg: float):
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2 * lat_rad)
        + 1.175 * math.cos(4 * lat_rad)
        - 0.0023 * math.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3 * lat_rad)
        + 0.118 * math.cos(5 * lat_rad)
    )
    return m_per_deg_lat, m_per_deg_lon


def polygon_area_xy(points_xy):
    area = 0.0
    n = len(points_xy)
    for i in range(n):
        x1, y1 = points_xy[i]
        x2, y2 = points_xy[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def generate_star_polygon_xy(n_vertices: int, target_area: float):
    if n_vertices < 3:
        raise ValueError("Polygon must have at least 3 vertices")

    base_r = math.sqrt(target_area / math.pi)
    angles = [2 * math.pi * i / n_vertices for i in range(n_vertices)]

    radii = []
    for i in range(n_vertices):
        rand_factor = 0.6 + 0.8 * random.random()
        if i % 2 == 0:
            r = base_r * rand_factor
        else:
            r = base_r * rand_factor * 0.7
        radii.append(r)

    pts = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]

    area_initial = polygon_area_xy(pts)
    scale = math.sqrt(target_area / area_initial)
    pts_scaled = [(x * scale, y * scale) for (x, y) in pts]

    return pts_scaled


def xy_to_latlon(points_xy, base_lat, base_lon):
    m_per_deg_lat, m_per_deg_lon = meters_per_degree(base_lat)
    latlon = []
    for x, y in points_xy:
        dlat = y / m_per_deg_lat
        dlon = x / m_per_deg_lon
        latlon.append([base_lat + dlat, base_lon + dlon])
    return latlon


# ----------------- Main logic -----------------

def main():
    while True:
        try:
            n = int(input("Enter number of sides (vertices) for the polygon: "))
            if n < 3:
                print("Number of sides must be >= 3")
                continue
            break
        except ValueError:
            print("Enter a valid integer.")

    pts_xy = generate_star_polygon_xy(n, TARGET_AREA_M2)
    area_xy = polygon_area_xy(pts_xy)
    polygon_latlon = xy_to_latlon(pts_xy, BASE_LAT, BASE_LON)

    planned_home = [BASE_LAT, BASE_LON, 195]

    plan_dict = {
        "fileType": "Plan",
        "geoFence": {
            "circles": [],
            "polygons": [
                {
                    "inclusion": True,
                    "polygon": polygon_latlon,
                    "version": 1
                }
            ],
            "version": 2
        },
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 15,
            "firmwareType": 3,
            "globalPlanAltitudeMode": 1,
            "hoverSpeed": 5,
            "items": [],
            "plannedHomePosition": planned_home,
            "vehicleType": 2,
            "version": 2
        },
        "rallyPoints": {
            "points": [],
            "version": 2
        },
        "version": 1
    }

    # ---------- NEW: ensure folder exists ----------------
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filename = f"polygon_{n}_sides.plan"
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, indent=4)

    print("\n----------------------------------")
    print(f"Polygon generated with {n} sides")
    print(f"Area ≈ {area_xy:.2f} m²")
    print(f"File saved at: {filepath}")
    print("----------------------------------")
    print("Coordinates (lat, lon):")
    for lat, lon in polygon_latlon:
        print(f"  [{lat:.8f}, {lon:.8f}]")


if __name__ == "__main__":
    main()
