from fastkml import kml
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

KML_FILE = "poly.kml"

def kml_poly(path):
    with open(path, "rt", encoding="utf-8") as f:
        raw = f.read()
    k = kml.KML()
    k.from_string(raw.encode("utf-8"))
    polygon_geom = None
    for feature in k.features:       
        for placemark in feature.features:   
            geom = getattr(placemark, "geometry", None)
            if geom is not None and geom.geom_type == "Polygon":
                polygon_geom = geom
                break
        if polygon_geom:
            break

    if polygon_geom is None:
        raise ValueError("No Polygon geometry found in KML")

    return polygon_geom.exterior.coords[:]

coords = list(kml_poly(KML_FILE))
lons  = [c[0] for c in coords]
lats  = [c[1] for c in coords]


if coords[0] != coords[-1]:
    lons.append(lons[0])
    lats.append(lats[0])

plt.figure(figsize=(7,7))
plt.plot(lons, lats, linewidth=2, color="blue")
plt.fill(lons, lats, alpha=0.3)  
plt.title("Polygon from KML")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.axis("equal")
plt.savefig("shape.png")
# plt.show()