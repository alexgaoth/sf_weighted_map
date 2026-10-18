"""The named places you can search for and click: DataSF's 117 "SF Find Neighborhoods",
each at a point inside its land, plus landmarks, stations and campuses at their OSM
positions (checked against data/raw/osm_places.json and the agents' OSM pull, 2026-10-17).
-> data/places.csv
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import shapely.ops

sys.path.insert(0, "src")
import geo

LANDMARKS = [
    ("Union Square", 37.78799, -122.40752), ("Ferry Building", 37.79554, -122.39343),
    ("Salesforce Transit Center", 37.78916, -122.39678), ("Oracle Park", 37.77859, -122.38927),
    ("Chase Center", 37.76797, -122.38754), ("Coit Tower", 37.80239, -122.40582),
    ("Pier 39", 37.80867, -122.40982), ("Fisherman's Wharf", 37.80800, -122.41770),
    ("Ghirardelli Square", 37.80587, -122.42295), ("Lombard Street", 37.80211, -122.41881),
    ("Palace of Fine Arts", 37.80292, -122.44839), ("Golden Gate Bridge toll plaza", 37.80750, -122.47500),
    ("Fort Point", 37.81051, -122.47710), ("Baker Beach", 37.79315, -122.48474),
    ("Sutro Baths", 37.78014, -122.51380), ("Ocean Beach", 37.76032, -122.50863),
    ("Conservatory of Flowers", 37.77261, -122.46022), ("Japanese Tea Garden", 37.77034, -122.47005),
    ("Cal Academy of Sciences", 37.76949, -122.46598), ("Dutch Windmill", 37.77044, -122.50941),
    ("Painted Ladies", 37.77622, -122.43279), ("Twin Peaks summit", 37.75246, -122.44759),
    ("Mount Davidson", 37.73810, -122.45370), ("Dolores Park", 37.75960, -122.42690),
    ("City Hall", 37.77927, -122.41924), ("Transamerica Pyramid", 37.79517, -122.40279),
    ("Dragon Gate", 37.79069, -122.40559), ("Moscone Center", 37.78223, -122.40173),
    ("Powell St BART", 37.78471, -122.40720), ("Montgomery St BART", 37.78923, -122.40147),
    ("Embarcadero BART", 37.79286, -122.39691), ("Civic Center BART", 37.77936, -122.41397),
    ("16th St Mission BART", 37.76507, -122.41968), ("24th St Mission BART", 37.75226, -122.41845),
    ("Glen Park BART", 37.73310, -122.43381), ("Balboa Park BART", 37.72087, -122.44671),
    ("4th & King Caltrain", 37.77636, -122.39494), ("22nd St Caltrain", 37.75729, -122.39240),
    ("SF State", 37.72410, -122.47990), ("Stonestown", 37.72745, -122.47490),
    ("City College", 37.72300, -122.45118), ("UCSF Parnassus", 37.76439, -122.45839),
    ("UCSF Mission Bay", 37.76433, -122.38887), ("SF General Hospital", 37.75560, -122.40480),
    ("SF Zoo", 37.73290, -122.50232), ("Presidio Main Post", 37.80188, -122.45623),
    ("Yerba Buena Island", 37.81120, -122.36588),
]

land = shapely.from_wkb(np.load("data/mesh.npz")["land_wkb"].tobytes())
hoods = gpd.read_file("data/raw/find_neighborhoods.geojson")
rows = []
for name, geom in zip(hoods["name"], hoods.geometry):
    poly = shapely.transform(geom, lambda c: np.column_stack(geo.to_xy(c[:, 0], c[:, 1])))
    on_land = poly.intersection(land)
    if on_land.is_empty:
        print(f"  {name}: no land, dropped")
        continue
    # the largest piece's inside point, so a neighbourhood never sits in its own lake
    big = max(shapely.get_parts(on_land), key=lambda p: p.area)
    p = big.point_on_surface()
    lon, lat = geo.to_ll(p.x, p.y)
    rows.append((name, float(lat), float(lon), "neighborhood"))
names = {r[0].lower() for r in rows}
for name, lat, lon in LANDMARKS:
    if name.lower() not in names:
        rows.append((name, lat, lon, "landmark"))
df = pd.DataFrame(rows, columns=["label", "lat", "lon", "kind"])
# a point on the sand can sit just past the city's shoreline; pull it onto land
x, y = geo.to_xy(df.lon.values, df.lat.values)
shapely.prepare(land)
for i in np.flatnonzero(~shapely.contains_xy(land, x, y)):
    q = shapely.ops.nearest_points(land, shapely.Point(x[i], y[i]))[0]
    q = shapely.Point(q.x + np.sign(x[i] - q.x) * -5, q.y + np.sign(y[i] - q.y) * -5)
    lon, lat = geo.to_ll(q.x, q.y)
    print(f"  {df.label[i]}: {np.hypot(x[i] - q.x, y[i] - q.y):.0f} m offshore, moved onto land")
    df.loc[i, ["lat", "lon"]] = float(lat), float(lon)
x, y = geo.to_xy(df.lon.values, df.lat.values)
assert shapely.contains_xy(land, x, y).all()
df.to_csv("data/places.csv", index=False)
print(f"{len(df)} places: {int((df.kind == 'neighborhood').sum())} neighborhoods, "
      f"{int((df.kind == 'landmark').sum())} landmarks")
