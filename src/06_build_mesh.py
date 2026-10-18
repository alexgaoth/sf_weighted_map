"""The sheet that warps: SF's land as a triangle mesh, the lines drawn on it, and the grid
the page's time field lives on.

Land is the city's shoreline polygon — the mainland plus Treasure and Yerba Buena Islands,
the only island a car can reach — less the lakes it counts as land. Alcatraz, the
Farallones and the scraps of SF-owned landfill across the bay are dropped.

Writes data/mesh.npz.
"""
import json
import sys

import geopandas as gpd
import numpy as np
import shapely
from scipy.ndimage import binary_dilation
from scipy.spatial import Delaunay

sys.path.insert(0, "src")
import geo

MESH_M = 70.0          # triangle size, metres; ~30k vertices
CELL = 40.0            # time grid, metres
LINE_SEG = 60.0        # drawn lines are cut this fine so they bend with the warp


def local(geom):
    return shapely.transform(geom, lambda c: np.column_stack(geo.to_xy(c[:, 0], c[:, 1])))


shore = gpd.read_file("data/raw/shoreline.geojson").geometry.iloc[0]
parts = sorted(shapely.get_parts(local(shore)), key=lambda p: -p.area)
land = shapely.union_all(parts[:2])            # mainland, then TI + YBI (2.3 km²)
assert parts[0].area > 100e6 and 2e6 < parts[1].area < 3e6, [p.area for p in parts[:3]]

# lakes: closed ways, and multipolygon relations stitched from their member lines
water = []
for e in json.load(open("data/raw/osm_water.json"))["elements"]:
    if e["type"] == "way" and len(e.get("geometry", [])) >= 4:
        c = np.array([(g["lon"], g["lat"]) for g in e["geometry"]])
        water.append(shapely.Polygon(np.column_stack(geo.to_xy(c[:, 0], c[:, 1]))))
    elif e["type"] == "relation":
        segs = []
        for m in e.get("members", []):
            if m.get("role") == "outer" and m.get("geometry"):
                c = np.array([(g["lon"], g["lat"]) for g in m["geometry"]])
                segs.append(shapely.LineString(np.column_stack(geo.to_xy(c[:, 0], c[:, 1]))))
        water += list(shapely.get_parts(shapely.polygonize(segs)))
water = [w.buffer(0) for w in water if w.is_valid or w.buffer(0).area > 0]
lakes = shapely.union_all([w for w in water if w.area > 8000])   # Stow Lake and up
land = land.difference(lakes).buffer(0)
# Piers, breakwaters and spits narrower than ~90 m are opened away. Their far ends are
# hundreds of metres of walking from any street, and drawn, each became a spike — the
# Embarcadero a comb of them. They are still in the router; they are just not the sheet.
OPEN = 45.0
before = land.area
land = land.buffer(-OPEN).buffer(OPEN).intersection(land)
land = shapely.union_all([p for p in shapely.get_parts(land) if p.area > 0.2e6])
print(f"land {land.area / 1e6:.2f} km² after {lakes.area / 1e6:.2f} km² of lakes "
      f"and {(before - land.area) / 1e6:.2f} km² of piers and spits")

# ---- mesh ---------------------------------------------------------------------------
outline = shapely.segmentize(shapely.simplify(land, 4.0), MESH_M)
bpts = np.vstack([np.asarray(g.coords) for g in shapely.get_parts(outline.boundary)])
x0, y0, x1, y1 = land.bounds
gx, gy = np.meshgrid(np.arange(x0, x1, MESH_M), np.arange(y0, y1, MESH_M * 0.866))
gx[1::2] += MESH_M / 2
inner = np.column_stack([gx.ravel(), gy.ravel()])
shapely.prepare(land)
inner = inner[shapely.contains_xy(land, inner[:, 0], inner[:, 1])]
inner = inner[shapely.distance(shapely.points(inner), outline.boundary) > MESH_M * 0.45]
pts = np.unique(np.vstack([bpts, inner]).round(2), axis=0)
tri = Delaunay(pts)
tv = pts[tri.simplices]
cen = tv.mean(axis=1)
longest = np.max([np.hypot(*(tv[:, i] - tv[:, (i + 1) % 3]).T) for i in range(3)], axis=0)
keep = shapely.contains_xy(land, cen[:, 0], cen[:, 1]) & (longest < MESH_M * 2.3)
tris = tri.simplices[keep]
used = np.unique(tris)
remap = np.full(len(pts), -1)
remap[used] = np.arange(len(used))
pts, tris = pts[used], remap[tris]
assert len(pts) < 65536, "mesh too large for uint16 indices"


# ---- lines --------------------------------------------------------------------------
def segments(geom):
    out = []
    for part in shapely.get_parts(shapely.segmentize(geom, LINE_SEG)):
        c = np.asarray(part.coords)
        if len(c) > 1:
            out.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(out)


coast = segments(shapely.simplify(land, 3.0).boundary)
hoods = gpd.read_file("data/raw/analysis_neighborhoods.geojson")
inner_lines = shapely.union_all([local(g).boundary for g in hoods.geometry])
# only where two neighbourhoods meet, inside the drawn land — the coast is drawn already
inner_lines = shapely.simplify(inner_lines.intersection(land.buffer(-25)), 4.0)
hood = segments(shapely.line_merge(inner_lines))

# ---- the time grid ------------------------------------------------------------------
gx0, gy0 = np.floor(x0 / CELL) * CELL - 3 * CELL, np.floor(y0 / CELL) * CELL - 3 * CELL
nx_, ny_ = int(np.ceil((x1 - gx0) / CELL)) + 4, int(np.ceil((y1 - gy0) / CELL)) + 4
cx, cy = np.meshgrid(gx0 + CELL * np.arange(nx_), gy0 + CELL * np.arange(ny_))
on = shapely.contains_xy(land, cx.ravel(), cy.ravel()).reshape(ny_, nx_)
# a mesh vertex samples the four cells around it, so every cell within two of land
# needs a value too
near = binary_dilation(on, iterations=2)
mask = np.where(on, 1, np.where(near, 2, 0)).astype(np.uint8)

np.savez_compressed("data/mesh.npz", x=pts[:, 0], y=pts[:, 1], tris=tris.astype(np.int32),
                    outline=coast, hood=hood, grid=np.array([gx0, gy0, CELL, nx_, ny_]),
                    land=mask.ravel(), land_wkb=np.frombuffer(shapely.to_wkb(land), np.uint8))
print(f"mesh {len(pts):,} vertices / {len(tris):,} triangles")
print(f"lines: coast {len(coast):,}, neighbourhoods {len(hood):,} segments")
print(f"grid {nx_} x {ny_} at {CELL:.0f} m: {int((mask == 1).sum()):,} land cells, "
      f"{int((mask > 0).sum()):,} active")
