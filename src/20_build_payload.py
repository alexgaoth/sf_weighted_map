"""Freeze everything the page routes on into data/payload.npz: the graph as CSR with its
costs quantised to whole deciseconds, the car profiles, the mesh, the line layers and the
time grid's land mask. 21_measure.py and the page both read exactly this.
"""
import sys

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, "src")
import costs
import geo

g = np.load("data/graph.npz")
tr = np.load("data/traffic.npz")
mesh = np.load("data/mesh.npz")

N = len(g["x"])
u, v = g["u"], g["v"]
order = np.lexsort((v, u))                     # CSR by source
u, v = u[order], v[order]


def ds(sec):
    """seconds -> uint16 deciseconds, never 0 (a zero-cost edge reads as no edge to scipy),
    65535 for an edge the mode may not use"""
    sec = sec[order]
    q = np.where(sec < 0, 65535, np.clip(np.round(sec * 10), 1, 65534))
    return q.astype(np.uint16)


cost_walk, cost_bike, cost_car = ds(g["walk"]), ds(g["bike"]), ds(g["car"])
car_prof = tr["edge_prof"][order].astype(np.uint16)
off = np.zeros(N + 1, np.uint32)
np.add.at(off, u + 1, 1)
off = np.cumsum(off).astype(np.uint32)
node_q = geo.quantize(g["x"], g["y"])

# Street lines for drawing, one segment per undirected edge. Major is the arterial network
# a driver would name; minor is every other street a car can use.
a, b = np.minimum(u, v), np.maximum(u, v)
cls = g["cls"][order]
car_ok = cost_car != 65535
seen = {}
for i in np.flatnonzero(car_ok):
    k = (a[i], b[i])
    seen[k] = min(seen.get(k, 9), cls[i])
# Only SF is drawn: a street segment is kept if both ends are on its land (or within
# 30 m of it — the edge of a pier, a street on the shoreline). Daly City's streets, the
# bridge decks over open water and Marin stay in the router and off the map.
land = shapely.from_wkb(mesh["land_wkb"].tobytes())
shore = land.buffer(30.0)
shapely.prepare(shore)
on_land = shapely.contains_xy(shore, g["x"], g["y"])
major = [k for k, c in seen.items() if c <= 2 and on_land[k[0]] and on_land[k[1]]]
minor = [k for k, c in seen.items() if c > 2 and on_land[k[0]] and on_land[k[1]]]


def seg_q(pairs):
    p = np.array(pairs, np.int64).reshape(-1, 2)
    return node_q[p.ravel()].reshape(-1, 2)


# Every grid cell a drawn vertex can sample must hold a time. A vertex reads the four
# cells around it, so mark those for every mesh and line vertex — a pier narrower than a
# cell has no cell centre on land, and its tip would otherwise interpolate into nothing.
gx0, gy0, cell, gnx, gny = mesh["grid"]
gnx, gny = int(gnx), int(gny)
mask = mesh["land"].astype(np.uint8).reshape(gny, gnx).copy()
lines_xy = [np.column_stack([mesh["x"], mesh["y"]]), mesh["outline"].reshape(-1, 2),
            mesh["hood"].reshape(-1, 2)]
for pairs in (major, minor):
    p_ = np.array(pairs, np.int64).ravel()
    lines_xy.append(np.column_stack([g["x"][p_], g["y"][p_]]))
pts = np.vstack(lines_xy)
i0 = np.clip(np.floor((pts[:, 0] - gx0) / cell).astype(int), 0, gnx - 2)
j0 = np.clip(np.floor((pts[:, 1] - gy0) / cell).astype(int), 0, gny - 2)
for di in (0, 1):
    for dj in (0, 1):
        sel = mask[j0 + dj, i0 + di] == 0
        mask[(j0 + dj)[sel], (i0 + di)[sel]] = 2
print(f"active cells {int((mask > 0).sum()):,} ({int((mask == 1).sum()):,} on land)")

places = pd.read_csv("data/places.csv")
pads = pd.read_csv("data/heliports.csv")
px, py = geo.to_xy(places.lon.values, places.lat.values)
hx, hy = geo.to_xy(pads.lon.values, pads.lat.values)
meta = costs.model()

np.savez_compressed(
    "data/payload.npz",
    q=np.array([geo.XMIN, geo.YMIN, geo.QSTEP]),
    node_q=node_q, node_snap=g["snap"].astype(np.uint8), off=off, tgt=v.astype(np.uint32),
    cost_walk=cost_walk, cost_bike=cost_bike, cost_car=cost_car, car_prof=car_prof,
    prof=tr["prof"].astype(np.uint16),
    grid=np.array([mesh["grid"][0], mesh["grid"][1], mesh["grid"][2], mesh["grid"][3], mesh["grid"][4]]),
    land=mask.ravel(),
    mesh_q=geo.quantize(mesh["x"], mesh["y"]), mesh_idx=mesh["tris"].astype(np.uint16).ravel(),
    line_outline=geo.quantize(*mesh["outline"].reshape(-1, 2).T),
    line_hood=geo.quantize(*mesh["hood"].reshape(-1, 2).T),
    line_major=seg_q(major), line_minor=seg_q(minor),
    walk=np.array(meta["walk"]),
    heli=np.array([meta["heli"][k] for k in ("dep", "fix", "cruise", "arr")]),
    pads=np.column_stack([hx, hy]),
    place_xy=np.column_stack([px, py]),
)
print(f"nodes {N:,}  edges {len(u):,}  profiles {tr['prof'].shape[0]}")
print(f"mesh {len(mesh['x']):,} vertices / {len(mesh['tris']):,} triangles")
print(f"streets: {len(major):,} major + {len(minor):,} minor segments")
print(f"places {len(places)}  heliports {len(pads)}")
