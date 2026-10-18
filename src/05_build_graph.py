"""OSM ways -> one directed street graph, costed for walking, cycling and driving.

Nodes are every intersection plus enough shape points that no edge is longer than ~50 m,
so the time field the page draws has a value every half-block. Each edge is costed by
walking its geometry in ~10 m pieces against the DEM, uphill and downhill separately.
Signals and stop signs come from SFMTA's own inventories (DataSF), which are complete
where OSM's tags are not, and cost time on the edge that arrives at them.

Writes data/graph.npz.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
import costs
import geo

MAX_EDGE, PIECE = 50.0, 10.0

# never part of any network here: indoor, unbuilt, or not a way at all
DROP = {"construction", "proposed", "abandoned", "razed", "platform", "bus_stop", "corridor",
        "elevator", "raceway", "rest_area", "services", "emergency_bay", "escape", "busway",
        "via_ferrata", "disused", "no", "planned"}
# Sidewalks, crossings and parking aisles run alongside a street that is already walkable.
# They are where most of SF's highway ways are, and dropping them halves the graph without
# moving a walking time by more than the width of a street.
DROP_FOOTWAY = {"sidewalk", "crossing", "access_aisle", "traffic_island"}
DROP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
CAR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
       "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
       "residential", "living_street", "service", "road"}
NO_FOOT = {"motorway", "motorway_link"}
FOOT_ONLY = {"footway", "pedestrian", "steps", "path", "track", "bridleway"}
# posted speed in mph where a way carries none; SF's default is 25
DEFAULT_MPH = {"motorway": 55, "motorway_link": 35, "trunk": 35, "trunk_link": 30,
               "primary": 30, "primary_link": 25, "secondary": 25, "secondary_link": 25,
               "tertiary": 25, "tertiary_link": 25, "unclassified": 25, "residential": 25,
               "living_street": 10, "service": 12, "road": 20}
# road class: 0 freeway, 1 major arterial, 2 minor arterial, 3 local
CLASS = {"motorway": 0, "motorway_link": 0, "trunk_link": 0, "trunk": 1, "primary": 1,
         "primary_link": 1, "secondary": 2, "secondary_link": 2, "tertiary": 2,
         "tertiary_link": 2}
NO = {"no", "private"}
YES = {"yes", "designated", "permissive"}


def mph(tag):
    if not tag:
        return None
    t = tag.split(";")[0].strip().lower()
    try:
        return float(t[:-3]) if t.endswith("mph") else float(t) / 1.609344   # bare = km/h
    except ValueError:
        return None


def access(t):
    """(car fwd, car bwd, bike fwd, bike bwd, walk) for one way."""
    hw = t["highway"]
    acc = t.get("access")
    vehicle = t.get("vehicle", acc)
    motor = t.get("motor_vehicle", vehicle)
    car = hw in CAR and t.get("motorcar", motor) not in NO and t.get("area") != "yes"
    if hw == "service" and t.get("service") in DROP_SERVICE:
        car = False
    bicycle = t.get("bicycle", vehicle)
    if hw in NO_FOOT or hw == "steps":
        bike = False
    elif hw in ("footway", "pedestrian", "bridleway"):
        # riding on the sidewalk is illegal for adults in SF; only signed paths allow it
        bike = bicycle in YES
    else:
        bike = bicycle not in NO and bicycle != "dismount"
    foot = t.get("foot", acc)
    # surface trunks (19th Avenue, Van Ness) are walkable; freeways and expressways are not
    walk = hw not in NO_FOOT and foot not in NO and t.get("expressway") != "yes"
    if hw == "trunk_link" and foot not in YES:
        walk = False
    ow = t.get("oneway", "no")
    implied = hw in ("motorway", "motorway_link") or t.get("junction") in ("roundabout", "circular")
    fwd_only = ow in ("yes", "true", "1") or (implied and ow not in ("no", "-1"))
    bwd_only = ow in ("-1", "reverse")
    cf, cb = car and not bwd_only, car and not fwd_only
    contra = t.get("oneway:bicycle") == "no" or str(t.get("cycleway", "")).startswith("opposite") \
        or any(t.get(f"cycleway:{s}:oneway") == "no" for s in ("left", "right", "both"))
    if hw in FOOT_ONLY:
        bf = bb = bike
    else:
        bf = bike and (not bwd_only or contra)
        bb = bike and (not fwd_only or contra)
    return cf, cb, bf, bb, walk


osm = json.load(open("data/raw/osm.json"))
coord, ntag, ways = {}, {}, []
for e in osm["elements"]:
    if e["type"] == "node":
        coord[e["id"]] = (e["lon"], e["lat"])
        if e.get("tags"):
            ntag[e["id"]] = e["tags"]
for e in osm["elements"]:
    if e["type"] != "way" or "highway" not in e.get("tags", {}):
        continue
    t = e["tags"]
    hw = t["highway"]
    if hw in DROP or t.get("footway") in DROP_FOOTWAY or t.get("indoor") == "yes":
        continue
    if hw == "service" and t.get("service") in DROP_SERVICE and t.get("foot") not in YES:
        continue
    acc = access(t)
    if not any(acc):
        continue
    nodes = [n for n in e["nodes"] if n in coord]
    if len(nodes) < 2:
        continue
    ways.append((e["id"], nodes, np.array([coord[n] for n in nodes]), t, acc))
print(f"{len(ways):,} usable ways")

# A node is kept if it ends a way, is shared by ways, or carries a control.
uses = defaultdict(int)
for _, nodes, _, _, _ in ways:
    for k, n in enumerate(nodes):
        uses[n] += 2 if k in (0, len(nodes) - 1) else 1
keep_osm = {n for n, c in uses.items() if c >= 2}

# the 2023 lidar where it has ground, the coarser default mosaic past the county line
dem = rasterio.open("data/raw/dem_lidar.tif")
band = dem.read(1).astype(np.float64)
base = rasterio.open("data/raw/dem.tif").read(1).astype(np.float64)
assert base.shape == band.shape
band = np.where(band <= -9000, base, band)
band[band <= -9000] = np.nan
inv = ~dem.transform
to_dem = Transformer.from_crs(4326, dem.crs, always_xy=True)


def elevation(lon, lat):
    """Bilinear DEM lookup, metres, from lon/lat."""
    ex, ey = to_dem.transform(np.asarray(lon, float), np.asarray(lat, float))
    c, r = inv * (ex, ey)
    c, r = np.asarray(c) - 0.5, np.asarray(r) - 0.5
    c0 = np.clip(np.floor(c).astype(int), 0, band.shape[1] - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, band.shape[0] - 2)
    fc, fr = np.clip(c - c0, 0, 1), np.clip(r - r0, 0, 1)
    z = (band[r0, c0] * (1 - fc) * (1 - fr) + band[r0, c0 + 1] * fc * (1 - fr)
         + band[r0 + 1, c0] * (1 - fc) * fr + band[r0 + 1, c0 + 1] * fc * fr)
    # the DEM ends at the county line; past it (Daly City) the ground is flat enough
    return np.nan_to_num(z, nan=0.0)


# ---- nodes --------------------------------------------------------------------------
node_id = {}            # osm id, or (way, a, j) for a shape point we invent
nx, ny, nz, nsnap = [], [], [], []


def node(key, lon, lat):
    i = node_id.get(key)
    if i is None:
        i = node_id[key] = len(nx)
        x, y = geo.to_xy(lon, lat)
        nx.append(float(x)); ny.append(float(y)); nz.append(float(elevation(lon, lat)))
        nsnap.append(0)
    return i


# ---- edges --------------------------------------------------------------------------
E = defaultdict(list)
for wid, nodes, ll, t, (cf, cb, bf, bb, wk) in ways:
    hw = t["highway"]
    flat = t.get("bridge", "no") != "no" or t.get("tunnel", "no") != "no" or t.get("covered") == "yes"
    steps = hw == "steps"
    v_car = (mph(t.get("maxspeed")) or DEFAULT_MPH.get(hw, 25)) * 0.44704
    cls = CLASS.get(hw, 3)
    # Where each mode may start or end a trip. A car stops only where it can pull over —
    # never on a freeway or a ramp — and nobody starts on a bridge deck or in a tunnel.
    stoppable = (cf or cb) and cls != 0 and t.get("expressway") != "yes"
    snap = 0 if flat else (1 if wk else 0) | (2 if (bf or bb) else 0) | (4 if stoppable else 0)
    xy = np.column_stack(geo.to_xy(ll[:, 0], ll[:, 1]))
    seg = np.hypot(*np.diff(xy, axis=0).T)
    # split at kept nodes, and at a shape point wherever the run would pass MAX_EDGE
    cut, run = [0], 0.0
    for k in range(1, len(nodes)):
        run += seg[k - 1]
        nxt = seg[k] if k < len(seg) else 0.0
        if nodes[k] in keep_osm or k == len(nodes) - 1 or run + nxt > MAX_EDGE:
            cut.append(k)
            run = 0.0
    for a, b in zip(cut[:-1], cut[1:]):
        pts, pxy = ll[a:b + 1], xy[a:b + 1]
        d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pxy, axis=0).T))])
        L = d[-1]
        if L < 0.5:
            continue
        # a run still longer than MAX_EDGE (sparse points on a long straight) splits evenly
        n_sub = max(1, int(np.ceil(L / MAX_EDGE)))
        n_piece = n_sub * max(1, int(np.ceil(L / n_sub / PIECE)))
        s = np.linspace(0, L, n_piece + 1)
        lon, lat = np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])
        z = elevation(lon, lat)
        if flat:                    # the DEM is the ground under a bridge, the hill over a tunnel
            z = np.linspace(z[0], z[-1], len(z))
        per = n_piece // n_sub
        ids = [node(nodes[a], *pts[0])]
        ids += [node((wid, a, j), lon[j * per], lat[j * per]) for j in range(1, n_sub)]
        ids += [node(nodes[b], *pts[-1])]
        for j in range(n_sub):
            dl = np.diff(s[j * per:(j + 1) * per + 1])
            dz = np.diff(z[j * per:(j + 1) * per + 1])
            slope = dz / dl
            length = float(dl.sum())
            u, v = ids[j], ids[j + 1]
            fwd = (float(np.sum(dl / costs.walk_speed(slope, steps))) if wk else -1,
                   float(np.sum(dl / costs.bike_speed(slope))) if bf else -1,
                   length / v_car if cf else -1)
            bwd = (float(np.sum(dl / costs.walk_speed(-slope, steps))) if wk else -1,
                   float(np.sum(dl / costs.bike_speed(-slope))) if bb else -1,
                   length / v_car if cb else -1)
            for (p, q), (w_, b_, c_) in (((u, v), fwd), ((v, u), bwd)):
                if w_ < 0 and b_ < 0 and c_ < 0:
                    continue
                for k_, val in (("u", p), ("v", q), ("len", length), ("walk", w_),
                                ("bike", b_), ("car", c_), ("cls", cls)):
                    E[k_].append(val)
            nsnap[u] |= snap
            nsnap[v] |= snap

N = len(nx)
nx, ny, nz = np.array(nx), np.array(ny), np.array(nz)
nsnap = np.array(nsnap, np.uint8)


def components(mode):
    ok = np.array(E[mode]) >= 0
    M = csr_matrix((np.ones(ok.sum()), (np.array(E["u"])[ok], np.array(E["v"])[ok])), shape=(N, N))
    _, lab = connected_components(M, directed=True, connection="strong")
    size = np.bincount(lab)
    return lab, size


# Without sidewalks and crosswalks, a plaza's or a park's footpaths can form an island —
# Union Square's did, and a walk from it reached nothing past its own paving. Every node
# stranded off the main walking network gets a short connector to the nearest node on it,
# if one is within STITCH metres: you can step off a path onto the street beside it.
STITCH = 20.0
lab, size = components("walk")
main = lab == np.argmax(size)
walk_nodes = np.unique(np.r_[np.array(E["u"])[np.array(E["walk"]) >= 0],
                             np.array(E["v"])[np.array(E["walk"]) >= 0]])
stranded = walk_nodes[~main[walk_nodes]]
mains = np.flatnonzero(main)
d, j = cKDTree(np.column_stack([nx[mains], ny[mains]])).query(
    np.column_stack([nx[stranded], ny[stranded]]), distance_upper_bound=STITCH)
hit = np.isfinite(d)
for a_, b_, dd in zip(stranded[hit], mains[j[hit]], d[hit]):
    for p, q in ((a_, b_), (b_, a_)):
        for k_, val in (("u", p), ("v", q), ("len", dd), ("walk", max(dd, 0.5) / costs.WALK_FLAT),
                        ("bike", -1), ("car", -1), ("cls", 3)):
            E[k_].append(val)
print(f"stitched {int(hit.sum()):,} of {len(stranded):,} stranded walking nodes "
      f"onto the main network")

# A trip may only start or end on a large connected piece of each mode's network: the
# mainland's, or Treasure Island's own streets. Dead-end one-ways and leftover scraps would
# otherwise swallow an origin whole.
MIN_PIECE = 300
for bit, mode in ((1, "walk"), (2, "bike"), (4, "car")):
    lab, size = components(mode)
    small = size[lab] < MIN_PIECE
    nsnap[small] &= ~np.uint8(bit)
    print(f"  {mode}: {int((size >= MIN_PIECE).sum())} pieces of {MIN_PIECE}+ nodes, "
          f"largest {size.max():,}")

u = np.array(E["u"], np.int64); v = np.array(E["v"], np.int64)
cost = {m: np.array(E[m], np.float64) for m in ("walk", "bike", "car")}

# ---- intersection controls ----------------------------------------------------------
# SFMTA's inventories, snapped to the nearest node a car can reach within 25 m (signals)
# or 20 m (stop signs, which stand back from the corner). OSM's own tags fill in the rest.
SIG, STOP = 1, 2
car_c = np.zeros(N, np.int8)
ped_c = np.zeros(N, np.int8)
car_nodes = np.unique(np.r_[u[cost["car"] >= 0], v[cost["car"] >= 0]])
tree = cKDTree(np.column_stack([nx[car_nodes], ny[car_nodes]]))


def snap_points(path, radius, keep=lambda p: True):
    feats = json.load(open(path))["features"]
    ll = np.array([f["geometry"]["coordinates"][:2] for f in feats
                   if f.get("geometry") and keep(f["properties"])])
    d, j = tree.query(np.column_stack(geo.to_xy(ll[:, 0], ll[:, 1])))
    return car_nodes[j[d < radius]], len(ll), int((d < radius).sum())


stops, n_st, k_st = snap_points("data/raw/stop_signs.geojson", 20.0)
car_c[stops] = STOP
# the inventory mixes in beacons, flashers, radar signs and signals not yet built
REAL_SIGNAL = {"SIGNAL", "SIGNAL (CONTRACTOR MAINTAINED)", "CALTRANS",
               "CALTRANS (BY CONTRACTOR CONSORTIUM GLC)"}
sigs, n_sg, k_sg = snap_points("data/raw/signals.geojson", 25.0,
                               lambda p: p.get("type") in REAL_SIGNAL)
car_c[sigs] = SIG
ped_c[sigs] = SIG
for osm_n, tg in ntag.items():
    i = node_id.get(osm_n)
    if i is None:
        continue
    if tg.get("highway") == "traffic_signals":
        car_c[i] = ped_c[i] = SIG
    elif tg.get("crossing") == "traffic_signals":
        ped_c[i] = SIG
    elif tg.get("highway") == "stop" and car_c[i] == 0:
        car_c[i] = STOP
print(f"signals: {k_sg:,}/{n_sg:,} SFMTA points snapped; stop signs: {k_st:,}/{n_st:,}")
print(f"controlled nodes: {(car_c == SIG).sum():,} signalised, {(car_c == STOP).sum():,} stops")

for m, kind in (("walk", ped_c), ("bike", np.maximum(car_c, ped_c)), ("car", car_c)):
    dly = np.select([kind[v] == SIG, kind[v] == STOP],
                    [costs.DELAY[m]["signal"], costs.DELAY[m]["stop"]], 0.0)
    ok = cost[m] >= 0
    cost[m][ok] += dly[ok]

np.savez_compressed("data/graph.npz", x=nx, y=ny, z=nz, snap=nsnap, u=u, v=v,
                    length=np.array(E["len"]), walk=cost["walk"], bike=cost["bike"],
                    car=cost["car"], cls=np.array(E["cls"], np.uint8), car_ctrl=car_c,
                    ped_ctrl=ped_c)
length = np.array(E["len"])
print(f"{N:,} nodes, {len(u):,} directed edges")
for m in ("walk", "bike", "car"):
    ok = cost[m] >= 0
    print(f"  {m:4s}: {ok.sum():,} edges, ~{length[ok].sum() / 2000:,.0f} km of street, "
          f"{length[ok].sum() / cost[m][ok].sum() * 3.6:.1f} km/h average incl. delays")
for bit, name in ((1, "walk"), (2, "bike"), (4, "car")):
    print(f"  snappable for {name}: {(nsnap & bit > 0).sum():,} nodes")
