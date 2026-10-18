"""How much slower each street drives, every hour of the week.

Three measured sources, one multiplier per edge per hour on top of the free-flow cost:

1. Shape through the day, per street: SFCTA's INRIX speeds on 740 one-way segments of
   the Congestion Management Program network (every arterial and the freeways), averaged
   over the last twelve months of weekdays. Each segment is read against its own night
   speed, so a multiplier of 1.3 at 5 pm means 30% longer than the same street at 2 am.
2. Shape through the week, citywide: DataSF taxi trips, ~1M of them, every day. The factor
   for (day, hour) is taxi time-per-km then, over the Mon–Fri mean at that hour — the
   same weekday mean SFCTA's numbers are. Weekday segments times this give all 168 hours.
3. Streets off the CMP network borrow from the arterials around them: an inverse-distance
   blend of nearby segments' slowdown, damped — side streets queue less than the
   arterials they feed. The damping is a choice, not a measurement; README says so.

How hard that slowdown bites, and the absolute level, are calibrated once against the
same taxi trips (22_validate_taxi.py): CONGESTION_GAIN and CAR_BIAS. Writes
data/traffic.npz.
"""
import json
import sys

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, "src")
import geo

# Both set by 22_validate_taxi.py --sweep against 12,096 taxi rides. Each hour's slowdown
# over free flow is scaled by the gain — 1.3 is where the taxi/model ratio is flattest
# across the 168 hours — and then everything by the bias, which centres it.
CONGESTION_GAIN = 1.3
CAR_BIAS = 0.986
# SFCTA draws each direction offset to its own side of the street, 10-20 m off OSM's
# centreline, so the match radius has to reach past that; direction and road class keep
# it from catching a parallel street or the freeway overhead
MATCH_M, MATCH_COS = 30.0, 0.85
AREA_CELL, AREA_REACH = 600.0, 1200.0
DAMP = {2: 0.8, 3: 0.5}     # minor arterials, local streets: share of the arterial slowdown
NIGHT = [0, 1, 2, 3, 4]

# ---- 1. weekday multipliers per CMP segment ------------------------------------------
d = pd.read_csv("data/raw/inrix_hourly.csv")
d = d[d.avg_speed > 0]
# mean travel time per mile across the months, then back to a speed
pace = d.assign(p=1 / d.avg_speed).groupby(["cmp_segid", "period"]).p.mean().unstack()
pace = pace.dropna(thresh=20).interpolate(axis=1, limit_direction="both")
night = pace[NIGHT].mean(axis=1)
m_seg = pace.div(night, axis=0).clip(0.8, 4.0)        # rows: segment, cols: hour 0..23
print(f"{len(m_seg)} CMP segments with data; median weekday multiplier by hour:")
print("  " + " ".join(f"{v:.2f}" for v in m_seg.median().values))

# ---- 2. day-of-week factors from taxi trips ----------------------------------------
frames = []
for m in ("2023-10", "2023-11", "2024-03", "2024-04"):
    t = pd.read_csv(f"data/raw/taxi_{m}.csv")
    frames.append(t)
t = pd.concat(frames, ignore_index=True)
n0 = len(t)
# SFMTA's own guidance: drop rows flagged DR, FTR, ST or ET
t = t[~t.qa_flags.fillna("").str.contains(r"DR|FTR|ST|ET")]
t = t[t.sfo_pickup.fillna(0) == 0]
start = pd.to_datetime(t.start_time_local)
dur = (pd.to_datetime(t.end_time_local) - start).dt.total_seconds()
dist = t.trip_distance_meters
B = geo.BBOX
inside = lambda la, lo: la.between(37.705, 37.835) & lo.between(-122.52, -122.355)
ok = (inside(t.pickup_location_latitude, t.pickup_location_longitude)
      & inside(t.dropoff_location_latitude, t.dropoff_location_longitude)
      & dist.between(1000, 20000) & dur.between(120, 5400)
      & (dist / dur * 3.6).between(3, 80))
t, start, dur, dist = t[ok], start[ok], dur[ok], dist[ok]
dow, hour = start.dt.dayofweek.values, start.dt.hour.values       # Monday = 0
slot = dow * 24 + hour
S = np.array([dur.values[slot == s].sum() / dist.values[slot == s].sum() for s in range(168)])
cnt = np.bincount(slot, minlength=168)
S = S.reshape(7, 24)
F = S / S[:5].mean(axis=0, keepdims=True)                 # 1.0 = the Mon–Fri mean
print(f"taxi trips: {len(t):,} of {n0:,} kept, {cnt.min()}–{cnt.max()} per hour of week")
print("  day factor at 8h:", " ".join(f"{v:.2f}" for v in F[:, 8]),
      " at 17h:", " ".join(f"{v:.2f}" for v in F[:, 17]))
np.save("data/taxi_week.npy", np.stack([S.ravel(), cnt]))

# ---- 3. match CMP segments to graph edges --------------------------------------------
g = np.load("data/graph.npz")
u, v, cls = g["u"], g["v"], g["cls"].astype(int)
car = g["car"] >= 0
x, y = g["x"], g["y"]
segs = json.load(open("data/raw/cmp_segments.geojson"))["features"]
lines, sid, is_fwy = [], [], []
for f in segs:
    gg = shapely.geometry.shape(f["geometry"])
    if gg.geom_type == "MultiLineString":
        gg = shapely.line_merge(gg)
        if gg.geom_type == "MultiLineString":
            gg = max(gg.geoms, key=lambda q: q.length)
    c = np.asarray(gg.coords)
    xy = np.column_stack(geo.to_xy(c[:, 0], c[:, 1]))
    # SFCTA draws most segments against the direction of travel; flip whichever way
    # makes the line run the way its label says
    want = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}[f["properties"]["direction"]]
    if np.dot(xy[-1] - xy[0], want) < 0:
        xy = xy[::-1]
    lines.append(shapely.LineString(xy))
    sid.append(f["properties"]["cmp_segid"])
    is_fwy.append(any(k in f["properties"]["cmp_name"] for k in ("I-80", "I-280", "US-101")))
lines = np.array(lines, dtype=object)
sid, is_fwy = np.array(sid), np.array(is_fwy)
have = np.isin(sid, m_seg.index.values)
tree = shapely.STRtree(lines)

ei = np.flatnonzero(car)
mid = np.column_stack([(x[u[ei]] + x[v[ei]]) / 2, (y[u[ei]] + y[v[ei]]) / 2])
vec = np.column_stack([x[v[ei]] - x[u[ei]], y[v[ei]] - y[u[ei]]])
vec /= np.maximum(np.hypot(*vec.T), 1e-6)[:, None]
pts = shapely.points(mid)
pi, lj = tree.query(pts, predicate="dwithin", distance=MATCH_M)
pos = shapely.line_locate_point(lines[lj], pts[pi])
a = shapely.get_coordinates(shapely.line_interpolate_point(lines[lj], np.maximum(pos - 5, 0)))
b = shapely.get_coordinates(shapely.line_interpolate_point(lines[lj], pos + 5))
tan = (b - a) / np.maximum(np.hypot(*(b - a).T), 1e-6)[:, None]
cos = np.einsum("ij,ij->i", tan, vec[pi])
dst = shapely.distance(pts[pi], lines[lj])
good = (cos > MATCH_COS) & have[lj] & (is_fwy[lj] == (cls[ei][pi] == 0))
match = np.full(len(ei), -1)
best = np.full(len(ei), np.inf)
for p_, l_, d_ in zip(pi[good], lj[good], dst[good]):
    if d_ < best[p_]:
        best[p_], match[p_] = d_, l_
edge_seg = np.full(len(u), -1)
edge_seg[ei] = match
L = g["length"]
print(f"matched {int((match >= 0).sum()):,} car edges ({L[ei][match >= 0].sum() / 1000:.0f} km) "
      f"to {len(np.unique(match[match >= 0]))} of {int(have.sum())} segments with data")

# ---- profiles ------------------------------------------------------------------------
# weekday hour h -> week slot (day, h): the segment's weekday shape times the city's day
seg_rows = {s: i for i, s in enumerate(m_seg.index.values)}
M_wk = m_seg.values                                              # nseg x 24
week = (M_wk[:, None, :] * F[None, :, :]).reshape(len(M_wk), 168)
profiles = [np.ones(168)]                                        # 0: unmatched, flat
prof_of_line = np.zeros(len(lines), int)
for j, s in enumerate(sid):
    if have[j]:
        prof_of_line[j] = len(profiles)
        profiles.append(week[seg_rows[s]])

# Area profiles from the arterials only — freeways congest on their own schedule
art = np.flatnonzero(have & ~is_fwy)
amid = shapely.get_coordinates(shapely.line_interpolate_point(lines[art], 0.5, normalized=True))
art_excess = np.stack([week[seg_rows[sid[j]]] for j in art]) - 1
city_excess = np.median(art_excess, axis=0)
gx0, gy0 = geo.XMIN, geo.YMIN
cell_of = lambda px, py: (((py - gy0) // AREA_CELL).astype(int) * 1000
                          + ((px - gx0) // AREA_CELL).astype(int))
area_prof = {}


def area_profile(key, damp):
    k = (key, damp)
    if k not in area_prof:
        cy_, cx_ = divmod(key, 1000)
        c = np.array([gx0 + (cx_ + 0.5) * AREA_CELL, gy0 + (cy_ + 0.5) * AREA_CELL])
        dd = np.hypot(*(amid - c).T)
        near = dd < AREA_REACH
        ex = (np.average(art_excess[near], axis=0, weights=1 / (dd[near] + 200) ** 2)
              if near.any() else city_excess)
        area_prof[k] = len(profiles)
        profiles.append(1 + damp * ex)
    return area_prof[k]


fwy_lines = np.flatnonzero(have & is_fwy)
fwy_tree = shapely.STRtree(lines[fwy_lines])
edge_prof = np.zeros(len(u), np.int64)
mx, my = (x[u] + x[v]) / 2, (y[u] + y[v]) / 2
cells = cell_of(mx, my)
for e in np.flatnonzero(car):
    if edge_seg[e] >= 0:
        edge_prof[e] = prof_of_line[edge_seg[e]]
    elif cls[e] == 0:
        # ramps and unmatched freeway queue with the freeway they join
        j = fwy_tree.nearest(shapely.Point(mx[e], my[e]))
        edge_prof[e] = prof_of_line[fwy_lines[j]]
    else:
        edge_prof[e] = area_profile(int(cells[e]), DAMP.get(cls[e], 1.0) if cls[e] >= 2 else 1.0)
prof = np.clip((1 + CONGESTION_GAIN * (np.stack(profiles) - 1)) * CAR_BIAS, 0.5, 6.0)
assert len(prof) < 65536
np.savez_compressed("data/traffic.npz", prof=np.round(prof * 1000).astype(np.uint16),
                    edge_prof=edge_prof.astype(np.uint16), F=F, m_seg=M_wk,
                    seg_ids=m_seg.index.values)
w = L[car]
city = np.average(prof[edge_prof[car]], axis=0, weights=w)
print(f"{len(prof)} profiles; length-weighted city multiplier: "
      f"min {city.min():.2f} (slot {city.argmin()}), max {city.max():.2f} (slot {city.argmax()})")
print("  Thu by hour:", " ".join(f"{v:.2f}" for v in city[72:96]))
print("  Sun by hour:", " ".join(f"{v:.2f}" for v in city[144:168]))
