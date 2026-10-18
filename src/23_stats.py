"""Everything the graphs page plots, measured with the page's own router.
-> data/stats.json
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from router import GONE, Router

R = Router()
places = pd.read_csv("data/places.csv")
xy = R.P["place_xy"]
NPL = len(xy)
LABEL = list(places.label)
SEEDS = {s: [R.seeds(s, x, y) for x, y in xy] for s in (0, 1, 2)}
ON_LAND = np.isin(R.act, R.land_cells)
curves = np.load("data/curves.npy")                       # place x mode x hour, minutes
idx = {l: i for i, l in enumerate(LABEL)}


def at_place(T, s, j):
    """node seconds -> seconds to place j, through its seeds"""
    ids, secs = SEEDS[s][j]
    return np.min(T[:, ids] + secs, axis=1) if len(ids) else np.full(len(T), np.inf)


WALK_T = R.trees_from(R.mode_costs("walk"), SEEDS[0])
BIKE_T = R.trees_from(R.mode_costs("bike"), SEEDS[1])
BIKE_G = R.grid("bike", BIKE_T)


def hour(h):
    slot = 4 * h + 2
    T = R.trees_from(R.car_costs(slot), SEEDS[2])
    Hh = R.heli_from(T, slot)
    Gc, Gh = R.grid("car", T), R.grid("heli", Hh)
    land = ON_LAND[None, :] & (Gc < GONE) & (BIKE_G < GONE)
    bike_wins = float(((BIKE_G <= Gc) & land).sum() / land.sum())
    flown = float(((Gh < Gc - 0.5) & land).sum() / land.sum())
    car_pp = np.stack([at_place(T, 2, j) for j in range(NPL)], axis=1) / 60     # i -> j
    heli_pp = np.stack([at_place(Hh, 2, j) for j in range(NPL)], axis=1) / 60
    return h, bike_wins, flown, car_pp.astype(np.float32), heli_pp.astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    bike_wins, flown = np.zeros(168), np.zeros(168)
    car_pp = np.zeros((168, NPL, NPL), np.float32)
    heli_pp = np.zeros((168, NPL, NPL), np.float32)
    with Pool(8) as pool:
        for h, bw, fl, cp, hp in pool.imap_unordered(hour, range(168)):
            bike_wins[h], flown[h], car_pp[h], heli_pp[h] = bw, fl, cp, hp
    print(f"168 hours routed ({time.time() - t0:.0f}s)")
    walk_pp = np.stack([at_place(WALK_T, 0, j) for j in range(NPL)], axis=1) / 60
    bike_pp = np.stack([at_place(BIKE_T, 1, j) for j in range(NPL)], axis=1) / 60

    # 1. the week, by mode: the typical place's median trip, each hour
    week = np.median(curves, axis=0)                      # mode x 168
    out["week"] = {m: week[i].round(2).tolist() for i, m in enumerate(("walk", "bike", "car", "heli"))}
    out["bike_wins"] = bike_wins.round(4).tolist()
    out["flown"] = flown.round(4).tolist()

    # 2. Brenda's trip, and the same from Bernal over the Bay Bridge to Treasure Island
    def trip(a, b):
        i, j = idx[a], idx[b]
        return dict(walk=round(float(walk_pp[i, j]), 1), bike=round(float(bike_pp[i, j]), 1),
                    car=car_pp[:, i, j].round(2).tolist(), heli=heli_pp[:, i, j].round(2).tolist())
    out["trips"] = {"Bernal Heights → Union Square": trip("Bernal Heights", "Union Square"),
                    "Union Square → Bernal Heights": trip("Union Square", "Bernal Heights"),
                    "Bernal Heights → Treasure Island": trip("Bernal Heights", "Treasure Island")}

    # 3. hills: the same pair of places on foot, one way and back
    off = ~np.eye(NPL, dtype=bool) & np.isfinite(walk_pp) & np.isfinite(walk_pp.T)
    with np.errstate(invalid="ignore"):                   # Treasure Island: inf - inf
        diff = np.where(off, walk_pp - walk_pp.T, 0)
    order = np.argsort(diff.ravel())[::-1]
    pairs = []
    for k in order:
        i, j = divmod(int(k), NPL)
        if len(pairs) == 12:
            break
        if any(p["from"] == LABEL[j] and p["to"] == LABEL[i] for p in pairs):
            continue
        pairs.append({"from": LABEL[i], "to": LABEL[j], "up": round(float(walk_pp[i, j]), 1),
                      "down": round(float(walk_pp[j, i]), 1),
                      "km": round(float(np.hypot(*(xy[i] - xy[j])) / 1000), 2)})
    out["hills"] = pairs
    asym = np.abs(diff[off]) / np.minimum(walk_pp, walk_pp.T)[off]
    out["hills_share_10pct"] = float((asym > 0.10).mean())

    # 4. centrality: median minutes from each neighbourhood to the rest of SF, by car, at the
    # city's calmest and worst hours — and by bike, which has no rush hour
    m = json.load(open("data/measure.json"))
    calm_h, peak_h = (m["calm"] - 2) // 4, (m["peak"] - 2) // 4
    hood = places.kind.values == "neighborhood"
    # a median over the cells a mode can reach means nothing when it reaches almost none:
    # Treasure Island on foot is "11 min" to its own shore and no way to anywhere else
    reach = {m: ((R.grid(m, T)[:, ON_LAND] < GONE).mean(axis=1))
             for m, T in (("walk", WALK_T), ("bike", BIKE_T))}
    rows = []
    for i in np.flatnonzero(hood):
        mode_med = lambda m, k: round(float(curves[i, k, 0]), 1) if reach[m][i] > 0.5 else None
        rows.append(dict(label=LABEL[i], calm=round(float(curves[i, 2, calm_h]), 1),
                         peak=round(float(curves[i, 2, peak_h]), 1),
                         bike=mode_med("bike", 1), walk=mode_med("walk", 0)))
    rows.sort(key=lambda r: r["peak"])
    out["centrality"] = rows
    out["calm_h"], out["peak_h"] = calm_h, peak_h

    # 5. the helicopter's best trips at the worst hour: minutes saved over driving
    save = car_pp[peak_h] - heli_pp[peak_h]
    order = np.argsort(save.ravel())[::-1]
    best = []
    for k in order[:400]:
        i, j = divmod(int(k), NPL)
        if len(best) == 8:
            break
        if any(b["from"] == LABEL[j] and b["to"] == LABEL[i] for b in best):
            continue
        best.append({"from": LABEL[i], "to": LABEL[j], "car": round(float(car_pp[peak_h, i, j]), 1),
                     "heli": round(float(heli_pp[peak_h, i, j]), 1)})
    out["heli_best"] = best
    allp = ~np.eye(NPL, dtype=bool)
    out["heli_share_peak"] = float((heli_pp[peak_h][allp] < car_pp[peak_h][allp] - 0.5).mean())
    out["heli_share_calm"] = float((heli_pp[calm_h][allp] < car_pp[calm_h][allp] - 0.5).mean())

    # 6. how each class of road slows through the week (length-weighted multipliers)
    tr = np.load("data/traffic.npz")
    g = np.load("data/graph.npz")
    car = g["car"] >= 0
    prof = tr["prof"].astype(float) / 1000
    by = {}
    for c, name in ((0, "freeway"), (1, "major arterial"), (2, "minor arterial"), (3, "local street")):
        sel = car & (g["cls"] == c)
        by[name] = np.average(prof[tr["edge_prof"][sel]], axis=0, weights=g["length"][sel]).round(3).tolist()
    out["by_class"] = by
    out["day_factor"] = np.load("data/traffic.npz")["F"].round(3).tolist()

    # 6b. parking: the model drives curb to curb, like a maps app. How many minutes of
    # finding a space does it take before the bike wins?
    PARK = list(range(0, 13))
    def bike_share(h):
        c = car_pp[h][pp_ok]
        return [round(float((bike_pp[pp_ok] <= c + p).mean()), 4) for p in PARK]
    pp_ok = allp & np.isfinite(walk_pp) & np.isfinite(bike_pp)
    out["parking"] = dict(minutes=PARK, peak=bike_share(peak_h), calm=bike_share(calm_h))

    # 7. headline numbers
    out["kpi"] = dict(
        walk=round(float(np.median(curves[:, 0, 0])), 1),
        bike=round(float(np.median(curves[:, 1, 0])), 1),
        car_calm=round(float(week[2].min()), 1), car_peak=round(float(week[2].max()), 1),
        heli_peak=round(float(week[3][peak_h]), 1),
        bike_beats_car_peak=round(float(bike_wins.max()), 3),
        bike_beats_car_calm=round(float(bike_wins.min()), 3),
        pairs_bike_beats_car_peak=round(float((bike_pp[pp_ok] <= car_pp[peak_h][pp_ok]).mean()), 3),
        places=NPL)
    # JSON has no Infinity: a trip with no way to make it (on foot to Treasure Island) is null
    def clean(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    json.dump(clean(out), open("data/stats.json", "w"), allow_nan=False)
    print(json.dumps(out["kpi"], indent=1))
    print("Brenda's trip, Thursday by car:", [round(v) for v in out["trips"]["Bernal Heights → Union Square"]["car"][72:96]])
    print("hills:", out["hills"][:3])
    print("heli best at peak:", out["heli_best"][:3], "share", out["heli_share_peak"])
    print(f"done ({time.time() - t0:.0f}s)")
