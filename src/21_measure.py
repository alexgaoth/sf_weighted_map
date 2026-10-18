"""Run the page's router offline over every named place and every hour of the week.

Writes data/curves.npy — median minutes from each place to the rest of SF, per mode, per
hour (the page's week strip) — and data/measure.json: each mode's colour span, the city's
calmest and worst quarter-hour, and the per-hour medians the graphs use. Hours are
sampled on the half hour, where an hourly profile reads its own value.

The 168 car hours run in parallel; each is independent. ~3 minutes on 8 cores.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "src")
import costs
from router import GONE, Router

R = Router()
xy = R.P["place_xy"]
NPL = len(xy)
SEEDS = {s: [R.seeds(s, x, y) for x, y in xy] for s in (0, 1, 2)}
cx = R.gx0 + (R.act % R.GW) * R.cell
cy = R.gy0 + (R.act // R.GW) * R.cell
ON_LAND = np.isin(R.act, R.land_cells)
RDIST = np.hypot(cx[None, :] - xy[:, 0:1], cy[None, :] - xy[:, 1:2])      # place x cell
SAMPLE = np.random.default_rng(7).random(RDIST.shape) < 0.05              # for the spans


def log_stretch(minutes):
    """log stretch on land cells over 300 m from the origin, where the ratio means
    something rather than measuring the walk to the nearest corner"""
    far = ON_LAND[None, :] & (RDIST > 300) & (minutes < GONE) & SAMPLE
    return np.log(minutes[far] * costs.VREF / RDIST[far])


def car_hour(h):
    slot = 4 * h + 2
    T = R.trees_from(R.car_costs(slot), SEEDS[2])
    Hh = R.heli_from(T, slot)
    Gc, Gh = R.grid("car", T), R.grid("heli", Hh)
    ls = (log_stretch(Gc), log_stretch(Gh)) if h % 3 == 0 else None
    return h, R.stats(Gc), R.stats(Gh), ls


if __name__ == "__main__":
    t0 = time.time()
    curves = np.zeros((NPL, 4, 168), np.float64)
    spans = {}
    for m, mode in ((0, "walk"), (1, "bike")):
        T = R.trees_from(R.mode_costs(mode), SEEDS[m])
        G = R.grid(mode, T)
        curves[:, m, :] = R.stats(G)[:, None]
        ls = log_stretch(G)
        spans[mode] = [float(np.percentile(ls, 1)), float(np.percentile(ls, 99))]
    print(f"walk, bike done ({time.time() - t0:.0f}s)")
    car_ls, heli_ls = [], []
    with Pool(8) as pool:
        for k, (h, sc, sh, ls) in enumerate(pool.imap_unordered(car_hour, range(168))):
            curves[:, 2, h], curves[:, 3, h] = sc, sh
            if ls is not None:
                car_ls.append(ls[0]); heli_ls.append(ls[1])
            if k % 24 == 23:
                print(f"  {k + 1}/168 hours ({time.time() - t0:.0f}s)")
    for mode, ls in (("car", car_ls), ("heli", heli_ls)):
        ls = np.concatenate(ls)
        spans[mode] = [float(np.percentile(ls, 1)), float(np.percentile(ls, 99))]

    city = np.median(curves[:, 2, :], axis=0)                 # the typical place, each hour
    calm, peak = int(np.argmin(city)), int(np.argmax(city))
    np.save("data/curves.npy", curves)
    json.dump(dict(spans=[spans[m] for m in ("walk", "bike", "car", "heli")],
                   calm=4 * calm + 2, peak=4 * peak + 2, city_car=city.round(3).tolist()),
              open("data/measure.json", "w"), indent=1)
    kmh = lambda v: round(costs.VREF * 0.06 / np.exp(v), 1)
    print("spans, as straight-line km/h at the two ends:",
          {m: [kmh(v) for v in s] for m, s in spans.items()})
    print(f"typical place, by car: calmest hour {calm} ({city[calm]:.1f} min), "
          f"worst {peak} ({city[peak]:.1f} min); {time.time() - t0:.0f}s")
    print("median of place medians:", {m: round(float(np.median(curves[:, i, :])), 1)
                                       for i, m in enumerate(("walk", "bike", "car", "heli"))})
