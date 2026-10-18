"""Hold the car model up against real trips: DataSF taxi rides, pickup to dropoff, at the
hour they actually happened. Taxis are not a perfect stand-in for "a car" — the meter
runs for the whole ride, points are rounded to ~100 m, and a cab does not always take the
fastest route — but they are the only door-to-door times on every day of the week.

Sweeps a congestion gain — each hour's slowdown over free flow scaled by it — and keeps
the one under which the taxi/model ratio is flattest through the week, then the bias that
centres it: CONGESTION_GAIN and CAR_BIAS in 07_traffic.py. ~10 minutes. Writes
data/validation.json for the graphs page.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import geo
from router import Router

N_PER_HOUR = 72
rng = np.random.default_rng(11)

t = pd.concat([pd.read_csv(f"data/raw/taxi_{m}.csv")
               for m in ("2023-10", "2023-11", "2024-03", "2024-04")], ignore_index=True)
t = t[~t.qa_flags.fillna("").str.contains(r"DR|FTR|ST|ET") & (t.sfo_pickup.fillna(0) == 0)]
start = pd.to_datetime(t.start_time_local)
t = t.assign(dur=(pd.to_datetime(t.end_time_local) - start).dt.total_seconds(),
             how=start.dt.dayofweek * 24 + start.dt.hour)
t = t[t.trip_distance_meters.between(1500, 15000) & t.dur.between(180, 4800)]
t = t.dropna(subset=["pickup_location_longitude", "pickup_location_latitude",
                     "dropoff_location_longitude", "dropoff_location_latitude"])
px, py = geo.to_xy(t.pickup_location_longitude.values, t.pickup_location_latitude.values)
dx, dy = geo.to_xy(t.dropoff_location_longitude.values, t.dropoff_location_latitude.values)
t = t.assign(px=px, py=py, dx=dx, dy=dy, crow=np.hypot(dx - px, dy - py))
R = Router()
land = np.zeros(len(t), bool)
# both ends on SF land, and a crow-flies distance the ~100 m rounding cannot swamp
cxi = np.floor((t.px.values - R.gx0) / R.cell).astype(int)
cyi = np.floor((t.py.values - R.gy0) / R.cell).astype(int)
cxj = np.floor((t.dx.values - R.gx0) / R.cell).astype(int)
cyj = np.floor((t.dy.values - R.gy0) / R.cell).astype(int)
ok = (cxi >= 0) & (cxi < R.GW) & (cyi >= 0) & (cyi < R.GH) & (cxj >= 0) & (cxj < R.GW) \
    & (cyj >= 0) & (cyj < R.GH)
lm = R.land.reshape(R.GH, R.GW)
ok[ok] &= (lm[cyi[ok], cxi[ok]] == 1) & (lm[cyj[ok], cxj[ok]] == 1)
t = t[ok & (t.crow.values > 1000)]
sample = t.sample(frac=1, random_state=1).groupby("how").head(N_PER_HOUR)
print(f"{len(t):,} usable trips; sampled {len(sample):,} across {sample.how.nunique()} hours")

# --sweep tries several gains on top of whatever 07_traffic.py shipped; without it this just
# checks the model as built
GAINS = [1.0, 1.3, 1.6, 1.9] if "--sweep" in sys.argv else [1.0]
free_prof = R.prof.copy()


def run(gain):
    """model seconds for every sampled trip, with each hour's slowdown scaled by gain"""
    R.prof = 1000 + gain * (free_prof - 1000)
    out = np.full(len(sample), np.nan)
    for h, g in sample.groupby("how"):
        T = R.trees_from(R.car_costs(4 * h + 2), [R.seeds(2, x, y) for x, y in zip(g.px, g.py)])
        for k, (i, x, y) in enumerate(zip(g.index, g.dx, g.dy)):
            ids, secs = R.seeds(2, x, y)
            out[sample.index.get_loc(i)] = np.min(T[k, ids] + secs) if len(ids) else np.nan
    R.prof = free_prof.copy()
    return out


tm = sample.groupby("how").dur.median().values.reshape(7, 24)
results = {}
for gain in GAINS:
    model = run(gain)
    ok = np.isfinite(model)
    ratio = sample.dur.values[ok] / model[ok]
    mm = pd.Series(model).groupby(sample.how.values).median().values.reshape(7, 24)
    hr = pd.Series(sample.dur.values / model).groupby(sample.how.values).median().values
    # how much of the week's swing the model reproduces: slope of log(taxi) on log(model)
    slope = np.polyfit(np.log(mm.ravel()), np.log(tm.ravel()), 1)[0]
    results[gain] = dict(bias=float(np.median(ratio)), corr=float(np.corrcoef(tm.ravel(), mm.ravel())[0, 1]),
                         swing=float(mm.max() / mm.min()), slope=float(slope),
                         hour_spread=float(np.std(np.log(hr))), model_hour=mm, ratio=ratio)
    print(f"gain {gain:.1f}: taxi/model median {results[gain]['bias']:.3f}, hour medians r = "
          f"{results[gain]['corr']:.3f}, swing {results[gain]['swing']:.2f}x (taxi "
          f"{tm.max() / tm.min():.2f}x), log-log slope {slope:.2f}, spread of hourly ratios "
          f"{results[gain]['hour_spread']:.3f}")
# the gain whose hour-to-hour ratios are flattest: the model then slows when the taxis do
best = min(GAINS, key=lambda g: results[g]["hour_spread"])
r = results[best]
ape = np.abs(1 / r["ratio"] * r["bias"] - 1)
print(f"best gain {best}: CAR_BIAS {r['bias']:.3f}; median abs error per trip {np.median(ape):.0%}")
json.dump(dict(gain=best, bias=r["bias"], n=int(len(sample)), corr=r["corr"],
               taxi_hour=tm.round(1).tolist(), model_hour=(r["model_hour"] * r["bias"]).round(1).tolist(),
               ratio_q=[float(np.quantile(r["ratio"], q)) for q in (.1, .25, .5, .75, .9)],
               mape=float(np.median(ape)),
               sweep={str(g): {k: v for k, v in results[g].items() if k not in ("model_hour", "ratio")}
                      for g in GAINS}),
          open("data/validation.json", "w"), indent=1)
