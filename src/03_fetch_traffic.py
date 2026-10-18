"""The two measured sources the car model is built from, into data/raw/.

  cmp_segments.geojson   SFCTA's 740 one-way Congestion Management Program segments
  inrix_hourly.csv       SFCTA's INRIX speeds on those segments, by hour, weekdays only,
                         one row per segment-hour-month for the last twelve months
  taxi_YYYY-MM.csv       DataSF taxi trips (SFMTA, public domain): real start and end times
                         and ~100 m pickup/dropoff points, every day of the week

SFCTA's API is the one behind congestion.sfcta.org; DataSF's taxi data runs Dec 2022 to
May 2024. Four months of it, away from the holidays, is ~1M trips.
"""
import json
import pathlib
import urllib.parse
import urllib.request

raw = pathlib.Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "sf-weighted-map/0.1 (personal, non-commercial)"}
API = "https://api.sfcta.org/api/"
MONTHS = [f"2025-{m:02d}" for m in (10, 11, 12)] + [f"2026-{m:02d}" for m in range(1, 10)]
TAXI = ["2023-10", "2023-11", "2024-03", "2024-04"]


def get(url, out, accept=None):
    if out.exists():
        print(f"{out} exists, skipping")
        return
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    body = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600).read()
    out.write_bytes(body)
    print(f"{out}: {len(body) / 1e6:.1f} MB")


seg = raw / "cmp_segments.geojson"
if not seg.exists():
    rows = json.loads(urllib.request.urlopen(urllib.request.Request(
        API + "cmp_segments_exp?select=cmp_segid,cmp_name,cmp_from,cmp_to,direction,length,geometry",
        headers=UA), timeout=300).read())
    feats = [dict(type="Feature", geometry=json.loads(r.pop("geometry")) if isinstance(r["geometry"], str)
                  else r.pop("geometry"), properties=r) for r in rows]
    seg.write_text(json.dumps(dict(type="FeatureCollection", features=feats)))
    print(f"{seg}: {len(feats)} segments")

get(API + "inrix_rt_hourly_exp?" + urllib.parse.urlencode(
    [("date", f"gte.{MONTHS[0]}"), ("date", f"lte.{MONTHS[-1]}"),
     ("select", "cmp_segid,date,period,avg_speed,refspd_inrix,sample_size"),
     ("order", "cmp_segid,date,period")]), raw / "inrix_hourly.csv", accept="text/csv")

for m in TAXI:
    y, mo = map(int, m.split("-"))
    end = f"{y + (mo == 12):04d}-{mo % 12 + 1:02d}-01T00:00:00"
    q = urllib.parse.urlencode({
        "$select": "start_time_local,end_time_local,pickup_location_latitude,"
                   "pickup_location_longitude,dropoff_location_latitude,"
                   "dropoff_location_longitude,trip_distance_meters,sfo_pickup,qa_flags",
        "$where": f"start_time_local >= '{m}-01T00:00:00' and start_time_local < '{end}'",
        "$limit": "1000000"})
    get("https://data.sf.gov/resource/m8hk-2ipk.csv?" + q, raw / f"taxi_{m}.csv")
