"""Assemble dist/ — a static site: index.html, the payload, and the graphs page.

The payload ships gzipped and the page inflates it itself, so the download is the same
size whether or not the host compresses .bin. Every buffer is padded to 8 bytes, because
the page views them in place and a Uint32Array cannot start at an odd offset.
"""
import gzip
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import costs

P = np.load("data/payload.npz")
model = costs.model()
measure = json.load(open("data/measure.json"))
curves = np.load("data/curves.npy")
places = pd.read_csv("data/places.csv")
pads = pd.read_csv("data/heliports.csv")

buffers, meta_buf, offset = [], {}, 0


def put(name, arr, dtype):
    global offset
    a = np.ascontiguousarray(arr, dtype=dtype)
    b = a.tobytes()
    meta_buf[name] = dict(offset=offset, count=int(a.size))
    pad = (-len(b)) % 8
    buffers.append(b + b"\0" * pad)
    offset += len(b) + pad


put("node_q", P["node_q"], np.uint16)
put("node_snap", P["node_snap"], np.uint8)
put("off", P["off"], np.uint32)
put("tgt", P["tgt"], np.uint32)
put("cost_walk", P["cost_walk"], np.uint16)
put("cost_bike", P["cost_bike"], np.uint16)
put("cost_car", P["cost_car"], np.uint16)
put("car_prof", P["car_prof"], np.uint16)
put("prof", P["prof"], np.uint16)
put("land", P["land"], np.uint8)
put("mesh_q", P["mesh_q"], np.uint16)
put("mesh_idx", P["mesh_idx"], np.uint16)
for k in ("outline", "hood", "major", "minor"):
    put(f"line_{k}", P[f"line_{k}"], np.uint16)
put("curves", np.clip(np.round(curves * 60), 0, 65535), np.uint16)   # seconds
blob = b"".join(buffers)

g = P["grid"]
xy = P["place_xy"]
meta = dict(
    q=dict(x0=float(P["q"][0]), y0=float(P["q"][1]), step=float(P["q"][2])),
    grid=dict(x0=float(g[0]), y0=float(g[1]), cell=float(g[2]), nx=int(g[3]), ny=int(g[4])),
    vref=model["vref"], walk=model["walk"], heli=model["heli"], snapMax=model["snapMax"],
    smooth=model["smooth"],
    nNodes=int(len(P["node_q"])), nProf=int(P["prof"].shape[0]),
    spans=measure["spans"], calm=measure["calm"], peak=measure["peak"],
    start=model["start"],
    places=[dict(label=r.label, x=round(float(x), 1), y=round(float(y), 1))
            for r, (x, y) in zip(places.itertuples(), xy)],
    heliports=[dict(name=r.name, x=round(float(x), 1), y=round(float(y), 1))
               for r, (x, y) in zip(pads.itertuples(), P["pads"])],
    buffers=meta_buf,
)

out = pathlib.Path("dist")
out.mkdir(exist_ok=True)
html = (open("src/map_template.html").read()
        .replace("{{META}}", json.dumps(meta, separators=(",", ":")))
        .replace("{{ROUTE_JS}}", open("src/route.js").read()))
assert "{{" not in html
(out / "index.html").write_text(html)
packed = gzip.compress(blob, 9)
(out / "sfmap.bin.gz").write_bytes(packed)

# the graphs page: everything 23_stats.py and 22_validate_taxi.py measured, inlined
stats = json.load(open("data/stats.json"))
validation = json.load(open("data/validation.json"))
validation = {k: validation[k] for k in ("bias", "n", "corr", "taxi_hour", "model_hour", "mape", "gain")}
graphs = (open("src/graphs_template.html").read()
          .replace("{{DATA}}", json.dumps(dict(stats=stats, validation=validation), separators=(",", ":")))
          .replace("{{N_PLACES}}", str(len(places)))
          .replace("{{N_HOODS}}", str(int((places.kind == "neighborhood").sum())))
          .replace("{{N_TAXI}}", f"{validation['n']:,}"))
assert "{{" not in graphs
(out / "graphs.html").write_text(graphs)
print(f"dist/graphs.html    {len(graphs) / 1024:.0f} KB")

# dist/ is the deployable artefact, so its host config ships inside it. The payload is
# pinned to octet-stream: some CDNs see .gz and set Content-Encoding, which would make the
# browser inflate it before the page does (the page sniffs for that too).
(out / "vercel.json").write_text(json.dumps({
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "headers": [
        {"source": "/sfmap.bin.gz", "headers": [
            {"key": "Content-Type", "value": "application/octet-stream"},
            {"key": "Cache-Control", "value": "public, max-age=3600"}]},
        {"source": "/og.png", "headers": [
            {"key": "Cache-Control", "value": "public, max-age=86400"}]},
    ],
}, indent=2) + "\n")

print(f"dist/index.html     {len(html) / 1024:.0f} KB")
print(f"dist/sfmap.bin.gz   {len(packed) / 1048576:.2f} MB "
      f"({len(blob) / 1048576:.2f} MB inflated, {len(packed) / len(blob):.0%})")
