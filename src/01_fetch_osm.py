"""Every highway in and just around San Francisco, from the Overpass API.

The box runs about a kilometre past SF's land border to the south, so a trip may cut
through Daly City or Brisbane as a real one would; only SF itself is drawn. Also pulls the
controlled nodes (signals, stops, signalised crossings), the named places used as
origins, anything tagged as a helipad, and the lakes the city's shoreline counts as land.
-> data/raw/osm.json, osm_places.json, osm_water.json
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, "src")
import geo

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.private.coffee/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
UA = "sf-weighted-map/0.1 (personal, non-commercial; github.com/alexgaoth)"
B = geo.BBOX
box = f"({B['s']},{B['w']},{B['n']},{B['e']})"

QUERIES = {
    "osm.json": f"""[out:json][timeout:300];
way["highway"]{box}->.w;
node["highway"~"^(traffic_signals|stop)$"]{box}->.c1;
node["crossing"="traffic_signals"]{box}->.c2;
(.w; node(w.w); .c1; .c2;);
out body qt;""",
    "osm_places.json": f"""[out:json][timeout:120];
area(id:3600111968)->.sf;
(
  node["place"~"^(neighbourhood|suburb|quarter)$"](area.sf){box};
  nwr["aeroway"~"^(helipad|heliport)$"]{box};
);
out center tags qt;""",
    # lakes and reservoirs, which the city's shoreline polygon counts as land
    "osm_water.json": f"""[out:json][timeout:120];
area(id:3600111968)->.sf;
(
  way["natural"="water"](area.sf);
  relation["natural"="water"](area.sf);
);
out geom qt;""",
}

raw = pathlib.Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
for name, q in QUERIES.items():
    out = raw / name
    if out.exists():
        print(f"{out} exists, skipping")
        continue
    for attempt in range(8):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": UA})
            body = urllib.request.urlopen(req, timeout=360).read()
            n = len(json.loads(body)["elements"])
            out.write_bytes(body)
            print(f"{out}: {len(body) / 1e6:.1f} MB, {n:,} elements from {url}")
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            wait = 15 * (attempt + 1)
            print(f"  {url} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise SystemExit(f"could not fetch {name}")
