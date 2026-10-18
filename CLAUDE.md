# sf_weighted_map

San Francisco redrawn so distance from any point is travel time — walk, bike, car by the
hour of the week, helicopter. The SF counterpart of `../weighted_map` (the US map).
`README.md` has the pipeline and the method; this file is only what bites.

## Commands

- Always `.venv/bin/python …` (CPython 3.12 via `uv`; system python is 3.14). `tools/` needs
  `npm install` once for the two headless checks.
- The pipeline runs in filename order. After touching anything upstream of the page, the
  minimum is `20 → 21 → 23 → 24`, then `25_check_parity.py` and look at the page.
- `python3 -m http.server -d dist` to view it. It **must** be served — the page fetches
  `sfmap.bin.gz` — and from `dist/`, which is the deployable artefact.
- `22_validate_taxi.py` checks the car model as built; `--sweep` re-runs the gain sweep and is
  the only way to re-derive `CONGESTION_GAIN`/`CAR_BIAS` (~10 min).

## Invariants kept by hand

- **The page's router and `src/router.py` must stay twins.** The graphs page, the week strip
  and the colour spans are all measured in Python; the map routes in JS. Change seeds,
  snapping, `fillGrid`, `carCostsAt` or `heliFrom` in one and you must change the other, then
  run `25_check_parity.py` (fails above 1 s; it reads 0.05 s today).
- Model constants live in `src/costs.py` and reach the page through `costs.model()` →
  META. Don't hard-code `VREF`, `SNAP_MAX`, `SMOOTH`, the walk speed or the heli constants in the
  template.
- `CONGESTION_GAIN` and `CAR_BIAS` in `07_traffic.py` were fitted with the profiles as they are.
  Changing the traffic model (matching, damping, the taxi factors) invalidates them; re-sweep.
- Car profiles are indexed by `uint16` (`car_prof`) — there are 1,464, more than a byte holds.
  The payload buffers are padded to 8 bytes because the page views `Uint32Array`s in place.
- The mesh must stay under 65,536 vertices (`uint16` indices); `06_build_mesh.py` asserts it.
- `lift()` in the vertex shader and `lift()` in the page's CPU code must agree — including the
  near-origin fade `smoothstep(200, 900, r)` — or the dots float off the relief.
- Hourly profile values are means over the hour, so they sit on the half hour; quarter-hours
  in between interpolate linearly. Python samples curves at `4h + 2` for the same reason.

## Gotchas

- **Unreachable is a channel, not a value.** The time texture is `RG16F`: minutes, and the
  share of unreachable cells. An earlier version stored a 30,000-minute sentinel in the time
  itself; bilinear filtering blended it into neighbours and flung vertices hundreds of km —
  the map rendered as a starburst. Unreachable cells now hold the field's largest real time.
- **Dropping sidewalks strands footpaths.** Sidewalks/crossings are dropped (they halve the
  graph), which left Union Square's plaza an island — a walk from it "reached" 1% of the city.
  `05_build_graph.py` stitches stranded walk nodes to the main network within 20 m, and only
  pieces of 300+ nodes may start or end a trip. Check `walk` reach after graph changes.
- **SFCTA segment geometry is drawn against the direction of travel** (724 of 740) and offset
  10–20 m to its side of the street. `07_traffic.py` flips by the `direction` label and matches
  at 30 m with a direction and road-class test. Matching at 15 m caught 175 km of 899.
- **DataSF moved** from data.sfgov.org to data.sf.gov; Socrata endpoints need `$limit` or they
  return 1,000 rows. Taxi `qa_flags` containing DR (duplicate!), FTR, ST or ET must be dropped
  — DR alone is ~38% of rows. Use end − start for duration, not `fare_time_milliseconds`.
  Timestamps are true local time (DST included) despite the field description.
- **The SFMTA signal inventory mixes in beacons, flashers, radar signs and unbuilt signals.**
  Keep `type` in SIGNAL, SIGNAL (CONTRACTOR MAINTAINED), CALTRANS, CALTRANS (BY CONTRACTOR …).
- **The lidar DEM is nodata outside SF** (and over water). The graph builder falls back to the
  default 3DEP mosaic there, or Daly City streets would get cliff-edge grades at the county line.
- **The shoreline polygon counts lakes as land.** `06_build_mesh.py` subtracts OSM water.
- `pgrep -f`/`pkill -f` patterns match the shell running them: an `until ! pgrep -f X` loop
  never ends, and `pkill -f "<pattern>"` kills its own command. Match on something the
  watcher's command line can't contain, or use the PID.

## Verifying the page

- `node tools/page.js <port> "<query>" out.png [w h] [eval.js]` loads the built page headless
  (SwiftShader), waits for `#boot` to go, optionally runs a script, screenshots, and prints
  `{boot_ms, errors, readout, eval}`. `window.__sfmap` exposes `solve`, `stats`, `goto`,
  `placeState`, `minutesAt` and `META` for scripted checks.
- Use `?t=1` (skip the dissolve), `h=0&tilt=90` for the flat 2-D sheet, `nochrome=1` for stills.
  Boot is ~1.5 s headless; an "errors" array that isn't empty is a real page error.
- Do **not** kill Chrome by name to clean up: the user's own Chrome is usually running.
  Headless leftovers can be found with `pgrep -af -- --headless`.
