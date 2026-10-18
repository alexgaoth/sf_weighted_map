# sf_weighted_map

The SF twin of `../weighted_map`. `README.md` has the pipeline and method; this is what bites.

## Commands

- `.venv/bin/python` (3.12 via `uv`; system python is 3.14). `cd tools && npm install` once.
- Scripts run in filename order; after any upstream change rerun `20 → 21 → 23 → 24`, then
  `25_check_parity.py`. `22_validate_taxi.py --sweep` (~10 min) is the only way to re-derive
  `CONGESTION_GAIN`/`CAR_BIAS`, which go stale whenever matching, damping or taxi factors change.
- View with `python3 -m http.server -d dist` — the page fetches `sfmap.bin.gz`; `file://` fails.

## Invariants kept by hand

- The page's router and `src/router.py` are twins (seeds, snapping, `fillGrid`, `carCostsAt`,
  `heliFrom`): change both, then run `25_check_parity.py` (fails above 1 s).
- `route.grid` holds exact times (tips, medians, stats); `route.disp` is smoothed by `SMOOTH`
  for geometry and framing only — never read a number off it.
- Constants live in `src/costs.py` and reach the page via `costs.model()` → META; never
  hard-code `VREF`, `SNAP_MAX`, `SMOOTH`, the walk speed or heli constants in the template.
- `car_prof` is uint16 (1,464 profiles) in `20_build_payload.py`, `24_build_dist.py` and the
  template's `grab` — change all three together.
- `lift()` in the shader and on the CPU must match, near-origin fade included, or dots float.

## Gotchas

- Unreachable is the texture's second channel (`RG16F`), never a sentinel time: filtering
  blended a 30,000-min sentinel into neighbours and flung vertices hundreds of km.
- Dropping sidewalks strands plaza footpaths (Union Square once reached 1% of SF on foot);
  after any graph change, check walk reach from a few origins before trusting medians.
- Taxi data: drop rows whose `qa_flags` hold DR (duplicates, ~38%), FTR, ST or ET; duration is
  end − start, not `fare_time_milliseconds`; timestamps are local time, DST included.

## Verifying the page

- `node tools/page.js <port> "<query>" out.png [w h] [eval.js]` boots the page headless
  (SwiftShader), runs an optional script, screenshots, and prints `{boot_ms, errors, readout,
  eval}`; `window.__sfmap` exposes `solve`, `stats`, `goto`, `placeState`, `minutesAt`, `META`.
- Plain `google-chrome --headless --screenshot --virtual-time-budget` hangs on this page.
- `?t=1` skips the dissolve, `h=0&tilt=90` is the flat sheet, `nochrome=1` hides the panels.
- Never kill Chrome by name (the user's own is running, whatever `../weighted_map` says);
  find headless leftovers with `pgrep -af -- --headless`.
