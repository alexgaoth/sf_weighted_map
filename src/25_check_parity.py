"""The page routes in JavaScript; the graphs were measured in Python. Load the built page
headless, have it compute medians for a spread of places, hours and modes, and compare
them with data/curves.npy. Fails if any differs by more than a second.

Needs `npm install` in tools/ once.
"""
import json
import subprocess
import sys

import numpy as np

PORT = 8743
srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1",
                        "-d", "dist"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    subprocess.run(["sh", "-c", f"until curl -sf -o /dev/null http://127.0.0.1:{PORT}/index.html; "
                                "do sleep 0.3; done"], check=True, timeout=30)
    run = subprocess.run(["node", "tools/page.js", str(PORT), "index.html?t=1&check=1", "-",
                          "1200", "800", "tools/parity_eval.js"],
                         capture_output=True, text=True, timeout=600, check=True)
finally:
    srv.terminate()
res = json.loads(run.stdout.strip().splitlines()[-1])
assert not res["errors"], res["errors"]
curves = np.load("data/curves.npy")
worst = 0.0
for p, h, meds in res["eval"]:
    worst = max(worst, float(np.max(np.abs(np.array(meds) - curves[p, :, h]))))
print(f"page booted in {res['boot_ms']} ms; {len(res['eval'])} place-hours x 4 modes; "
      f"worst |page - Python| = {worst * 60:.2f} s")
sys.exit(0 if worst * 60 < 1.0 else 1)
