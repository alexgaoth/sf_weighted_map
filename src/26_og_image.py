"""Render dist/og.png — the link preview, straight from the live page: Bernal Heights by
car at Thursday 4:45 pm, tilted just enough to show the relief. Needs `npm install` in tools/ once."""
import pathlib
import subprocess
import sys

PORT = 8744
QUERY = "index.html?t=1&nochrome=1&tilt=58&spin=0&h=0.3"
dist = pathlib.Path("dist")
srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1",
                        "-d", str(dist)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    subprocess.run(["sh", "-c", f"until curl -sf -o /dev/null http://127.0.0.1:{PORT}/index.html; "
                                "do sleep 0.3; done"], check=True, timeout=30)
    subprocess.run(["node", "tools/page.js", str(PORT), QUERY, str(dist / "og.png"), "1200", "630"],
                   check=True, capture_output=True, timeout=300)
finally:
    srv.terminate()
print(f"dist/og.png  {(dist / 'og.png').stat().st_size / 1024:.0f} KB")
