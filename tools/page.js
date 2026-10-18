// Load the built page headless, wait for it to boot, optionally evaluate a script in it,
// and screenshot it. Prints one JSON line: {boot_ms, errors, readout, eval}.
//   node tools/page.js <port> "<query>" <out.png> [width height] [eval.js]
// Uses the system Chrome; SwiftShader stands in for a GPU.
const puppeteer = require("puppeteer-core");
const fs = require("fs");
(async () => {
  const [port, query, out, w = "1400", h = "900", evalFile] = process.argv.slice(2);
  const exe = ["/usr/bin/google-chrome-stable", "/usr/bin/google-chrome", "/usr/bin/chromium"]
    .find((p) => fs.existsSync(p));
  const browser = await puppeteer.launch({ executablePath: exe, headless: "new",
    args: ["--no-sandbox", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
           "--hide-scrollbars", "--force-color-profile=srgb"] });
  const page = await browser.newPage();
  await page.setViewport({ width: +w, height: +h, deviceScaleFactor: 1 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && !/favicon|404/.test(m.text()) && errors.push(m.text()));
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${port}/${query}`, { waitUntil: "domcontentloaded" });
  // ready when #boot fades (the map) or immediately (the graphs page has none)
  await page.waitForFunction(() => { const b = document.getElementById("boot");
    return !b || b.classList.contains("gone") || /could not|needs|failed/.test(b.textContent); },
    { timeout: 120000, polling: 250 });
  const boot_ms = Date.now() - t0;
  await new Promise((r) => setTimeout(r, 2500));
  const res = evalFile ? await page.evaluate(fs.readFileSync(evalFile, "utf8")) : null;
  if (out !== "-") await page.screenshot({ path: out });
  const readout = await page.evaluate(() => ["ro-where", "ro-big", "ro-sub"].map(
    (id) => document.getElementById(id)?.textContent).join(" | "));
  console.log(JSON.stringify({ boot_ms, errors, readout, eval: res }));
  await browser.close();
})();
