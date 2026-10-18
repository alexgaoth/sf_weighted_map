// in the page: median minutes to the rest of SF for a spread of places, hours and modes
(() => {
  const m = window.__sfmap, out = [];
  const n = m.META.places.length;
  for (let p = 0; p < n; p += 17) for (const h of [3, 41, 88, 113, 149]) {
    const r = m.solve(m.placeState(p, 2, 4 * h + 2));
    out.push([p, h, [0, 1, 2, 3].map((k) => m.stats(r.grid[k]).med)]);
  }
  return out;
})()
