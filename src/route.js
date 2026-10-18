// Shortest-path trees over the street graph, run in the page. Inlined into the map by
// the build. `25_check_parity.py` runs the live page against router.py, its Python twin.
//
// The graph is CSR by source node: edges off[u]..off[u+1]-1 leave u, going to tgt[e].
// Every cost is in deciseconds and already includes the delay of the intersection the
// edge arrives at, so a search here is plain Dijkstra with nothing per node.

// A binary heap of (key, node) on typed arrays. Stale entries are left in and skipped on
// pop, which beats a decrease-key on graphs this sparse.
function makeHeap(cap) {
  let key = new Float64Array(cap), val = new Int32Array(cap), n = 0;
  return {
    get size() { return n; },
    clear() { n = 0; },
    push(k, v) {
      if (n === key.length) {
        const k2 = new Float64Array(n * 2), v2 = new Int32Array(n * 2);
        k2.set(key); v2.set(val); key = k2; val = v2;
      }
      let i = n++;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (key[p] <= k) break;
        key[i] = key[p]; val[i] = val[p]; i = p;
      }
      key[i] = k; val[i] = v;
    },
    // returns the node; the key it was pushed with is left in `top`
    pop() {
      const v = val[0]; this.top = key[0];
      const k = key[--n], x = val[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && key[c + 1] < key[c]) c++;
        if (key[c] >= k) break;
        key[i] = key[c]; val[i] = val[c]; i = c;
      }
      key[i] = k; val[i] = x;
      return v;
    },
    top: 0,
  };
}

// Fill `dist` (seconds) from a set of seeds. `cost[e]` is deciseconds for edge e under the
// current mode and hour, negative for an edge this mode may not use. `dist` must be a
// Float64Array: the heap keys are doubles, and a Float32 store rounds a node's distance
// below its own key, so the `du > dist[u]` test discards it and it is never expanded.
function dijkstra(G, seeds, seedSecs, cost, dist, heap) {
  dist.fill(Infinity);
  heap.clear();
  for (let i = 0; i < seeds.length; i++) {
    const s = seeds[i], d = seedSecs[i] * 10;
    if (d < dist[s]) { dist[s] = d; heap.push(d, s); }
  }
  const off = G.off, tgt = G.tgt;
  while (heap.size) {
    const u = heap.pop(), du = heap.top;
    if (du > dist[u]) continue;
    for (let e = off[u], end = off[u + 1]; e < end; e++) {
      const c = cost[e];
      if (c < 0) continue;
      const v = tgt[e], nd = du + c;
      if (nd < dist[v]) { dist[v] = nd; heap.push(nd, v); }
    }
  }
  for (let i = 0; i < dist.length; i++) dist[i] *= 0.1;
  return dist;
}

if (typeof module !== "undefined") module.exports = { makeHeap, dijkstra };
