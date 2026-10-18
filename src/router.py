"""The page's router, in Python, for everything measured offline: the week curves, the
colour spans, the graphs. It reads the very arrays the page ships (data/payload.npz,
written by 20_build_payload.py) and repeats the page's arithmetic step for step, so a
number on the graphs page is a number the map would show. `25_check_parity.py` holds
the two to each other.

Mirrors, in map_template.html: carCostsAt, seeds, fillGrid, heliFrom, stats.
"""
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
from costs import SNAP_MAX

GONE = 30000.0
K3 = 3


class Router:
    def __init__(self, path="data/payload.npz"):
        P = np.load(path, allow_pickle=False)
        self.P = P
        q = P["q"]                                     # x0, y0, step
        self.nx = (q[0] + P["node_q"][:, 0] * q[2]).astype(np.float32).astype(np.float64)
        self.ny = (q[1] + P["node_q"][:, 1] * q[2]).astype(np.float32).astype(np.float64)
        self.N = len(self.nx)
        self.src = np.repeat(np.arange(self.N), np.diff(P["off"]))
        self.tgt = P["tgt"].astype(np.int64)
        self.cost = {"walk": P["cost_walk"].astype(np.float64), "bike": P["cost_bike"].astype(np.float64)}
        self.car_free = P["cost_car"].astype(np.float64)
        self.car_prof = P["car_prof"].astype(np.int64)
        self.prof = P["prof"].astype(np.float64).reshape(-1, 168)
        self.walk = float(P["walk"])
        self.snap = P["node_snap"]
        g = P["grid"]                                  # x0, y0, cell, nx, ny
        self.gx0, self.gy0, self.cell, self.GW, self.GH = g[0], g[1], g[2], int(g[3]), int(g[4])
        self.land = P["land"]
        self.act = np.flatnonzero(self.land)
        self.land_cells = np.flatnonzero(self.land == 1)
        cx = self.gx0 + (self.act % self.GW) * self.cell
        cy = self.gy0 + (self.act // self.GW) * self.cell
        # the K nearest snappable nodes per cell, per snapping class (1 walk, 2 bike, 4 car)
        self.trees, self.cell_node, self.cell_walk = {}, {}, {}
        for s, bit in enumerate((1, 2, 4)):
            ids = np.flatnonzero(self.snap & bit)
            tree = cKDTree(np.column_stack([self.nx[ids], self.ny[ids]]))
            d, j = tree.query(np.column_stack([cx, cy]), k=K3, distance_upper_bound=SNAP_MAX)
            self.trees[s] = (tree, ids)
            miss = ~np.isfinite(d)
            self.cell_node[s] = ids[np.where(miss, 0, j)]
            self.cell_walk[s] = np.where(miss, np.inf, d / self.walk)
        self.heli = {k: float(P["heli"][i]) for i, k in enumerate(("dep", "fix", "cruise", "arr"))}
        self.pads = P["pads"]                          # n x 2, metres
        self._mats = {}

    # ---- costs ------------------------------------------------------------------------
    def car_costs(self, slot):
        h = slot / 4 - 0.5
        h0 = int(np.floor(h)); f = h - h0
        a, b = h0 % 168, (h0 + 1) % 168
        mul = (self.prof[:, a] + (self.prof[:, b] - self.prof[:, a]) * f) / 1000
        c = self.car_free * mul[self.car_prof]
        c[self.car_free == 65535] = -1
        return c

    def mode_costs(self, mode, slot=None):
        if mode in ("walk", "bike"):
            c = self.cost[mode].copy()
            c[c == 65535] = -1
            return c
        return self.car_costs(slot)

    # ---- search -----------------------------------------------------------------------
    def seeds(self, snap_class, x, y):
        tree, ids = self.trees[snap_class]
        d, j = tree.query([x, y], k=K3, distance_upper_bound=SNAP_MAX)
        ok = np.isfinite(d)
        return ids[j[ok]], d[ok] / self.walk

    def trees_from(self, costs, seed_lists):
        """Node seconds from each of several seed sets. A virtual source per set carries
        the walk-in to each seed, so this is one call to scipy for all of them."""
        ok = costs >= 0
        u, v, w = self.src[ok], self.tgt[ok], costs[ok]
        # duplicate (u, v) pairs would be summed by scipy; the page takes the cheaper
        key = u * self.N + v
        order = np.lexsort((w, key))
        key, u, v, w = key[order], u[order], v[order], w[order]
        first = np.r_[True, key[1:] != key[:-1]]
        u, v, w = u[first], v[first], w[first]
        S = len(seed_lists)
        su = np.concatenate([np.full(len(ids), self.N + k) for k, (ids, _) in enumerate(seed_lists)])
        sv = np.concatenate([ids for ids, _ in seed_lists])
        sw = np.concatenate([secs * 10 for _, secs in seed_lists])
        # a seed sitting exactly on its node would be a zero, which csgraph reads as no edge
        sw = np.maximum(sw, 1e-9)
        M = csr_matrix((np.r_[w, sw], (np.r_[u, su], np.r_[v, sv])), shape=(self.N + S,) * 2)
        D = dijkstra(M, directed=True, indices=np.arange(self.N, self.N + S))
        return D[:, :self.N] * 0.1

    # ---- grids ------------------------------------------------------------------------
    def grid(self, mode, T):
        """Node seconds -> minutes on the active cells (GONE where unreachable)."""
        s = {"walk": 0, "bike": 1}.get(mode, 2)
        t = T[..., self.cell_node[s]] + self.cell_walk[s]
        best = t.min(axis=-1)
        return np.where(np.isfinite(best), best / 60, GONE)

    def heli_from(self, carT, slot, pad_trees=None):
        """carT: (S, N) node seconds by car from S origins; returns heli node seconds."""
        pad_seeds = [self.seeds(2, *p) for p in self.pads]
        if pad_trees is None:
            pad_trees = self.trees_from(self.car_costs(slot), pad_seeds)
        n = len(self.pads)
        at = np.stack([np.min(carT[:, ids] + secs, axis=1) for ids, secs in pad_seeds], axis=1)
        out = carT.copy()
        for j in range(n):
            land = np.full(len(carT), np.inf)
            for i in range(n):
                if i == j:
                    continue
                km = np.hypot(*(self.pads[i] - self.pads[j]))
                land = np.minimum(land, at[:, i] + self.heli["dep"] + self.heli["fix"]
                                  + km / self.heli["cruise"] + self.heli["arr"])
            out = np.minimum(out, land[:, None] + pad_trees[j][None, :])
        return out

    def stats(self, cells_minutes):
        """median over land, matching the page: the n>>1-th of the sorted reachable values."""
        on_land = np.isin(self.act, self.land_cells)
        v = cells_minutes[..., on_land]
        v = np.where(v >= GONE, np.nan, v)
        srt = np.sort(v, axis=-1)
        n = np.sum(np.isfinite(v), axis=-1)
        return np.take_along_axis(srt, (n >> 1)[..., None], axis=-1)[..., 0]
