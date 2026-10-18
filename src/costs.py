"""How fast each mode moves over a short piece of street, given its slope.

Everything here is per metre of horizontal distance on a slope s = rise / run, positive
uphill in the direction of travel. Edges are costed by summing these over ~10 m pieces,
so a street that climbs and then drops is charged for both, not for its net grade.
"""
import numpy as np

# ---- walking ------------------------------------------------------------------------
# Tobler's hiking function, scaled so flat ground is 1.34 m/s (4.8 km/h, the usual urban
# figure) rather than Tobler's 5.0 km/h. Its peak sits on a gentle descent, and it slows
# on steep descents as well as climbs, which is what an SF hill actually does to you.
WALK_FLAT = 1.34
TOBLER_K = WALK_FLAT / (6.0 / 3.6 * np.exp(-3.5 * 0.05))
# Stairways are not slopes. Fruin's measured horizontal speeds on stairs, which Tobler on
# a 60% grade would undercut by half.
STAIRS_UP, STAIRS_DOWN = 0.50, 0.60


def walk_speed(s, steps=False):
    s = np.asarray(s, float)
    if steps:
        return np.where(s >= 0, STAIRS_UP, STAIRS_DOWN)
    return TOBLER_K * 6.0 / 3.6 * np.exp(-3.5 * np.abs(s + 0.05))


# ---- cycling ------------------------------------------------------------------------
# A commuter on an ordinary bike: 150 W sustained, 90 kg with the bike, upright.
# Flat out that is 7 m/s, but nobody rides 25 km/h between stop signs, so the flat cruise
# is capped at 5 m/s (18 km/h) and gentle descents at 8.3 m/s (30 km/h). Steeper than 6%
# the rider is braking, not coasting: the cap falls to 9 km/h by a 30% drop. Where the
# power model falls below walking pace the rider gets off and pushes, a little slower
# than walking.
BIKE_W, BIKE_KG, CRR, CDA, RHO, G = 150.0, 90.0, 0.006, 0.5, 1.2, 9.81
BIKE_CRUISE, BIKE_DESCENT, PUSH = 5.0, 8.3, 0.9


def _power_speed(s):
    """Solve P = (m g (sin + Crr cos)) v + 0.5 rho CdA v^3 for v, by bisection."""
    th = np.arctan(s)
    a = 0.5 * RHO * CDA
    b = BIKE_KG * G * (np.sin(th) + CRR * np.cos(th))
    lo, hi = np.zeros_like(s), np.full_like(s, 30.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_fast = a * mid ** 3 + b * mid > BIKE_W
        hi = np.where(too_fast, mid, hi)
        lo = np.where(too_fast, lo, mid)
    return (lo + hi) / 2


def bike_speed(s):
    s = np.asarray(s, float)
    braking = np.maximum(2.5, BIKE_DESCENT - 25.0 * np.maximum(0.0, -s - 0.06))
    v = np.minimum(_power_speed(s), np.where(s < -0.01, braking, BIKE_CRUISE))
    return np.maximum(v, PUSH * walk_speed(s))


# ---- intersections -------------------------------------------------------------------
# Seconds lost passing through a controlled node, folded into the cost of the edge that
# arrives at it. SF's stop signs are overwhelmingly all-way, so a stop costs every
# approach. Signal delay is the random-arrival expectation for a ~75 s cycle.
DELAY = {"walk": {"signal": 12.0, "stop": 0.0},
         "bike": {"signal": 15.0, "stop": 4.0},
         "car": {"signal": 18.0, "stop": 7.0}}


# ---- helicopter ---------------------------------------------------------------------
# A what-if: a helicopter waiting at whichever of SF's five pads suits you. Every pad in
# the city is private-use, so nobody can actually do this. Drive to a pad, board and spin
# up, fly the straight line, shut down and get out, then drive again from the far pad.
HELI = {"dep": 600.0,          # s: at the pad, briefed, aboard, rotors up to speed
        "fix": 180.0,          # s: take-off and climb, approach and landing
        "cruise": 61.7,        # m/s: 120 kn, a light twin like an EC135 or a Bell 407
        "arr": 180.0}          # s: rotors down, out, into the car waiting at the pad

# ---- the map ------------------------------------------------------------------------
VREF = 500.0                   # metres of map per minute of travel: the 1.0x scale, 30 km/h
# The longest walk in from the nearest usable node. Every mainland cell is within this of a
# street or path — the fenced-off Hunters Point Shipyard needs most of it — while Treasure
# Island sits 2.5 km of water from the mainland's, so on foot or by bike the island reads
# as unreachable instead of being swum to.
SNAP_MAX = 1000.0
# Gaussian sigma, in 40 m grid cells, of the field the sheet is drawn from (never the
# numbers shown): street-scale noise in time would otherwise crumple the warp.
SMOOTH = 2.5
# Thursday 4:45 pm from Bernal Heights, by car — the trip a reply to the original post
# asked about: "south of Bernal Heights to ... Union Square must be really bad on a
# Thursday 4:30-5:00pm".
START = {"from": "Bernal Heights", "mode": 2, "when": 3 * 96 + 16 * 4 + 3}


def model():
    return dict(vref=VREF, walk=WALK_FLAT, heli=HELI, start=START, snapMax=SNAP_MAX,
                smooth=SMOOTH)


if __name__ == "__main__":
    for g in (-0.30, -0.15, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        print(f"grade {g:+.2f}   walk {walk_speed(g) * 3.6:4.1f} km/h   "
              f"bike {bike_speed(g) * 3.6:4.1f} km/h")
