"""One flat frame for the whole city. At SF's size a local equirectangular projection is
off by under 0.1% corner to corner, and the page uses the very same two constants, so
Python and the shader agree to the metre without either doing any trigonometry.
"""
import numpy as np

LON0, LAT0 = -122.4350, 37.7700
# metres per degree on the WGS84 ellipsoid at LAT0
_p = np.radians(LAT0)
KX = 111412.84 * np.cos(_p) - 93.5 * np.cos(3 * _p)
KY = 111132.954 - 559.822 * np.cos(2 * _p) + 1.175 * np.cos(4 * _p)

# Routing covers SF plus about a kilometre past its land border, so a trip may duck
# through Daly City or Brisbane the way a real one would; only SF itself is drawn.
BBOX = dict(s=37.6990, w=-122.5300, n=37.8350, e=-122.3450)
QSTEP = 0.25                     # metres per unit when coordinates ship as uint16


def to_xy(lon, lat):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


def to_ll(x, y):
    return np.asarray(x) / KX + LON0, np.asarray(y) / KY + LAT0


XMIN, YMIN = to_xy(BBOX["w"], BBOX["s"])
XMAX, YMAX = to_xy(BBOX["e"], BBOX["n"])
assert (XMAX - XMIN) / QSTEP < 65536 and (YMAX - YMIN) / QSTEP < 65536


def quantize(x, y):
    q = np.empty((len(x), 2), np.uint16)
    q[:, 0] = np.round((np.asarray(x) - XMIN) / QSTEP)
    q[:, 1] = np.round((np.asarray(y) - YMIN) / QSTEP)
    return q
