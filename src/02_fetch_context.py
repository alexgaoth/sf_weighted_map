"""Everything else the map stands on, from public servers, into data/raw/.

  dem.tif, dem_lidar.tif   USGS 3DEP, 10 m, UTM 10N. The lidar copy is the 2023 SF flight
                           (water and everything outside SF are nodata); the default mosaic
                           fills in past the county line.
  shoreline.geojson        DataSF "SF Shoreline and Islands" — land only
  analysis_neighborhoods   DataSF, 41 polygons (the drawn boundaries)
  find_neighborhoods       DataSF "SF Find Neighborhoods", 117 polygons (the named places)
  signals, stop_signs      SFMTA inventories on DataSF
and writes data/heliports.csv, the five pads inside SF (FAA ADDS / OurAirports).
"""
import json
import pathlib
import urllib.parse
import urllib.request

raw = pathlib.Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "sf-weighted-map/0.1 (personal, non-commercial)"}

DEM = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
       "exportImage?bbox=542230,4172630,557310,4187710&bboxSR=32610&imageSR=32610"
       "&size=1508,1508&format=tiff&pixelType=F32&noData=-9999"
       "&interpolation=RSP_BilinearInterpolation&compression=LZ77"
       "&renderingRule=%7B%22rasterFunction%22%3A%22None%22%7D")
# The four rasters of the CA_SanFrancisco_B23 lidar. If USGS rebuilds the mosaic these ids
# change; look them up again at .../ImageServer/query?where=Name='CA_SanFrancisco_B23'
LOCK = ("&mosaicRule=" + urllib.parse.quote(json.dumps(
    {"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [66659, 71693, 77435, 98650]},
    separators=(",", ":"))))
SODA = "https://data.sf.gov/resource/{}.geojson?$limit=50000"
FILES = {
    "dem.tif": DEM + "&f=image",
    "dem_lidar.tif": DEM + LOCK + "&f=image",
    "shoreline.geojson": SODA.format("txuc-3kzm"),
    "analysis_neighborhoods.geojson": SODA.format("j2bu-swwd"),
    "find_neighborhoods.geojson": SODA.format("gfpk-269f"),
    "signals.geojson": SODA.format("ybh5-27n2"),
    "stop_signs.geojson": SODA.format("4542-gpa3"),
}
for name, url in FILES.items():
    out = raw / name
    if out.exists():
        print(f"{out} exists, skipping")
        continue
    body = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
    out.write_bytes(body)
    print(f"{out}: {len(body) / 1e6:.1f} MB")

# The heliports inside SF, chosen by hand from FAA ADDS and OurAirports — so they live in
# data/, committed, not in data/raw/. All are private-use, and Alcatraz's is closed and left
# out; the map treats the rest as a what-if, a helicopter waiting at whichever pad suits you.
pads = pathlib.Path("data/heliports.csv")
if not pads.exists():
    pads.write_text(
        "ident,name,lat,lon\n"
        "CA26,Hall of Justice,37.77510,-122.40429\n"
        "18CN,UCSF Mission Bay,37.76578,-122.39037\n"
        "CN02,SF VA Medical Center,37.78352,-122.50680\n"
        "16CA,Police Pistol Range,37.71998,-122.49915\n"
        "US-0225,Treasure Island,37.81990,-122.36477\n")
    print(f"{pads}: 5 pads (FAA ADDS + OurAirports, checked 2026-10-17)")
