import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

DISTRICTS_GEOJSON = Path(__file__).resolve().parent.parent / "data" / "external" / "community_districts.geojson"


@lru_cache(maxsize=4)
def load_polygons(geojson_path=DISTRICTS_GEOJSON):
    """[(boro_cd, outer ring path, hole paths, bbox min, bbox max)] for every polygon part."""
    polys = []
    for f in json.loads(Path(geojson_path).read_text())["features"]:
        code = int(f["properties"]["boro_cd"])
        geom = f["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for rings in parts:
            outer = np.asarray(rings[0], dtype=float)
            holes = tuple(MplPath(np.asarray(r, dtype=float)) for r in rings[1:])
            polys.append((code, MplPath(outer), holes, outer.min(0), outer.max(0)))
    return polys


def assign_districts(lat, lon, geojson_path=DISTRICTS_GEOJSON):
    """boro_cd of the polygon containing each point (None outside every polygon or if lat/lon is missing)."""
    pts = np.column_stack([np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)])
    out = np.full(len(pts), None, dtype=object)
    todo = ~np.isnan(pts).any(axis=1)
    for code, path, holes, lo, hi in load_polygons(Path(geojson_path)):
        cand = todo & (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
        if not cand.any():
            continue
        idx = np.flatnonzero(cand)
        inside = path.contains_points(pts[idx])
        for hole in holes:
            inside &= ~hole.contains_points(pts[idx])
        hit = idx[inside]
        out[hit] = code
        todo[hit] = False
    return out
