#!/usr/bin/env python3
"""Numeric core for the WGS 84 / LAEA Mozambique defining note.

Single source of truth for every quantitative claim in the note. Three clients
consume these functions:

    python reproduce.py     prints all sections and writes results.csv
    figures.py              imports the distortion functions for the figures
    report.qmd              calls the table functions at render time, so the
                            document cannot drift from the computation

The module is self-contained and offline. It reads the bundled national
boundary by path and verifies its SHA-256 before use, so the reported numbers
reproduce exactly and are independent of any mutable upstream dataset.

Dependencies:
    pyproj  >= 3.7   (PROJ >= 9.5)
    numpy   >= 1.24
    shapely >= 2.0
Reference environment: pyproj 3.7.2 / PROJ 9.5.1.

Two independent distortion computations are provided and cross-checked:
    omega_pyproj  authoritative; PROJ Tissot factors
    omega_svd     projection-agnostic; singular values of the local Jacobian
                  built from central geodesic differences, making no assumption
                  that the Tissot axes align with meridian and parallel

Methodological limitations: distortion is evaluated pointwise on regular grids;
onshore statistics use a 0.1 degree grid and the bundled 1:10M boundary,
adequate for distributional statistics but not for boundary-pixel adjudication.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pyproj import CRS, Geod, Proj
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import LambertAzimuthalEqualAreaConversion
from shapely.geometry import Point, shape
from shapely.prepared import prep

# =========================================================================== #
# SECTION 0. Configuration
# =========================================================================== #
HERE = Path(__file__).parent
BOUNDARY_FILE = HERE / "data" / "mozambique_boundary_ne10m.geojson"
BOUNDARY_SHA256 = "b7b0ac538b11665b680a300bba35a889f027dec612b467655640dfa9404baccc"

WGS84 = Geod(ellps="WGS84")

# All coordinates in this module are (lon, lat), degrees. One convention only.
LAEA_PROJ = "+proj=laea +lat_0=-18.5 +lon_0=35.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m"

# Equal-area candidates. All are true equal-area on the WGS 84 ellipsoid EXCEPT
# where noted. PROJ's transverse cylindrical equal area (tcea) is a spherical-only
# implementation, so it does NOT preserve area on the ellipsoid; it is therefore
# excluded from the comparison and discussed only in prose (non-registrable too).
CANDIDATES = {
    "LAEA (this CRS)": LAEA_PROJ,
    "Albers MZ -13/-24": "+proj=aea +lat_1=-13 +lat_2=-24 +lat_0=-18.5 +lon_0=35.5 +datum=WGS84 +units=m",
    "Bonne lat_1=-18.5": "+proj=bonne +lat_1=-18.5 +lon_0=35.5 +datum=WGS84 +units=m",
    "Africa Albers 102022": "+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +datum=WGS84 +units=m",
    "CEA lat_ts=-18.5": "+proj=cea +lat_ts=-18.5 +lon_0=35.5 +datum=WGS84 +units=m",
}
REGISTRABLE = ("LAEA (this CRS)", "Albers MZ -13/-24", "Bonne lat_1=-18.5",
               "Africa Albers 102022")

# Display names used in the rendered tables
LABELS = {
    "LAEA (this CRS)": "**LAEA 18.5°S / 35.5°E (this CRS)**",
    "Albers MZ -13/-24": "Albers Equal Area, parallels 13°/24° S",
    "Bonne lat_1=-18.5": "Bonne, standard parallel 18.5° S (EPSG method 9827)",
    "Africa Albers 102022": "Africa Albers (ESRI:102022, 20°N/23°S)",
    "CEA lat_ts=-18.5": "Lambert Cylindrical EA, lat_ts 18.5°S",
}

# Extreme points evaluated for Table 1. The onshore points are the actual extreme
# vertices of the bundled 1:10M boundary; the two "corner" points are the corners
# of the EPSG extent 1167 bounding box (offshore), not land.
EXTREMES = {  # (lon, lat); onshore points are exact 1:10M boundary vertices
    "N tip": (40.4539, -10.4690),    # northernmost onshore vertex
    "S tip": (32.3522, -26.8603),    # southernmost onshore vertex
    "W edge": (30.2138, -14.9885),   # westernmost onshore vertex
    "E edge": (40.8480, -14.7050),   # easternmost onshore vertex
    "NE corner": (43.03, -10.09),    # offshore bounding-box corner
    "SE corner": (43.03, -27.71),    # offshore bounding-box corner
}
TEST_POINTS = {  # (lon, lat); exact sexagesimal inputs as repeating decimals
    "Maputo": (32.5833333333, -25.9666666667),
    "Tete": (33.5833333333, -16.1666666667),
    "Pemba": (40.5000000000, -12.9666666667),
}
TEST_POINT_DMS = {
    "Maputo": ("25\u00b058\u203200.000\u2033 S", "32\u00b035\u203200.000\u2033 E"),
    "Tete": ("16\u00b010\u203200.000\u2033 S", "33\u00b035\u203200.000\u2033 E"),
    "Pemba": ("12\u00b058\u203200.000\u2033 S", "40\u00b030\u203200.000\u2033 E"),
}

ORIGIN = (35.5, -18.5)                        # (lon, lat)
EXTENT_1167 = (30.21, -27.71, 43.03, -10.09)  # (W, S, E, N) EPSG extent, Mozambique

ONSHORE_LONS = np.arange(30.2, 40.9, 0.1)
ONSHORE_LATS = np.arange(-26.9, -10.4, 0.1)


_PROJ_CACHE: dict[str, Proj] = {}


def get_proj(spec) -> Proj:
    """Return a cached Proj for a PROJ string, EPSG code, or CRS (built once)."""
    key = str(spec)
    if key not in _PROJ_CACHE:
        crs = spec if isinstance(spec, CRS) else CRS.from_user_input(spec)
        _PROJ_CACHE[key] = Proj(crs)
    return _PROJ_CACHE[key]


CRS_SCOPE = "Statistical analysis."


def authoritative_wkt() -> str:
    """Emit the authoritative WKT2:2019 for the proposed CRS.

    Built with pyproj so it is guaranteed conformant to ISO 19162:2019: the base
    is the WGS 84 datum ENSEMBLE (not a plain DATUM, which is the WKT2:2015 form),
    the coordinate system is Cartesian easting/northing E,N (EPSG CS 4400), and
    the method carries EPSG code 9820. A USAGE block with scope, area and bounding
    box is appended. The result round-trips through CRS.from_wkt.
    """
    conv = LambertAzimuthalEqualAreaConversion(
        latitude_natural_origin=-18.5, longitude_natural_origin=35.5,
        false_easting=0, false_northing=0)
    conv_d = conv.to_json_dict()
    conv_d["name"] = "Mozambique Lambert Azimuthal Equal Area"
    crs = ProjectedCRS(
        conversion=LambertAzimuthalEqualAreaConversion.from_json_dict(conv_d),
        geodetic_crs=CRS.from_epsg(4326),
        name="WGS 84 / LAEA Mozambique")
    wkt = crs.to_wkt(version="WKT2_2019", pretty=True)
    wkt = wkt.replace('AXIS["(E)",east', 'AXIS["easting (E)",east')
    wkt = wkt.replace('AXIS["(N)",north', 'AXIS["northing (N)",north')
    # Complete the WGS 84 ensemble membership to the current EPSG dataset (v11.022):
    # G2296 (datum 1383) was added in 2024. This local PROJ bundles an earlier EPSG
    # version, so the member is appended here; the ensemble list is reference-only
    # (see the note in report.qmd) and not part of the frozen normative content.
    if "G2296" not in wkt:
        wkt = wkt.replace(
            'MEMBER["World Geodetic System 1984 (G2139)"],',
            'MEMBER["World Geodetic System 1984 (G2139)"],\n'
            '            MEMBER["World Geodetic System 1984 (G2296)"],')
    usage = (',\n    USAGE[\n        SCOPE["' + CRS_SCOPE + '"],\n'
             '        AREA["Mozambique - onshore and offshore."],\n'
             '        BBOX[-27.71,30.21,-10.09,43.03]]')
    return wkt.rstrip()[:-1] + usage + ']'


# =========================================================================== #
# SECTION 1. Distortion: two independent computations
# =========================================================================== #
def omega_pyproj(spec, lon, lat):
    """Tissot maximum angular deformation omega, degrees. Authoritative.

    Accepts scalars or arrays. NaN where the projection is singular.
    """
    f = get_proj(spec).get_factors(np.asarray(lon, float), np.asarray(lat, float))
    a = np.asarray(f.tissot_semimajor, float)
    b = np.asarray(f.tissot_semiminor, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2.0 * np.degrees(np.arcsin((a - b) / (a + b)))


def omega_svd(spec, lon, lat, eps_m=500.0):
    """Independent omega via SVD of the local Jacobian (central geodesic steps).

    Projection-agnostic: makes no assumption that the Tissot principal axes
    align with meridian and parallel. Returns (omega_degrees, areal_scale).
    """
    P = get_proj(spec)
    e_lon, e_lat, _ = WGS84.fwd(lon, lat, 90.0, eps_m)
    w_lon, w_lat, _ = WGS84.fwd(lon, lat, 270.0, eps_m)
    n_lon, n_lat, _ = WGS84.fwd(lon, lat, 0.0, eps_m)
    s_lon, s_lat, _ = WGS84.fwd(lon, lat, 180.0, eps_m)
    xe, ye = P(e_lon, e_lat)
    xw, yw = P(w_lon, w_lat)
    xn, yn = P(n_lon, n_lat)
    xs, ys = P(s_lon, s_lat)
    jac = np.array([[(xe - xw) / (2 * eps_m), (xn - xs) / (2 * eps_m)],
                    [(ye - yw) / (2 * eps_m), (yn - ys) / (2 * eps_m)]])
    s1, s2 = np.linalg.svd(jac, compute_uv=False)
    return 2.0 * math.degrees(math.asin((s1 - s2) / (s1 + s2))), s1 * s2


# =========================================================================== #
# SECTION 2. Boundary and grids
# =========================================================================== #
def load_boundary():
    """Load the bundled boundary, aborting if its SHA-256 does not match."""
    blob = BOUNDARY_FILE.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != BOUNDARY_SHA256:
        raise SystemExit(
            f"Boundary hash mismatch.\n  expected {BOUNDARY_SHA256}\n  got      {got}\n"
            "The bundled boundary file has changed; results would not reproduce."
        )
    geom = shape(json.loads(blob)["features"][0]["geometry"])
    return geom, prep(geom)


def onshore_mask(prepared, lons=None, lats=None):
    """Flattened (lon, lat) arrays and a boolean onshore mask."""
    lons = ONSHORE_LONS if lons is None else lons
    lats = ONSHORE_LATS if lats is None else lats
    LO, LA = np.meshgrid(lons, lats)
    flat = np.fromiter(
        (prepared.contains(Point(x, y)) for x, y in zip(LO.ravel(), LA.ravel())),
        dtype=bool, count=LO.size,
    )
    return LO.ravel(), LA.ravel(), flat


# =========================================================================== #
# SECTION 3. Tables (pure data; consumed by report.qmd and by main)
# =========================================================================== #
def table1_extremes():
    """Table 1: omega at domain extreme points, every candidate projection."""
    rows = []
    for name, spec in CANDIDATES.items():
        row = {"projection": LABELS[name]}
        for label, pt in EXTREMES.items():
            row[label] = round(float(omega_pyproj(spec, *pt)), 3)
        row["max"] = max(v for k, v in row.items() if k != "projection")
        rows.append(row)
    return rows


def table2_onshore(prepared):
    """Table 2: onshore distribution of omega, registrable candidates."""
    lon_flat, lat_flat, mask = onshore_mask(prepared)
    rows = []
    for name in REGISTRABLE:
        w = omega_pyproj(CANDIDATES[name], lon_flat[mask], lat_flat[mask])
        w = w[np.isfinite(w)]
        rows.append({
            "projection": LABELS[name],
            "median": round(float(np.median(w)), 3),
            "p95": round(float(np.percentile(w, 95)), 3),
            "max": round(float(np.max(w)), 3),
        })
    return rows, int(mask.sum())




def table_test_points():
    """Section 8: projected test points and inverse round-trip error."""
    P = get_proj(LAEA_PROJ)
    rows = []
    for name, (lon, lat) in TEST_POINTS.items():
        easting, northing = P(lon, lat)
        lon2, lat2 = P(easting, northing, inverse=True)
        dms_lat, dms_lon = TEST_POINT_DMS[name]
        rows.append({
            "location": name, "latitude": dms_lat, "longitude": dms_lon,
            "easting_m": round(easting, 2), "northing_m": round(northing, 2),
            "roundtrip_mas": round(max(abs(lon2 - lon), abs(lat2 - lat)) * 3.6e6, 4),
        })
    return rows


def crosscheck_africa_albers():
    """Section 3: ESRI:102022 verified by two independent methods."""
    crs = CRS.from_authority("ESRI", 102022)
    rows = []
    for lat in (-10, -23):
        auth = float(omega_pyproj(crs, 35.5, lat))
        independent, areal = omega_svd(crs, 35.5, lat)
        factors = get_proj(crs).get_factors(35.5, lat)
        rows.append({
            "latitude": lat,
            "omega_pyproj": round(auth, 3),
            "omega_svd": round(independent, 3),
            "parallel_scale": round(float(np.asarray(factors.parallel_scale)), 4),
            "meridional_scale": round(float(np.asarray(factors.meridional_scale)), 4),
            "areal_scale": round(areal, 5),
        })
    return rows


def crosscheck_laea():
    """Independent verification of the proposed CRS at the extreme points."""
    rows = []
    for name, (lon, lat) in EXTREMES.items():
        auth = float(omega_pyproj(LAEA_PROJ, lon, lat))
        independent, areal = omega_svd(LAEA_PROJ, lon, lat)
        rows.append({"point": name, "omega_pyproj": round(auth, 4),
                     "omega_svd": round(independent, 4),
                     "areal_scale": round(areal, 5)})
    return rows


def origin_distances():
    """Section 2.2: geodesic distance from the natural origin to each extreme."""
    rows = []
    for label, (lon, lat) in EXTREMES.items():
        _, _, dist = WGS84.inv(*ORIGIN, lon, lat)
        rows.append({"point": label, "distance_km": round(dist / 1000)})
    return rows


def origin_optimality(prepared):
    """Show the adopted origin is near the min-max optimum for onshore omega.

    Compares three origins by their maximum onshore angular deformation: the
    adopted (rounded) origin, the exact onshore midpoint (unrounded), and the
    min-max optimal origin found by a coarse grid search. Demonstrates that
    rounding to the half degree does not cost accuracy (it in fact improves on
    the exact midpoint) and that the adopted origin is within a few percent of
    the optimum, so the choice is justified, not merely aesthetic.
    """
    lon_flat, lat_flat, mask = onshore_mask(prepared)
    lons, lats = lon_flat[mask], lat_flat[mask]

    def onshore_max(o_lon, o_lat):
        spec = (f"+proj=laea +lat_0={o_lat} +lon_0={o_lon} "
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m")
        return float(np.nanmax(omega_pyproj(spec, lons, lats)))

    adopted = onshore_max(*ORIGIN)
    midpoint = onshore_max(35.53, -18.66)          # onshore midpoint (0.01 deg)
    best_max, best_origin = 1e9, None
    for o_lat in np.arange(-19.4, -18.0, 0.05):
        for o_lon in np.arange(35.6, 37.4, 0.05):
            m = onshore_max(o_lon, o_lat)
            if m < best_max:
                best_max, best_origin = m, (round(o_lon, 2), round(o_lat, 2))
    return {
        "adopted_max": round(adopted, 3),
        "midpoint_max": round(midpoint, 3),
        "optimal_max": round(best_max, 3),
        "optimal_lon": best_origin[0],
        "optimal_lat": best_origin[1],
        "gap_percent": round((adopted - best_max) / best_max * 100, 1),
    }


# =========================================================================== #
# SECTION 4. Emitters
# =========================================================================== #
def plain(text, width=None):
    """Strip Markdown emphasis for console display."""
    out = str(text).replace("**", "")
    return out[:width] if width else out


def md_table(rows, columns=None, headers=None, suffix=None, fmt=None):
    """Render a list of dicts as a GitHub-flavoured Markdown table.

    suffix: dict {column: text} appended to each cell, e.g. the degree sign.
    fmt:    dict {column: format spec} applied to numeric cells, e.g. ".3f",
            so that trailing zeros are preserved and columns align.
    """
    if not rows:
        return ""
    columns = list(rows[0]) if columns is None else list(columns)
    headers = columns if headers is None else list(headers)
    suffix, fmt = suffix or {}, fmt or {}
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            spec = fmt.get(col)
            text = format(value, spec) if (spec and isinstance(value, (int, float))) else str(value)
            cells.append(f"{text}{suffix.get(col, '')}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_csv(sections, path=None):
    """Flatten every section into one machine-diffable CSV."""
    path = Path(path) if path else HERE / "output" / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [{"section": section, **row}
            for section, rows in sections.items() for row in rows]
    fields = sorted({k for r in flat for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return path, len(flat)


def all_sections(prepared):
    """Every table in one dict, for CSV emission and for the document."""
    t2, n_onshore = table2_onshore(prepared)
    return {
        "Table 1": table1_extremes(),
        "Table 2": t2,
        "Section 8": table_test_points(),
        "Section 3 crosscheck": crosscheck_africa_albers(),
        "CRS crosscheck": crosscheck_laea(),
        "Section 2.2": origin_distances(),
        "Origin optimality": [origin_optimality(prepared)],
    }, n_onshore


def verify(sections):
    """Assertions that must hold for the note's claims to stand."""
    worst = max(abs(r["omega_pyproj"] - r["omega_svd"])
                for r in sections["CRS crosscheck"])
    assert worst < 0.01, f"Independent methods disagree by {worst:.4f} deg"
    assert all(abs(r["areal_scale"] - 1.0) < 1e-6
               for r in sections["CRS crosscheck"]), "Areal scale is not unity"
    assert all(r["roundtrip_mas"] < 1.0
               for r in sections["Section 8"]), "Round-trip error too large"
    return worst


# =========================================================================== #
# SECTION 5. Command-line reproduction
# =========================================================================== #
def main() -> None:
    geom, prepared = load_boundary()
    print(f"boundary OK  sha256={BOUNDARY_SHA256[:12]}...  parts={len(geom.geoms)}")

    sections, n_onshore = all_sections(prepared)

    print("\n[Table 1] omega at extreme points (degrees)")
    for row in sections["Table 1"]:
        pts = {k: v for k, v in row.items() if k not in ("projection", "max")}
        print(f"  {plain(row['projection']):<40} {pts}  max {row['max']}")

    print(f"\n[Table 2] onshore distribution ({n_onshore} samples, 0.1 deg grid)")
    for row in sections["Table 2"]:
        print(f"  {plain(row['projection']):<40} median {row['median']:.3f} "
              f"p95 {row['p95']:.3f}  max {row['max']:.3f}")


    print("\n[Section 8] test points and inverse round-trip")
    for row in sections["Section 8"]:
        print(f"  {row['location']:<7} E={row['easting_m']:.2f}  N={row['northing_m']:.2f}"
              f"  roundtrip {row['roundtrip_mas']:.4f} mas")

    print("\n[Section 3] Africa Albers cross-check (ESRI:102022)")
    for row in sections["Section 3 crosscheck"]:
        print(f"  lat {row['latitude']:>4}: pyproj {row['omega_pyproj']:.3f}  "
              f"svd {row['omega_svd']:.3f}  k {row['parallel_scale']:.4f}  "
              f"h {row['meridional_scale']:.4f}  areal {row['areal_scale']:.5f}")

    worst = verify(sections)
    print(f"\n[cross-check] proposed CRS: max |pyproj - svd| = {worst:.4f} deg  PASS")

    print("\n[Section 2.2] geodesic distance from origin")
    for row in sections["Section 2.2"]:
        print(f"  origin -> {row['point']}: {row['distance_km']} km")

    path, count = write_results_csv(sections)
    print(f"\nwrote {path.name}  ({count} rows)")


if __name__ == "__main__":
    main()
