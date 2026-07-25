#!/usr/bin/env python3
"""Regenerate every figure in the WGS 84 / LAEA Mozambique defining note.
 
All distortion mathematics is imported from reproduce.py, so the figures and
the tables are computed by exactly the same functions and cannot drift apart.
The national boundary is the bundled, hash-verified file loaded by that module.
 
Dependencies:
    pyproj  >= 3.7   (PROJ >= 9.5)
    numpy   >= 1.24
    shapely >= 2.0
    matplotlib >= 3.8
Reference environment: pyproj 3.7.2 / PROJ 9.5.1 / matplotlib 3.10.
 
Outputs (written next to this script):
    fig1_omega_field.png     distortion field of the proposed CRS
    fig2_omega_cdf.png       onshore cumulative distribution, registrable CRSs
    fig3_country_fill.png    the country in each projection, common scale
 
Methodological limitations: distortion is evaluated pointwise on regular grids;
onshore statistics use a 0.1 degree grid and the bundled 1:10M boundary.
"""
from __future__ import annotations
 
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from pyproj import CRS, Proj, Transformer
from shapely.ops import transform as shapely_transform
 
from reproduce import (
    CANDIDATES,
    COMPARED,
    EXTENT_1167,
    EXTREMES,
    LAEA_PROJ,
    ORIGIN,
    TEST_POINTS,
    get_proj,
    load_boundary,
    omega_pyproj,
    onshore_mask,
)
 
HERE = Path(__file__).parent
DPI = 200
 
# Shared cartographic conventions
CMAP = "YlOrRd"
CAP = 1.5          # colour scale cap, degrees; keeps the low-distortion contest visible
ISOLINES = (0.25, 0.5, 1.0)
 
 
# --------------------------------------------------------------------------- #
# Small geometry helpers (no geopandas dependency)
# --------------------------------------------------------------------------- #
def polygon_rings(geom):
    """Yield exterior coordinate arrays of a Polygon or MultiPolygon."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        yield np.asarray(poly.exterior.coords)
 
 
def draw_outline(ax, geom, **kwargs):
    for ring in polygon_rings(geom):
        ax.plot(ring[:, 0], ring[:, 1], **kwargs)
 
 
def clip_patch(ax, geom):
    """Invisible patch of the country used to clip filled contours."""
    verts, codes = [], []
    for ring in polygon_rings(geom):
        verts.extend(map(tuple, ring))
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1))
    patch = PathPatch(MplPath(verts, codes), transform=ax.transData,
                      facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    return patch
 
 
def segment_midpoints(contour_set, min_vertices=4):
    """One representative point per contour segment.
 
    Matplotlib stores each level as a compound path whose disconnected
    segments are separated by MOVETO codes. Radially symmetric fields cut the
    country in several places, and short segments are skipped by the automatic
    labeller, so positions are supplied explicitly.
    """
    points = []
    for path in contour_set.get_paths():
        verts, codes = path.vertices, path.codes
        if codes is None:
            segments = [verts]
        else:
            starts = np.flatnonzero(codes == MplPath.MOVETO)
            segments = np.split(verts, starts[1:]) if starts.size > 1 else [verts]
        for seg in segments:
            if len(seg) >= min_vertices:
                points.append(tuple(seg[len(seg) // 2]))
    return points
 
 
def project_geom(geom, proj_spec):
    tr = Transformer.from_crs(4326, get_proj(proj_spec).crs, always_xy=True).transform
    return shapely_transform(tr, geom)
 
 
# --------------------------------------------------------------------------- #
# Figure 1: distortion field of the proposed CRS
# --------------------------------------------------------------------------- #
def figure1(geom):
    lons = np.arange(29.5, 43.6, 0.06)
    lats = np.arange(-28.2, -9.5, 0.06)
    LO, LA = np.meshgrid(lons, lats)
    W = omega_pyproj(LAEA_PROJ, LO.ravel(), LA.ravel()).reshape(LO.shape)
 
    fig, ax = plt.subplots(figsize=(6.0, 7.0))
    filled = ax.contourf(LO, LA, W, levels=np.arange(0, 0.66, 0.04), cmap="YlGnBu")
    lines = ax.contour(LO, LA, W, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                       colors="0.25", linewidths=0.5)
    ax.clabel(lines, fmt="%.1f°", fontsize=7)
    draw_outline(ax, geom, color="crimson", linewidth=1.1)
 
    west, south, east, north = EXTENT_1167
    ax.add_patch(Rectangle((west, south), east - west, north - south,
                           fill=False, edgecolor="dimgray", linestyle="--", linewidth=0.9))
    ax.plot(*ORIGIN, marker="*", color="black", markersize=14, linestyle="none",
            label="Natural origin (35.5°E, 18.5°S)")
    for name, (lon, lat) in TEST_POINTS.items():
        ax.plot(lon, lat, "k^", markersize=5)
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(5, 3), fontsize=8)
 
    fig.colorbar(filled, ax=ax, shrink=0.75,
                 label="Maximum angular deformation ω (degrees)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("WGS 84 / LAEA Mozambique: shape distortion field", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect(1 / np.cos(np.radians(18.5)))
    fig.tight_layout()
    return fig
 
 
# --------------------------------------------------------------------------- #
# Figure 2: onshore cumulative distribution
# --------------------------------------------------------------------------- #
def figure2(prepared):
    lons = np.arange(30.2, 40.9, 0.1)
    lats = np.arange(-26.9, -10.4, 0.1)
    lon_flat, lat_flat, mask = onshore_mask(prepared, lons, lats)
 
    colours = {"LAEA (this CRS)": "#1a6390",
               "Albers MZ -13/-24": "#c0392b",
               "Bonne lat_1=-18.5": "#8e44ad",
               "GLANCE Africa 10592": "#2e8b57",
               "Africa Albers 102022": "#7f8c8d"}
    labels = {"LAEA (this CRS)": "WGS 84 / LAEA Mozambique (this CRS)",
              "Albers MZ -13/-24": "Albers Equal Area 13°/24° S",
              "Bonne lat_1=-18.5": "Bonne, standard parallel 18.5° S",
              "GLANCE Africa 10592": "WGS 84 / GLANCE Africa (EPSG:10592)",
              "Africa Albers 102022": "Africa Albers (ESRI:102022)"}
 
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(50, color="0.55", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(0.0055, 51.5, "half the country", fontsize=7, color="0.4")
 
    for i, name in enumerate(COMPARED):
        w = omega_pyproj(CANDIDATES[name], lon_flat[mask], lat_flat[mask])
        w = np.sort(w[np.isfinite(w)])
        cdf = np.arange(1, w.size + 1) / w.size * 100
        ax.plot(w, cdf, label=labels[name], color=colours[name],
                linewidth=1.8, zorder=3)
        median, worst = float(np.median(w)), float(w[-1])
        ax.plot([median], [50], "o", color=colours[name], markersize=5, zorder=4)
        # mark where the whole country is covered
        ax.plot([worst], [100], "|", color=colours[name], markersize=9,
                markeredgewidth=1.8, zorder=4)
        # place each worst-value label above its own marker, in the white space over
        # the 100% line; stagger the close pair (Albers 0.71 / Bonne 0.66) in height
        ax.annotate(f"{worst:.2f}°", (worst, 100), textcoords="offset points",
                    xytext=(0, 4 + (i % 2) * 12), ha="center", va="bottom",
                    fontsize=7.5, color=colours[name])
 
    ax.set_xscale("log")
    ax.set_xlim(0.005, 10)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Shape distortion ω (degrees, log scale)")
    ax.set_ylabel("Percentage of the country\nbelow that distortion (%)")
    ax.set_title("How much of Mozambique stays below a given shape distortion\n"
                 f"({int(mask.sum())} sample points; every curve covers the same "
                 "country and reaches 100%)", fontsize=9.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig
 
 
# --------------------------------------------------------------------------- #
# Figure 3: the country in each projection, common metric scale
# --------------------------------------------------------------------------- #
def figure3(geom, prepared):
    panels = [
        ("LAEA (this CRS)", "WGS 84 / LAEA Mozambique", "Isotropic, minimal, uniform"),
        ("Albers MZ -13/-24", "Albers Equal Area 13°/24° S", "Low, banded by latitude"),
        ("Africa Albers 102022", "Africa Albers (ESRI:102022)", "Continental fallback: strong"),
    ]
 
    # Sampling grid and onshore mask, shared by every panel
    lons = np.arange(29.3, 41.8, 0.07)
    lats = np.arange(-27.7, -9.6, 0.07)
    LO, LA = np.meshgrid(lons, lats)
    _, _, mask_flat = onshore_mask(prepared, lons, lats)
    inside = mask_flat.reshape(LO.shape)
 
    # Common metric window so the country is the same size in every panel
    geoms, half = {}, 0.0
    for key, _, _ in panels:
        gp = project_geom(geom, CANDIDATES[key])
        minx, miny, maxx, maxy = gp.bounds
        geoms[key] = (gp, (minx + maxx) / 2, (miny + maxy) / 2)
        half = max(half, (maxx - minx) / 2, (maxy - miny) / 2)
    half *= 1.15
 
    meridians = np.arange(30, 42.1, 2.0)
    parallels = np.arange(-10, -27.1, -2.0)
    grat_lat = np.linspace(-27.8, -9.5, 120)
    grat_lon = np.linspace(29.2, 41.9, 120)
    norm = Normalize(0, CAP)
 
    # Panel max labels are computed on the standard 0.1 deg onshore grid (the same
    # grid as Table 2 / reproduce.py), NOT on this figure's finer display grid, so
    # the numbers on the figure agree with the numbers in the table.
    olon, olat, omask = onshore_mask(prepared)
    panel_max = {key: float(np.nanmax(omega_pyproj(CANDIDATES[key], olon[omask], olat[omask])))
                 for key, _, _ in panels}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.4))
    for ax, (key, title, caption) in zip(axes, panels):
        P = get_proj(CANDIDATES[key])
        gp, cx, cy = geoms[key]
        X, Y = P(LO.ravel(), LA.ravel())
        X = X.reshape(LO.shape)
        Y = Y.reshape(LO.shape)
        W = omega_pyproj(CANDIDATES[key], LO.ravel(), LA.ravel()).reshape(LO.shape)
        W_in = np.where(inside, W, np.nan)   # isolines and labels stay inside the country
 
        for lon in meridians:
            gx, gy = P(np.full_like(grat_lat, lon), grat_lat)
            ax.plot(gx, gy, color="0.85", linewidth=0.6, zorder=1)
        for lat in parallels:
            gx, gy = P(grat_lon, np.full_like(grat_lon, lat))
            ax.plot(gx, gy, color="0.85", linewidth=0.6, zorder=1)
 
        filled = ax.contourf(X, Y, W, levels=np.linspace(0, CAP, 16),
                             cmap=CMAP, norm=norm, extend="max", zorder=2)
        filled.set_clip_path(clip_patch(ax, gp))
        lines = ax.contour(X, Y, W_in, levels=ISOLINES, colors="0.2",
                           linewidths=0.7, zorder=3)
        ax.clabel(lines, manual=segment_midpoints(lines), fmt="%.2g°",
                  fontsize=7, inline=True)
        draw_outline(ax, gp, color="black", linewidth=1.0, zorder=4)
 
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
 
        bar_x = cx - half + 0.08 * 2 * half
        bar_y = cy - half + 0.07 * 2 * half
        ax.plot([bar_x, bar_x + 200_000], [bar_y, bar_y], color="black",
                linewidth=2.5, zorder=5, solid_capstyle="butt")
        ax.text(bar_x + 100_000, bar_y + 0.02 * 2 * half, "200 km",
                ha="center", fontsize=7, zorder=5)
 
        ax.set_title(f"{title}\nmax shape distortion = {panel_max[key]:.2f}°",
                     fontsize=10.5, pad=8)
        ax.text(0.5, -0.02, caption, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color="0.25")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.6")
 
    mappable = ScalarMappable(norm=norm, cmap=CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=axes, location="bottom", fraction=0.045,
                        shrink=0.5, aspect=42, pad=0.09, extend="max")
    cbar.set_label(f"Shape (angular) distortion ω, degrees  ·  scale capped at {CAP}°"
                   "  ·  AREA preserved exactly in all three", fontsize=9.2)
    fig.suptitle("Shape distortion of Mozambique: same scale in every panel",
                 fontsize=13.5, y=0.99)
    return fig
 
 
FIGURES = {
    "fig1_omega_field": figure1,
    "fig2_omega_cdf": figure2,
    "fig3_country_fill": figure3,
}
 
 
def build(name, geom=None, prepared=None):
    """Build one figure by name and return the Matplotlib Figure."""
    if geom is None or prepared is None:
        geom, prepared = load_boundary()
    fn = FIGURES[name]
    args = {"fig1_omega_field": (geom,), "fig2_omega_cdf": (prepared,),
            "fig3_country_fill": (geom, prepared)}[name]
    return fn(*args)
 
 
def main(formats=("png", "pdf")) -> None:
    """Save every figure as raster (PNG) and vector (PDF) for print quality."""
    geom, prepared = load_boundary()
    print("boundary verified; generating figures")
    outdir = HERE / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        fig = build(name, geom, prepared)
        tight = {"bbox_inches": "tight"} if name == "fig3_country_fill" else {}
        for ext in formats:
            out = outdir / f"{name}.{ext}"
            fig.savefig(out, dpi=DPI, **tight)
            print(f"  wrote {out.name}")
        plt.close(fig)
 
 
if __name__ == "__main__":
    main()