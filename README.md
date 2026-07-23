# WGS 84 / Mozambique LAEA

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21512485.svg)](https://doi.org/10.5281/zenodo.21512485)

Defining note and reproducibility package for a national equal-area projected
coordinate reference system for Mozambique, submitted as a change request to the
IOGP EPSG Geodetic Parameter Dataset.

The CRS is a Lambert Azimuthal Equal Area projection (EPSG method 9820) centred
on 18.5&deg;S, 35.5&deg;E, the midpoint of the national territory. It preserves
area exactly, removes the UTM 36S/37S zone seam, and keeps shape distortion
below 0.37&deg; over the onshore territory.

![Shape distortion in three equal-area projections](output/fig3_country_fill.png)

## What is here

| File | Role |
|---|---|
| `report.qmd` | The defining note. Tables, figures and several sentences are computed at render time. |
| `reproduce.py` | Numeric core: distortion functions, tables, assertions. |
| `figures.py` | Figure builders, importing the same functions. |
| `data/mozambique_boundary_ne10m.geojson` | National boundary, SHA-256 verified at load. |
| `submission/` | The IOGP EPSG data submission template as sent. |
| `output/results.csv` | Every computed value, flat, versioned as a regression baseline. |

## Reproduce the numbers

```bash
conda env create -f environment.yml
conda activate MZ_LAEA
python reproduce.py     # prints every section, writes output/results.csv
python figures.py       # writes figures to output/ as PNG and PDF
```

Neither step needs Quarto. To render the note:

```bash
quarto render report.qmd --to html    # also: pdf, docx
```

## Design

Nothing in the document is transcribed. `report.qmd` imports the modules above,
so a table cannot disagree with the code that produced it. The boundary file is
hash-verified before use, and the document's setup chunk calls
`reproduce.verify()`, which asserts that two independent distortion computations
agree, that the areal scale is unity, and that inverse round-trip error is
sub-milliarcsecond. A failed check aborts the render rather than producing a
document containing a false claim.

Angular deformation is computed two ways: from the PROJ Tissot factors
(authoritative), and independently from the singular values of the local
Jacobian built from central geodesic differences, which assumes nothing about
the orientation of the Tissot axes.

Reproduced independently on Linux with Python 3.12 and Windows with Python 3.13,
pyproj 3.7.2 / PROJ 9.5.1.

## License

CC-BY-4.0. See `LICENSE`.
