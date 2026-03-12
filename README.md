# CurveLab

Interactive curve fitting application for 1D experimental data, built on [lmfit](https://lmfit.github.io/lmfit-py/).

CurveLab provides a desktop GUI (Tkinter) and a Jupyter notebook widget for loading data, building composite models, fitting curves, and analyzing results -- all without writing code.

## Features

- **34 built-in models** -- Gaussian, Lorentzian, Voigt, PseudoVoigt, exponential, polynomial (up to degree 7), spline, step, sine, and more. Custom expressions supported.
- **Composite models** -- combine components with `+`, `*`, `-`, `/` operators
- **14 fitting methods** -- Levenberg-Marquardt, Nelder-Mead, differential evolution, basin-hopping, MCMC (emcee), brute-force grid search, ODR, and more
- **10 file formats** -- CSV, TSV, TXT/DAT, Excel, ODS, JSON, Parquet, HDF5, SQLite, plus clipboard paste
- **Multi-series / multi-session** -- plot multiple datasets, run multiple fit sessions per series, batch fit across all series
- **Statistical analysis** -- confidence intervals, correlation/covariance matrices, diagnostic plots, bootstrap CI, profile likelihood, F-test, model comparison (AIC/BIC), uncertainty propagation, 2D confidence contours
- **Data tools** -- auto peak detection, derivative/integral, Savitzky-Golay smoothing, outlier detection (MAD-based sigma-clipping), point exclusion, column calculator, data simulation
- **Export** -- parameters (CSV), fit reports, curve data, plots (PNG/PDF/SVG), lmfit ModelResult (.sav)
- **Workspace persistence** -- save/load entire sessions as `.clw` files
- **Jupyter support** -- full-featured `CurveLabWidget` with ipywidgets

## Quick Start

```bash
git clone https://github.com/your-username/curvelab.git
cd curvelab
pip install -e .
curvelab
```

In a Jupyter notebook:

```python
%matplotlib widget
from curvelab.notebook import CurveLabWidget
CurveLabWidget()
```

## Installation

Requires Python 3.10+.

```bash
# Core install
pip install -e .

# With all optional dependencies
pip install -e ".[notebook,excel,ods,hdf5,odr]"
```

| Extra | Packages | Enables |
|-------|----------|---------|
| `notebook` | ipywidgets, ipympl | Jupyter notebook widget |
| `excel` | openpyxl | Excel .xlsx/.xls files |
| `ods` | odfpy | LibreOffice .ods files |
| `hdf5` | tables | HDF5 .h5/.hdf5 files |
| `odr` | odrpack | Orthogonal Distance Regression |

See [docs/installation.md](docs/installation.md) for detailed instructions.

## Documentation

- [Installation Guide](docs/installation.md)
- [User Manual](docs/user_manual.md) -- comprehensive guide with workflows and references

## Architecture

```
DataManager --> SeriesRecord --> PlotManager (display)
                            --> FitManager --> lmfit.Model --> FitResult
```

Core logic is GUI-agnostic. The two frontends (`app.py` for Tkinter, `notebook.py` for Jupyter) coordinate between the core modules:

| Module | Role |
|--------|------|
| `fit_manager.py` | Composite model building, auto-guess, fitting, GOF metrics |
| `data_manager.py` | Tabular data loading with auto-detection |
| `models.py` | Registry of 34 built-in lmfit models |
| `plot_manager.py` | Matplotlib figure management (dual-axis, error bars, bands) |
| `preprocessing.py` | Data cleaning (mask, range filter, NaN/inf, sort) |
| `session.py` | Dataclasses: SeriesRecord, FitSession, FitResult |
| `workspace.py` | JSON serialization for save/load |

## Dependencies

**Core:** numpy, pandas, matplotlib, lmfit

**Optional:** ipywidgets/ipympl (notebook), openpyxl (Excel), odfpy (ODS), tables (HDF5), odrpack (ODR)

## License

GPL 2

