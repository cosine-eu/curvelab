# CurveLab

Interactive curve fitting application for 1D experimental data, built on [lmfit](https://lmfit.github.io/lmfit-py/).

CurveLab provides a desktop GUI (Tkinter) and a Jupyter notebook widget for loading data, building composite models, fitting curves, and analyzing results -- all without writing code.

![curvelab](https://raw.githubusercontent.com/cosine-eu/curvelab/main/docs/curvelab.png "Curvelab GUI")

This project is also [an experiment in AI-assisted coding](docs/ai_assisted_coding_experiment.md).

## Features

- **34 built-in models** -- Gaussian, Lorentzian, Voigt, PseudoVoigt, exponential, polynomial (up to degree 7), spline, step, sine, and more, plus Bose and Fermi when lmfit >= 1.3 is installed. Custom expressions supported.
- **Composite models** -- combine components with `+`, `*`, `-`, `/` operators
- **14 fitting methods** -- Trust Region Reflective (default), Levenberg-Marquardt, Nelder-Mead, differential evolution, basin-hopping, MCMC (emcee), brute-force grid search, ODR, and more
- **9 file formats** -- CSV, TSV, TXT/DAT, Excel, ODS, JSON, Parquet, HDF5, SQLite, plus clipboard paste
- **Multi-series / multi-session** -- plot multiple datasets, run multiple fit sessions per series, batch fit across all series
- **Statistical analysis** -- confidence intervals, correlation/covariance matrices, diagnostic plots, bootstrap CI, profile likelihood, F-test, model comparison (AIC/BIC), uncertainty propagation, 2D confidence contours
- **Data tools** -- auto peak detection, derivative/integral, Savitzky-Golay smoothing, outlier detection (MAD-based sigma-clipping), point exclusion, column calculator, data simulation
- **Export** -- parameters (CSV), fit reports, curve data, plots (PNG/PDF/SVG), lmfit ModelResult (.sav)
- **Workspace persistence** -- save/load entire sessions as `.clw` files
- **Jupyter support** -- full-featured `CurveLabWidget` with ipywidgets

## Quick Start

```bash
pip install curvelab
curvelab
```

Or from a clone, for development:

```bash
git clone https://github.com/cosine-eu/curvelab.git
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

See the [Installation Guide](docs/installation.md) for detailed instructions, optional dependencies, and troubleshooting.

## Documentation

- [Installation Guide](docs/installation.md) -- prerequisites, setup, optional dependencies, troubleshooting
- [User Manual](docs/user_manual.md) -- comprehensive guide covering all features, workflows, and references
- [AI-Assisted Coding Experiment](docs/ai_assisted_coding_experiment.md) -- background on how this project was built

## Architecture

```
DataManager --> SeriesRecord --> PlotManager (display)
                            --> FitManager --> lmfit.Model --> FitResult
```

Core logic is GUI-agnostic. The two frontends (`app.py` for Tkinter, `notebook.py` for Jupyter) coordinate between the core modules. See the [User Manual](docs/user_manual.md) for details.

| Module | Role |
|--------|------|
| `app.py` | Main Tkinter application, central coordinator |
| `notebook.py` | Jupyter notebook widget (ipywidgets) |
| `fit_manager.py` | Composite model building, auto-guess, fitting, GOF metrics |
| `data_manager.py` | Tabular data loading with auto-detection |
| `models.py` | Registry of 34+ built-in lmfit models |
| `plot_manager.py` | Matplotlib figure management (dual-axis, error bars, bands) |
| `preprocessing.py` | Data cleaning (mask, range filter, NaN/inf, sort) |
| `session.py` | Dataclasses: SeriesRecord, FitSession, FitResult |
| `workspace.py` | JSON serialization for save/load |
| `ui_panels.py` | Tkinter UI panels (data, fit, plot controls, results) |
| `ui_dialogs_analysis.py` | Statistical analysis dialogs (CI, contours, bootstrap, etc.) |
| `ui_dialogs_data.py` | Data tool dialogs (peaks, smooth, derivative, simulate) |

## Dependencies

**Core:** numpy, pandas, matplotlib, scipy, lmfit, asteval, uncertainties, numdifftools, emcee, tqdm

**Optional:** ipywidgets/ipympl (notebook), openpyxl (Excel), odfpy (ODS), tables (HDF5), pyarrow (Parquet), odrpack (ODR), pytest/pytest-cov (test)

See the [Installation Guide](docs/installation.md) for how to install optional dependency groups.

## BSD-3-Clause license

Copyright 2026 Giuseppe Vacanti <giuseppe@vacanti.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

    3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.