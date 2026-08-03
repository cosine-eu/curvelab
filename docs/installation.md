# CurveLab Installation Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing from PyPI](#installing-from-pypi)
3. [Cloning the Repository](#cloning-the-repository)
4. [Installing in Development Mode](#installing-in-development-mode)
5. [Optional Dependencies](#optional-dependencies)
6. [Running CurveLab](#running-curvelab)
7. [Using CurveLab in Jupyter](#using-curvelab-in-jupyter)
8. [Verifying the Installation](#verifying-the-installation)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

CurveLab requires **Python 3.10 or later** and **pip**. Verify your Python
version before proceeding:

```bash
python --version
# Should print Python 3.10.x or later
```

If you need to install Python, visit <https://www.python.org/downloads/> or
use your system package manager.

A working Tk installation is required for the desktop GUI. On most systems
this is included with Python. If not, install it via your package manager:

```bash
# Debian / Ubuntu
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (via Homebrew Python)
brew install python-tk
```

---

## Installing from PyPI

For normal use, install the released package:

```bash
pip install curvelab
```

Optional dependency groups use the same names as below, quoted so the shell
does not interpret the brackets:

```bash
pip install "curvelab[excel]"
```

---

## Cloning the Repository

To work on CurveLab itself, clone the repository and change into its
directory:

```bash
git clone https://github.com/cosine-eu/curvelab.git
cd curvelab
```

---

## Installing in Development Mode

Install CurveLab in editable (development) mode so that changes to the
source are immediately reflected without reinstalling:

```bash
pip install -e .
```

This installs the core dependencies automatically:

| Package       | Purpose                                            |
|--------------|----------------------------------------------------|
| numpy         | Numerical arrays and computation                   |
| pandas        | Tabular data loading and manipulation              |
| matplotlib    | Plotting (uses the TkAgg backend)                  |
| scipy         | Optimizers, smoothing filters, statistical tests   |
| lmfit         | Non-linear least-squares curve fitting             |
| asteval       | Sandboxed expression evaluation                    |
| uncertainties | Uncertainty propagation                            |
| numdifftools  | Uncertainties for the scalar and global minimizers |
| emcee         | MCMC sampling (the `emcee` fitting method)         |
| tqdm          | Progress bars for MCMC sampling                    |

The same list is mirrored in `requirements.txt` at the repository root.

---

## Optional Dependencies

CurveLab supports several optional dependency groups for additional file
formats and features. Install them individually or all at once.

### Individual groups

```bash
# Jupyter notebook widget (ipywidgets + ipympl)
pip install -e ".[notebook]"

# Excel file support (.xlsx, .xls)
pip install -e ".[excel]"

# OpenDocument Spreadsheet support (.ods)
pip install -e ".[ods]"

# HDF5 file support (.h5, .hdf5, .hdf)
pip install -e ".[hdf5]"

# Parquet file support (.parquet)
pip install -e ".[parquet]"

# Orthogonal Distance Regression (ODR) fitting method
pip install -e ".[odr]"

# Test runner (needed to run the test suite)
pip install -e ".[test]"
```

### All optional dependencies at once

```bash
pip install -e ".[notebook,excel,ods,hdf5,parquet,odr,test]"
```

### Summary of optional groups

| Group      | Packages              | Enables                                |
|------------|----------------------|----------------------------------------|
| `notebook` | ipywidgets, ipympl   | `CurveLabWidget` for Jupyter notebooks |
| `excel`    | openpyxl             | Loading .xlsx files                    |
| `ods`      | odfpy                | Loading .ods (LibreOffice) files       |
| `hdf5`     | tables (PyTables)    | Loading .h5, .hdf5, .hdf files         |
| `parquet`  | pyarrow              | Loading .parquet files                 |
| `odr`      | odrpack              | ODR fitting method for errors in X and Y |
| `test`     | pytest, pytest-cov   | Running the test suite                 |

Legacy `.xls` workbooks are offered in the file dialog but need the `xlrd`
package, which no group installs; add it yourself if you have such files:

```bash
pip install xlrd
```

---

## Running CurveLab

### Desktop GUI

After installation, you can launch CurveLab in two ways:

```bash
# Using the installed console script
curvelab

# Using Python's module runner
python -m curvelab
```

Both commands open the main application window.

### Using CurveLab in Jupyter

To use CurveLab as an interactive widget inside a Jupyter notebook, you must
first install the notebook optional dependencies and then configure the
matplotlib backend:

```bash
pip install -e ".[notebook]"
```

In your notebook:

```python
%matplotlib widget

from curvelab.notebook import CurveLabWidget

w = CurveLabWidget()
w
```

The widget provides the same core functionality as the desktop application:
data loading, model building, fitting, and result inspection, all inline
within the notebook.

**Important**: The `%matplotlib widget` backend (provided by ipympl) is
required. Do not use `%matplotlib inline` or `%matplotlib tk` -- these
backends are not compatible with the interactive widget.

---

## Verifying the Installation

### Quick check

Run the following command to verify CurveLab can be imported:

```bash
python -c "import curvelab; print('CurveLab imported successfully')"
```

### Check optional dependencies

```bash
python -c "
import curvelab
print('Core: OK')

try:
    import openpyxl
    print('Excel support: OK')
except ImportError:
    print('Excel support: not installed (pip install -e \".[excel]\")')

try:
    import odf
    print('ODS support: OK')
except ImportError:
    print('ODS support: not installed (pip install -e \".[ods]\")')

try:
    import tables
    print('HDF5 support: OK')
except ImportError:
    print('HDF5 support: not installed (pip install -e \".[hdf5]\")')

try:
    import pyarrow
    print('Parquet support: OK')
except ImportError:
    print('Parquet support: not installed (pip install -e \".[parquet]\")')

try:
    import odrpack
    print('ODR support: OK')
except ImportError:
    print('ODR support: not installed (pip install -e \".[odr]\")')

try:
    import ipywidgets, ipympl
    print('Notebook support: OK')
except ImportError:
    print('Notebook support: not installed (pip install -e \".[notebook]\")')
"
```

### Run the test suite

The suite runs under coverage by default, so install the `test` group first:

```bash
pip install -e ".[test]"
python -m pytest tests/
```

---

## Troubleshooting

### "No module named tkinter"

The Tk toolkit is not installed. See the [Prerequisites](#prerequisites)
section for platform-specific installation instructions.

### TkAgg backend conflict in Jupyter

If you see an error about the TkAgg backend when using `CurveLabWidget` in
Jupyter, make sure you are using the `widget` backend:

```python
%matplotlib widget
```

Do **not** mix `%matplotlib tk` and `%matplotlib widget` in the same
notebook session. If you have already set a different backend, restart the
kernel and set `%matplotlib widget` before importing CurveLab.

### "No module named openpyxl" when loading .xlsx files

Install the Excel optional dependency:

```bash
pip install -e ".[excel]"
```

The same pattern applies to other optional formats -- see the
[Optional Dependencies](#optional-dependencies) table for the correct
group name.

### "ODR requires the 'odrpack' package"

The ODR fitting method requires the odrpack package:

```bash
pip install -e ".[odr]"
```

### Matplotlib window does not appear

On some Linux systems with Wayland, the TkAgg backend may not display
correctly. Try setting the display backend:

```bash
export GDK_BACKEND=x11
curvelab
```

### "No module named pytest_cov" when running the tests

`pyproject.toml` enables coverage for every test run, so pytest-cov must be
present. Install the `test` group:

```bash
pip install -e ".[test]"
```

### A feature reports a missing package

Only the optional groups above are ever missing -- scipy, emcee,
uncertainties, numdifftools, asteval, and tqdm are installed as core
dependencies. CurveLab shows an informative error message naming the
package if a feature requires one that is not installed.
