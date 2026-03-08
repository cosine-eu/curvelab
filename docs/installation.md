# CurveLab Installation Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cloning the Repository](#cloning-the-repository)
3. [Installing in Development Mode](#installing-in-development-mode)
4. [Optional Dependencies](#optional-dependencies)
5. [Running CurveLab](#running-curvelab)
6. [Using CurveLab in Jupyter](#using-curvelab-in-jupyter)
7. [Verifying the Installation](#verifying-the-installation)
8. [Troubleshooting](#troubleshooting)

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

## Cloning the Repository

Clone the CurveLab repository and change into its directory:

```bash
git clone <repository-url> curvelab
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

| Package      | Purpose                                |
|-------------|----------------------------------------|
| numpy        | Numerical arrays and computation       |
| pandas       | Tabular data loading and manipulation  |
| matplotlib   | Plotting (uses the TkAgg backend)      |
| lmfit        | Non-linear least-squares curve fitting |

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

# Orthogonal Distance Regression (ODR) fitting method
pip install -e ".[odr]"
```

### All optional dependencies at once

```bash
pip install -e ".[notebook,excel,ods,hdf5,odr]"
```

### Summary of optional groups

| Group      | Packages              | Enables                                |
|------------|----------------------|----------------------------------------|
| `notebook` | ipywidgets, ipympl   | `CurveLabWidget` for Jupyter notebooks |
| `excel`    | openpyxl             | Loading .xlsx and .xls files           |
| `ods`      | odfpy                | Loading .ods (LibreOffice) files       |
| `hdf5`     | tables (PyTables)    | Loading .h5, .hdf5, .hdf files         |
| `odr`      | odrpack              | ODR fitting method for errors in X and Y |

### Additional optional packages

Some analysis features use packages that are not listed as formal
dependencies but will be used if available:

| Package        | Feature                                          |
|---------------|--------------------------------------------------|
| scipy          | Smoothing filters, peak detection, statistical tests, integration |
| emcee          | MCMC sampling (the `emcee` fitting method)       |
| uncertainties  | Uncertainty propagation dialog                   |
| asteval        | Expression evaluation in Column Calculator and Uncertainty Propagation |

These are typically installed as dependencies of lmfit or can be installed
separately:

```bash
pip install scipy emcee uncertainties asteval
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
    import odfpy
    print('ODS support: OK')
except ImportError:
    print('ODS support: not installed (pip install -e \".[ods]\")')

try:
    import tables
    print('HDF5 support: OK')
except ImportError:
    print('HDF5 support: not installed (pip install -e \".[hdf5]\")')

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

```bash
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

### Import errors for scipy, emcee, or uncertainties

These packages are used by specific analysis features. Install them
as needed:

```bash
pip install scipy emcee uncertainties asteval
```

CurveLab will show an informative error message if a feature requires a
package that is not installed.
