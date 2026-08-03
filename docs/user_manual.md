# CurveLab User Manual

*Version 0.11.1 -- A comprehensive guide to interactive curve fitting and statistical analysis*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Data Management](#3-data-management)
4. [Building Models](#4-building-models)
5. [Fitting Methods](#5-fitting-methods)
6. [Weighting and Objective Functions](#6-weighting-and-objective-functions)
7. [Interpreting Fit Results](#7-interpreting-fit-results)
8. [Parameter Editing and Constraints](#8-parameter-editing-and-constraints)
9. [Confidence Intervals and Bands](#9-confidence-intervals-and-bands)
10. [Correlation and Covariance Matrices](#10-correlation-and-covariance-matrices)
11. [Diagnostic Plots](#11-diagnostic-plots)
12. [2D Confidence Contours](#12-2d-confidence-contours)
13. [Profile Likelihood](#13-profile-likelihood)
14. [Bootstrap Confidence Intervals](#14-bootstrap-confidence-intervals)
15. [MCMC Sampling with emcee](#15-mcmc-sampling-with-emcee)
16. [Brute-Force Grid Search](#16-brute-force-grid-search)
17. [Model Comparison and Selection](#17-model-comparison-and-selection)
18. [F-Test for Nested Models](#18-f-test-for-nested-models)
19. [Global Fitting](#19-global-fitting)
20. [Batch Fitting](#20-batch-fitting)
21. [Uncertainty Propagation](#21-uncertainty-propagation)
22. [Find Peaks](#22-find-peaks)
23. [Derivative and Integral Analysis](#23-derivative-and-integral-analysis)
24. [Smooth and Outlier Detection](#24-smooth-and-outlier-detection)
25. [Point Exclusion](#25-point-exclusion)
26. [Simulate Data](#26-simulate-data)
27. [Evaluate Model](#27-evaluate-model)
28. [Column Calculator](#28-column-calculator)
29. [Session Management](#29-session-management)
30. [Multi-Series Workflow](#30-multi-series-workflow)
31. [Plot Controls and Visualization](#31-plot-controls-and-visualization)
32. [Export and Import](#32-export-and-import)
33. [Workspace Persistence](#33-workspace-persistence)
34. [Jupyter Notebook Widget](#34-jupyter-notebook-widget)
35. [Settings](#35-settings)
36. [Keyboard Shortcuts](#36-keyboard-shortcuts)
37. [Typical Workflows](#37-typical-workflows)
38. [References](#38-references)

---

## 1. Introduction

CurveLab is an interactive GUI application for fitting analytical models to
one-dimensional experimental data. Built on top of
[lmfit](https://lmfit.github.io/lmfit-py/) (Newville et al., 2014),
CurveLab provides a graphical interface for the complete curve fitting
workflow: loading data, composing models from built-in and custom
components, running fits with a choice of 14 optimization algorithms, and
performing post-fit statistical diagnostics.

### What CurveLab is for

- Fitting analytical models to 1-D experimental data (spectra, kinetics,
  decay curves, dose-response, etc.)
- Comparing alternative models via information criteria (AIC, BIC) and
  the F-test
- Estimating parameter uncertainties through covariance, profile
  likelihood, bootstrap, and MCMC methods
- Propagating fitted uncertainties into derived quantities
- Exploring parameter space via grid search and confidence contour maps
- Simultaneous (global) fitting of shared parameters across datasets
- Batch fitting the same model to many datasets at once

### Key features

- 34 built-in model components (peaks, polynomials, decay, step functions,
  splines, and user-defined expressions), plus 2 additional models
  (Bose, Fermi) when lmfit >= 1.3 is installed
- 14 fitting methods (local, global, grid, Bayesian, and ODR)
- Multi-series and multi-session architecture for systematic model
  comparison
- Interactive point exclusion, peak detection, smoothing, and outlier
  removal
- Full workspace save/load for reproducible analysis
- Available as both a desktop application (Tkinter) and a Jupyter notebook
  widget (ipywidgets)

---

## 2. Getting Started

### Launching the application

```bash
# Console script (after pip install)
curvelab

# Module runner
python -m curvelab
```

See [docs/installation.md](installation.md) for installation instructions.

### Example data

The source repository carries test data files in the `test-data/`
directory. These include CSV files for all built-in models (e.g.,
`model_gaussian.csv`, `model_voigt.csv`, `model_exponential.csv`) as well
as multi-peak and composite model examples. They are useful for learning
how each model behaves and for testing your fitting workflow. They are not
part of the installed package -- clone the repository to get them, or use
**Analysis > Simulate Data** to generate equivalent data from any model
(see [Section 26](#26-simulate-data)).

### The interface at a glance

The window is divided into three regions:

| Region | Contents |
|--------|----------|
| **Left pane (top)** | Data panel -- file loading, dataset selection, column mapping, series list, and style controls |
| **Left pane (bottom)** | Fit panel -- series/session selectors, model components, fitting method, and fit controls |
| **Right pane (top)** | Matplotlib plot canvas with toolbar and coordinate display |
| **Right pane (middle)** | Plot controls -- scales, toggles, axis labels, fit range |
| **Right pane (bottom)** | Fit results -- GOF summary, parameter table, and fit report |

Four menus are available in the menu bar:

| Menu | Items |
|------|-------|
| **File** | Save/Load Workspace, Paste Data, Export Parameters/Report/Curve Data, Export/Import Model Result, Save Plot, Quit |
| **Analysis** | Confidence Intervals, Correlation/Covariance Matrix, Diagnostic Plots, 2D Contours, Profile Likelihood, Bootstrap CI, Global Fit, Uncertainty Propagation, Model Comparison, F-Test, Simulate Data, Evaluate Model, Find Peaks, Derivative/Integral, Smooth/Outlier Detection, Clear Exclusions (Active Series), Clear Exclusions (All Series) |
| **Settings** | Fonts, Show numeric warnings |
| **Help** | About CurveLab |

### A minimal workflow

1. Click **Load File** and select a data file (CSV, Excel, etc.).
2. Select X and Y columns, and optionally Y-error and X-error columns.
3. Click **Add Series**, then **Plot**.
4. In the Fit panel, click **New** to create a fit session ("Fit 1").
5. Select a model (e.g., Gaussian) and click **Add Component**.
6. Click **Auto Guess** to estimate starting parameter values.
7. Click **Fit** to run the optimizer.
8. Inspect the fitted parameters, goodness-of-fit statistics, and
   residuals in the results panel.

---

## 3. Data Management

### Supported file formats

| Format | Extensions | Required package |
|--------|-----------|-----------------|
| CSV | `.csv` | (built-in) |
| TSV | `.tsv` | (built-in) |
| Plain text | `.txt`, `.dat` | (built-in) |
| Excel | `.xlsx`, `.xls` | openpyxl (`pip install -e ".[excel]"`) |
| OpenDocument | `.ods` | odfpy (`pip install -e ".[ods]"`) |
| JSON | `.json` | (built-in) |
| Parquet | `.parquet` | pyarrow (`pip install -e ".[parquet]"`) |
| HDF5 | `.h5`, `.hdf5`, `.hdf` | tables (`pip install -e ".[hdf5]"`) |
| SQLite | `.sqlite`, `.db` | (built-in) |

For plain text files (`.txt`, `.dat`), CurveLab reads whitespace-separated
columns and treats lines starting with `#` as comments. If the last comment
line before data looks like column headers (non-numeric tokens matching the
column count), those are used as column names. Otherwise, columns are named
`col_0`, `col_1`, etc.

For SQLite files, all tables are loaded as separate datasets named
`filename::table_name`.

CurveLab auto-detects whether the first row is a header. If every value in
the first row is numeric, default column names are assigned and the row is
treated as data.

### Clipboard paste (Ctrl+V)

You can paste tabular data directly from a spreadsheet or text editor. The
delimiter is auto-detected from: tab, comma, semicolon, whitespace (in
that priority order). Headers are auto-detected the same way as file
loading. Pasted datasets are named "clipboard", "clipboard (2)", etc.

### Multiple datasets

Multiple files can be loaded simultaneously. Each gets a unique name
(duplicates are suffixed, e.g., "data.csv (2)"). The **Dataset** dropdown
switches between them, and the **Remove** button next to the dropdown
unloads a dataset and all of its associated series and fit sessions.

### Series

A *series* is a specific (X, Y) column pair drawn from a dataset, optionally
accompanied by Y-error and X-error columns. You can add multiple series from
the same dataset (e.g., different Y columns against the same X) and they
will all be plotted together.

Each series has style controls:

| Property | Options |
|----------|---------|
| **Marker** | o, s, ^, v, D, x, +, ., *, h |
| **Line style** | None, solid (-), dashed (--), dash-dot (-.), dotted (:) |
| **Color** | 14 preset xkcd colors, or auto-assigned (the blank entry) |
| **Label** | Free-text label for the legend |

To modify the style of an existing series, select it in the series listbox,
change the style controls, and click **Update Style**. The plot refreshes
immediately.

The **Show/Hide** button toggles per-series visibility. Hidden series
disappear from the plot but their fit sessions and data remain intact.
The listbox shows a `[hidden]` prefix for hidden series.

The **Remove** button below the style controls takes the selected series
off the plot and discards its fit sessions. If the series has any, you are
asked to confirm first — to hide a series without losing its fits, use
**Show/Hide** instead.

In the Fit panel, a **Show/Hide All** button next to the Series dropdown
toggles the active series and all of its fit sessions at once. This is
convenient for temporarily hiding an entire dataset and its fits when
comparing results across multiple series.

---

## 4. Building Models

### Composite model construction

CurveLab follows lmfit's composite model architecture. Models are built by
combining components with algebraic operators:

| Operator | Meaning | Example use case |
|----------|---------|------------------|
| `+` | Additive | Gaussian peak on a linear background |
| `*` | Multiplicative | Absorption dip modulating a continuum |
| `-` | Subtractive | Difference of two profiles |
| `/` | Division | Normalizing by a baseline shape |

Components are added via the **Add Component** button. When a single
component is present, its parameters have no prefix (e.g., `center`,
`sigma`). When a second component is added, both components receive
prefixes to avoid name collisions (e.g., `gaussian1_center`,
`lorentzian1_center`).

The operator dropdown next to the model selector controls how the new
component combines with the existing model. The first component's
operator is always ignored.

### Built-in models (34 + 2 conditional)

#### Peak and line-shape models

| Model | Description | Key parameters |
|-------|-------------|----------------|
| **Gaussian** | Normal distribution peak | amplitude, center, sigma |
| **Lorentzian** | Cauchy distribution / Breit-Wigner line | amplitude, center, sigma |
| **Voigt** | Convolution of Gaussian and Lorentzian | amplitude, center, sigma, gamma |
| **PseudoVoigt** | Linear combination of Gaussian and Lorentzian | amplitude, center, sigma, fraction |
| **Pearson4** | Pearson type IV distribution | amplitude, center, sigma, expon, skew |
| **Pearson7** | Pearson type VII (generalized Lorentzian) | amplitude, center, sigma, expon |
| **SplitLorentzian** | Asymmetric Lorentzian with different left/right widths | amplitude, center, sigma, sigma_r |
| **SkewedGaussian** | Gaussian with skewness parameter | amplitude, center, sigma, gamma |
| **SkewedVoigt** | Voigt with additional skewness | amplitude, center, sigma, gamma, skew |
| **Moffat** | Astronomical PSF model | amplitude, center, sigma, beta |
| **StudentT** | Student's t-distribution profile | amplitude, center, sigma |
| **BreitWigner** | Relativistic Breit-Wigner resonance | amplitude, center, sigma, q |
| **Doniach** | Doniach-Sunjic line shape (XPS) | amplitude, center, sigma, gamma |
| **ExponentialGaussian** | Exponentially modified Gaussian (EMG) | amplitude, center, sigma, gamma |

#### Oscillatory and decay models

| Model | Description |
|-------|-------------|
| **DampedOscillator** | Damped harmonic motion response |
| **DampedHarmonicOscillator** | Classical damped harmonic oscillator response function |
| **Sine** | Sinusoidal: `amplitude * sin(2*pi*frequency*x + shift)` |
| **Exponential** | Exponential decay: `amplitude * exp(-x / decay)` |
| **PowerLaw** | Power law: `amplitude * x^exponent` |

#### Polynomial and background models

| Model | Description |
|-------|-------------|
| **Constant** | Single constant value (`c`) |
| **Linear** | Slope + intercept (`slope * x + intercept`) |
| **Quadratic** | Degree-2 polynomial |
| **Polynomial2** through **Polynomial7** | Polynomials of degree 2 through 7 |

#### Distribution and special models

| Model | Description |
|-------|-------------|
| **Lognormal** | Log-normal distribution |
| **ThermalDistribution** | Bose-Einstein, Fermi-Dirac, or Maxwell-Boltzmann distribution |
| **Bose** | Bose-Einstein distribution (requires lmfit >= 1.3) |
| **Fermi** | Fermi-Dirac step function (requires lmfit >= 1.3) |
| **Step** | Step function (linear, arctan, error function, or logistic forms) |
| **Rectangle** | Product of two step functions (box shape) |

#### Flexible models

| Model | Description | Notes |
|-------|-------------|-------|
| **Spline** | Natural cubic spline | Prompts for number of knots (4--100). Knots are evenly spaced over the data range. |
| **Expression** | User-defined mathematical expression | Prompts for a formula using `x` as the independent variable. Free symbols become parameters. |

The **Expression** model is particularly powerful. You can type any formula
using `x` as the independent variable, for example:

```
a * exp(-b * x) * cos(c * x + d)
```

CurveLab creates an lmfit `ExpressionModel` with the free symbols (`a`,
`b`, `c`, `d`) as adjustable parameters. Standard numpy functions are
available in expressions.

### Auto-guessing initial values

The **Auto Guess** button calls lmfit's `model.guess(data, x=x)` on each
component to estimate sensible starting values from the data. This works
well for peaks (Gaussian, Lorentzian, Voigt, etc.) where the data range
provides clear hints about center, amplitude, and width. Components that
do not implement `guess()` fall back to default parameter values.

Good initial guesses are essential for gradient-based optimizers. If the
auto-guess is poor, adjust parameters manually in the results table before
fitting.

---

## 5. Fitting Methods

CurveLab exposes 14 optimization algorithms. All are provided by lmfit's
`Minimizer` class, which wraps scipy's optimization routines and external
packages.

### Local optimizers

| Method | Algorithm | Bounds | Covariance | Speed |
|--------|-----------|--------|------------|-------|
| `least_squares` **(default)** | Trust Region Reflective | Yes | Yes | Fast |
| `leastsq` | Levenberg-Marquardt (MINPACK) | No | Yes | Fast |
| `nelder` | Nelder-Mead simplex | No | No | Moderate |
| `powell` | Powell's conjugate directions | No | No | Moderate |
| `cobyla` | Constrained Optimization BY Linear Approximation | Yes | No | Moderate |
| `lbfgsb` | Limited-memory BFGS with bounds | Yes | No | Fast |

#### Trust Region Reflective (`least_squares`) -- default

The default fitting method. Natively supports box constraints (parameter
bounds) and computes the covariance matrix from the Jacobian. Use this
method for most fitting tasks, especially when parameters have physically
meaningful bounds.

See: Branch, M.A., Coleman, T.F. & Li, Y. (1999), *A subspace, interior,
and conjugate gradient method for large-scale bound-constrained minimization
problems*, SIAM J. Scientific Computing 21(1).

#### Levenberg-Marquardt (`leastsq`)

Interpolates between steepest descent and Gauss-Newton steps, adapting as
it approaches a minimum. Computes the covariance matrix analytically from
the Jacobian. Does not support parameter bounds. Requires a reasonably good
initial guess.

See: More, J.J. (1978), *The Levenberg-Marquardt algorithm: implementation
and theory*, Numerical Analysis, Lecture Notes in Mathematics vol. 630.

#### Nelder-Mead (`nelder`)

A derivative-free simplex algorithm. Does not require gradient computation
and handles noisy or discontinuous objective functions. Slow convergence in
high dimensions and does not produce a covariance matrix.

See: Nelder, J.A. & Mead, R. (1965), *A simplex method for function
minimization*, Computer Journal 7(4).

#### Powell's method (`powell`)

A direction-set method that performs sequential line minimizations along
conjugate directions. Derivative-free and often faster than Nelder-Mead.

See: Powell, M.J.D. (1964), *An efficient method for finding the minimum of
a function of several variables without calculating derivatives*, Computer
Journal 7(2).

### Global optimizers

| Method | Algorithm | Bounds required | Speed |
|--------|-----------|----------------|-------|
| `differential_evolution` | Population-based stochastic optimizer | Yes | Slow |
| `basinhopping` | Local optimizer + random perturbations | No | Slow |
| `dual_annealing` | Dual annealing (generalized simulated annealing) | Yes | Slow |
| `shgo` | Simplicial Homology Global Optimization | Yes | Slow |
| `ampgo` | Adaptive Memory Programming for Global Optimization | No | Slow |

The bounds-requiring methods (`differential_evolution`, `dual_annealing`,
`shgo`, and grid `brute`) are checked before the fit starts: if any varying
parameter lacks a finite min *and* max, CurveLab reports which parameters need
bounds instead of letting the fit fail partway through. Set the bounds in the
Fit Results table (the **Min**/**Max** columns) first.

The Fit panel also disables the controls a method doesn't use — for example
ODR ignores the Weights, Max nfev, and Scale-covariance controls, so they grey
out while it is selected. A method whose backend package isn't installed (only
ODR's `odrpack` today, since the rest of the fitting stack is required) is
reported clearly rather than failing at run time.

#### Differential evolution (`differential_evolution`)

Maintains a population of candidate solutions that evolve through mutation,
crossover, and selection. Robust for multi-modal landscapes. Bounds are
required on all free parameters.

See: Storn, R. & Price, K. (1997), *Differential Evolution -- A Simple and
Efficient Heuristic for Global Optimization over Continuous Spaces*, J.
Global Optimization 11(4).

#### Basin-hopping (`basinhopping`)

Combines a local optimizer with random perturbations using a Metropolis
acceptance criterion. Good for escaping local minima.

See: Wales, D.J. & Doye, J.P.K. (1997), *Global Optimization by
Basin-Hopping*, J. Physical Chemistry A 101(28).

### Grid-based methods

#### Brute force (`brute`)

Evaluates the objective function on a regular grid over the parameter space.
All free parameters must have finite bounds. See
[Section 16](#16-brute-force-grid-search) for details.

### Bayesian / sampling methods

#### MCMC (`emcee`)

Uses the affine-invariant ensemble sampler from the emcee package
(Foreman-Mackey et al., 2013). Samples the posterior distribution of
parameters rather than finding a single point estimate. See
[Section 15](#15-mcmc-sampling-with-emcee) for details.

### Orthogonal Distance Regression

#### ODR (`odr`)

Uses the odrpack package to perform Orthogonal Distance Regression, which
accounts for errors in both X and Y. Unlike standard least squares (which
minimizes vertical distances), ODR minimizes the orthogonal distances from
data points to the model curve, weighted by the measurement uncertainties
in both variables.

Requires installation of the optional `odr` dependency:

```bash
pip install -e ".[odr]"
```

See: Boggs, P.T. & Rogers, J.E. (1990), *Orthogonal Distance Regression*,
Contemporary Mathematics 112.

### Hidden series warning

If you attempt to fit a hidden series, CurveLab displays a warning that the
fit curve will be plotted but the underlying data points are not visible.
This helps avoid confusion when the fit curve appears without any data.

### Slow methods and threading

Methods that tend to be slow (differential evolution, basin-hopping,
dual annealing, shgo, ampgo, emcee, brute) run in a background thread so
the UI remains responsive. The **Fit** button changes to an **Abort**
button during the fit, allowing early termination.

### Max function evaluations

The **Max nfev** field in the Fit panel limits the number of objective
function evaluations. Leave it blank for the default (algorithm-dependent).
Setting a limit is useful for expensive models or when you want a quick
exploratory fit.

### Scale covariance

The **Scale covariance by reduced chi-squared** checkbox (enabled by
default) controls whether the covariance matrix is scaled by the reduced
chi-squared. When enabled, the reported standard errors assume the model is
correct and the residual scatter is due to unknown measurement errors.
When disabled, the standard errors reflect the raw Jacobian and are
appropriate when you have reliable error bars on your data.

### Starting values and repeated fits

Each fit starts from the current parameter values — the Auto Guess result,
or whatever you have typed into the **Value** column — and clicking **Fit**
again starts from those same values, not from the previous fit's output. So
repeated fits are reproducible: the answer doesn't drift when you fit twice.
This applies to ordinary and ODR fits alike. Fixed (Vary = No) parameters
keep the value you set; only the varying parameters are reset to the start.

To deliberately refine — start the next fit from the last result — click
**Use as Start** below the parameter table. It copies the current fitted
values into the starting values (the Initial column updates to match), so the
next **Fit** continues from there. Repeat Use as Start → Fit to iterate. This
is the opt-in replacement for the old automatic chaining; raising **Max nfev**
is the other way to push a slow fit further.

### When a fit reports no uncertainties

If the **StdErr** column comes back empty, CurveLab warns you that the fit
could not estimate uncertainties. This means the covariance matrix is
singular — the parameters are not all identifiable from the data. The fitted
*curve* can still look excellent (good chi-squared and R-squared) while
individual parameter values are meaningless, because the model can trade one
parameter off against another with no change to the curve.

A classic example is fitting a **Breit-Wigner (Fano)** line to a symmetric
peak: the amplitude and the asymmetry parameter `q` are degenerate (only their
combination is determined), so `q` runs off toward large values and the
amplitude collapses toward zero. The remedies are to fix the undetermined
parameter (set **Vary** to No), remove a redundant component, or use a simpler
model — here, a Lorentzian.

---

## 6. Weighting and Objective Functions

### Weight modes

The weight mode controls how measurement uncertainties influence the fit.
The residual vector passed to the optimizer is element-wise
`r_i = w_i * (y_i - f(x_i))`, where `w_i` is the weight for point `i`.

| Mode | Weight `w_i` | Use case |
|------|-------------|----------|
| **1/yerr (default)** | `1 / sigma_i` | Standard weighted least squares when Y-errors represent 1-sigma uncertainties. Minimizes chi-squared. |
| **1/yerr^2** | `1 / sigma_i^2` | Occasionally used in spectroscopic fitting conventions. |
| **1/y** | `1 / y_i` | Relative-error weighting; useful for data spanning several orders of magnitude. |
| **No weights** | `1` | Unweighted (ordinary) least squares. All points contribute equally. |
| **yerr as weights** | `sigma_i` | Passes errors directly as weights (for specialized cost functions). |
| **Effective variance** | `1 / sqrt(sigma_y^2 + (df/dx * sigma_x)^2)` | Accounts for X-errors via error propagation. The model derivative df/dx is estimated numerically at each iteration. |

**Effective variance weighting** (Orear, 1982) is the correct approach when
both X and Y have measurement errors and the model has significant slope.
It approximates the full errors-in-variables problem by projecting X-errors
onto Y-errors through the model derivative.

See:
- Orear, J. (1982), *Least squares when both variables have uncertainties*,
  American Journal of Physics 50(10).
- York, D. et al. (2004), *Unified equations for the slope, intercept, and
  standard errors of the best straight line*, American Journal of Physics
  72(3).

### Objective

The **Objective** selector chooses what is minimized. Not every objective
applies to every method, so its choices follow the selected **Method** —
you can only pick a combination that actually does something. The default,
**Least squares**, minimizes the sum of squared (weighted) residuals.

**With `least_squares`** the objective offers scipy's robust loss functions,
which reshape how large residuals contribute and make the fit resistant to
outliers:

| Objective | scipy loss | Effect |
|-----------|-----------|--------|
| **Least squares** | `linear` | Standard least squares; sensitive to outliers. |
| **Soft L1** | `soft_l1` | Smooth approximation to L1; mild outlier resistance. |
| **Huber** | `huber` | Quadratic near zero, linear in the tails. |
| **Cauchy** | `cauchy` | Strong outlier rejection (Lorentzian-tailed M-estimator). |
| **Arctan** | `arctan` | Caps the contribution of the largest residuals. |

For every objective except plain Least squares, the **f_scale** entry (next to
the selector) sets the residual value beyond which points are down-weighted.
The default is 1.0, which suits weighted residuals of order one; increase it
for un-weighted data whose residuals are larger.

**With the scalar minimizers** (Nelder-Mead, Powell, differential evolution,
etc.) the objective offers the *reduce functions*, which collapse the residual
vector to a single number the optimizer minimizes:

| Objective | Formula | Properties |
|-----------|---------|------------|
| **Chi-square (default)** | `sum(r_i^2)` | Maximum likelihood for Gaussian errors. Sensitive to outliers. |
| **Neg. entropy** | `-sum(r_i * log(|r_i|))` | Robust to outliers; encourages smooth residual distributions. |
| **Cauchy log-pdf** | `sum(log(1 + r_i^2))` | Heavy-tailed loss. Strongly down-weights large residuals. |

**`leastsq`, `emcee`, and `odr`** each have a single fixed objective
(least squares, the log-posterior, and orthogonal distance respectively), so
the selector shows one entry and is disabled.

> Robust fitting on the default method used to be impossible: the old
> Reduce menu was ignored by `least_squares`. The Objective control now maps
> to scipy's `loss` there, so Cauchy/Huber robustness works with the default
> method.

See:
- Huber, P.J. (1981), *Robust Statistics*, Wiley.
- Hampel, F.R. et al. (1986), *Robust Statistics: The Approach Based on
  Influence Functions*, Wiley.

---

## 7. Interpreting Fit Results

### The Fit Results table

After a fit, the results panel shows three sections:

1. **GOF summary line** -- key statistics displayed at the top
2. **Parameter table** -- one row per parameter with columns:

| Column | Description |
|--------|-------------|
| **Name** | Parameter name (prefixed by component, e.g., `gaussian1_center`) |
| **Value** | Best-fit value |
| **Initial** | Value before the fit (for comparison) |
| **StdErr** | Standard error from the covariance matrix |
| **Min** | Lower bound |
| **Max** | Upper bound |
| **Vary** | Whether the parameter was free (Yes/No) |
| **Expr** | Constraint expression, if any |

3. **Fit report** -- the full lmfit fit report text

### Goodness-of-fit statistics

| Statistic | Symbol | Description |
|-----------|--------|-------------|
| Chi-squared | chi2 | Sum of squared weighted residuals. |
| Reduced chi-squared | chi2/nu | chi2 divided by degrees of freedom (N_data - N_params). Should be approximately 1 for a good fit with correct error bars. |
| R-squared | R2 | Coefficient of determination. Measures the fraction of variance explained by the model. |
| AIC | AIC | Akaike Information Criterion: `N * ln(chi2/N) + 2k`. Lower is better. |
| BIC | BIC | Bayesian Information Criterion: `N * ln(chi2/N) + k * ln(N)`. Penalizes complexity more heavily than AIC for large N. |

**Interpreting reduced chi-squared**:
- chi2/nu >> 1: the model does not describe the data, or the error bars are
  underestimated.
- chi2/nu ~ 1: good fit with correctly estimated errors.
- chi2/nu << 1: the model is over-fitting, or the error bars are
  overestimated.

See: Bevington, P.R. & Robinson, D.K. (2003), *Data Reduction and Error
Analysis for the Physical Sciences*, 3rd edition, McGraw-Hill.

### Copy Table

The **Copy Table** button copies the parameter table to the clipboard in
tab-separated format, suitable for pasting into a spreadsheet.

---

## 8. Parameter Editing and Constraints

### Editing parameters

Double-click any editable cell in the Fit Results parameter table to modify
it. Editable columns are: Value, Min, Max, Vary, and Expr. Press Enter to
confirm or Escape to cancel.

Each parameter has five attributes:

| Attribute | Description |
|-----------|-------------|
| **value** | Current (or initial) value |
| **min** | Lower bound (`-inf` = unbounded) |
| **max** | Upper bound (`inf` = unbounded) |
| **vary** | Whether the parameter is free (Yes) or fixed (No) |
| **expr** | Algebraic constraint expression |

### Constraint expressions

Expressions are lmfit's constraint mechanism. A parameter can be defined as
a function of other parameters. For example:

| Expression | Effect |
|------------|--------|
| `gaussian1_sigma` | Forces this parameter to equal `gaussian1_sigma` |
| `2.3548 * gaussian1_sigma` | Sets this parameter to the FWHM of a Gaussian |
| `lorentzian1_center + 1.5` | Fixes a peak offset of 1.5 units |

Expressions can use arithmetic, `abs()`, `min()`, `max()`, `sin()`, etc.
When a parameter has an expression, it is no longer free -- its value is
determined by the expression.

See: [lmfit constraints documentation](https://lmfit.github.io/lmfit-py/constraints.html).

### Parameter hints

Parameter edits are stored as "hints" that survive model rebuilds. If you
add or remove a component, the values you have set on remaining parameters
are preserved.

### Undo / Redo

Parameter edits are tracked per session. Use **Ctrl+Z** to undo and
**Ctrl+Shift+Z** (or **Ctrl+Z** with Shift) to redo. The undo/redo stacks
are cleared when the fit session is cleared.

---

## 9. Confidence Intervals and Bands

**Note**: The statistics dialogs display the session name in their title
bars (and for Model Comparison and F-Test, the series label) to help you
track which session's results you are viewing. The data tools (Simulate
Data, Evaluate Model, Find Peaks, Derivative/Integral, Smooth/Outlier
Detection) use a plain title.

### Profile likelihood confidence intervals

**Access**: Analysis > Confidence Intervals

CurveLab wraps lmfit's `conf_interval()` function, which computes
confidence intervals by the F-test / profile likelihood method. Unlike
covariance-based errors (which assume a parabolic chi-squared surface),
profile likelihood intervals trace the actual chi-squared surface.

**How it works**: For each parameter, the algorithm fixes the parameter at
a series of values, re-optimizes all other parameters, and records the
chi-squared. The confidence limits are the parameter values where the
chi-squared exceeds the minimum by an amount determined by the F-distribution.

By default, 1-sigma (68.3%), 2-sigma (95.4%), and 3-sigma (99.7%) intervals
are computed.

**When to use**: When you suspect the chi-squared surface is asymmetric
(e.g., near parameter bounds, or for parameters in non-linear positions
like exponential decay constants). This is the recommended method for
reporting parameter uncertainties in scientific publications.

See:
- Venzon, D.J. & Moolgavkar, S.H. (1988), *A method for computing
  profile-likelihood-based confidence intervals*, Applied Statistics 37(1).

### Confidence bands on the plot

The **Conf. band** checkbox in the plot controls displays a shaded region
around the fit curve representing the n-sigma uncertainty in the model
prediction. The sigma level can be set to 1, 2, or 3 using the dropdown
next to the checkbox.

The confidence band is computed from the parameter covariance matrix and
the model Jacobian:

```
sigma_f(x) = sqrt( J(x)^T * Cov * J(x) )
```

where J(x) is the gradient of the model with respect to parameters. The
band is `f(x) +/- n * sigma_f(x)`.

---

## 10. Correlation and Covariance Matrices

### Correlation matrix

**Access**: Analysis > Correlation Matrix

After a fit, the Correlation Matrix dialog displays the Pearson correlation
coefficients between all pairs of fitted parameters:

```
C[i,j] = Cov[i,j] / sqrt(Cov[i,i] * Cov[j,j])
```

Values range from -1 (perfectly anti-correlated) to +1 (perfectly
correlated).

**Interpreting correlations**:
- |C| > 0.9: Strong correlation. The parameters are not independently
  determined by the data. Consider whether the model is over-parameterized.
- |C| ~ 0.5: Moderate correlation. Acceptable in most cases.
- |C| ~ 0: Parameters are independently determined.

See: Press, W.H. et al. (2007), *Numerical Recipes*, Section 15.6.

### Covariance matrix

**Access**: Analysis > Covariance Matrix

Displays the full parameter covariance matrix in a monospace-formatted
dialog. The diagonal elements are the variances (squared standard errors)
and the off-diagonal elements quantify parameter covariances.

The covariance matrix is computed from the Jacobian at the best-fit point:

```
Cov = (J^T W J)^{-1}
```

where J is the Jacobian matrix and W is the weight matrix.

---

## 11. Diagnostic Plots

**Access**: Analysis > Diagnostic Plots

After fitting, the Diagnostic Plots dialog opens a four-panel figure plus
residual statistics. These plots are standard tools in regression analysis
for assessing whether the assumptions underlying the fit are satisfied.

### Panel 1: Residuals vs. Fitted Values

Plots `y_i - f(x_i)` against `f(x_i)`.

- **Random scatter around zero**: The model is adequate.
- **Systematic patterns**: The model is missing a systematic effect.
- **Funnel shape**: Heteroscedasticity (non-constant variance).

### Panel 2: Normal Q-Q Plot

Ordered standardized residuals vs. quantiles of a standard normal
distribution.

- **Points on the diagonal**: Residuals are normally distributed.
- **S-shape**: Heavy-tailed distribution. Consider a robust reduce
  function.
- **Individual outliers**: Points far from the line deserve investigation.

See: Wilk, M.B. & Gnanadesikan, R. (1968), *Probability Plotting Methods
for the Analysis of Data*, Biometrika 55(1).

### Panel 3: Scale-Location Plot

Plots `sqrt(|standardized residuals|)` against fitted values. A more
sensitive test for heteroscedasticity.

- **Flat trend**: Constant variance (good).
- **Upward trend**: Variance increases with the fitted value.

### Panel 4: Autocorrelation Function (ACF)

Autocorrelation of residuals for lags 0 through 20, with 95% confidence
bounds (dashed lines at +/- 1.96/sqrt(N)).

- **All bars within bounds**: No significant autocorrelation (good).
- **Significant positive lag-1**: The model may be missing a slowly varying
  component.
- **Oscillating pattern**: The model may be missing a periodic component.

See: Box, G.E.P. & Jenkins, G.M. (1976), *Time Series Analysis*, Holden-Day.

### Residual statistics

Below the plots, CurveLab reports:

- **Durbin-Watson statistic**: Tests for first-order autocorrelation.
  Values near 2 indicate no autocorrelation; values < 1.5 suggest positive
  autocorrelation; values > 2.5 suggest negative autocorrelation.
- **Runs test**: Tests whether the signs of the residuals form a random
  sequence. A low p-value indicates systematic misfit.

---

## 12. 2D Confidence Contours

**Access**: Analysis > 2D Confidence Contours

Computes and displays 2D confidence regions for any pair of fitted
parameters. This is the two-parameter extension of the profile likelihood
method.

### How it works

CurveLab uses `lmfit.conf_interval2d()` to compute the chi-squared surface
on a 2D grid. For each grid point, the two selected parameters are fixed
and all other parameters are re-optimized.

### Usage

1. Select two parameters from the dropdown menus.
2. Choose the grid resolution (5--50 points per axis).
3. Click **Compute**.

### Interpretation

- **Elliptical contours**: Parameters are approximately normal. The tilt
  indicates correlation.
- **Non-elliptical contours**: The chi-squared surface is non-parabolic.
  Covariance-based errors are unreliable.
- **Banana-shaped contours**: Strong non-linear correlation.
- **Multiple minima**: The contour map may reveal secondary minima.

See: Avni, Y. (1976), *Energy spectra of X-ray clusters of galaxies*, ApJ
210, 642.

---

## 13. Profile Likelihood

**Access**: Analysis > Profile Likelihood

Displays chi-squared profile traces for each parameter. For each parameter,
the profile shows how chi-squared varies as that parameter is moved away
from its best-fit value while all other parameters are re-optimized.

The plot shows:
- The chi-squared profile (blue circles connected by lines)
- The minimum chi-squared (red dashed line)
- The 1-sigma threshold at chi2_min + 1 (orange dotted line)

The points where the profile crosses the threshold define the 1-sigma
confidence interval for that parameter. This method is more reliable than
covariance-based errors when the chi-squared surface is non-parabolic.

See:
- Venzon, D.J. & Moolgavkar, S.H. (1988), *A method for computing
  profile-likelihood-based confidence intervals*, Applied Statistics 37(1).

---

## 14. Bootstrap Confidence Intervals

**Access**: Analysis > Bootstrap CI

Bootstrap methods estimate parameter uncertainties by resampling the data
and refitting many times. This provides empirical distributions of the
parameter estimates that do not rely on assumptions about the shape of the
chi-squared surface.

### Bootstrap types

| Type | Method |
|------|--------|
| **Residual bootstrap** | Resamples the residuals (with replacement), adds them to the fitted curve, and refits. Assumes the residual distribution is correct but the specific ordering does not matter. |
| **Case bootstrap** | Resamples (x, y, yerr) rows with replacement and refits. Makes fewer assumptions but may break structure in the data. |

### Usage

1. Run a fit first (to establish the best-fit model and residuals).
2. Open the Bootstrap CI dialog.
3. Set the number of bootstrap resamples (default: 200).
4. Select the bootstrap type.
5. Click **Run**.

### Output

The dialog displays:
- Histograms of each parameter's bootstrap distribution
- Mean, standard deviation, and 95% confidence intervals (2.5th and 97.5th
  percentiles)
- Vertical lines marking the mean and confidence bounds

See:
- Efron, B. & Tibshirani, R.J. (1993), *An Introduction to the Bootstrap*,
  Chapman & Hall/CRC.
- Davison, A.C. & Hinkley, D.V. (1997), *Bootstrap Methods and their
  Application*, Cambridge University Press.

---

## 15. MCMC Sampling with emcee

### What is MCMC?

Markov Chain Monte Carlo is a class of algorithms that sample from a
probability distribution by constructing a Markov chain. In curve fitting,
MCMC samples the posterior distribution of model parameters given the data,
yielding full probability distributions rather than point estimates.

### The emcee sampler

CurveLab uses the [emcee](https://emcee.readthedocs.io/) package
(Foreman-Mackey et al., 2013), which implements the affine-invariant
ensemble sampler of Goodman & Weare (2010).

### Workflow

1. First run a standard fit (e.g., `least_squares`) to find a good starting
   point.
2. Select **emcee** as the method and click **Fit**.
3. The fit runs in a background thread (can be aborted).
4. After completion, the **MCMC Summary** dialog appears automatically,
   showing for each parameter: Median, Mean, Standard deviation, 2.5% and
   97.5% quantiles (95% credible interval).

### Practical considerations

- Ensure convergence by running enough steps.
- A good starting point from a prior least-squares fit is essential.
- If Y-errors are provided, the likelihood is Gaussian with known
  variances. Without errors, a uniform likelihood is used.

See:
- Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*, PASP
  125(925).
- Goodman, J. & Weare, J. (2010), *Ensemble samplers with affine
  invariance*, Comm. Applied Math. Comp. Sci. 5(1).
- Gelman, A. et al. (2013), *Bayesian Data Analysis*, 3rd edition, Chapman
  & Hall/CRC.

---

## 16. Brute-Force Grid Search

### Overview

The brute-force method evaluates the objective function on a regular grid
spanning the parameter space. It is useful for:

- Mapping the global structure of the objective function
- Finding a good starting point for local optimization
- Verifying that the local optimizer found the global minimum

### Requirements

You **must** set finite bounds (`min` and `max`) on all free parameters
before running a brute-force search. CurveLab validates this and shows a
warning if any parameter has infinite bounds.

### Candidates dialog

After a brute-force fit, the Brute Candidates dialog appears automatically,
listing all grid points sorted by score. You can:

- Browse the candidates and their parameter values
- Click **Load Selected** to transfer a candidate's parameters into the
  model
- Switch to `least_squares` and re-fit for local refinement

### Typical use

1. Set finite bounds on 2--3 key parameters.
2. Fix other parameters or set narrow bounds.
3. Run **brute** to survey the landscape.
4. Load the best candidate.
5. Switch to **least_squares** and re-fit.

See: Press, W.H. et al. (2007), *Numerical Recipes*, Section 10.5.

---

## 17. Model Comparison and Selection

**Access**: Analysis > Model Comparison

Displays a table comparing all fit sessions on the active series:

| Column | Description |
|--------|-------------|
| Session | Session name |
| Model | Composite model description |
| N params | Number of free parameters |
| chi2 | Chi-squared statistic |
| chi2/nu | Reduced chi-squared |
| AIC | Akaike Information Criterion |
| BIC | Bayesian Information Criterion |

The session with the lowest AIC is highlighted in green.

### Information criteria

#### AIC (Akaike Information Criterion)

```
AIC = N * ln(chi2/N) + 2k
```

where `k` is the number of free parameters and `N` the number of data
points. Lower AIC indicates a better model. This is lmfit's definition,
based on the Gaussian log-likelihood; only differences between AIC values
are meaningful, not the absolute number.

**Delta-AIC interpretation** (Burnham & Anderson, 2002):
- Delta < 2: Models are essentially equivalent
- 2 < Delta < 10: Substantially less support for the higher-AIC model
- Delta > 10: Essentially no support for the higher-AIC model

See: Akaike, H. (1974), *A new look at the statistical model
identification*, IEEE Trans. Automatic Control 19(6).

#### BIC (Bayesian Information Criterion)

```
BIC = N * ln(chi2/N) + k * ln(N)
```

where `N` is the number of data points. BIC penalizes model complexity more
heavily than AIC for large datasets.

See: Schwarz, G. (1978), *Estimating the dimension of a model*, Annals of
Statistics 6(2).

---

## 18. F-Test for Nested Models

**Access**: Analysis > F-Test (Nested Models)

The F-test compares two nested models (where the simpler model is a special
case of the more complex one) to determine whether the additional parameters
significantly improve the fit.

### Usage

1. Fit at least two sessions on the same series (one with fewer parameters
   than the other).
2. Open the F-Test dialog.
3. Select the reduced model (fewer parameters) and the full model (more
   parameters).
4. Click **Compute**.

### Output

The dialog reports:
- Number of parameters and chi-squared for each model
- Number of extra parameters and residual degrees of freedom
- **F-statistic** and **p-value**
- Interpretation: whether the extra parameters significantly improve the
  fit (at p < 0.01, p < 0.05, or not significant)

### Interpretation

The F-statistic is:

```
F = ((chi2_reduced - chi2_full) / (p2 - p1)) / (chi2_full / (N - p2))
```

where p1 and p2 are the number of parameters in the reduced and full
models, and N is the number of data points.

A low p-value (typically < 0.05) indicates that the extra parameters
significantly improve the fit and the more complex model is justified.

See: Bevington, P.R. & Robinson, D.K. (2003), *Data Reduction and Error
Analysis*, Chapter 11.

---

## 19. Global Fitting

**Access**: Analysis > Global Fit

Global (simultaneous) fitting optimizes a single set of shared parameters
across multiple datasets at once. This is essential when individual datasets
cannot constrain all parameters, but jointly they can.

### Workflow

1. Plot two or more series.
2. Set up a model on one series and (optionally) fit it.
3. Open the Global Fit dialog from the Analysis menu.
4. Select which series to include (checkboxes).
5. Select which parameters to share across series (checkboxes).
   Checked parameters will have a single value across all series.
   Unchecked parameters get independent values per series (prefixed
   `s0_`, `s1_`, etc.).
6. Click **Fit**.

### Example use cases

- Fitting spectra at different temperatures with a shared peak position
  but varying amplitudes
- Fitting kinetic data at multiple concentrations with a shared rate
  constant
- Fitting scattering data from multiple angles with a shared structural
  model

After the fit, results are distributed back to the individual series
sessions and the UI updates to show all fit curves.

See:
- Beechem, J.M. (1992), *Global analysis of biochemical and biophysical
  data*, Methods in Enzymology 210.

---

## 20. Batch Fitting

The **Batch Fit** button applies the active session's model to all plotted
series in a single operation. This is useful when you have multiple datasets
that should be fit with the same model but independent parameters.

### Workflow

1. Set up and fit a model on one series.
2. Click **Batch Fit**.
3. CurveLab clones the model components to each other series, calls
   auto-guess for initial values, and runs the fit.
4. A model comparison table summarizes the results across all series.

Each series gets its own independent fit result. Parameters are **not**
shared across series. For shared parameters, use Global Fitting instead
(see [Section 19](#19-global-fitting)).

---

## 21. Uncertainty Propagation

**Access**: Analysis > Uncertainty Propagation

After fitting, you often need to compute derived quantities from the fitted
parameters (e.g., the FWHM of a peak from its sigma, or the area under a
curve). The Uncertainty Propagation dialog lets you evaluate arbitrary
expressions using the fitted parameters and their uncertainties.

### How it works

CurveLab uses the [uncertainties](https://pythonhosted.org/uncertainties/)
package, which implements automatic differentiation for error propagation.
Each fitted parameter is represented as a `ufloat` (a number with an
associated uncertainty), and arithmetic operations on ufloats automatically
propagate uncertainties through the chain rule.

### Usage

1. Run a fit to obtain parameters with standard errors.
2. Open the Uncertainty Propagation dialog.
3. The dialog lists all available variables with their nominal values and
   uncertainties.
4. Type an expression (e.g., `2.3548 * sigma` for the FWHM of a
   Gaussian).
5. Press Enter to evaluate.

### Available functions

All functions from `uncertainties.umath` are available: `sin`, `cos`,
`exp`, `log`, `sqrt`, `atan2`, etc.

### Example expressions

| Expression | Meaning |
|------------|---------|
| `2.3548 * sigma` | FWHM of a Gaussian |
| `amplitude * sigma * sqrt(2 * 3.14159)` | Area under a Gaussian |
| `gaussian1_center - gaussian2_center` | Peak separation |
| `gaussian1_amplitude / lorentzian1_amplitude` | Amplitude ratio |

### Limitations

Uncertainty propagation assumes linear error propagation (first-order
Taylor expansion). For highly non-linear functions or large relative
uncertainties, MCMC-based propagation is more accurate.

See:
- JCGM 100:2008, *Evaluation of measurement data -- Guide to the
  expression of uncertainty in measurement* (GUM).
- Ku, H.H. (1966), *Notes on the use of propagation of error formulas*,
  J. Research of the National Bureau of Standards 70C(4).

---

## 22. Find Peaks

**Access**: Analysis > Find Peaks

Automatically detects peaks in the active series data and adds model
components for each detected peak.

### Usage

1. Plot a series with visible peaks.
2. Open the Find Peaks dialog.
3. Optionally adjust:
   - **Prominence**: Minimum peak prominence (default: 10% of data range)
   - **Min distance**: Minimum distance between peaks in data points
   - **Model**: Peak model to add (Gaussian, Lorentzian, Voigt, or
     PseudoVoigt)
4. Click **Detect** to find peaks. A table shows the center, amplitude,
   and estimated width of each detected peak.
5. Click **Add to Model** to add one component per detected peak to the
   active session's model.

Peak widths are estimated using `scipy.signal.peak_widths` at half maximum.
Initial parameter guesses (center, amplitude, sigma) are set from the
detected peak properties.

---

## 23. Derivative and Integral Analysis

**Access**: Analysis > Derivative / Integral

Displays the numerical derivative and cumulative integral of the fitted
curve in a two-panel plot.

### Output

- **Top panel**: dy/dx computed via `numpy.gradient`
- **Bottom panel**: Cumulative integral computed via
  `scipy.integrate.cumulative_trapezoid`
- **Total area**: Displayed above the plots. If the model has multiple
  components, the area under each component is listed separately.

This is useful for determining the total area under a peak, finding
inflection points, or understanding the rate of change of the fitted
function.

---

## 24. Smooth and Outlier Detection

**Access**: Analysis > Smooth / Outlier Detection

This dialog provides tools for smoothing data, detecting outliers, and
exporting processed data.

### Smoothing methods

| Method | Description | Parameters |
|--------|-------------|------------|
| **Savitzky-Golay** | Polynomial smoothing filter that preserves peak shapes | Window size (odd), polynomial order (1--7) |
| **Moving Average** | Uniform filter | Window size |
| **Median Filter** | Non-linear filter robust to spikes | Window size (odd) |
| **Gaussian Filter** | Gaussian kernel smoothing | Sigma (width) |

The **Show smooth overlay** checkbox displays the smoothed curve as a dashed
orange line on the main plot.

See: Savitzky, A. & Golay, M.J.E. (1964), *Smoothing and Differentiation
of Data by Simplified Least Squares Procedures*, Analytical Chemistry 36(8).

### Outlier detection

When enabled, outlier detection uses an iterative sigma-clipping algorithm:

1. Compute residuals: `r_i = y_i - y_smooth_i`
2. Estimate the scatter using the **Median Absolute Deviation (MAD)**:
   `sigma_est = 1.4826 * median(|r_i - median(r_i)|)`
3. Flag points where `|r_i - median(r_i)| > threshold * sigma_est`
4. Optionally iterate: re-smooth using only inlier points, then re-flag.

The MAD-based sigma estimate (factor 1.4826) is consistent with the standard
deviation for Gaussian data but is highly resistant to outliers.

Parameters:
- **Sigma**: Threshold in units of estimated sigma (default: 3.0)
- **Iterations**: Number of iterative rejection passes (default: 1)

See: Hampel, F.R. (1974), *The influence curve and its role in robust
estimation*, JASA 69(346).

### Actions

- **Apply Exclusions**: Mark detected outliers as excluded points (see
  [Section 25](#25-point-exclusion)).
- **Export Smoothed**: Create a new series from the smoothed data.
- **Export Baseline-Sub.**: Create a new series from `y - y_smooth`
  (baseline-subtracted data).

---

## 25. Point Exclusion

CurveLab supports interactive point exclusion for removing individual data
points from the fit without deleting them from the dataset.

### Click-to-exclude

1. Enable the **Exclude pts** checkbox in the plot controls.
2. Click on individual data points in the plot. Clicking a point toggles
   its exclusion status.
3. Excluded points appear as small gray markers on the plot.
4. Excluded points are not used in subsequent fits.

### Bulk exclusion via outlier detection

Use the Smooth / Outlier Detection dialog (Analysis > Smooth / Outlier
Detection) and click **Apply Exclusions** to exclude all detected outliers
at once.

### Clearing exclusions

Two options are available under the Analysis menu:

- **Clear Exclusions (Active Series)**: Restores all excluded points for
  the currently selected series only.
- **Clear Exclusions (All Series)**: Restores all excluded points across
  every series in the workspace.

### How exclusions work

Exclusions are stored as a boolean mask on the series record (`True` =
included, `False` = excluded). The mask is applied before fitting: excluded
points are filtered out of the data arrays passed to the optimizer. The mask
is preserved in workspace files.

---

## 26. Simulate Data

**Access**: Analysis > Simulate Data

CurveLab can generate synthetic data from the current model, with
configurable noise. This is useful for:

- Testing whether a fitting method can recover known parameters
- Exploring how noise levels affect parameter uncertainties
- Generating example data for demonstrations
- Monte Carlo studies of estimator bias

### Usage

1. Set up a model with desired parameter values (or fit real data first).
2. Open the Simulate Data dialog.
3. Configure:
   - **x_min, x_max, N points**: Range and number of points
   - **Gaussian noise**: Additive Gaussian noise with specified sigma
   - **Poisson noise**: Poisson-distributed noise scaled by a factor
     (appropriate for count data)
   - **X-jitter**: Gaussian perturbation of X values with specified sigma
4. Click **Generate**.

The simulated data appears as a new series (named "Simulated 1", etc.) and
is automatically plotted. If the active series is an empty placeholder, the
simulated data replaces it.

### Simulation without loading data

CurveLab can simulate data even without loading any data file. When you
create a new session without any series, a placeholder series is
created automatically. Set up your model, adjust parameters, and use
Simulate to generate the dataset.

---

## 27. Evaluate Model

**Access**: Analysis > Evaluate Model

Evaluates the fitted model at user-specified x values and displays the
results in a table.

### Input formats

- **Comma or space separated**: `1.0, 2.0, 3.0` or `1.0 2.0 3.0`
- **Range syntax**: `start:stop:npoints` (e.g., `0:10:100` for 100 evenly
  spaced points from 0 to 10)

### Output

A two-column table showing x and y values, displayed in monospace font. The
**Copy** button copies the results to the clipboard.

---

## 28. Column Calculator

**Access**: The **Column Calc...** button in the Data panel.

Creates new columns in the active dataset from mathematical expressions
operating on existing columns.

### Usage

1. Enter a name for the new column.
2. Enter an expression using existing column names as variables.
3. Click **Preview** to see the first 10 values.
4. Click **Apply** to create the column.

### Available functions

The following numpy functions and constants are available in expressions:

`abs`, `sqrt`, `log`, `log2`, `log10`, `exp`, `sin`, `cos`, `tan`,
`arcsin`, `arccos`, `arctan`, `arctan2`, `sinh`, `cosh`, `tanh`,
`floor`, `ceil`, `round`, `sign`, `clip`, `diff`, `cumsum`,
`mean`, `std`, `min`, `max`, `where`, `isnan`, `isinf`,
`pi`, `e`, `inf`, `nan`

### Examples

| Expression | Result |
|------------|--------|
| `log(intensity)` | Natural log of the "intensity" column |
| `col_0 / col_1` | Ratio of two columns |
| `sqrt(x**2 + y**2)` | Euclidean distance |
| `where(y > 0, y, nan)` | Replace negative values with NaN |

Note: Functions like `diff` reduce the array length by 1, which will
produce an error since all columns must have the same length.

When you create a new column, the column dropdowns in the Data panel are
refreshed but your existing Y-error and X-error selections are preserved.

---

## 29. Session Management

### Multiple fit sessions

Each series can have multiple *fit sessions*, allowing you to compare
different models or fitting strategies side by side. Sessions are
independent: each has its own model components, parameter values, fit
result, undo/redo history, and plot color.

| Action | Description |
|--------|-------------|
| **New** | Creates a new session with a default name ("Fit 1", "Fit 2", ...). You can enter a custom name. |
| **Rename** | Changes the session name |
| **Delete** | Removes a session, its model, fit result, and fit curve |
| **Show/Hide** | Toggles the session's fit curve visibility |
| **Clear Model** | Empties the active session: removes its model components, fit result, fit curve, and undo history. The session itself (name and color) remains, ready for a new model. Use **Delete** to remove the session entirely. |

Sessions are listed in the Fit panel's session listbox. The header above
the list reads "Sessions for: *series-label*" to indicate which series the
sessions belong to. Hidden sessions show a `[hidden]` prefix. The active
session (selected) is the one whose parameters are displayed in the results
table and modified by fit operations.

### Fit colors

Sessions are automatically assigned colors from a 16-color palette (xkcd
colors: red, bright blue, green, purple, orange, magenta, teal, gold,
coral, navy blue, lime green, lavender, hot pink, olive, sky blue, salmon).
Colors cycle if more than 16 sessions exist.

### Component curves

For composite models (multiple components), CurveLab displays individual
component curves as dashed lines in distinct colors from a secondary
12-color palette. This helps visually decompose overlapping peaks or
identify the contribution of background terms.

---

## 30. Multi-Series Workflow

CurveLab supports plotting and fitting multiple series simultaneously.

### Adding multiple series

You can add series from different datasets or from different column pairs
within the same dataset. Each series gets its own entry in the series
listbox and can have independent style settings.

### Switching between series

Use the **Series** dropdown in the Fit panel to switch between series. When
you select a different series, the session list, model components, and
fit results update to show the selected series' state.

When you add a new series and click Plot, the newly plotted series is
automatically selected in the Series dropdown so that new sessions and
fit operations target it immediately.

### Series identification

Each series is identified by a unique key: `dataset::x_column::y_column`.
This key is displayed in the Series dropdown and used internally to track
fit sessions.

---

## 31. Plot Controls and Visualization

### Scale controls

| Control | Options | Notes |
|---------|---------|-------|
| X scale | linear, log | Log scale is useful for power-law data |
| Y scale | linear, log | Log scale is common for decay and spectral data |

### Display toggles

| Toggle | Description |
|--------|-------------|
| **Data** | Show/hide all data points (fit curves remain visible) |
| **Grid** | Show/hide grid lines (enabled by default) |
| **Equal Axes** | Equal aspect ratio (useful for spatial data). Unchecking restores the original view. The residuals axis always uses auto aspect. |
| **Legend** | Show/hide the legend (enabled by default) |
| **Params** | Annotate the plot with fitted parameter values and GOF statistics |
| **Fit visible range** | Restrict the fit to the currently visible x-range on the plot |
| **x: min -- max** | Manually specify the x-range for fitting |
| **Residuals** | Show residual subplot below the main plot |
| **Conf. band** | Display confidence band around fit curves |
| **n sigma** | Confidence band width: 1, 2, or 3 sigma |
| **Wt. resid** | Show weighted residuals (divided by yerr) instead of raw residuals |
| **Exclude pts** | Enable click-to-exclude mode for data points |

### Title and axis labels

A plot title and custom X and Y axis labels can be entered in the plot
controls. They update when you press Enter or the field loses focus.

### Axis limits

The **X Range** and **Y Range** entry pairs pin the plot limits. An empty
field means that side is scaled automatically, so you can pin just one
side (e.g. only a maximum). Clearing all four fields restores full
autoscaling. Pinned limits persist across replots and are saved in the
workspace. Note that the fit x-range control (row above) is independent:
it restricts which points are *fitted*, not what is displayed.

### Coordinate display

The current mouse position in data coordinates is displayed below the plot
canvas.

### Matplotlib toolbar

The standard matplotlib navigation toolbar is included above the plot,
providing zoom, pan, home, and save functionality.

### Fit range control

You can restrict fitting to a subset of the x-range:

- **Fit visible range**: Check this box to fit only the data within the
  current plot view (as set by zoom/pan).
- **x: min -- max**: Enter explicit x-range bounds. These take priority
  over the "Fit visible range" checkbox.

---

## 32. Export and Import

### File menu exports

| Export | Format | Description |
|--------|--------|-------------|
| **Export Parameters** | CSV | Parameter name, value, stderr, min, max, vary |
| **Export Fit Report** | Text | Full lmfit fit report |
| **Export Curve Data** | CSV/TSV | Two tables: (1) dense fit curve with components and uncertainty, (2) data points with residuals and weighted residuals |
| **Save Plot** | PNG, PDF, SVG | Plot image at 150 DPI |
| **Export Model Result** | .sav | lmfit ModelResult serialized via `save_modelresult()` |

### Import

| Import | Format | Description |
|--------|--------|-------------|
| **Import Model Result** | .sav | Load a previously saved lmfit ModelResult. Displays its parameters, GOF statistics, and fit report. |

---

## 33. Workspace Persistence

### Saving and loading

CurveLab workspaces (`.clw` files) are JSON documents that capture the
complete application state.

**Save**: Ctrl+S or File > Save Workspace
**Load**: Ctrl+O or File > Load Workspace

### What is preserved

| State | Preserved? |
|-------|-----------|
| Data file paths (re-loaded on open) | Yes |
| Clipboard/simulated data (embedded) | Yes |
| Column selections and series styles | Yes |
| Series visibility | Yes |
| Point exclusion masks | Yes |
| All fit sessions and their colors | Yes |
| Model components and expressions | Yes |
| Parameter values, bounds, expressions, hints | Yes |
| Fit results (curves, params, GOF, report) | Yes |
| Component curves | Yes |
| Init param values | Yes |
| Plot control settings | Yes |
| Font settings | Yes |
| Brute-force candidates | Yes |
| lmfit ModelResult (reconstructed on load) | Yes |
| Confidence intervals | No (recompute via Analysis menu) |
| MCMC flatchain | No (rerun emcee) |
| Undo/redo history | No |

### Immediate analysis after loading

When you load a workspace, CurveLab automatically reconstructs the lmfit
`ModelResult` object for each completed fit session. This means analysis
tools (Confidence Intervals, Correlation Matrix, Diagnostic Plots, 2D
Contours, Profile Likelihood, Uncertainty Propagation, etc.) are available
immediately after loading -- you do not need to re-run the fit. The
DataPanel series list is also fully populated so all series appear in the
listbox.

### Portability

Workspaces store file paths for loaded datasets. They are portable as long
as the data files remain at the same paths. If a data file cannot be found
during loading, a warning is shown and that series is skipped.

---

## 34. Jupyter Notebook Widget

CurveLab provides a full-featured widget for Jupyter notebooks via the
`CurveLabWidget` class.

### Setup

```python
%matplotlib widget

from curvelab.notebook import CurveLabWidget

w = CurveLabWidget(figsize=(9, 5))
w
```

### Requirements

- The `notebook` optional dependency group must be installed:
  `pip install -e ".[notebook]"`
- The `%matplotlib widget` backend (ipympl) must be active. Do not use
  `%matplotlib inline` or `%matplotlib tk`.

### Features

The notebook widget covers the core fitting workflow:

- Data loading (via file upload button or programmatic DataFrame loading)
- Column selection and series management
- Model building with the same built-in models
- Parameter table with inline editing
- Plot with residuals and confidence bands
- Workspace save/load
- Numeric warning suppression by default (controllable via the
  `show_warnings` constructor argument)

### Differences from the desktop application

The widget has not yet been reworked to match the current desktop app, so
it lags behind in these respects:

- 13 fitting methods -- **ODR is not available** in the notebook.
- The old **Reduce** dropdown is still present instead of the method-aware
  **Objective** control and its `f_scale` entry, so scipy's robust loss
  functions cannot be selected (see [Section 6](#6-weighting-and-objective-functions)).
- No **Use as Start** button, and no warning when a fit reports no
  uncertainties.
- Of the analysis tools only **Model Comparison** is available; confidence
  intervals, contours, bootstrap, profile likelihood, F-test, global fit,
  uncertainty propagation and the data tools are desktop-only.

Use the desktop application when you need any of these.

The widget layout uses ipywidgets `VBox`, `HBox`, `Tab`, and accordion
containers to organize the controls in a notebook-friendly layout.

---

## 35. Settings

### Fonts

**Access**: Settings > Fonts

CurveLab allows separate font customization for the UI and the plot.

| Setting | Description | Default |
|---------|-------------|---------|
| UI Font Family | Font for buttons, labels, and tables | TkDefaultFont |
| UI Font Size | Size for UI elements | 10 |
| Plot Font Family | Font for axis labels, ticks, and annotations | sans-serif |
| Plot Font Size | Size for plot text | 10 |

Available sizes: 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 24.

Font settings are preserved in workspace files.

### Numeric warning suppression

**Access**: Settings > Show numeric warnings

By default, CurveLab suppresses `RuntimeWarning` messages from lmfit,
scipy, and uncertainties. These warnings are common during iterative
fitting (e.g., overflow in exponential functions, division by zero in
Jacobian evaluation) and are generally harmless -- the optimizer recovers
automatically.

If you want to see these warnings (for debugging or to diagnose convergence
issues), check the **Show numeric warnings** option in the Settings menu.
The setting is per-session and is not saved in workspace files.

In the Jupyter notebook widget, warning suppression is controlled by the
`show_warnings` constructor argument:

```python
w = CurveLabWidget(show_warnings=True)  # show all warnings
```

---

## 36. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+S** | Save workspace |
| **Ctrl+O** | Load workspace |
| **Ctrl+V** | Paste data from clipboard |
| **Ctrl+Z** | Undo parameter edit |
| **Ctrl+Shift+Z** | Redo parameter edit |

---

## 37. Typical Workflows

### Workflow 1: Simple peak fitting

1. Load a data file (CSV with x, y, yerr columns).
2. Select columns, click Add Series, click Plot.
3. Click New (session) > select Gaussian > click Add Component.
4. Click Auto Guess, then Fit (the default method is `least_squares`).
5. Check reduced chi-squared. If >> 1, add a Linear background component
   and refit.
6. Run Analysis > Diagnostic Plots to verify residual assumptions.
7. Export the fit report via File > Export Fit Report.

### Workflow 2: Model selection

1. Load data and plot.
2. Create session "Gaussian": add Gaussian + Linear, fit.
3. Create session "Voigt": add Voigt + Linear, fit.
4. Create session "PseudoVoigt": add PseudoVoigt + Linear, fit.
5. Open Analysis > Model Comparison and select the model with lowest
   AIC/BIC.
6. Use Analysis > F-Test to test whether the additional parameters in the
   Voigt are justified over the simpler Gaussian.

### Workflow 3: Robust parameter uncertainties

1. Fit with `least_squares` (the default) to get a starting point.
2. Run Analysis > Confidence Intervals for profile likelihood uncertainties.
3. Run Analysis > Bootstrap CI for non-parametric uncertainties.
4. Switch method to `emcee` and refit for full posterior distributions.
5. Compare covariance-based, profile likelihood, bootstrap, and MCMC
   uncertainties. If they agree, the Gaussian approximation is adequate.
   If not, report the MCMC credible intervals.

### Workflow 4: Global analysis

1. Load multiple datasets (e.g., spectra at different temperatures).
2. Add and plot all series.
3. Set up a model on the first series and fit it.
4. Open Analysis > Global Fit, select all series, check "shared" for
   parameters that should be common (e.g., peak position), and run.
5. Report shared and per-series parameters separately.

### Workflow 5: Multi-peak detection and fitting

1. Load a spectrum with multiple peaks.
2. Open Analysis > Find Peaks. Adjust prominence and distance as needed.
3. Click Detect, then Add to Model.
4. Add a Linear or Constant background component.
5. Click Auto Guess, then Fit.
6. Inspect the component curves to verify the decomposition.

### Workflow 6: Data preprocessing

1. Load noisy data and plot.
2. Open Analysis > Smooth / Outlier Detection.
3. Choose a smoothing method and adjust the window size.
4. Enable outlier detection and adjust the sigma threshold.
5. Click Apply Exclusions to exclude outliers, or Export Smoothed to create
   a clean series.
6. Fit the cleaned data.

### Workflow 7: Exploring parameter space

1. Set bounds on 2--3 key parameters.
2. Run brute force to survey the landscape.
3. Inspect the candidates list to identify local minima.
4. Load the best candidate, switch to `least_squares`, and refine.
5. Run Analysis > 2D Confidence Contours on the most correlated parameter
   pair to visualize the chi-squared surface.

### Workflow 8: Monte Carlo parameter recovery

1. Set up a model with known "true" parameter values.
2. Use Analysis > Simulate Data to generate synthetic data with realistic
   noise.
3. Fit the simulated data.
4. Compare fitted parameters to true values.
5. Repeat (generate new simulation, fit again) to assess estimator bias
   and coverage of confidence intervals.

---

## 38. References

### Software

- **lmfit**: Newville, M. et al. (2014), *LMFIT: Non-Linear Least-Square
  Minimization and Curve-Fitting for Python*, Zenodo.
  <https://lmfit.github.io/lmfit-py/>

- **emcee**: Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*,
  PASP 125(925), 306--312.
  <https://emcee.readthedocs.io/>

- **uncertainties**: Lebigot, E.O., *Uncertainties: a Python package for
  calculations with uncertainties*.
  <https://pythonhosted.org/uncertainties/>

- **scipy**: Virtanen, P. et al. (2020), *SciPy 1.0: Fundamental Algorithms
  for Scientific Computing in Python*, Nature Methods 17, 261--272.

- **odrpack**: Boggs, P.T. & Rogers, J.E. (1990), *Orthogonal Distance
  Regression*, Contemporary Mathematics 112.

### Textbooks

- **Bevington & Robinson** (2003), *Data Reduction and Error Analysis for
  the Physical Sciences*, 3rd edition, McGraw-Hill. The classic
  undergraduate text on fitting and error analysis.

- **Press et al.** (2007), *Numerical Recipes: The Art of Scientific
  Computing*, 3rd edition, Cambridge University Press. Comprehensive
  coverage of optimization algorithms and statistical methods.

- **Gelman et al.** (2013), *Bayesian Data Analysis*, 3rd edition, Chapman
  & Hall/CRC. The standard reference for Bayesian inference and MCMC.

- **Burnham & Anderson** (2002), *Model Selection and Multimodel Inference:
  A Practical Information-Theoretic Approach*, 2nd edition, Springer.
  Definitive guide to AIC and model selection.

- **Efron & Tibshirani** (1993), *An Introduction to the Bootstrap*,
  Chapman & Hall/CRC. The standard reference for bootstrap methods.

- **Sivia & Skilling** (2006), *Data Analysis: A Bayesian Tutorial*, 2nd
  edition, Oxford University Press. Accessible introduction to Bayesian
  methods for physical scientists.

- **Huber** (1981), *Robust Statistics*, Wiley. Foundational text on
  M-estimators and resistant fitting methods.

- **Chatfield** (2004), *The Analysis of Time Series: An Introduction*,
  6th edition, Chapman & Hall/CRC. Covers autocorrelation analysis and
  time series diagnostics.

### Key articles

- Akaike, H. (1974), *A new look at the statistical model identification*,
  IEEE Trans. Automatic Control 19(6), 716--723.

- Avni, Y. (1976), *Energy spectra of X-ray clusters of galaxies*, ApJ 210,
  642--646.

- Boggs, P.T. & Rogers, J.E. (1990), *Orthogonal Distance Regression*,
  Contemporary Mathematics 112, 183--194.

- Branch, M.A., Coleman, T.F. & Li, Y. (1999), *A subspace, interior, and
  conjugate gradient method for large-scale bound-constrained minimization
  problems*, SIAM J. Scientific Computing 21(1), 1--23.

- Breit, G. & Wigner, E. (1936), *Capture of slow neutrons*, Physical
  Review 49(7), 519.

- Doniach, S. & Sunjic, M. (1970), *Many-electron singularity in X-ray
  photoemission and X-ray line spectra from metals*, J. Physics C 3(2), 285.

- Efron, B. & Tibshirani, R.J. (1993), *An Introduction to the Bootstrap*,
  Chapman & Hall/CRC.

- Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*, PASP
  125(925), 306--312.

- Goodman, J. & Weare, J. (2010), *Ensemble samplers with affine
  invariance*, Comm. Applied Math. Comp. Sci. 5(1), 65--80.

- Grushka, E. (1972), *Characterization of exponentially modified Gaussian
  peaks in chromatography*, Analytical Chemistry 44(11), 1733--1738.

- Hampel, F.R. (1974), *The influence curve and its role in robust
  estimation*, JASA 69(346), 383--393.

- Kass, R.E. & Raftery, A.E. (1995), *Bayes Factors*, JASA 90(430),
  773--795.

- Ku, H.H. (1966), *Notes on the use of propagation of error formulas*,
  J. Research of the National Bureau of Standards 70C(4), 263--273.

- Moffat, A.F.J. (1969), *A theoretical investigation of focal stellar
  images in the photographic emulsion*, Astronomy & Astrophysics 3, 455.

- More, J.J. (1978), *The Levenberg-Marquardt algorithm: implementation and
  theory*, Numerical Analysis, Lecture Notes in Mathematics vol. 630,
  Springer.

- Nelder, J.A. & Mead, R. (1965), *A simplex method for function
  minimization*, Computer Journal 7(4), 308--313.

- Orear, J. (1982), *Least squares when both variables have uncertainties*,
  American Journal of Physics 50(10), 912--916.

- Pearson, K. (1895), *Contributions to the mathematical theory of
  evolution. II. Skew variation in homogeneous material*, Phil. Trans.
  Royal Soc. A 186, 343--414.

- Powell, M.J.D. (1964), *An efficient method for finding the minimum of a
  function of several variables without calculating derivatives*, Computer
  Journal 7(2), 155--162.

- Savitzky, A. & Golay, M.J.E. (1964), *Smoothing and Differentiation of
  Data by Simplified Least Squares Procedures*, Analytical Chemistry 36(8),
  1627--1639.

- Schwarz, G. (1978), *Estimating the dimension of a model*, Annals of
  Statistics 6(2), 461--464.

- Storn, R. & Price, K. (1997), *Differential Evolution -- A Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces*,
  J. Global Optimization 11(4), 341--359.

- Thompson, P., Cox, D.E. & Hastings, J.B. (1987), *Rietveld refinement of
  Debye-Scherrer synchrotron X-ray data from Al2O3*, J. Applied
  Crystallography 20(2), 79--83.

- Venzon, D.J. & Moolgavkar, S.H. (1988), *A method for computing
  profile-likelihood-based confidence intervals*, Applied Statistics 37(1),
  87--94.

- Wales, D.J. & Doye, J.P.K. (1997), *Global Optimization by
  Basin-Hopping*, J. Physical Chemistry A 101(28), 5111--5116.

- Wertheim, G.K. (1975), *Deconvolution and smoothing: Applications in
  ESCA*, J. Electron Spectroscopy 6(3), 239--251.

- Wilk, M.B. & Gnanadesikan, R. (1968), *Probability Plotting Methods for
  the Analysis of Data*, Biometrika 55(1), 1--17.

- York, D. et al. (2004), *Unified equations for the slope, intercept, and
  standard errors of the best straight line*, American Journal of Physics
  72(3), 367--375.
