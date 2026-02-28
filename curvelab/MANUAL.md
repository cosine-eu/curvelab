# CurveLab User Manual

*A comprehensive guide to data fitting and statistical analysis*

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Data Management](#3-data-management)
4. [Building Models](#4-building-models)
5. [Fitting Methods](#5-fitting-methods)
6. [Weighting and Objective Functions](#6-weighting-and-objective-functions)
7. [Interpreting Fit Results](#7-interpreting-fit-results)
8. [Confidence Intervals](#8-confidence-intervals)
9. [Correlation Analysis](#9-correlation-analysis)
10. [MCMC Sampling with emcee](#10-mcmc-sampling-with-emcee)
11. [Brute-Force Grid Search](#11-brute-force-grid-search)
12. [Model Comparison and Selection](#12-model-comparison-and-selection)
13. [Diagnostic Plots](#13-diagnostic-plots)
14. [2D Confidence Contours](#14-2d-confidence-contours)
15. [Global Fitting](#15-global-fitting)
16. [Uncertainty Propagation](#16-uncertainty-propagation)
17. [Simulation Mode](#17-simulation-mode)
18. [Batch Fitting](#18-batch-fitting)
19. [Session Management](#19-session-management)
20. [Plot Controls and Visualization](#20-plot-controls-and-visualization)
21. [Workspace Persistence](#21-workspace-persistence)
22. [Keyboard Shortcuts](#22-keyboard-shortcuts)
23. [Typical Workflows](#23-typical-workflows)
24. [References](#24-references)

---

## 1. Introduction

CurveLab is an interactive curve-fitting application built on top of
[lmfit](https://lmfit.github.io/lmfit-py/) (Newville et al., 2014), a
Python library for non-linear least-squares minimization and curve fitting.
CurveLab provides a graphical interface for the full lmfit workflow: loading
data, composing models from built-in and custom components, running fits
with a choice of optimization algorithms, and performing post-fit
statistical diagnostics.

The application supports multiple datasets, multiple series per dataset, and
multiple independent fit sessions per series, allowing systematic comparison
of models and methods within a single workspace.

### What CurveLab is for

- Fitting analytical models to 1-D experimental data
- Comparing alternative models via information criteria (AIC, BIC)
- Estimating parameter uncertainties through covariance, profile
  likelihood, and MCMC methods
- Propagating fitted uncertainties into derived quantities
- Exploring parameter space via grid search and contour maps
- Simultaneous (global) fitting of shared parameters across datasets

---

## 2. Getting Started

### Launching

```python
from curvelab.app import CurveLabApp
CurveLabApp.launch()
```

### The interface at a glance

The window is divided into three regions:

| Region | Contents |
|--------|----------|
| **Left pane** | Data panel (top) and Fit panel (bottom) |
| **Center** | Matplotlib plot canvas |
| **Right pane** | Plot controls (top) and Fit results table (bottom) |

### A minimal workflow

1. **Load** a data file (CSV, Excel, JSON, Parquet, or TSV).
2. Select X and Y columns, optionally Y-error and X-error columns.
3. Click **Add Series**, then **Plot**.
4. In the Fit panel, click **New Session** to create "Fit 1".
5. Select a model (e.g., Gaussian) and click **Add Component**.
6. Click **Fit** to run the optimizer.
7. Inspect fitted parameters, goodness-of-fit statistics, and residuals.

---

## 3. Data Management

### Supported file formats

| Format | Extensions | Library |
|--------|-----------|---------|
| CSV | `.csv` | pandas `read_csv` |
| TSV | `.tsv` | pandas `read_csv` (tab delimiter) |
| Excel | `.xlsx`, `.xls` | pandas `read_excel` |
| JSON | `.json` | pandas `read_json` |
| Parquet | `.parquet` | pandas `read_parquet` |

CurveLab auto-detects whether the first row is a header. If every value in
the first row is numeric, it assigns default column names (`col_0`, `col_1`,
...) and treats the row as data.

### Clipboard paste (Ctrl+V)

Paste tabular data directly from a spreadsheet. The delimiter is
auto-detected in this order: tab, comma, semicolon, whitespace.

### Multiple datasets

Multiple files can be loaded simultaneously. Each gets a unique name
(duplicates are suffixed, e.g., "data.csv (2)"). The dataset dropdown
switches between them, and the **Remove** button next to the dropdown
unloads a dataset.

### Series

A *series* is a specific (X, Y) column pair drawn from a dataset, optionally
accompanied by Y-error and X-error columns. You can add multiple series from
the same dataset (e.g., different Y columns against the same X) and they
will all be plotted together.

Each series has independent style controls:

- **Marker**: o, s, ^, v, D, x, +, ., *, h
- **Line style**: None, solid, dashed, dash-dot, dotted
- **Color**: 15 preset xkcd colors, or auto-assigned
- **Label**: Free-text label for the legend

The **Show/Hide** button toggles per-series visibility. Hidden series
disappear from the plot but their fit curves remain visible. The listbox
shows a `[hidden]` prefix for hidden series.

---

## 4. Building Models

### Composite model construction

CurveLab follows lmfit's composite model architecture
([lmfit Models documentation](https://lmfit.github.io/lmfit-py/model.html)).
Models are built by combining components with algebraic operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `+` | Additive | Gaussian peak on a linear background |
| `*` | Multiplicative | Absorption dip modulating a continuum |
| `-` | Subtractive | Difference of two profiles |

Components are added via the **Add Component** button. Each component is
automatically prefixed (`c0_`, `c1_`, ...) to avoid parameter name
collisions.

### Built-in models

CurveLab exposes 34 model types, most of which wrap lmfit's built-in model
library:

#### Peak / line-shape models

| Model | Description | Key parameters | Reference |
|-------|-------------|----------------|-----------|
| **Gaussian** | Normal distribution peak | amplitude, center, sigma | — |
| **Lorentzian** | Cauchy distribution / Breit-Wigner line | amplitude, center, sigma | — |
| **Voigt** | Convolution of Gaussian and Lorentzian | amplitude, center, sigma, gamma | Wertheim (1975) |
| **PseudoVoigt** | Linear combination of Gaussian and Lorentzian | amplitude, center, sigma, fraction | Thompson, Cox & Hastings (1987) |
| **Pearson4** | Pearson type IV distribution | amplitude, center, sigma, expon, skew | Pearson (1895) |
| **Pearson7** | Pearson type VII (generalized Lorentzian) | amplitude, center, sigma, expon | — |
| **SplitLorentzian** | Asymmetric Lorentzian with different widths | amplitude, center, sigma, sigma_r | — |
| **SkewedGaussian** | Gaussian with skewness parameter | amplitude, center, sigma, gamma | — |
| **SkewedVoigt** | Voigt with additional skewness | amplitude, center, sigma, gamma, skew | — |
| **Moffat** | Used in astronomical PSF modeling | amplitude, center, sigma, beta | Moffat (1969) |
| **StudentT** | Student's t-distribution profile | amplitude, center, sigma | — |
| **BreitWigner** | Relativistic Breit-Wigner resonance | amplitude, center, sigma, q | Breit & Wigner (1936) |
| **Doniach** | Doniach-Sunjic line shape (XPS) | amplitude, center, sigma, gamma | Doniach & Sunjic (1970) |
| **ExponentialGaussian** | Exponentially modified Gaussian (EMG) | amplitude, center, sigma, gamma | Grushka (1972) |

#### Oscillatory / decay models

| Model | Description |
|-------|-------------|
| **DampedOscillator** | Damped harmonic motion |
| **DampedHarmonicOscillator** | Classical DHO response function |
| **Sine** | Sinusoidal: `amplitude * sin(2pi * frequency * x + shift)` |

#### Polynomial / background models

| Model | Description |
|-------|-------------|
| **Constant** | Single constant value |
| **Linear** | Slope + intercept |
| **Quadratic** | Degree-2 polynomial |
| **Polynomial2** through **Polynomial7** | Polynomials of degree 2-7 |

#### Distribution / special models

| Model | Description |
|-------|-------------|
| **Exponential** | `amplitude * exp(-x / decay)` |
| **PowerLaw** | `amplitude * x^exponent` |
| **Lognormal** | Log-normal distribution |
| **ThermalDistribution** | Bose-Einstein, Fermi-Dirac, or Maxwell-Boltzmann |
| **Step** | Step function (linear, atan, erf, logistic forms) |
| **Rectangle** | Product of two step functions |

#### Flexible models

| Model | Description |
|-------|-------------|
| **Spline** | Natural cubic spline with configurable knots (4-100) |
| **Expression** | User-defined mathematical expression using standard Python/numpy syntax |

The **Expression** model is particularly powerful: you can type any formula
using `x` as the independent variable (e.g., `a * exp(-b * x) * cos(c * x + d)`)
and CurveLab will create an lmfit `ExpressionModel` with the free symbols
as parameters.

### Auto-guessing initial values

When a component is added, CurveLab calls lmfit's `model.guess(data, x=x)`
to estimate sensible starting values from the data. This works well for
peaks (Gaussian, Lorentzian, Voigt, etc.) where the data range provides
clear hints about center, amplitude, and width. Components that do not
implement `.guess()` fall back to lmfit's default parameter values.

Good initial guesses are essential for gradient-based optimizers. If the
auto-guess is poor, adjust parameters manually in the results table before
fitting.

### Parameter constraints

Each parameter has five editable attributes:

| Attribute | Description |
|-----------|-------------|
| **value** | Current (or initial) value |
| **min** | Lower bound (`-inf` = unbounded) |
| **max** | Upper bound (`inf` = unbounded) |
| **vary** | Whether the parameter is free (Yes) or fixed (No) |
| **expr** | Algebraic constraint expression |

**Expressions** are lmfit's constraint mechanism. A parameter can be defined
as a function of other parameters. For example, setting `c1_sigma` to
`c0_sigma` forces two peaks to share the same width. Expressions can use
arithmetic, `abs()`, `min()`, `max()`, `sin()`, etc. See the
[lmfit constraints documentation](https://lmfit.github.io/lmfit-py/constraints.html).

Double-click any editable cell in the Fit Results table to modify it. Press
Enter to confirm, Escape to cancel.

### Undo / Redo

Parameter edits are tracked per session. Use **Ctrl+Z** to undo and
**Ctrl+Shift+Z** to redo.

---

## 5. Fitting Methods

CurveLab exposes eight optimization algorithms, all provided by lmfit's
`Minimizer` class, which in turn wraps scipy's optimization routines and
external packages.

### Local optimizers

#### Levenberg-Marquardt (`leastsq`)

The default method. Implements the Levenberg-Marquardt algorithm from
MINPACK via `scipy.optimize.leastsq`. This is the workhorse of non-linear
least-squares fitting: it interpolates between steepest descent and
Gauss-Newton steps, adapting as it approaches a minimum.

- **Strengths**: Fast convergence near the solution; computes the covariance
  matrix analytically from the Jacobian.
- **Weaknesses**: Requires a reasonably good initial guess; can converge to
  local minima; does not support bound constraints directly.
- **When to use**: First choice for well-conditioned problems with a good
  initial guess.

See: Moré (1978), *The Levenberg-Marquardt algorithm: implementation and
theory*, in Numerical Analysis, Lecture Notes in Mathematics vol. 630.

#### Trust Region Reflective (`least_squares`)

Uses `scipy.optimize.least_squares` with the "trf" method. Unlike
`leastsq`, it natively supports box constraints (parameter bounds).

- **Strengths**: Handles bounds elegantly; robust for problems where
  `leastsq` fails near boundaries.
- **Weaknesses**: Slightly slower than `leastsq` for unconstrained problems.
- **When to use**: When parameters have physically meaningful bounds.

See: Branch, Coleman & Li (1999), *A subspace, interior, and conjugate
gradient method for large-scale bound-constrained minimization problems*,
SIAM J. Scientific Computing 21(1).

#### Nelder-Mead (`nelder`)

A derivative-free simplex algorithm (`scipy.optimize.minimize` with
`method='Nelder-Mead'`). It maintains a simplex of N+1 points in
N-dimensional parameter space and iteratively reflects, expands, or
contracts it toward the minimum.

- **Strengths**: Does not require gradient computation; handles noisy or
  discontinuous objective functions.
- **Weaknesses**: Slow convergence in high dimensions; no covariance matrix
  estimate; can stall on ridges.
- **When to use**: When gradients are unreliable or the objective is noisy.

See: Nelder & Mead (1965), *A simplex method for function minimization*,
Computer Journal 7(4).

#### Powell's method (`powell`)

A direction-set method (`scipy.optimize.minimize` with `method='Powell'`)
that performs sequential line minimizations along conjugate directions.

- **Strengths**: Derivative-free; often faster than Nelder-Mead.
- **Weaknesses**: No covariance matrix; sensitive to initial directions.
- **When to use**: Alternative to Nelder-Mead when simplex methods stall.

See: Powell (1964), *An efficient method for finding the minimum of a
function of several variables without calculating derivatives*, Computer
Journal 7(2).

### Global optimizers

#### Differential evolution (`differential_evolution`)

A population-based stochastic optimizer
(`scipy.optimize.differential_evolution`). It maintains a population of
candidate solutions that evolve through mutation, crossover, and selection.

- **Strengths**: Global optimizer; robust for multi-modal landscapes;
  supports bounds.
- **Weaknesses**: Slow (many function evaluations); bounds are required.
- **When to use**: When the parameter space has multiple local minima and
  you need the global optimum. Always set reasonable parameter bounds first.

See: Storn & Price (1997), *Differential Evolution - A Simple and Efficient
Heuristic for Global Optimization over Continuous Spaces*, J. Global
Optimization 11(4).

#### Basin-hopping (`basinhopping`)

Combines a local optimizer with random perturbations
(`scipy.optimize.basinhopping`). At each step it makes a random
displacement, runs a local minimization, then accepts or rejects the new
minimum using a Metropolis criterion.

- **Strengths**: Escapes local minima; good for problems with many shallow
  basins.
- **Weaknesses**: Requires tuning of step size and temperature; slower than
  pure local methods.
- **When to use**: When you suspect multiple local minima but
  differential evolution is too slow.

See: Wales & Doye (1997), *Global Optimization by Basin-Hopping and the
Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110
Atoms*, J. Physical Chemistry A 101(28).

### Grid-based methods

#### Brute force (`brute`)

Evaluates the objective function on a regular grid over the parameter space
(`scipy.optimize.brute`). Returns the grid point with the lowest value and
optionally a list of ordered candidates.

See [Section 11](#11-brute-force-grid-search) for detailed usage.

### Bayesian / sampling methods

#### Markov Chain Monte Carlo (`emcee`)

Uses the affine-invariant ensemble sampler from the
[emcee](https://emcee.readthedocs.io/) package (Foreman-Mackey et al.,
2013). Rather than finding a single best-fit point, it samples the posterior
distribution of the parameters, yielding full probability distributions.

See [Section 10](#10-mcmc-sampling-with-emcee) for detailed usage.

### Max function evaluations

The **Max nfev** field in the Fit panel limits the number of objective
function evaluations. Leave it blank for the default (algorithm-dependent).
Setting a limit is useful for expensive models or when you want a quick
exploratory fit.

### Threaded fitting

Methods that tend to be slow (differential evolution, basin-hopping, emcee)
run in a background thread so the UI remains responsive. An **Abort** button
appears during the fit, allowing early termination.

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
| **1/y** | `1 / y_i` | Relative-error weighting; useful for data spanning several orders of magnitude (e.g., decay curves). |
| **No weights** | `1` | Unweighted (ordinary) least squares. All points contribute equally. |
| **yerr as weights** | `sigma_i` | Passes errors directly as weights (unusual; for specialized cost functions). |
| **Effective variance** | `1 / sqrt(sigma_y^2 + (df/dx * sigma_x)^2)` | Accounts for X-errors via error propagation. The model derivative is estimated numerically. |

**Effective variance weighting** (Orear, 1982) is the correct approach when
both X and Y have measurement errors and the model has significant slope.
It approximates the full errors-in-variables problem by projecting X-errors
onto Y-errors through the model derivative. For details see:

- Orear, J. (1982), *Least squares when both variables have uncertainties*,
  American Journal of Physics 50(10).
- York, D. et al. (2004), *Unified equations for the slope, intercept, and
  standard errors of the best straight line*, American Journal of Physics
  72(3).

### Reduce functions

The reduce function transforms the residual vector into a scalar objective.
The default is the sum of squares (chi-squared). Alternative reduce
functions provide robustness against outliers.

| Function | Formula | Properties |
|----------|---------|------------|
| **Chi-square (default)** | `sum(r_i^2)` | Maximum likelihood estimator for Gaussian errors. Sensitive to outliers. |
| **Neg. entropy** | `-sum(r_i * log(r_i))` for `r_i > 0` | Robust to outliers; encourages smooth residual distributions. |
| **Cauchy log-pdf** | `-sum(log(1 + r_i^2))` | Heavy-tailed loss function. Downweights large residuals. Related to Lorentzian loss. |

The Cauchy reduce function is equivalent to the Cauchy (Lorentzian) loss in
robust statistics. It produces an M-estimator that is highly resistant to
outliers. See:

- Huber, P.J. (1981), *Robust Statistics*, Wiley.
- Hampel, F.R. et al. (1986), *Robust Statistics: The Approach Based on
  Influence Functions*, Wiley.

---

## 7. Interpreting Fit Results

### The Fit Results table

After a fit, the results table shows one row per parameter:

| Column | Description |
|--------|-------------|
| **Name** | Parameter name (prefixed by component, e.g., `c0_center`) |
| **Value** | Best-fit value |
| **Initial** | Value before the fit (for comparison) |
| **Stderr** | Standard error from the covariance matrix |
| **Min** | Lower bound |
| **Max** | Upper bound |
| **Vary** | Whether the parameter was free |
| **Expr** | Constraint expression, if any |

### Goodness-of-fit statistics

The fit report (accessible via **Export Report**) and the parameter
annotation on the plot display:

| Statistic | Symbol | Description |
|-----------|--------|-------------|
| Chi-squared | chi2 | Sum of squared weighted residuals |
| Reduced chi-squared | chi2/nu | chi2 divided by degrees of freedom (N_data - N_params). Should be ~1 for a good fit with correct error bars. |
| AIC | AIC | Akaike Information Criterion: `chi2 + 2k` (k = number of free parameters). Lower is better. |
| BIC | BIC | Bayesian Information Criterion: `chi2 + k * ln(N)`. Penalizes complexity more than AIC for large N. |
| R-squared | R2 | Coefficient of determination (not meaningful for non-linear models, shown for reference). |

**Interpreting reduced chi-squared**:
- chi2/nu >> 1: the model does not describe the data, or the error bars are
  underestimated.
- chi2/nu ~ 1: good fit with correctly estimated errors.
- chi2/nu << 1: the model is over-fitting, or the error bars are
  overestimated.

See: Bevington, P.R. & Robinson, D.K. (2003), *Data Reduction and Error
Analysis for the Physical Sciences*, 3rd edition, McGraw-Hill. Chapter 11.

### Standard errors and covariance

For gradient-based optimizers (`leastsq`, `least_squares`), lmfit computes
the covariance matrix from the Jacobian at the best-fit point:

    Cov = (J^T W J)^{-1}

where J is the Jacobian matrix and W is the weight matrix. The standard
error on each parameter is `sqrt(Cov[i,i])`. These errors assume the model
is correct and the residuals are normally distributed, which should be
verified using the diagnostic plots (Section 13).

The covariance-based errors are *local* approximations and may be unreliable
when the chi-squared surface is highly non-parabolic. In such cases, use
profile likelihood confidence intervals (Section 8) or MCMC (Section 10).

---

## 8. Confidence Intervals

### Profile likelihood method

CurveLab wraps lmfit's `conf_interval()` function, which computes
confidence intervals by the F-test / profile likelihood method. Unlike
covariance-based errors (which assume a parabolic chi-squared surface),
profile likelihood intervals trace the actual chi-squared surface.

**How it works**: For each parameter, the algorithm fixes the parameter at
a series of values, re-optimizes all other parameters, and records the
chi-squared. The confidence limits are the parameter values where

    chi2(p) = chi2_min * (1 + F(alpha, 1, N-k) / (N-k))

This accounts for parameter correlations and non-linear effects.

**Access**: Fit panel > **Confidence Intervals** button.

**Configurable sigma levels**: By default, 1-sigma (68.3%), 2-sigma
(95.4%), and 3-sigma (99.7%) intervals are computed. Custom sigma values
can be specified.

**Output**: A report showing, for each parameter, the lower and upper
bounds at each confidence level.

**When to use**: When you suspect the chi-squared surface is asymmetric
(e.g., near parameter bounds, or for parameters in non-linear positions
like exponential decay constants). This is the recommended method for
publishing parameter uncertainties in scientific papers.

See:
- Venzon, D.J. & Moolgavkar, S.H. (1988), *A method for computing
  profile-likelihood-based confidence intervals*, Applied Statistics 37(1).
- The lmfit documentation:
  [Confidence Interval](https://lmfit.github.io/lmfit-py/confidence.html)

---

## 9. Correlation Analysis

### The correlation matrix

After a fit, the **Correlations** button displays the parameter correlation
matrix. Each entry C[i,j] is the Pearson correlation coefficient between
parameters i and j:

    C[i,j] = Cov[i,j] / sqrt(Cov[i,i] * Cov[j,j])

Values range from -1 (perfectly anti-correlated) to +1 (perfectly
correlated). The diagonal is always 1.

**Interpreting correlations**:
- |C| > 0.9: Strong correlation. The parameters are not independently
  determined by the data. Consider whether the model is over-parameterized,
  or whether one parameter can be fixed or constrained.
- |C| ~ 0.5: Moderate correlation. Acceptable in most cases but be aware
  that the marginal uncertainties may be larger than the formal errors
  suggest.
- |C| ~ 0: Parameters are independent.

High correlations often indicate that the model has redundant degrees of
freedom. For example, fitting a Gaussian with both amplitude and sigma free
while the area is the quantity of interest will show strong
amplitude-sigma correlation.

See: Press, W.H. et al. (2007), *Numerical Recipes: The Art of Scientific
Computing*, 3rd edition, Cambridge University Press. Section 15.6.

---

## 10. MCMC Sampling with emcee

### What is MCMC?

Markov Chain Monte Carlo is a class of algorithms that sample from a
probability distribution by constructing a Markov chain whose stationary
distribution is the desired posterior. In the context of curve fitting,
MCMC samples the posterior distribution of model parameters given the data
and prior information.

### The emcee sampler

CurveLab uses the [emcee](https://emcee.readthedocs.io/) package
(Foreman-Mackey et al., 2013), which implements the affine-invariant
ensemble sampler of Goodman & Weare (2010). This sampler is:

- **Affine-invariant**: Performance does not depend on the coordinate system
  or scaling of parameters.
- **Ensemble-based**: Uses multiple "walkers" that explore the parameter
  space simultaneously and share information.
- **Parallelizable**: Walkers are updated in parallel within each step.

### Workflow

1. First run a standard fit (e.g., `leastsq`) to find a good starting
   point.
2. Select **emcee** as the method and click **Fit**.
3. The walkers are initialized in a small ball around the best-fit values.
4. After the run completes, the **MCMC Summary** dialog appears
   automatically, showing for each parameter:
   - Median, Mean, Standard deviation
   - 2.5% and 97.5% quantiles (95% credible interval)

### Interpreting MCMC results

The key output is the *flatchain*: a table of parameter values sampled from
the posterior. From this you can compute:

- **Marginal distributions**: Histogram of each parameter.
- **Credible intervals**: Central intervals containing a given fraction of
  the posterior mass.
- **Joint distributions**: 2D scatter plots or contours showing parameter
  correlations.

MCMC uncertainties are generally more reliable than covariance-based errors
for non-linear models, especially when:
- The chi-squared surface is asymmetric
- Parameters have correlated uncertainties
- The posterior is multi-modal

### Practical considerations

- **Convergence**: Ensure the chains have converged by running enough steps.
  Look for stationarity in the trace plots.
- **Burn-in**: The initial samples (before convergence) should be discarded.
  lmfit handles this via the `burn` and `thin` parameters.
- **Weighted vs. unweighted**: If Y-errors are provided, the likelihood is
  Gaussian with known variances. Without errors, a uniform likelihood is
  used.

See:
- Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*, PASP 125(925).
- Goodman, J. & Weare, J. (2010), *Ensemble samplers with affine
  invariance*, Communications in Applied Mathematics and Computational
  Science 5(1).
- Hogg, D.W. & Foreman-Mackey, D. (2018), *Data Analysis Recipes: Using
  Markov Chain Monte Carlo*, ApJS 236(1).
- Gelman, A. et al. (2013), *Bayesian Data Analysis*, 3rd edition, Chapman
  & Hall/CRC. The standard reference for Bayesian methods.

---

## 11. Brute-Force Grid Search

### Overview

The brute-force method evaluates the objective function on a regular grid
spanning the parameter space. It is useful for:

- Mapping the global structure of the objective function
- Finding a good starting point for local optimization
- Verifying that the local optimizer found the global minimum

### How it works

For each free parameter, a grid of `Ns` points is created between `min` and
`max`. The objective function is evaluated at every grid point (the total
number of evaluations is `Ns^k` where `k` is the number of free
parameters). Results are sorted by objective value.

**Important**: You must set finite bounds (`min` and `max`) on all free
parameters before running a brute-force search. The number of free
parameters should be small (2-4) to keep the grid manageable.

### Candidates dialog

After a brute-force fit, the **Brute Candidates** button opens a dialog
listing all grid points sorted by score. You can:

- Browse the candidates and their parameter values
- Click **Load** to transfer a candidate's parameters back into the model
- Run a local optimizer (`leastsq`) from the loaded candidate to refine it

### Typical use

1. Set bounds on 2-3 key parameters.
2. Fix other parameters or set narrow bounds.
3. Run **brute** to survey the landscape.
4. Load the best candidate.
5. Switch to **leastsq** and re-fit for final refinement.

See: `scipy.optimize.brute` documentation. The method is described in
Press et al. (2007), *Numerical Recipes*, Section 10.5.

---

## 12. Model Comparison and Selection

### The Model Comparison dialog

The **Compare Sessions** button (Fit panel) displays a table comparing all
fit sessions on the active series:

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

    AIC = chi2 + 2k

where `k` is the number of free parameters. AIC estimates the relative
quality of a model by balancing goodness of fit against complexity. A lower
AIC indicates a better model.

**Delta-AIC interpretation** (Burnham & Anderson, 2002):
- Delta < 2: Models are essentially equivalent
- 2 < Delta < 10: Substantially less support for the higher-AIC model
- Delta > 10: Essentially no support for the higher-AIC model

See: Akaike, H. (1974), *A new look at the statistical model
identification*, IEEE Trans. Automatic Control 19(6).

#### BIC (Bayesian Information Criterion)

    BIC = chi2 + k * ln(N)

where `N` is the number of data points. BIC penalizes model complexity more
heavily than AIC for large datasets (when `ln(N) > 2`, i.e., `N > 7`).

**Delta-BIC interpretation** (Kass & Raftery, 1995):
- Delta < 2: Not worth mentioning
- 2 < Delta < 6: Positive evidence against the higher-BIC model
- 6 < Delta < 10: Strong evidence
- Delta > 10: Very strong evidence

See:
- Schwarz, G. (1978), *Estimating the dimension of a model*, Annals of
  Statistics 6(2).
- Kass, R.E. & Raftery, A.E. (1995), *Bayes Factors*, JASA 90(430).

#### Which criterion to use?

- **AIC** is preferred when prediction accuracy is the goal (e.g.,
  interpolation, forecasting).
- **BIC** is preferred when model identification is the goal (e.g.,
  determining the physical mechanism), because it is consistent (selects
  the true model as N -> infinity).

A thorough discussion is found in:
- Burnham, K.P. & Anderson, D.R. (2002), *Model Selection and Multimodel
  Inference*, 2nd edition, Springer.

---

## 13. Diagnostic Plots

After fitting, the **Diagnostics** button opens a four-panel diagnostic
figure. These plots are standard tools in regression analysis for assessing
whether the assumptions underlying the fit are satisfied.

### Panel 1: Residuals vs. Fitted Values

Plots the raw residuals `y_i - f(x_i)` against the fitted values `f(x_i)`.

**What to look for**:
- **Random scatter around zero**: The model is adequate.
- **Systematic patterns** (curvature, trends): The model is missing a
  systematic effect. Consider adding a polynomial term or a different
  functional form.
- **Funnel shape** (increasing spread): Heteroscedasticity. The variance of
  the residuals depends on the fitted value. Consider weighting by `1/y` or
  transforming the data.

### Panel 2: Normal Q-Q Plot

Plots the ordered standardized residuals against the quantiles of a standard
normal distribution. If the residuals are normally distributed, points lie
on the diagonal line.

**What to look for**:
- **S-shape**: Heavy-tailed distribution (more outliers than expected). May
  indicate that a robust reduce function (Cauchy) is appropriate.
- **Banana shape**: Skewed distribution. The error model may be wrong.
- **Individual outliers**: Points far from the line deserve investigation.

See: Wilk, M.B. & Gnanadesikan, R. (1968), *Probability Plotting Methods
for the Analysis of Data*, Biometrika 55(1).

### Panel 3: Scale-Location Plot

Plots `sqrt(|standardized residuals|)` against fitted values. This is a
more sensitive test for heteroscedasticity than the residuals-vs-fitted
plot.

**What to look for**:
- **Flat trend**: Constant variance (homoscedasticity). Good.
- **Upward trend**: Variance increases with the fitted value.
- **Downward trend**: Variance decreases with the fitted value.

### Panel 4: Autocorrelation Function (ACF)

Displays the autocorrelation of the residuals as a bar chart for lags 0
through 20, with 95% confidence bounds shown as dashed horizontal lines.

**What to look for**:
- **All bars within the confidence bounds** (except lag 0, which is always
  1): No significant autocorrelation. Good.
- **Significant positive autocorrelation at lag 1**: Adjacent residuals
  tend to have the same sign. The model may be missing a slowly varying
  component.
- **Oscillating autocorrelation**: The model may be missing a periodic
  component.

Autocorrelated residuals violate the independence assumption of least
squares and cause the formal uncertainties to be underestimated. If
significant autocorrelation is found, consider adding model components or
using a correlated-error model.

See:
- Box, G.E.P. & Jenkins, G.M. (1976), *Time Series Analysis: Forecasting
  and Control*, Holden-Day.
- Chatfield, C. (2004), *The Analysis of Time Series: An Introduction*,
  6th edition, Chapman & Hall/CRC.

---

## 14. 2D Confidence Contours

### Overview

The **Contour** button computes and displays 2D confidence regions for any
pair of parameters. This is the two-parameter extension of the profile
likelihood method described in Section 8.

### How it works

CurveLab uses `lmfit.conf_interval2d()` to compute the chi-squared surface
on a 2D grid. For each grid point, the two selected parameters are fixed
and all other parameters are re-optimized. The resulting chi-squared values
are displayed as a filled contour map.

### Usage

1. Select two parameters from the dropdown menus.
2. Choose the grid resolution (5-50 points per axis; higher resolution
   gives smoother contours but takes longer).
3. Click **Compute**.

The contour levels correspond to delta-chi-squared values for 1-sigma,
2-sigma, and 3-sigma confidence regions (delta-chi2 = 2.30, 6.18, 11.83
for 2 parameters).

### Interpretation

- **Elliptical contours**: The parameters are approximately normally
  distributed. The orientation of the ellipse indicates the correlation:
  tilted ellipses mean the parameters are correlated.
- **Non-elliptical contours**: The chi-squared surface is non-parabolic.
  Covariance-based errors are unreliable; use MCMC or profile likelihood
  intervals instead.
- **Banana-shaped contours**: Strong non-linear correlation between
  parameters.
- **Multiple minima**: The contour map may reveal secondary minima that
  the local optimizer missed.

See:
- Avni, Y. (1976), *Energy spectra of X-ray clusters of galaxies*, ApJ
  210, 642. (Tabulates delta-chi-squared values for joint parameter
  confidence regions.)
- Press et al. (2007), *Numerical Recipes*, Section 15.6.

---

## 15. Global Fitting

### What is global fitting?

Global (simultaneous) fitting optimizes a single set of shared parameters
across multiple datasets at once. This is essential when datasets
individually cannot constrain all parameters, but jointly they can.

### Example use cases

- Fitting a series of spectra taken at different temperatures with a shared
  peak position but varying amplitudes
- Fitting kinetic data at multiple concentrations with a shared rate
  constant
- Fitting scattering data from multiple angles with a shared structural
  model

### Workflow

1. Plot two or more series.
2. Click **Global Fit** in the Fit panel.
3. In the dialog:
   - Select which series to include (checkboxes).
   - Select which parameters to share across series (checkboxes).
4. Click **Run Global Fit**.

### How it works

CurveLab creates a combined objective function that concatenates the
residuals from all selected series. Parameters are prefixed with series
indices (`s0_`, `s1_`, ...) for non-shared parameters. Shared parameters
appear once and are linked across all series via lmfit constraint
expressions.

The total chi-squared is the sum of the individual chi-squared values:

    chi2_total = sum_j sum_i w_{j,i}^2 * (y_{j,i} - f_j(x_{j,i}))^2

where j indexes the series and i indexes the data points within each
series. Each series can have its own weight mode.

After the fit, the results are distributed back to the individual series
sessions. A model comparison table summarizes the per-series
goodness of fit.

See:
- Beechem, J.M. (1992), *Global analysis of biochemical and biophysical
  data*, Methods in Enzymology 210.
- Knutson, J.R. et al. (1983), *Simultaneous analysis of multiple
  fluorescence decay curves*, Chemical Physics Letters 102(6).

---

## 16. Uncertainty Propagation

### Overview

After fitting, you often need to compute derived quantities from the fitted
parameters (e.g., the FWHM of a peak from its sigma, or the area under a
curve). The **Uncertainty Propagation** dialog lets you evaluate arbitrary
expressions using the fitted parameters and their uncertainties.

### How it works

CurveLab uses the [uncertainties](https://pythonhosted.org/uncertainties/)
package, which implements automatic differentiation for error propagation.
Each fitted parameter is represented as a `ufloat` (a number with an
associated uncertainty), and arithmetic operations on ufloats automatically
propagate uncertainties through the chain rule.

### Usage

1. Run a fit to obtain parameters with standard errors.
2. Click **Uncertainty Propagation** in the Fit panel.
3. The dialog shows all available variables (parameter names with their
   nominal values and uncertainties).
4. Type an expression in the input field (e.g., `2.3548 * c0_sigma` for
   the FWHM of a Gaussian).
5. Press Enter to evaluate. The result shows the nominal value and
   propagated uncertainty.

### Available functions

All functions from `uncertainties.umath` are available: `sin`, `cos`,
`exp`, `log`, `sqrt`, `atan2`, etc. Standard arithmetic operators work as
expected.

### Example expressions

| Expression | Meaning |
|------------|---------|
| `2.3548 * c0_sigma` | FWHM of a Gaussian |
| `c0_amplitude * c0_sigma * sqrt(2 * 3.14159)` | Area under a Gaussian |
| `c0_center - c1_center` | Peak separation |
| `c0_amplitude / c1_amplitude` | Amplitude ratio |

### Limitations

Uncertainty propagation assumes:
- Linear error propagation (first-order Taylor expansion). For highly
  non-linear functions or large relative uncertainties, MCMC-based
  propagation is more accurate.
- Gaussian uncertainties. If the posterior is significantly non-Gaussian,
  use MCMC.

See:
- Ku, H.H. (1966), *Notes on the use of propagation of error formulas*,
  J. Research of the National Bureau of Standards 70C(4).
- JCGM 100:2008, *Evaluation of measurement data - Guide to the expression
  of uncertainty in measurement* (GUM).

---

## 17. Simulation Mode

### Overview

CurveLab can generate synthetic data from the current model, with
configurable noise. This is useful for:

- Testing whether a model and fitting method can recover known parameters
- Exploring how noise levels affect parameter uncertainties
- Generating example data for demonstrations
- Monte Carlo studies of estimator bias and coverage

### Usage

1. Set up a model with the desired parameter values (or fit real data
   first).
2. Click **Simulate** in the Fit panel.
3. Configure:
   - **x_min, x_max, N**: Range and number of points for the independent
     variable.
   - **Gaussian noise sigma**: Standard deviation of additive Gaussian
     noise.
   - **Poisson noise scale**: If > 0, Poisson-distributed noise is added
     (scaled by this factor). Appropriate for count data.
   - **X-jitter sigma_x**: Standard deviation of Gaussian perturbation
     applied to X values, simulating positional uncertainty.
4. Click **Generate**.

The simulated data appears as a new series (named "Simulated 1", etc.) and
is automatically plotted.

### Simulation without data

CurveLab can simulate even without loading any data file. If no series
exists, a placeholder series is created automatically when you start a fit
session. Set up your model, adjust parameters to desired values, and use
**Simulate** to generate the synthetic dataset.

---

## 18. Batch Fitting

### Overview

The **Batch Fit** button applies the active session's model to all plotted
series in a single operation. This is useful when you have multiple datasets
that should be fit with the same model but independent parameters.

### Workflow

1. Set up and fit a model on one series.
2. Click **Batch Fit**.
3. CurveLab clones the model components to each other series, auto-guesses
   initial values, and runs the fit.
4. A model comparison table summarizes the results across all series.

Each series gets its own independent fit result. Parameters are not shared
(for shared parameters, use Global Fitting instead).

---

## 19. Session Management

### Multiple fit sessions

Each series can have multiple *fit sessions*, allowing you to compare
different models or fitting strategies side by side. Sessions are
independent: each has its own model components, parameter values, fit
result, and plot color.

| Action | Description |
|--------|-------------|
| **New Session** | Creates a new session ("Fit 1", "Fit 2", ...) |
| **Rename** | Changes the session name |
| **Delete** | Removes a session and its fit curve |
| **Show/Hide** | Toggles the session's fit curve visibility |

Sessions are listed in the Fit panel's session listbox. Hidden sessions show
a `[hidden]` prefix. The active session (highlighted) is the one whose
parameters are displayed in the results table and modified by fit
operations.

### Fit colors

Sessions are automatically assigned colors from a 16-color palette (xkcd
colors: red, bright blue, green, purple, orange, ...). Colors cycle if more
than 16 sessions exist.

### Component curves

For composite models, CurveLab can display individual component curves as
dashed lines. Each component gets a distinct color from a secondary
12-color palette. This helps visually decompose overlapping peaks or
identify the contribution of background terms.

---

## 20. Plot Controls and Visualization

### Scale controls

| Control | Options | Notes |
|---------|---------|-------|
| X scale | linear, log | Log scale is useful for power-law data |
| Y scale | linear, log | Log scale is common for decay and spectral data |

### Display toggles

| Toggle | Description |
|--------|-------------|
| **Grid** | Show/hide grid lines |
| **Equal** | Equal aspect ratio (useful for spatial data) |
| **Legend** | Show/hide the legend |
| **Data** | Global show/hide of all data points (fit curves remain visible) |
| **Residuals** | Show residual subplot below the main plot |
| **Conf. band** | Display 1-sigma confidence band around fit curves |
| **Params** | Annotate the plot with fitted parameter values and GOF statistics |

### Axis labels

Custom X and Y axis labels can be set in the plot controls panel. These are
preserved in workspace files.

### Confidence band

When enabled, the confidence band shows a shaded region around the fit
curve representing the 1-sigma uncertainty in the model prediction. This is
computed from the parameter covariance matrix and the model Jacobian:

    sigma_f(x) = sqrt(J(x)^T * Cov * J(x))

where J(x) is the gradient of the model with respect to parameters
evaluated at x. The band is `f(x) +/- sigma_f(x)`.

### Residual subplot

When enabled, a subplot appears below the main plot showing the residuals
`y_i - f(x_i)` (or weighted residuals if weights are used). This provides a
quick visual check for systematic deviations.

### Plot export

**File > Export Plot** saves the current figure. Supported formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| PNG | `.png` | Raster, 150 DPI default |
| PDF | `.pdf` | Vector, publication quality |
| SVG | `.svg` | Vector, web-friendly |

---

## 21. Workspace Persistence

### Saving and loading

CurveLab workspaces (`.clw` files) are JSON documents that capture the
complete application state:

- **Data sources**: File paths for all loaded datasets (re-loaded on open)
- **Series records**: Column selections, styles, visibility flags
- **Fit sessions**: Model components, parameter hints, fit results,
  session visibility, colors
- **Plot controls**: Scales, grid, legend, residuals, axis labels
- **Font settings**: UI and plot font families and sizes
- **Active selection**: Which series and session were selected

**Save**: Ctrl+S or File > Save Workspace
**Load**: Ctrl+O or File > Load Workspace

### Portability

Workspaces store file paths, so they are portable as long as the data files
remain at the same paths. If a data file cannot be found during loading, a
warning is shown and that series is skipped.

### What is preserved

| State | Preserved? |
|-------|-----------|
| Data file paths | Yes |
| Column selections | Yes |
| Series styles | Yes |
| Series visibility | Yes |
| All fit sessions | Yes |
| Parameter values, bounds, expressions | Yes |
| Fit results (curves, params, report, GOF) | Yes |
| Confidence intervals | No (recompute) |
| MCMC flatchain | No (rerun emcee) |
| Undo/redo history | No |

---

## 22. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+S | Save workspace |
| Ctrl+O | Load workspace |
| Ctrl+V | Paste data from clipboard |
| Ctrl+Z | Undo parameter edit |
| Ctrl+Shift+Z | Redo parameter edit |

---

## 23. Typical Workflows

### Workflow 1: Simple peak fitting

1. Load data file (CSV with x, y, yerr columns).
2. Select columns, Add Series, Plot.
3. New Session > Add Gaussian component.
4. Fit with `leastsq`.
5. Check reduced chi-squared. If >> 1, add a Linear background component
   and refit.
6. Run Diagnostics to verify residual assumptions.
7. Export the fit report.

### Workflow 2: Model selection

1. Load data and plot.
2. Create Session "Gaussian", add Gaussian + Linear, fit.
3. Create Session "Voigt", add Voigt + Linear, fit.
4. Create Session "PseudoVoigt", add PseudoVoigt + Linear, fit.
5. Click **Compare Sessions** and select the model with lowest AIC/BIC.

### Workflow 3: Robust parameter uncertainties

1. Fit with `leastsq` to get a starting point.
2. Run **Confidence Intervals** for profile likelihood uncertainties.
3. Run `emcee` for full posterior distributions.
4. Compare covariance-based, profile likelihood, and MCMC uncertainties.
5. If they agree, the Gaussian approximation is adequate. If not, report
   the MCMC credible intervals.

### Workflow 4: Global analysis

1. Load multiple datasets (e.g., spectra at different temperatures).
2. Plot all series.
3. Set up a model on the first series and fit it.
4. Click **Global Fit**, select all series, check "shared" for parameters
   that should be common (e.g., peak position).
5. Run the global fit.
6. Report shared and per-series parameters separately.

### Workflow 5: Monte Carlo parameter recovery

1. Set up a model with known "true" parameter values.
2. Use **Simulate** to generate synthetic data with realistic noise.
3. Fit the simulated data.
4. Compare fitted parameters to true values.
5. Repeat (generate new simulation, fit again) to assess estimator
   bias and coverage of confidence intervals.

### Workflow 6: Exploring parameter space

1. Set bounds on 2-3 key parameters.
2. Run **brute** force to survey the landscape.
3. Inspect the candidates list to identify local minima.
4. Load the best candidate, switch to `leastsq`, and refine.
5. Run **Contour** on the most correlated parameter pair to visualize
   the chi-squared surface.

---

## 24. References

### Software

- **lmfit**: Newville, M. et al. (2014), *LMFIT: Non-Linear Least-Square
  Minimization and Curve-Fitting for Python*, Zenodo.
  https://lmfit.github.io/lmfit-py/

- **emcee**: Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*,
  PASP 125(925), 306-312.
  https://emcee.readthedocs.io/

- **uncertainties**: Lebigot, E.O., *Uncertainties: a Python package for
  calculations with uncertainties*.
  https://pythonhosted.org/uncertainties/

- **scipy**: Virtanen, P. et al. (2020), *SciPy 1.0: Fundamental Algorithms
  for Scientific Computing in Python*, Nature Methods 17, 261-272.

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
  IEEE Trans. Automatic Control 19(6), 716-723.

- Avni, Y. (1976), *Energy spectra of X-ray clusters of galaxies*, ApJ 210,
  642-646.

- Breit, G. & Wigner, E. (1936), *Capture of slow neutrons*, Physical
  Review 49(7), 519.

- Doniach, S. & Sunjic, M. (1970), *Many-electron singularity in X-ray
  photoemission and X-ray line spectra from metals*, J. Physics C 3(2), 285.

- Foreman-Mackey, D. et al. (2013), *emcee: The MCMC Hammer*, PASP
  125(925), 306-312.

- Goodman, J. & Weare, J. (2010), *Ensemble samplers with affine
  invariance*, Comm. Applied Math. Comp. Sci. 5(1), 65-80.

- Grushka, E. (1972), *Characterization of exponentially modified Gaussian
  peaks in chromatography*, Analytical Chemistry 44(11), 1733-1738.

- Hogg, D.W. & Foreman-Mackey, D. (2018), *Data Analysis Recipes: Using
  Markov Chain Monte Carlo*, ApJS 236(1), 11.

- Kass, R.E. & Raftery, A.E. (1995), *Bayes Factors*, JASA 90(430),
  773-795.

- Moffat, A.F.J. (1969), *A theoretical investigation of focal stellar
  images in the photographic emulsion*, Astronomy & Astrophysics 3, 455.

- Moré, J.J. (1978), *The Levenberg-Marquardt algorithm: implementation and
  theory*, in Numerical Analysis, Lecture Notes in Mathematics vol. 630,
  Springer.

- Nelder, J.A. & Mead, R. (1965), *A simplex method for function
  minimization*, Computer Journal 7(4), 308-313.

- Orear, J. (1982), *Least squares when both variables have uncertainties*,
  American Journal of Physics 50(10), 912-916.

- Pearson, K. (1895), *Contributions to the mathematical theory of
  evolution. II. Skew variation in homogeneous material*, Phil. Trans.
  Royal Soc. A 186, 343-414.

- Schwarz, G. (1978), *Estimating the dimension of a model*, Annals of
  Statistics 6(2), 461-464.

- Storn, R. & Price, K. (1997), *Differential Evolution - A Simple and
  Efficient Heuristic for Global Optimization over Continuous Spaces*,
  J. Global Optimization 11(4), 341-359.

- Thompson, P., Cox, D.E. & Hastings, J.B. (1987), *Rietveld refinement of
  Debye-Scherrer synchrotron X-ray data from Al2O3*, J. Applied
  Crystallography 20(2), 79-83.

- Wales, D.J. & Doye, J.P.K. (1997), *Global Optimization by
  Basin-Hopping*, J. Physical Chemistry A 101(28), 5111-5116.

- Wertheim, G.K. (1975), *Deconvolution and smoothing: Applications in
  ESCA*, J. Electron Spectroscopy 6(3), 239-251.

- York, D. et al. (2004), *Unified equations for the slope, intercept, and
  standard errors of the best straight line*, American Journal of Physics
  72(3), 367-375.
