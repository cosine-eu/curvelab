"""Curve fitting logic using lmfit."""

import ast
import copy
from dataclasses import dataclass

import numpy as np
from lmfit import CompositeModel, Model, Parameters
from lmfit.minimizer import reduce_chisquare
from lmfit.model import ModelResult
from lmfit.models import SplineModel

from .models import create_expression_model, create_model, create_spline_model
# Re-exports; canonical location is session.py
from .session import FitResult, MIN_ERROR, STRONG_CORRELATION


def _reduce_cauchylogpdf(r):
    """Cauchy log-pdf reduce function for outlier-tolerant fitting."""
    return np.sum(np.log1p(r * r))


# Scalar objectives for the scalar minimizers. A "Neg. entropy" entry was
# removed: neither the formula used here nor lmfit's own negentropy has its
# minimum at zero residual, so it was not a usable fit objective.
REDUCE_FUNCTIONS = {
    "Chi-square (default)": None,
    "Cauchy log-pdf": _reduce_cauchylogpdf,
}

# Weight modes double as UI labels and dispatch keys, so they are named once.
# lmfit multiplies the residual by the weight, so a weight of 1/sigma is
# what makes the objective chi-square. Two former modes, "1/yerr^2" and
# "yerr as weights", minimized sum r^2/sigma^4 and sum (sigma r)^2, for which
# stderr, chi-square, AIC and BIC are all meaningless; they were removed.
WEIGHT_INV_YERR = "1/yerr (default)"
WEIGHT_INV_Y = "1/y"
WEIGHT_NONE = "No weights"
WEIGHT_EFFECTIVE_VARIANCE = "Effective variance"

WEIGHT_MODES = [
    WEIGHT_INV_YERR,
    WEIGHT_INV_Y,
    WEIGHT_NONE,
    WEIGHT_EFFECTIVE_VARIANCE,
]
DEFAULT_WEIGHT_MODE = WEIGHT_INV_YERR
# Modes whose weights come from yerr (and so silently fall back to unit
# weights when a dataset has none).
_YERR_WEIGHT_MODES = {WEIGHT_INV_YERR, WEIGHT_EFFECTIVE_VARIANCE}

# Minimizers offered in the UI. The slow ones run on a background thread.
FIT_METHODS = [
    "leastsq", "least_squares", "nelder", "powell",
    "cobyla", "lbfgsb",
    "differential_evolution", "basinhopping",
    "dual_annealing", "shgo", "ampgo",
    "brute", "emcee", "odr",
]
SLOW_METHODS = {
    "emcee", "brute", "differential_evolution", "basinhopping",
    "dual_annealing", "shgo", "ampgo",
}
DEFAULT_FIT_METHOD = "least_squares"

# --- Method capabilities and objectives ---------------------------------
#
# Not every Method + Objective combination is meaningful: reduce_fcn is only
# consumed by the scalar minimizers, least_squares takes a scipy robust
# `loss` instead, emcee optimizes a log-posterior, and ODR minimizes
# orthogonal distance. Each method therefore declares which *objective kind*
# it supports, plus which shared controls it honors, so the UI can offer only
# valid choices and disable the rest. Verified against lmfit 1.3.4.

# Objective kinds (how the residual becomes the thing minimized):
OBJ_RESIDUAL = "residual"      # array residual, no scalarizing (leastsq)
OBJ_LOSS = "loss"             # scipy robust loss on least_squares
OBJ_SCALAR = "scalar"         # reduce_fcn scalarizes (nelder, powell, ...)
OBJ_POSTERIOR = "posterior"    # emcee log-posterior
OBJ_ODR = "odr"               # orthogonal distance regression


@dataclass(frozen=True)
class MethodCaps:
    """What a fit method supports, so the UI can constrain the choices."""

    objective: str                          # one of the OBJ_* kinds
    needs_bounds: bool = False              # finite min/max on varied params
    honors_weights: bool = True
    honors_max_nfev: bool = True
    honors_scale_covar: bool = True
    requires: tuple[str, ...] = ()          # importable modules the method needs


_SCALAR = MethodCaps(objective=OBJ_SCALAR)
_SCALAR_BOUNDED = MethodCaps(objective=OBJ_SCALAR, needs_bounds=True)

METHOD_CAPS: dict[str, MethodCaps] = {
    "leastsq": MethodCaps(objective=OBJ_RESIDUAL),
    "least_squares": MethodCaps(objective=OBJ_LOSS),
    "nelder": _SCALAR,
    "powell": _SCALAR,
    "cobyla": _SCALAR,
    "lbfgsb": _SCALAR,
    "basinhopping": _SCALAR,
    "ampgo": _SCALAR,
    "differential_evolution": _SCALAR_BOUNDED,
    "dual_annealing": _SCALAR_BOUNDED,
    "shgo": _SCALAR_BOUNDED,
    "brute": _SCALAR_BOUNDED,
    "emcee": MethodCaps(objective=OBJ_POSTERIOR),
    "odr": MethodCaps(
        objective=OBJ_ODR, honors_weights=False,
        honors_max_nfev=False, honors_scale_covar=False,
        requires=("odrpack",),
    ),
}


# Objective choices per kind. For OBJ_SCALAR the label maps to a reduce_fcn
# (reusing REDUCE_FUNCTIONS); for OBJ_LOSS it maps to a scipy loss name.
LEAST_SQUARES_DEFAULT = "Least squares"
# label -> scipy least_squares `loss` value
LOSS_FUNCTIONS: dict[str, str] = {
    LEAST_SQUARES_DEFAULT: "linear",
    "Soft L1": "soft_l1",
    "Huber": "huber",
    "Cauchy": "cauchy",
    "Arctan": "arctan",
}
# Robust losses (everything except plain linear least squares) take an
# f_scale; the UI enables its entry only for these.
ROBUST_LOSSES = frozenset(LOSS_FUNCTIONS) - {LEAST_SQUARES_DEFAULT}
DEFAULT_F_SCALE = 1.0

# Single-choice objective labels for the kinds that offer no alternatives.
_FIXED_OBJECTIVE_CHOICES = {
    OBJ_RESIDUAL: ["Least squares"],
    OBJ_POSTERIOR: ["Log-posterior"],
    OBJ_ODR: ["Orthogonal distance"],
}


def objective_choices(method: str) -> list[str]:
    """Objective labels valid for a method (first is the default)."""
    kind = METHOD_CAPS[method].objective
    if kind == OBJ_LOSS:
        return list(LOSS_FUNCTIONS)
    if kind == OBJ_SCALAR:
        return list(REDUCE_FUNCTIONS)
    return list(_FIXED_OBJECTIVE_CHOICES[kind])


def validate_fit_setup(method: str, params, has_xerr: bool,
                       weight_mode: str) -> tuple[list[str], list[str]]:
    """Check a fit configuration before running it.

    Returns (errors, warnings). Errors mean the fit cannot run as configured
    (missing backend package, or a bounds-requiring method with an unbounded
    varying parameter). Warnings mean it will run but not as the user might
    expect (effective-variance weighting without x-errors).
    """
    import importlib.util

    errors: list[str] = []
    warnings: list[str] = []
    caps = METHOD_CAPS.get(method)
    if caps is None:
        return errors, warnings

    for module in caps.requires:
        if importlib.util.find_spec(module) is None:
            errors.append(
                f"The '{method}' method needs the '{module}' package, which "
                f"is not installed."
            )

    if caps.needs_bounds and params is not None:
        unbounded = [
            name for name, par in params.items()
            if par.vary and not (np.isfinite(par.min) and np.isfinite(par.max))
        ]
        if unbounded:
            errors.append(
                f"The '{method}' method needs finite min and max bounds on "
                f"every varying parameter. Missing bounds: "
                f"{', '.join(unbounded)}."
            )

    if weight_mode == WEIGHT_EFFECTIVE_VARIANCE and not has_xerr:
        warnings.append(
            "Effective-variance weighting needs x-errors; this series has "
            "none, so 1/yerr weighting is used instead."
        )

    if method == "odr" and not has_xerr:
        errors.append(
            "Orthogonal distance regression needs x-errors; this series has "
            "none. Without them ODR silently assumes an x-error of 1 in data "
            "units and absorbs the residuals into x. Add an x-error column, "
            "or use a least-squares method."
        )

    return errors, warnings


def objective_kwargs(method: str, label: str, f_scale: float = DEFAULT_F_SCALE) -> dict:
    """fit_kws contribution for a method's chosen objective.

    Returns {"reduce_fcn": fn} for scalar methods, {"loss": ..., "f_scale":
    ...} for least_squares, or {} for methods whose objective is fixed.
    Unknown labels fall back to the method's default (empty kwargs), so a
    stale selection can never inject an invalid argument.
    """
    kind = METHOD_CAPS[method].objective
    if kind == OBJ_LOSS:
        loss = LOSS_FUNCTIONS.get(label, "linear")
        kws = {"loss": loss}
        if loss != "linear":
            kws["f_scale"] = f_scale
        return kws
    if kind == OBJ_SCALAR:
        fn = REDUCE_FUNCTIONS.get(label)
        return {"reduce_fcn": fn} if fn is not None else {}
    return {}


# Points in the dense grid used for smooth fit curves.
N_DENSE = 500

# Central-difference step for df/dx in effective-variance weights: relative
# to |x|, with a floor so x == 0 still gets a usable step.
DFDX_REL_STEP = 1e-8
DFDX_MIN_STEP = 1e-10
# Effective-variance weights are iterated to self-consistency: refit with
# weights re-evaluated at the fitted parameters until no varying parameter
# moves by more than this relative amount, or the round limit is hit.
EFFECTIVE_VARIANCE_MAX_ROUNDS = 10
EFFECTIVE_VARIANCE_RTOL = 1e-6
# Central-difference step for the ODR band Jacobian (df/dbeta).
ODR_JAC_REL_STEP = 1e-6
ODR_JAC_MIN_STEP = 1e-10
# Stand-in for an infinite parameter bound in odrpack, which needs finite
# bounds: the largest finite double.
ODR_UNBOUNDED = float(np.finfo(float).max)


def make_gof(chisqr=None, redchi=None, rsquared=None, aic=None, bic=None,
             objective=None) -> dict:
    """Goodness-of-fit dict with the canonical key names and order.

    Statistics that a fit doesn't provide (ODR has no AIC/BIC) are left
    out, so consumers never have to format a None. "objective" is the
    minimized value when the objective was not chi-square (a robust loss
    or a reduce function); chi-square is then still reported, as the
    weighted sum of squares at the solution, but it is not what was
    minimized.
    """
    values = {
        "chi-squared": chisqr,
        "reduced chi-squared": redchi,
        "R-squared": rsquared,
        "AIC": aic,
        "BIC": bic,
        "objective": objective,
    }
    return {k: v for k, v in values.items() if v is not None}


# scipy.optimize.least_squares loss functions rho(z), z = (r/f_scale)^2; the
# minimized cost is 0.5 * f_scale^2 * sum(rho(z)).
_SCIPY_LOSSES = {
    "linear": lambda z: z,
    "soft_l1": lambda z: 2.0 * (np.sqrt(1.0 + z) - 1.0),
    "huber": lambda z: np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0),
    "cauchy": lambda z: np.log1p(z),
    "arctan": lambda z: np.arctan(z),
}


def objective_value(residual: np.ndarray, fit_kws: dict) -> float | None:
    """The value the minimizer actually minimized, when it was not
    chi-square: sum(reduce_fcn) for a scalar reduce function, or scipy's
    robust cost for a least_squares loss. None when the objective was
    chi-square, so callers can tell the two cases apart.

    lmfit's chisqr/redchi/aic/bic are always the plain (weighted) sum of
    squares, whatever the objective; after a robust fit they are not the
    minimized quantity and AIC/BIC cannot rank models fitted with it.
    """
    fit_kws = fit_kws or {}
    reduce_fcn = fit_kws.get("reduce_fcn")
    if callable(reduce_fcn):
        return float(reduce_fcn(np.asarray(residual, dtype=float)))
    loss = fit_kws.get("loss", "linear")
    if loss and loss != "linear":
        rho = _SCIPY_LOSSES.get(loss)
        if rho is None:
            return None
        f_scale = float(fit_kws.get("f_scale", 1.0))
        z = (np.asarray(residual, dtype=float) / f_scale) ** 2
        return float(0.5 * f_scale ** 2 * np.sum(rho(z)))
    return None


@dataclass(frozen=True)
class FitOptions:
    """The fit settings read from the Fit panel, in one object rather than a
    positional tuple: the objective keywords (see objective_kwargs), the
    weight mode, an optional evaluation cap, the confidence-band sigma, and
    whether the covariance is scaled by reduced chi-square."""
    objective_kws: dict
    weight_mode: str = DEFAULT_WEIGHT_MODE
    max_nfev: int | None = None
    band_sigma: int = 1
    scale_covar: bool = True

    def run_fit_kwargs(self) -> dict:
        """Keyword arguments for FitManager.run_fit."""
        return dict(objective_kws=self.objective_kws, weight_mode=self.weight_mode,
                    max_nfev=self.max_nfev, band_sigma=self.band_sigma,
                    scale_covar=self.scale_covar)


def param_info(value, stderr=None, minimum=-np.inf, maximum=np.inf,
               vary=True, expr="") -> dict:
    """One parameter's entry for FitResult.params.

    at_bound is "min", "max" or None. A varying parameter that ended on a
    bound is not at an interior minimum, so its stderr (the curvature there)
    does not describe its uncertainty.
    """
    return {
        "value": value,
        "stderr": stderr,
        "min": minimum,
        "max": maximum,
        "vary": vary,
        "expr": expr,
        "at_bound": _at_bound(value, minimum, maximum) if vary else None,
    }


def _at_bound(value, minimum, maximum) -> str | None:
    """Which bound (if any) a value sits on, to within rounding."""
    if value is None:
        return None
    for tag, bound in (("min", minimum), ("max", maximum)):
        if bound is not None and np.isfinite(bound) and np.isclose(
                value, bound, rtol=1e-9, atol=1e-12):
            return tag
    return None


def convergence_lines(success: bool, message: str, nfev, params: dict,
                      strong_correlations=()) -> list[str]:
    """Report lines describing how the minimizer ended, which varying
    parameters sit on a bound, and which pairs are strongly correlated;
    prepended to every fit report."""
    status = "converged" if success else "NOT converged"
    detail = f" -- {message}" if message else ""
    evals = f" ({nfev} function evaluations)" if nfev is not None else ""
    lines = ["[[Convergence]]", f"    {status}{detail}{evals}"]
    bounded = [f"{name} (at {info['at_bound']})" for name, info in params.items()
               if info.get("at_bound")]
    if bounded:
        lines.append("    at a bound, stderr not meaningful: " + ", ".join(bounded))
    if strong_correlations:
        pairs = ", ".join(f"{a}/{b} (rho = {rho:+.3f})" for a, b, rho in strong_correlations)
        lines.append(f"    strongly correlated (|rho| > {STRONG_CORRELATION}): {pairs}")
    return lines


# Stochastic global optimizers accept a seed through fit_kws. A fixed
# default makes a repeated fit reproduce itself; brute and shgo are
# deterministic and take none.
SEEDED_METHODS = {"differential_evolution", "basinhopping", "dual_annealing"}
DEFAULT_GLOBAL_SEED = 0

# Sampling diagnostics targets (Foreman-Mackey et al. 2013, emcee docs).
EMCEE_ACCEPTANCE_RANGE = (0.2, 0.5)
EMCEE_MIN_CHAIN_OVER_TAU = 50
# Sampler settings offered in the MCMC Settings dialog. burn discards the
# first steps of every walker (lmfit's default of 0 averages over the
# burn-in); seed makes a run reproducible.
DEFAULT_EMCEE_SETTINGS = {"steps": 1000, "nwalkers": 100, "burn": 200, "thin": 1, "seed": 0}


def emcee_diagnostics(result, kws: dict) -> dict:
    """JSON-safe summary of an emcee run: sampler settings, mean acceptance
    fraction, per-parameter autocorrelation time tau, and the chain length
    in units of the longest tau. Values the user must judge the sample by,
    which lmfit computes but the report never showed."""
    chain = getattr(result, "chain", None)
    steps_kept = int(chain.shape[0]) if chain is not None and chain.ndim >= 2 else None
    acc = getattr(result, "acceptance_fraction", None)
    acor = getattr(result, "acor", None)
    names = list(getattr(result, "var_names", []) or [])
    tau = None
    if acor is not None:
        try:
            tau = {n: float(t) for n, t in zip(names, np.atleast_1d(acor))}
        except Exception:
            tau = None
    steps = int(kws.get("steps", 1000))
    ratio = None
    if tau and steps_kept:
        worst = max(tau.values())
        ratio = float(steps / worst) if worst > 0 else None
    return {
        "steps": steps,
        "nwalkers": int(kws.get("nwalkers", 100)),
        "burn": int(kws.get("burn", 0)),
        "thin": int(kws.get("thin", 1)),
        "seed": kws.get("seed"),
        "is_weighted": bool(kws.get("is_weighted", True)),
        "steps_kept": steps_kept,
        "acceptance_fraction": float(np.mean(acc)) if acc is not None else None,
        "autocorr_time": tau,
        "steps_over_tau": ratio,
    }


def emcee_lines(mcmc: dict) -> list[str]:
    """Report lines for an emcee result: what the table's numbers are and
    whether the sample can be trusted."""
    lo, hi = EMCEE_ACCEPTANCE_RANGE
    lines = [
        "[[MCMC (emcee)]]",
        "    values are posterior medians; StdErr is half the 16th-84th "
        "percentile range (a credible, not a confidence, interval)",
        f"    walkers {mcmc['nwalkers']}, steps {mcmc['steps']}, burn "
        f"{mcmc['burn']}, thin {mcmc['thin']}"
        + (f", seed {mcmc['seed']}" if mcmc.get("seed") is not None else ""),
    ]
    acc = mcmc.get("acceptance_fraction")
    if acc is not None:
        ok = lo <= acc <= hi
        lines.append(f"    mean acceptance fraction {acc:.2f}"
                     + ("" if ok else f"  (outside the {lo}-{hi} target: the walkers "
                        "are moving too rarely or too freely)"))
    tau = mcmc.get("autocorr_time")
    if tau:
        lines.append("    autocorrelation time: "
                     + ", ".join(f"{n} {t:.1f}" for n, t in tau.items()))
        ratio = mcmc.get("steps_over_tau")
        if ratio is not None:
            need = EMCEE_MIN_CHAIN_OVER_TAU
            lines.append(f"    steps / longest tau = {ratio:.0f}"
                         + ("" if ratio >= need else f"  (below {need}: run more steps, "
                            "the sample is not yet converged)"))
    else:
        lines.append("    autocorrelation time could not be estimated: the chain "
                     "is too short to judge convergence")
    if not mcmc.get("is_weighted", True):
        lines.append("    __lnsigma is the fitted log of the noise scale "
                     "(unweighted residuals): the data's sigma, not a model parameter")
    return lines


def strong_correlations(params) -> list[list]:
    """[name_a, name_b, rho] for each varying pair with |rho| above
    STRONG_CORRELATION, strongest first. Such pairs trade off against each
    other: only a combination is well determined by the data."""
    seen = set()
    out = []
    for name, par in params.items():
        if not par.vary or not par.correl:
            continue
        for other, rho in par.correl.items():
            key = tuple(sorted((name, other)))
            if key in seen or rho is None or abs(rho) <= STRONG_CORRELATION:
                continue
            seen.add(key)
            out.append([key[0], key[1], float(rho)])
    out.sort(key=lambda t: -abs(t[2]))
    return out


@dataclass
class FitComponent:
    """A single model component in a composite fit."""

    name: str  # Registry name, e.g. "Gaussian"
    prefix: str  # e.g. "gauss1_"
    operator: str = "+"  # "+" (sum) or "*" (multiply); ignored for first component
    expression: str = ""  # Only used when name == "Expression"


def _objective_to_serializable(fit_kws: dict) -> dict:
    """The objective part of the fit keywords in JSON-safe form: a reduce
    function is stored by name so a workspace can record it."""
    out = {}
    for key, value in (fit_kws or {}).items():
        if key == "reduce_fcn" and callable(value):
            out[key] = value.__name__
        elif key in ("reduce_fcn", "loss", "f_scale"):
            out[key] = value
    return out


def _objective_from_serializable(data: dict | None) -> dict:
    """Inverse of _objective_to_serializable. An unknown reduce-function
    name (removed in a later version) is dropped, giving chi-square."""
    out = {}
    for key, value in (data or {}).items():
        if key == "reduce_fcn" and isinstance(value, str):
            fn = next((f for f in REDUCE_FUNCTIONS.values()
                       if f is not None and f.__name__ == value), None)
            if fn is not None:
                out[key] = fn
        elif key in ("loss", "f_scale"):
            out[key] = value
    return out


def _expr_names(expr: str) -> set[str]:
    """Names referenced by a parameter expression (an lmfit/asteval string)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _rewrite_expr_names(expr: str, mapping: dict[str, str]) -> str:
    """Return expr with every referenced name in mapping replaced by its
    mapped name, leaving functions and other names untouched."""

    class _Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            new = mapping.get(node.id)
            return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node) if new else node

    tree = _Rename().visit(ast.parse(expr, mode="eval"))
    return ast.unparse(tree)


class FitManager:
    """Builds composite lmfit models, auto-guesses parameters, and runs fits."""

    def __init__(self):
        self.components: list[FitComponent] = []
        self._model: Model | None = None
        self._params: Parameters | None = None
        self._last_result: ModelResult | None = None
        self._param_hints: dict[str, dict] = {}
        self._name_counters: dict[str, int] = {}
        # Values each fit starts from. Kept separate from _params (which
        # holds the last fit's result) so repeated fits are reproducible
        # instead of chaining from the previous result. Set by auto_guess and
        # by user value edits; keyed by parameter name.
        self._start_values: dict[str, float] = {}

    def add_component(
        self, model_name: str, operator: str = "+", expression: str = ""
    ) -> FitComponent:
        """Add a model component. operator is '+' or '*'; ignored for first component."""
        # Monotonic per-name counter so a freed suffix (from remove_component)
        # is never reused and can't collide with a still-present component.
        count = self._name_counters.get(model_name, 0) + 1
        self._name_counters[model_name] = count
        if len(self.components) == 0:
            # Single component: no prefix for cleaner parameter names
            prefix = ""
        else:
            if len(self.components) == 1 and self.components[0].prefix == "":
                # Retroactively add prefix to first component. Its hints and
                # start values are keyed by the bare names and must follow,
                # or the user's edits silently revert on rebuild.
                old = self.components[0]
                old.prefix = f"{old.name.lower()}1_"
                self._reprefix_param_state("", old.prefix)
            prefix = f"{model_name.lower()}{count}_"

        comp = FitComponent(
            name=model_name, prefix=prefix, operator=operator, expression=expression
        )
        self.components.append(comp)
        self._rebuild_model()
        return comp

    def remove_component(self, index: int):
        """Remove a component by index and rebuild.

        Parameter hints are keyed by prefixed parameter name, so they follow
        the component list: the removed component's hints are dropped, and a
        lone survivor's hints are re-keyed when it loses its prefix.
        """
        if not (0 <= index < len(self.components)):
            return
        removed = self.components.pop(index)
        self._drop_param_state(removed.prefix)
        if len(self.components) == 1:
            # Single component: no prefix for cleaner parameter names
            old_prefix = self.components[0].prefix
            self.components[0].prefix = ""
            self._reprefix_param_state(old_prefix, "")
        self._rebuild_model()

    def _drop_param_state(self, prefix: str):
        """Forget hints and start values belonging to a component prefix.

        Stale start values are not harmless: if the prefix is later reused
        (a lone component regains bare names), _reset_to_start_values would
        resurrect them over the current values.
        """
        for mapping in (self._param_hints, self._start_values):
            for name in [n for n in mapping if n.startswith(prefix)]:
                del mapping[name]

    def _reprefix_param_state(self, old_prefix: str, new_prefix: str):
        """Re-key hints and start values from one component prefix to
        another, so they keep applying to the same parameters."""
        if old_prefix == new_prefix:
            return
        for mapping in (self._param_hints, self._start_values):
            for name in [n for n in mapping if n.startswith(old_prefix)]:
                mapping[new_prefix + name[len(old_prefix):]] = mapping.pop(name)

    def edit_expression(self, index: int, new_expr: str):
        """Update the expression of an Expression component and rebuild."""
        if 0 <= index < len(self.components):
            comp = self.components[index]
            if comp.name == "Expression":
                comp.expression = new_expr
                self._rebuild_model()

    def clear_components(self):
        self.components.clear()
        self._model = None
        self._params = None
        self._last_result = None
        self._param_hints.clear()
        self._name_counters.clear()
        self._start_values.clear()

    def set_start_value(self, name: str, value: float):
        """Record the value the next fit should start this parameter from.
        Called when the user edits a value so a manual start isn't discarded
        by the reset-to-guess behavior in run_fit."""
        self._start_values[name] = value

    def capture_start_values(self):
        """Adopt the current parameter values as the fit starting point,
        e.g. to reuse a fit's result as the next fit's guess."""
        if self._params is not None:
            self._start_values = {n: p.value for n, p in self._params.items()}

    def _prepare_emcee_kws(self, kws: dict, weights) -> None:
        """Fill in the emcee keywords that must follow from the fit setup.

        is_weighted: whether the residual is already (data - model)/sigma.
        It follows from the weights actually in use, not from whether the
        series has a yerr column: 'No weights' with yerr present is an
        unweighted residual, and emcee must then fit the noise scale
        (__lnsigma) rather than assume sigma = 1.

        pos: lmfit starts every walker at value * (1 + 1e-4 N(0,1)), so a
        parameter whose value is exactly 0 gets identical walkers and the
        ensemble cannot move in that direction. Walkers are placed here
        with an absolute spread for zero-valued parameters, inside the
        bounds, using the caller's seed for reproducibility.
        """
        kws.setdefault("is_weighted", weights is not None)
        if "pos" in kws:
            return
        rng = np.random.default_rng(kws.get("seed"))
        names = [n for n, p in self._params.items() if p.vary and not p.expr]
        values = [self._params[n].value for n in names]
        lows = [self._params[n].min for n in names]
        highs = [self._params[n].max for n in names]
        if not kws["is_weighted"] and "__lnsigma" not in self._params:
            names.append("__lnsigma")
            values.append(0.01)
            lows.append(-np.inf)
            highs.append(np.inf)
        nwalkers = int(kws.get("nwalkers", 100))
        values = np.asarray(values, dtype=float)
        spread = np.where(values != 0, 1e-4 * np.abs(values), 1e-4)
        pos = values + spread * rng.standard_normal((nwalkers, len(values)))
        lo = np.asarray(lows, dtype=float)
        hi = np.asarray(highs, dtype=float)
        margin = np.where(np.isfinite(hi - lo), 1e-6 * np.abs(hi - lo), 0.0)
        pos = np.clip(pos, lo + margin, hi - margin)
        kws["pos"] = pos

    def _reset_to_start_values(self):
        """Reset each varying parameter to its remembered starting value so
        repeated fits are reproducible instead of chaining from the previous
        result. A parameter without a recorded start (first fit, or one added
        since the last guess) has its current value captured now. Fixed
        parameters are user-set constants that don't drift, so they are left
        untouched."""
        for name, par in self._params.items():
            if not par.vary:
                continue
            if name in self._start_values:
                par.set(value=self._start_values[name])
            else:
                self._start_values[name] = par.value

    def _build_component_model(self, comp: FitComponent, x_data: np.ndarray | None = None):
        """Build a single component model. Uses x_data for Spline if available."""
        if comp.name == "Expression":
            return create_expression_model(comp.expression, prefix=comp.prefix)
        elif comp.name == "Spline":
            # Parse knot count from expression field (e.g. "knots:8")
            n_knots = 8
            if comp.expression and comp.expression.startswith("knots:"):
                try:
                    n_knots = int(comp.expression.split(":")[1])
                except (ValueError, IndexError):
                    pass
            if x_data is not None:
                return create_spline_model(n_knots, x_data, prefix=comp.prefix)
            else:
                # Placeholder knots when no data available yet
                xknots = np.linspace(0, 1, n_knots)
                return SplineModel(xknots=xknots, prefix=comp.prefix)
        else:
            return create_model(comp.name, prefix=comp.prefix)

    def _assemble_model(self, models: list):
        """Combine individual component models using operators."""
        composite = models[0]
        for comp, m in zip(self.components[1:], models[1:]):
            if comp.operator == "*":
                composite = composite * m
            elif comp.operator == "-":
                composite = composite - m
            elif comp.operator == "/":
                composite = composite / m
            else:
                composite = composite + m
        return composite

    def _rebuild_model(self):
        """Rebuild the composite model from current components."""
        if not self.components:
            self._model = None
            self._params = None
            return

        models = [self._build_component_model(c) for c in self.components]
        self._model = self._assemble_model(models)
        self._params = self._model.make_params()

        # ExpressionModel params default to -inf; set to 1.0 so fits don't NaN
        for par in self._params.values():
            if par.value == float("-inf"):
                par.set(value=1.0)

        # Re-apply stored parameter hints so edits survive rebuilds
        for name, hints in self._param_hints.items():
            if name in self._params:
                self._params[name].set(**hints)

    def _rebuild_model_with_data(self, x: np.ndarray):
        """Rebuild model using real x-data (needed for Spline knots)."""
        if not self.components:
            self._model = None
            self._params = None
            return

        models = [self._build_component_model(c, x_data=x) for c in self.components]
        self._model = self._assemble_model(models)

        # Preserve existing param values where possible
        old_params = self._params
        self._params = self._model.make_params()
        if old_params is not None:
            for name, par in old_params.items():
                if name in self._params:
                    self._params[name].set(
                        value=par.value, min=par.min, max=par.max,
                        vary=par.vary, expr=par.expr,
                    )
        for par in self._params.values():
            if par.value == float("-inf"):
                par.set(value=1.0)

        # Re-apply stored parameter hints so edits survive rebuilds
        for name, hints in self._param_hints.items():
            if name in self._params:
                self._params[name].set(**hints)

    @property
    def _has_spline(self) -> bool:
        return any(c.name == "Spline" for c in self.components)

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def params(self) -> Parameters | None:
        return self._params

    def auto_guess(self, x: np.ndarray, y: np.ndarray) -> Parameters:
        """Auto-guess parameters for each component using lmfit's guess().

        Additively-combined components are guessed sequentially against the
        residual after subtracting previously-guessed components, so e.g.
        three summed Gaussians don't all guess the same dominant peak.
        Components combined with *, -, / are guessed against the original
        data, since subtracting their contribution isn't meaningful.

        Only updates value/min/max from guess; preserves user-edited vary,
        expr, and param_hints.
        """
        if self._model is None:
            raise ValueError("No model defined")

        if self._has_spline:
            self._rebuild_model_with_data(x)

        residual = np.asarray(y, dtype=float).copy()

        for i, comp in enumerate(self.components):
            m = self._build_component_model(comp, x_data=x)
            is_additive = i == 0 or comp.operator == "+"
            target = residual if is_additive else y
            try:
                guessed = m.guess(target, x=x)
                for pname, par in guessed.items():
                    if pname not in self._params:
                        continue
                    if self._params[pname].expr:
                        # A derived parameter (fwhm, height, or a user tie)
                        # takes its value from its expression. Parameter.set
                        # with a value clears the expression, which would
                        # freeze it at the guess and drop it from the fit.
                        continue
                    self._params[pname].set(
                        value=par.value, min=par.min, max=par.max
                    )
                if is_additive:
                    try:
                        residual = residual - m.eval(guessed, x=x)
                    except Exception:
                        pass  # Keep prior residual if this component can't be evaluated yet
            except NotImplementedError:
                pass  # Model doesn't implement guess()

        # The guess is the starting point every subsequent fit resets to.
        self._start_values = {name: par.value for name, par in self._params.items()}
        return self._params

    def clone_components_to(self, target: "FitManager"):
        """Copy this manager's component list and parameter hints (fixed
        values, bounds) into target, rebuilding its model."""
        if target is self:
            # Batch fit clones the source model onto every series, including
            # the one it came from. Without this guard clear_components()
            # would empty the list being copied, destroying the model.
            return
        target.clear_components()
        for comp in self.components:
            target.add_component(comp.name, operator=comp.operator, expression=comp.expression)

        # add_component can assign a target component a different prefix
        # than the source's (e.g. if a middle component was removed from
        # the source's history), so hints are remapped by position rather
        # than copied verbatim. Longest prefix first so a component with
        # an empty prefix doesn't swallow every hint name.
        prefix_pairs = sorted(
            zip((c.prefix for c in self.components), (c.prefix for c in target.components)),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        def remap(name):
            for old_prefix, new_prefix in prefix_pairs:
                if name.startswith(old_prefix):
                    return new_prefix + name[len(old_prefix):]
            return None

        for name, hints in self._param_hints.items():
            new_name = remap(name)
            if new_name is not None:
                target.set_param_hint(new_name, **hints)
        # The start values travel too, so the clone's first fit begins where
        # the source's would, not from the model defaults.
        for name, value in self._start_values.items():
            new_name = remap(name)
            if new_name is not None and new_name in (target.params or {}):
                target._start_values[new_name] = value

    def set_param_hint(self, name: str, **kwargs):
        """Store a parameter hint that survives model rebuilds.

        A value given here also becomes the fit's starting value; see
        set_param.
        """
        self._param_hints[name] = {**self._param_hints.get(name, {}), **kwargs}
        if self._model is not None:
            self._model.set_param_hint(name, **kwargs)
        if self._params is not None and name in self._params:
            self._params[name].set(**kwargs)
        if "value" in kwargs:
            self._start_values[name] = kwargs["value"]

    def set_param(self, name: str, **kwargs):
        """Set parameter attributes (value, min, max, vary, expr).

        A value set this way is a deliberate choice of where the next fit
        should start, so it is recorded as the start value here rather than
        relying on every caller to remember set_start_value: run_fit resets
        to the start values, and a caller that forgot would have its edit
        silently discarded.
        """
        if self._params is not None and name in self._params:
            self._params[name].set(**kwargs)
        if "value" in kwargs:
            self._start_values[name] = kwargs["value"]

    def model_description(self) -> str:
        """Compact one-line description, e.g. 'Gaussian + Linear'."""
        return " ".join(
            c.name if i == 0 else f"{c.operator} {c.name}"
            for i, c in enumerate(self.components)
        )

    def component_labels(self) -> list[str]:
        """Verbose per-component labels for UI list display."""
        labels = []
        for i, c in enumerate(self.components):
            if c.name == "Expression" and c.expression:
                display = f"Expression: {c.expression}"
                if c.prefix:
                    display = f"{display} ({c.prefix})"
            elif c.name == "Spline" and c.expression:
                display = f"Spline [{c.expression}]"
                if c.prefix:
                    display = f"{display} ({c.prefix})"
            else:
                display = f"{c.name} ({c.prefix})" if c.prefix else c.name
            if i > 0:
                display = f"{c.operator} {display}"
            labels.append(display)
        return labels

    @staticmethod
    def params_to_info(params) -> dict[str, dict]:
        """Convert lmfit Parameters to {name: {value, stderr, min, max, vary, expr}}."""
        return {
            name: param_info(par.value, par.stderr, par.min, par.max,
                             par.vary, par.expr or "")
            for name, par in params.items()
        }

    @staticmethod
    def _effective_variance_weights(eval_at, x, yerr, xerr):
        """w = 1/sqrt(yerr² + (df/dx)²·xerr²), with df/dx by central
        difference. eval_at(x) evaluates the model at the given x."""
        h = np.maximum(np.abs(x) * DFDX_REL_STEP, DFDX_MIN_STEP)
        dfdx = (eval_at(x + h) - eval_at(x - h)) / (2 * h)
        denom = np.maximum(np.sqrt(yerr**2 + (dfdx * xerr) ** 2), MIN_ERROR)
        return 1.0 / denom

    @staticmethod
    def _compute_weights(y, yerr, weight_mode):
        """Compute weights array from y, yerr, and the selected weight mode.

        Effective variance is not handled here: it depends on the model
        derivative, so it is computed per fit (and per iteration for a
        global fit) by _effective_variance_weights.
        """
        if weight_mode == WEIGHT_NONE:
            return None
        if weight_mode == WEIGHT_INV_Y:
            # sigma proportional to |y|. A zero y would get an infinite
            # weight, so zeros are floored at the smallest non-zero |y|:
            # they count like the smallest measured magnitude, not like an
            # exact constraint.
            abs_y = np.abs(y)
            positive = abs_y[abs_y > 0]
            if positive.size == 0:
                return None
            return 1.0 / np.maximum(abs_y, positive.min())
        # Default: WEIGHT_INV_YERR
        if yerr is not None:
            return 1.0 / np.maximum(np.abs(yerr), MIN_ERROR)
        return None

    def run_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        xerr: np.ndarray | None = None,
        n_dense: int = N_DENSE,
        method: str = DEFAULT_FIT_METHOD,
        iter_cb=None,
        fit_kws: dict | None = None,
        reduce_fcn=None,
        objective_kws: dict | None = None,
        weight_mode: str = DEFAULT_WEIGHT_MODE,
        max_nfev: int | None = None,
        band_sigma: int = 1,
        scale_covar: bool = True,
    ) -> FitResult:
        """Run the fit and return results.

        objective_kws (from objective_kwargs()) carries the method-appropriate
        objective: a scipy ``loss``/``f_scale`` for least_squares, or a
        ``reduce_fcn`` for the scalar minimizers. The standalone reduce_fcn
        argument is kept for direct callers; objective_kws wins if both give
        the same key.
        """
        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        # Rebuild with real x-data if model contains a Spline component
        if self._has_spline:
            self._rebuild_model_with_data(x)

        # Start from the remembered guess/user values, not the previous
        # result, so repeated fits are reproducible and don't drift along a
        # degenerate direction. emcee is the exception: it samples the
        # posterior around wherever the parameters are, and the useful place
        # is the least-squares optimum the user has just found.
        if method != "emcee":
            self._reset_to_start_values()

        # Snapshot initial parameter values before fitting
        init_values = {name: par.value for name, par in self._params.items()}

        weights = self._compute_weights(y, yerr, weight_mode)

        iterate_weights = False
        if weight_mode == WEIGHT_EFFECTIVE_VARIANCE:
            if xerr is not None and yerr is not None:
                weights = self._effective_variance_weights(
                    lambda xv: self._model.eval(self._params, x=xv), x, yerr, xerr
                )
                iterate_weights = True
            elif yerr is not None:
                weights = 1.0 / np.maximum(np.abs(yerr), MIN_ERROR)
            else:
                weights = None

        kws = dict(fit_kws or {})
        if reduce_fcn is not None:
            kws["reduce_fcn"] = reduce_fcn
        if objective_kws:
            kws.update(objective_kws)
        if method == "emcee":
            self._prepare_emcee_kws(kws, weights)
        elif method in SEEDED_METHODS:
            kws.setdefault("seed", DEFAULT_GLOBAL_SEED)

        fit_kwargs = dict(
            method=method, nan_policy="omit",
            iter_cb=iter_cb, fit_kws=kws,
            scale_covar=scale_covar,
        )
        if max_nfev is not None:
            fit_kwargs["max_nfev"] = max_nfev

        self._last_result = self._model.fit(
            y, self._params, x=x, weights=weights,
            **fit_kwargs,
        )

        if iterate_weights:
            # The effective-variance weights depend on df/dx, which was
            # evaluated at the starting parameters. Left there, the answer
            # depends on the initial guess. Re-evaluate at the fitted
            # parameters and refit until the parameters stop moving (Orear's
            # iterated scheme), so the weights and the fit are consistent.
            for _ in range(EFFECTIVE_VARIANCE_MAX_ROUNDS):
                prev = np.array([p.value for p in self._last_result.params.values() if p.vary])
                fitted = self._last_result.params
                weights = self._effective_variance_weights(
                    lambda xv: self._model.eval(fitted, x=xv), x, yerr, xerr
                )
                self._last_result = self._model.fit(
                    y, fitted, x=x, weights=weights, **fit_kwargs,
                )
                new = np.array([p.value for p in self._last_result.params.values() if p.vary])
                scale = np.maximum(np.abs(new), np.abs(prev))
                scale[scale == 0] = 1.0
                if np.all(np.abs(new - prev) / scale < EFFECTIVE_VARIANCE_RTOL):
                    break

        # Dense x-grid for smooth curve
        x_dense = np.linspace(x.min(), x.max(), n_dense)
        y_fit_dense = self._last_result.eval(x=x_dense)

        # Confidence band on dense grid
        y_uncertainty = None
        try:
            y_uncertainty = self._last_result.eval_uncertainty(x=x_dense, sigma=band_sigma)
        except Exception:
            pass  # Covariance matrix not available

        # Component curves on dense grid
        component_curves = {}
        if len(self.components) > 1:
            comps = self._last_result.eval_components(x=x_dense)
            for key, vals in comps.items():
                component_curves[key] = vals

        params_info = self.params_to_info(self._last_result.params)
        correlated = strong_correlations(self._last_result.params)
        success = bool(getattr(self._last_result, "success", True))
        message = str(getattr(self._last_result, "message", "") or "")
        nfev = getattr(self._last_result, "nfev", None)
        # nan_policy="omit" silently drops points where the model is NaN
        # (an Expression with log or sqrt of a negative, say). The fit then
        # ran on fewer points than the user gave it, and chi-square, AIC
        # and BIC refer to that smaller set.
        n_omitted = int(len(y) - self._last_result.ndata) if self._last_result.ndata else 0
        if n_omitted > 0:
            message = (message + " | " if message else "") + (
                f"{n_omitted} point(s) where the model is not finite were "
                f"omitted; statistics refer to the remaining {self._last_result.ndata}")

        # Update internal params with fitted values. An unweighted emcee run
        # adds __lnsigma (the fitted noise scale) to the result's params; it
        # is reported with the result but must not become a model parameter
        # that every later fit carries along.
        self._params = self._last_result.params.copy()
        self._params.pop("__lnsigma", None)

        objective = objective_value(self._last_result.residual, kws)
        gof = make_gof(
            chisqr=self._last_result.chisqr,
            redchi=self._last_result.redchi,
            rsquared=self._last_result.rsquared,
            # AIC/BIC are functions of chi-square, so after a robust or
            # reduce-function fit they describe a quantity that was not
            # minimized and would rank an outlier-dominated chi-square fit
            # above the robust one. Left out in that case.
            aic=self._last_result.aic if objective is None else None,
            bic=self._last_result.bic if objective is None else None,
            objective=objective,
        )

        # Capture brute-force candidates
        candidates = None
        if hasattr(self._last_result, "candidates") and self._last_result.candidates:
            candidates = []
            for cand in self._last_result.candidates:
                entry = {"score": cand.score}
                entry["params"] = {
                    name: cand.params[name].value for name in cand.params
                }
                candidates.append(entry)

        # Capture emcee flatchain and sampling diagnostics
        flatchain = None
        mcmc = None
        if hasattr(self._last_result, "flatchain") and self._last_result.flatchain is not None:
            flatchain = self._last_result.flatchain
            mcmc = emcee_diagnostics(self._last_result, kws)

        return FitResult(
            x_dense=x_dense,
            y_fit_dense=y_fit_dense,
            x_data=x,
            y_data=y,
            y_fit_data=self._last_result.best_fit,
            yerr_data=yerr,
            params=params_info,
            report="\n".join(
                convergence_lines(success, message, nfev, params_info, correlated)
                + (emcee_lines(mcmc) if mcmc else [])
                + ["", self._last_result.fit_report()]),
            gof=gof,
            component_curves=component_curves,
            y_uncertainty=y_uncertainty,
            candidates=candidates,
            flatchain=flatchain,
            mcmc=mcmc,
            init_params=init_values,
            errorbars=bool(getattr(self._last_result, "errorbars", True)),
            success=success,
            message=message,
            nfev=nfev,
            n_omitted=n_omitted,
            strong_correlations=correlated,
            xerr_data=xerr,
            fit_weights=None if weights is None else np.asarray(weights, dtype=float),
            fit_options={
                "method": method,
                "weight_mode": weight_mode,
                "objective": _objective_to_serializable(kws),
                "scale_covar": scale_covar,
            },
        )

    def refit_from_result(self, result: FitResult) -> None:
        """Re-run the fit from a stored result to reconstruct _last_result.

        Used after loading a workspace so the analysis tools (CIs, contours,
        profiles, ...) work without a manual refit. The refit starts from
        the saved best-fit values with the saved method, weights, objective
        and scale_covar, so it lands on the same optimum and the analysis
        describes the fit the table shows. Results from before the options
        were recorded, and methods that cannot be re-run in place (emcee,
        the global optimizers, ODR), fall back to the default least-squares
        method with default weights from the saved values; for a chi-square
        fit that is the same optimum, for emcee it is the least-squares fit
        nearest the posterior medians.
        """
        if self._model is None or self._params is None:
            return
        x, y = result.x_data, result.y_data
        yerr, xerr = result.yerr_data, result.xerr_data
        if self._has_spline:
            self._rebuild_model_with_data(x)

        # Start at the saved optimum.
        for name, info in (result.params or {}).items():
            if name in self._params and not self._params[name].expr:
                self._params[name].set(value=info["value"])

        opts = result.fit_options or {}
        method = opts.get("method", DEFAULT_FIT_METHOD)
        if method in SLOW_METHODS or method == "odr" or method not in METHOD_CAPS:
            method = DEFAULT_FIT_METHOD
        weight_mode = opts.get("weight_mode", DEFAULT_WEIGHT_MODE)
        if weight_mode not in WEIGHT_MODES:
            weight_mode = DEFAULT_WEIGHT_MODE
        scale_covar = bool(opts.get("scale_covar", True))
        fit_kws = _objective_from_serializable(opts.get("objective"))

        weights = self._compute_weights(y, yerr, weight_mode)
        if weight_mode == WEIGHT_EFFECTIVE_VARIANCE:
            if xerr is not None and yerr is not None:
                params = self._params
                weights = self._effective_variance_weights(
                    lambda xv: self._model.eval(params, x=xv), x, yerr, xerr
                )
            elif yerr is not None:
                weights = 1.0 / np.maximum(np.abs(yerr), MIN_ERROR)
            else:
                weights = None

        self._last_result = self._model.fit(
            y, self._params, x=x, weights=weights,
            method=method, nan_policy="omit", fit_kws=fit_kws,
            scale_covar=scale_covar,
        )
        self._params = self._last_result.params

    def run_global_fit(
        self,
        datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]],
        shared_params: set[str],
        method: str = DEFAULT_FIT_METHOD,
        max_nfev: int | None = None,
        weight_mode: str = DEFAULT_WEIGHT_MODE,
        objective_kws: dict | None = None,
        scale_covar: bool = True,
        iter_cb=None,
    ) -> list[FitResult]:
        """Run a global fit across multiple datasets with shared parameters.

        Each dataset is a tuple of (x, y, yerr, xerr). objective_kws,
        scale_covar and iter_cb mean the same as for run_fit, so the
        Objective and Scale-covariance controls and the abort button apply
        to a global fit too. Like run_fit, the fit starts from the
        remembered start values.
        Parameters that are in *shared_params* get one entry in the combined
        Parameters object.  Non-shared params get per-dataset prefixed entries
        (s0_, s1_, ...).  Returns one FitResult per dataset.

        A derived parameter (one with an expression, such as fwhm or height)
        is not a free parameter, so *shared_params* is ignored for it: it is
        shared exactly when every parameter its expression refers to is
        shared, and otherwise gets per-dataset entries whose expressions are
        rewritten to the per-dataset names.
        """
        from lmfit import Parameters, minimize

        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        # Same start as a single fit, so repeated global fits are
        # reproducible instead of chaining from the previous result.
        self._reset_to_start_values()

        n = len(datasets)
        base_params = self._params
        base_names = list(base_params.keys())

        # Resolve which names are shared, letting derived parameters follow
        # their dependencies. Iterate because an expression may refer to
        # another derived parameter.
        shared = {nm for nm in base_names
                  if nm in shared_params and not base_params[nm].expr}
        derived = [nm for nm in base_names if base_params[nm].expr]
        changed = True
        while changed:
            changed = False
            for nm in derived:
                deps = _expr_names(base_params[nm].expr) & set(base_names)
                if nm not in shared and deps <= shared:
                    shared.add(nm)
                    changed = True
        shared_params = shared

        def per_dataset_expr(expr: str, i: int) -> str | None:
            if not expr:
                return None
            return _rewrite_expr_names(
                expr, {nm: f"s{i}_{nm}" for nm in base_names if nm not in shared_params}
            )

        # Build combined Parameters
        combined = Parameters()
        for name in base_names:
            bp = base_params[name]
            if name in shared_params:
                combined.add(name, value=bp.value, min=bp.min, max=bp.max,
                             vary=bp.vary, expr=bp.expr or None)
            else:
                for i in range(n):
                    pname = f"s{i}_{name}"
                    combined.add(pname, value=bp.value, min=bp.min, max=bp.max,
                                 vary=bp.vary, expr=per_dataset_expr(bp.expr, i))

        # Every dataset must be weighted on the same footing: a dataset
        # without yerr gets unit weights, i.e. residuals in y units, which
        # concatenated with another dataset's 1/sigma residuals lets one of
        # them dominate silently and makes the combined chi-square a sum of
        # unlike quantities.
        if weight_mode in _YERR_WEIGHT_MODES:
            have = [yerr is not None for _, _, yerr, _ in datasets]
            if any(have) and not all(have):
                missing = [i + 1 for i, h in enumerate(have) if not h]
                raise ValueError(
                    f"Weight mode '{weight_mode}' needs y-errors on every "
                    f"dataset, but dataset(s) {missing} of {n} have none. "
                    "Give every series y-errors, or choose 'No weights'."
                )

        # Pre-compute constant weights (everything except Effective variance
        # with xerr, which depends on the model derivative per iteration)
        use_effective_variance = weight_mode == WEIGHT_EFFECTIVE_VARIANCE
        precomputed_weights = [
            self._compute_weights(y, yerr, weight_mode)
            for _, y, yerr, _ in datasets
        ]

        def objective(params):
            all_resid = []
            for i, (x, y, yerr, xerr) in enumerate(datasets):
                # Build per-dataset params
                kw = {}
                for name in base_names:
                    if name in shared_params:
                        kw[name] = params[name].value
                    else:
                        kw[name] = params[f"s{i}_{name}"].value
                # Evaluate model
                y_model = self._model.eval(x=x, **kw)
                resid = y - y_model
                weights = precomputed_weights[i]
                # Effective variance: recompute per iteration (depends on df/dx)
                if use_effective_variance and xerr is not None and yerr is not None:
                    weights = self._effective_variance_weights(
                        lambda xv: self._model.eval(x=xv, **kw), x, yerr, xerr
                    )
                if weights is not None:
                    resid = resid * weights
                all_resid.append(resid)
            return np.concatenate(all_resid)

        # objective_kws holds either a reduce_fcn (a Minimizer argument) or
        # a scipy loss/f_scale (forwarded by minimize as fit keywords).
        kws = dict(objective_kws or {})
        if max_nfev is not None:
            kws["max_nfev"] = max_nfev

        mini_result = minimize(objective, combined, method=method,
                               nan_policy="omit", scale_covar=scale_covar,
                               iter_cb=iter_cb, **kws)

        # The combined objective, when it was not chi-square (see run_fit).
        global_objective = objective_value(mini_result.residual, kws)

        # Build per-dataset FitResults
        results = []
        for i, (x, y, yerr, xerr) in enumerate(datasets):
            # Extract per-dataset param values
            params_info = {}
            init_values = {}
            for name in base_names:
                bp = base_params[name]
                if name in shared_params:
                    p = mini_result.params[name]
                else:
                    p = mini_result.params[f"s{i}_{name}"]
                # Report the user's expression, not the s{i}_-rewritten one.
                params_info[name] = param_info(
                    p.value, p.stderr, p.min, p.max, p.vary, bp.expr or ""
                )
                init_values[name] = bp.value

            # Evaluate model for this dataset
            eval_kw = {name: params_info[name]["value"] for name in base_names}
            x_dense = np.linspace(x.min(), x.max(), N_DENSE)
            y_fit_data = self._model.eval(x=x, **eval_kw)
            y_fit_dense = self._model.eval(x=x_dense, **eval_kw)

            gof = make_gof(
                chisqr=mini_result.chisqr,
                redchi=mini_result.redchi,
                rsquared=1 - np.sum((y - y_fit_data) ** 2) / np.sum((y - y.mean()) ** 2),
                aic=mini_result.aic if global_objective is None else None,
                bic=mini_result.bic if global_objective is None else None,
                objective=global_objective,
            )

            # Generate a report string
            lines = convergence_lines(
                bool(getattr(mini_result, "success", True)),
                str(getattr(mini_result, "message", "") or ""),
                getattr(mini_result, "nfev", None), params_info) + [""]
            lines.append(f"Global Fit — Dataset {i + 1}/{n}")
            lines.append(f"  Method: {method}")
            lines.append(f"  chi-squared = {gof['chi-squared']:.6g}")
            lines.append(f"  reduced chi-squared = {gof['reduced chi-squared']:.6g}")
            lines.append(f"  R-squared = {gof['R-squared']:.6g}")
            lines.append("")
            for name, info in params_info.items():
                shared_tag = " [shared]" if name in shared_params else ""
                stderr_str = f" +/- {info['stderr']:.6g}" if info['stderr'] is not None else ""
                lines.append(f"  {name}{shared_tag}: {info['value']:.6g}{stderr_str}")
            report = "\n".join(lines)

            results.append(FitResult(
                x_dense=x_dense,
                y_fit_dense=y_fit_dense,
                x_data=x,
                y_data=y,
                y_fit_data=y_fit_data,
                yerr_data=yerr,
                params=params_info,
                report=report,
                gof=gof,
                init_params=init_values,
                success=bool(getattr(mini_result, "success", True)),
                message=str(getattr(mini_result, "message", "") or ""),
                nfev=getattr(mini_result, "nfev", None),
                xerr_data=xerr,
                fit_weights=(None if precomputed_weights[i] is None
                             else np.asarray(precomputed_weights[i], dtype=float)),
            ))

        return results

    def ci_minimizer(self):
        """The last result, prepared for lmfit's confidence-interval tools.

        conf_interval, conf_interval2d and the profile traces re-minimize
        with ``Minimizer.leastsq``, which forwards the stored fit keywords
        to scipy's leastsq. Keywords that belong to another method (a
        robust ``loss``/``f_scale`` for least_squares, ``is_weighted`` for
        emcee) make that call raise, and a raise part-way through leaves
        parameters with vary=False. So hand the tools a shallow copy with
        those keywords cleared. The intervals are chi-square based in any
        case, whatever objective the fit itself used.
        """
        if self._last_result is None:
            raise ValueError("No fit result available")
        if self._last_result.method == "emcee":
            # The tools profile chi-square from result.chisqr, but an emcee
            # result's "best" values are posterior medians, not a chi-square
            # minimum, so the profile never brackets the target and the
            # root finder fails. The posterior already carries the answer.
            raise ValueError(
                "Profile-likelihood confidence intervals are not defined for "
                "an emcee result. Use the posterior errors in the parameter "
                "table (16th-84th percentile half-width), or refit with a "
                "least-squares method first."
            )
        mini = copy.copy(self._last_result)
        mini.kws = {}
        return mini

    # --- Confidence intervals -------------------------------------------
    #
    # lmfit's conf_interval profiles one parameter at a time and turns each
    # profile point into a probability with prob_func. Its default,
    # f_compare, is an F-test on chi2_new/chi2_best, which rescales the
    # threshold by reduced chi-square: the interval then matches a StdErr
    # computed with scale_covar=True. When the user has turned Scale
    # covariance off (real sigma), the matching threshold is a plain
    # delta-chi2, otherwise the CI and the StdErr disagree silently.

    def _ci_scaled(self) -> bool:
        """Whether the last fit's errors are reduced-chi-square scaled."""
        return bool(getattr(self._last_result, "scale_covar", True))

    def _ci_prob_func(self):
        """prob_func for lmfit's CI tools matching the fit's scale_covar."""
        if self._ci_scaled():
            return None                      # lmfit's f_compare
        from scipy.stats import chi2

        def delta_chi2_compare(best_fit, new_fit):
            nfix = best_fit.nvarys - new_fit.nvarys
            return chi2.cdf(max(new_fit.chisqr - best_fit.chisqr, 0.0), nfix)
        return delta_chi2_compare

    def _prob_to_chi2(self, probs, best_chi2, nfree):
        """Invert the CI probability back to the profile's chi-square:
        exactly the inverse of the prob_func in use."""
        from scipy.stats import chi2, f
        probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0 - 1e-12)
        if self._ci_scaled():
            return best_chi2 * (1.0 + f.ppf(probs, 1, nfree) / nfree)
        return best_chi2 + chi2.ppf(probs, 1)

    def ci_threshold(self, sigma: float = 1.0) -> float:
        """The delta-chi2 above the minimum that marks the sigma-level
        interval under the prob_func in use: sigma^2 for a plain delta-chi2
        (scale_covar off), chi2_best * F_ppf(p, 1, nu) / nu for the F-test
        (scale_covar on), which is close to sigma^2 * chi2_nu."""
        from scipy.stats import chi2, f, norm
        if self._last_result is None:
            raise ValueError("No fit result available")
        prob = 2.0 * norm.cdf(sigma) - 1.0
        if self._ci_scaled():
            nfree = self._last_result.nfree
            return float(self._last_result.chisqr * f.ppf(prob, 1, nfree) / nfree)
        return float(chi2.ppf(prob, 1))

    def joint_ci_thresholds(self, nfix: int = 2, sigmas=(1, 2, 3)) -> dict[str, float]:
        """delta-chi2 levels marking the joint sigma-level regions for nfix
        parameters varied together, under the CI method in use (see
        ci_threshold): chi2.ppf(p, nfix) unscaled, or the F-test level
        chi2_best * nfix * F_ppf(p, nfix, nu) / nu when scale_covar is on.
        Joint regions are wider than one-parameter intervals: 1-sigma for
        two parameters is delta-chi2 = 2.30, not 1."""
        from scipy.stats import chi2, f, norm
        if self._last_result is None:
            raise ValueError("No fit result available")
        levels = {}
        for sigma in sigmas:
            prob = 2.0 * norm.cdf(sigma) - 1.0
            if self._ci_scaled():
                nfree = self._last_result.nfree
                levels[f"{sigma}\u03c3"] = float(
                    self._last_result.chisqr * nfix * f.ppf(prob, nfix, nfree) / nfree)
            else:
                levels[f"{sigma}\u03c3"] = float(chi2.ppf(prob, nfix))
        return levels

    def compute_ci_contour(self, x_name: str, y_name: str, nx: int = 10, ny: int = 10):
        """delta-chi2 surface over two parameters, the others re-optimized
        at each grid point. Returns (x_values, y_values, delta_chi2_grid,
        levels) where levels are the joint 1/2/3-sigma thresholds from
        joint_ci_thresholds, so a caller can draw labeled contours."""
        from lmfit import conf_interval2d
        mini = self.ci_minimizer()
        xs, ys, chi2_grid = conf_interval2d(mini, mini, x_name, y_name, nx=nx, ny=ny,
                                            chi2_out=True)
        best = min(float(mini.chisqr), float(np.nanmin(chi2_grid)))
        return xs, ys, np.asarray(chi2_grid) - best, self.joint_ci_thresholds(2)

    def compute_confidence_intervals(self, sigmas=None) -> str:
        """Compute confidence intervals and return a CI report string."""
        from lmfit import conf_interval, ci_report
        mini = self.ci_minimizer()
        ci = conf_interval(mini, mini, sigmas=sigmas, prob_func=self._ci_prob_func())
        note = ("Intervals from an F-test on chi-square ratios, i.e. scaled by "
                "reduced chi-square (Scale covariance on)."
                if self._ci_scaled() else
                "Intervals from delta chi-square, unscaled (Scale covariance off).")
        return note + "\n\n" + ci_report(ci)

    def compute_ci_profiles(self, sigmas=None) -> dict[str, list[tuple[float, float]]]:
        """Compute profile likelihood traces for each parameter.

        Returns {param_name: [(param_value, chi_squared), ...]}, where the
        chi-square is the profile's own re-minimized chi-square, recovered by
        inverting the CI probability exactly.
        """
        from lmfit import conf_interval

        mini = self.ci_minimizer()
        ci_result, trace = conf_interval(mini, mini, sigmas=sigmas, trace=True,
                                         prob_func=self._ci_prob_func())
        best_chi2 = mini.chisqr
        nfree = mini.nfree

        profiles = {}
        for pname, tdata in trace.items():
            if not isinstance(tdata, dict) or pname not in tdata or "prob" not in tdata:
                continue
            param_vals = np.asarray(tdata[pname])
            probs = np.asarray(tdata["prob"])
            chi2_vals = np.where(probs > 0, self._prob_to_chi2(probs, best_chi2, nfree),
                                 best_chi2)
            # Sort by parameter value for clean plotting
            order = np.argsort(param_vals)
            profiles[pname] = list(zip(param_vals[order], chi2_vals[order]))
        return profiles

    def run_odr(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        xerr: np.ndarray | None = None,
        n_dense: int = N_DENSE,
        band_sigma: int = 1,
    ) -> FitResult:
        """Run orthogonal distance regression using the odrpack package.

        xerr is required: odrpack treats a missing x weight as an x-error of
        1 in data units, which lets the x-adjustments absorb the residuals
        and makes the fit and its errors meaningless. yerr is optional
        (unit y weights then), as for the other methods.
        """
        try:
            from odrpack import odr_fit
        except ImportError:
            raise ImportError(
                "ODR requires the 'odrpack' package. Install with: pip install odrpack"
            )

        if self._model is None or self._params is None:
            raise ValueError("No model defined")
        if xerr is None:
            raise ValueError(
                "Orthogonal distance regression needs x-errors; this series "
                "has none. Add an x-error column, or use a least-squares method."
            )

        if self._has_spline:
            self._rebuild_model_with_data(x)

        # Reset to the remembered start so repeated ODR fits are reproducible.
        self._reset_to_start_values()

        # Snapshot initial parameter values
        init_values = {name: par.value for name, par in self._params.items()}

        # Build parameter arrays for odrpack
        param_names = list(self._params.keys())
        vary_mask = [self._params[n].vary for n in param_names]
        beta0 = np.array([self._params[n].value for n in param_names])

        # Build bounds
        lower = np.array([self._params[n].min for n in param_names])
        upper = np.array([self._params[n].max for n in param_names])
        has_bounds = np.any(np.isfinite(lower)) or np.any(np.isfinite(upper))
        # odrpack has no notion of an open bound, so an infinite one is
        # emulated with the largest finite double.
        lower = np.where(np.isfinite(lower), lower, -ODR_UNBOUNDED)
        upper = np.where(np.isfinite(upper), upper, ODR_UNBOUNDED)

        # Fix non-varied parameters
        fix_beta = np.array([0 if v else 1 for v in vary_mask], dtype=float)

        # Wrap lmfit model to odrpack's f(x, beta) signature
        model = self._model

        def odr_func(x_arr, beta):
            kw = {name: beta[i] for i, name in enumerate(param_names)}
            return model.eval(x=x_arr, **kw)

        # Weights: odrpack uses weight = 1/variance
        weight_y = None
        if yerr is not None:
            safe_yerr = np.maximum(np.abs(yerr), MIN_ERROR)
            weight_y = 1.0 / (safe_yerr ** 2)

        weight_x = None
        if xerr is not None:
            safe_xerr = np.maximum(np.abs(xerr), MIN_ERROR)
            weight_x = 1.0 / (safe_xerr ** 2)

        odr_kwargs = dict(
            weight_x=weight_x,
            weight_y=weight_y,
            fix_beta=fix_beta,
        )
        if has_bounds:
            odr_kwargs["bounds"] = (lower, upper)

        result = odr_fit(odr_func, x, y, beta0, **odr_kwargs)

        if not result.success:
            raise RuntimeError(f"ODR failed: {result.stopreason}")

        # Update parameters with ODR results
        for i, name in enumerate(param_names):
            self._params[name].set(value=result.beta[i])
            self._params[name].stderr = result.sd_beta[i] if vary_mask[i] else None

        # Dense curve
        x_dense = np.linspace(x.min(), x.max(), n_dense)
        best_kw = {name: result.beta[i] for i, name in enumerate(param_names)}
        y_fit_dense = model.eval(x=x_dense, **best_kw)
        y_fit_data = model.eval(x=x, **best_kw)

        # Confidence band by linear error propagation through the full
        # parameter covariance: var(y) = J C J^T with J = df/dbeta on the
        # dense grid. odrpack's cov_beta is unscaled; the covariance that
        # matches sd_beta is cov_beta * res_var. Using only the diagonal
        # (sd_beta) would drop the parameter correlations, which for a line
        # fitted away from x = 0 are close to -1 and make the band several
        # times too wide.
        y_uncertainty = None
        varied = [i for i, v in enumerate(vary_mask) if v]
        if varied and np.any(result.sd_beta[varied] > 0):
            try:
                cov = np.asarray(result.cov_beta)[np.ix_(varied, varied)] * result.res_var
                jac = np.empty((n_dense, len(varied)))
                for col, i in enumerate(varied):
                    name = param_names[i]
                    h = max(abs(result.beta[i]) * ODR_JAC_REL_STEP, ODR_JAC_MIN_STEP)
                    kw_plus = dict(best_kw, **{name: result.beta[i] + h})
                    kw_minus = dict(best_kw, **{name: result.beta[i] - h})
                    jac[:, col] = (model.eval(x=x_dense, **kw_plus)
                                   - model.eval(x=x_dense, **kw_minus)) / (2 * h)
                var_y = np.einsum("ij,jk,ik->i", jac, cov, jac)
                y_uncertainty = band_sigma * np.sqrt(np.maximum(var_y, 0.0))
            except Exception:
                pass

        # Component curves
        component_curves = {}
        if len(self.components) > 1:
            try:
                if isinstance(model, CompositeModel):
                    for comp in model.components:
                        comp_kw = {n: best_kw[n] for n in comp.param_names if n in best_kw}
                        component_curves[comp.prefix] = comp.eval(x=x_dense, **comp_kw)
            except Exception:
                pass

        # Build params info
        params_info = {}
        for i, name in enumerate(param_names):
            params_info[name] = param_info(
                result.beta[i],
                result.sd_beta[i] if vary_mask[i] else None,
                self._params[name].min, self._params[name].max,
                vary_mask[i],
            )

        # GOF
        n_data = len(x)
        n_vary = sum(vary_mask)
        dof = n_data - n_vary
        chisqr = result.sum_square
        redchi = chisqr / dof if dof > 0 else float("inf")
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - np.sum((y - y_fit_data) ** 2) / ss_tot if ss_tot > 0 else 0

        gof = make_gof(chisqr=chisqr, redchi=redchi, rsquared=r_squared)

        # Report (run_odr raised above if odrpack reported failure)
        stop_reason = " / ".join(result.stopreason) if isinstance(
            result.stopreason, (list, tuple)) else str(result.stopreason)
        lines = convergence_lines(True, stop_reason, result.nfev, params_info) + [""]
        lines.append("Orthogonal Distance Regression (odrpack)")
        lines.append(f"  Stop reason: {stop_reason}")
        lines.append(f"  Function evals: {result.nfev}")
        lines.append(f"  Iterations: {result.niter}")
        lines.append(f"  Sum of squares: {chisqr:.6g}")
        lines.append(f"  Residual variance: {result.res_var:.6g}")
        lines.append("")
        for name, info in params_info.items():
            stderr_str = f" +/- {info['stderr']:.6g}" if info['stderr'] is not None else " (fixed)"
            lines.append(f"  {name}: {info['value']:.6g}{stderr_str}")
        report = "\n".join(lines)

        return FitResult(
            x_dense=x_dense,
            y_fit_dense=y_fit_dense,
            x_data=x,
            y_data=y,
            y_fit_data=y_fit_data,
            yerr_data=yerr,
            params=params_info,
            report=report,
            gof=gof,
            component_curves=component_curves,
            y_uncertainty=y_uncertainty,
            init_params=init_values,
            xerr_data=xerr,
            message=stop_reason,
            nfev=result.nfev,
        )

    def run_bootstrap(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        n_boot: int = 200,
        method: str = DEFAULT_FIT_METHOD,
        boot_type: str = "residual",
        weight_mode: str = DEFAULT_WEIGHT_MODE,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], int]:
        """Run bootstrap resampling and return parameter distributions.

        boot_type:
          - "residual": resample the residuals and add them to the fitted
            curve. For a weighted fit the residuals are resampled in
            *weighted* form (r_i w_i, of order one everywhere) and scaled
            back by each target point's 1/w_j: raw residuals from
            low-sigma points would otherwise land on high-sigma points and
            vice versa, distorting the distributions.
          - "case": resample (x, y, yerr) rows with replacement.

        Each resample is refitted with the last fit's objective (robust
        loss or reduce function) and scale_covar, so the distributions
        describe the fit the user ran. seed makes the resampling
        reproducible; None draws a fresh sequence.

        Returns ({param_name: array of successful resample values}, n_failed),
        where n_failed counts resamples whose fit raised or reported
        success=False (dropped from the distributions -- a high count means
        the reported confidence intervals rest on fewer samples than n_boot
        and may be unreliable).
        """
        if self._last_result is None or self._model is None:
            raise ValueError("Run a fit first")

        rng = np.random.default_rng(seed)
        last = self._last_result
        best_params = last.params
        y_fit = last.best_fit
        residuals = y - y_fit
        fit_weights = last.weights if last.weights is not None else None
        # The objective the fit used: a scipy loss travels in the stored
        # keywords, a reduce function in the result's reduce_fcn attribute
        # (lmfit's default there is reduce_chisquare, i.e. none).
        fit_kws = {k: v for k, v in (last.kws or {}).items() if k in ("loss", "f_scale")}
        reduce_fcn = getattr(last, "reduce_fcn", None)
        if reduce_fcn is not None and reduce_fcn is not reduce_chisquare:
            fit_kws["reduce_fcn"] = reduce_fcn
        scale_covar = bool(getattr(last, "scale_covar", True))

        param_names = [n for n, p in best_params.items() if p.vary]
        distributions: dict[str, list[float]] = {n: [] for n in param_names}
        n_failed = 0
        n_pts = len(x)

        for _ in range(n_boot):
            if boot_type == "case":
                idx = rng.integers(0, n_pts, size=n_pts)
                x_b, y_b = x[idx], y[idx]
                yerr_b = yerr[idx] if yerr is not None else None
            else:
                idx = rng.integers(0, n_pts, size=n_pts)
                if fit_weights is not None:
                    w = np.asarray(fit_weights, dtype=float)
                    y_b = y_fit + (residuals * w)[idx] / w
                else:
                    y_b = y_fit + residuals[idx]
                x_b = x
                yerr_b = yerr

            weights = self._compute_weights(y_b, yerr_b, weight_mode)
            try:
                params_copy = best_params.copy()
                result = self._model.fit(
                    y_b, params_copy, x=x_b, weights=weights,
                    method=method, nan_policy="omit", fit_kws=fit_kws,
                    scale_covar=scale_covar,
                )
                if not getattr(result, "success", True):
                    n_failed += 1
                    continue
                for n in param_names:
                    distributions[n].append(result.params[n].value)
            except Exception:
                n_failed += 1
                continue

        return {n: np.array(v) for n, v in distributions.items()}, n_failed

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the current model at given x values using fitted parameters."""
        if self._model is None or self._params is None:
            raise RuntimeError("No model or parameters available.")
        return self._model.eval(self._params, x=x)

    def get_covariance_matrix(self) -> tuple[list[str], np.ndarray] | None:
        """Extract the covariance matrix from the last fit result.

        Returns (param_names, cov_matrix) or None if unavailable.
        """
        if self._last_result is None:
            return None
        if self._last_result.covar is None:
            return None
        var_names = [n for n, p in self._last_result.params.items() if p.vary]
        return var_names, self._last_result.covar

    def get_correlations(self) -> dict[str, dict[str, float]]:
        """Extract parameter correlations from the last fit result."""
        if self._last_result is None:
            raise ValueError("No fit result available")
        correlations = {}
        for name, par in self._last_result.params.items():
            if par.correl is not None:
                correlations[name] = dict(par.correl)
        return correlations

    def serialize(self) -> dict:
        """Serialize component list for workspace persistence."""
        data = {
            "components": [
                {
                    "name": c.name,
                    "prefix": c.prefix,
                    "operator": c.operator,
                    "expression": c.expression,
                }
                for c in self.components
            ],
        }
        if self._param_hints:
            data["param_hints"] = self._param_hints
        if self._start_values:
            data["start_values"] = dict(self._start_values)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "FitManager":
        """Reconstruct a FitManager from serialized data."""
        fm = cls()
        for comp in data.get("components", []):
            fm.add_component(
                comp["name"],
                operator=comp.get("operator", "+"),
                expression=comp.get("expression", ""),
            )
        # Restore parameter hints
        for name, hints in data.get("param_hints", {}).items():
            fm.set_param_hint(name, **hints)
        # Restore the start values (after the hints, which also set starts).
        # Workspaces from before this field simply have none recorded.
        for name, value in data.get("start_values", {}).items():
            if fm.params is not None and name in fm.params:
                fm._start_values[name] = float(value)
        return fm
