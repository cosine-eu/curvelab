"""Per-series fit session dataclasses (no GUI imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Floor for magnitudes used as divisors or log arguments, so zero-valued
# errors/residuals can't produce inf weights or log-of-zero.
MIN_ERROR = 1e-12

# |rho| above this between two varying parameters is flagged in the report
# and the results panel.
STRONG_CORRELATION = 0.9


def make_series_id(dataset: str, x_col: str, y_col: str) -> str:
    """Canonical id for a plotted series: 'dataset::x_col::y_col'."""
    return f"{dataset}::{x_col}::{y_col}"


def make_session_key(series_id: str, session_name: str) -> str:
    """Canonical key for a fit session: 'series_id::session_name'."""
    return f"{series_id}::{session_name}"


@dataclass
class FitResult:
    """Results from a fit."""

    x_dense: np.ndarray
    y_fit_dense: np.ndarray
    x_data: np.ndarray
    y_data: np.ndarray
    y_fit_data: np.ndarray
    yerr_data: np.ndarray | None
    params: dict[str, dict]  # name -> {value, stderr, min, max, vary}
    report: str
    gof: dict[str, float] = field(default_factory=dict)
    component_curves: dict[str, np.ndarray] = field(default_factory=dict)
    y_uncertainty: np.ndarray | None = None  # 1-sigma band on dense grid
    candidates: list[dict] | None = None  # brute-force candidates
    flatchain: object | None = None  # emcee DataFrame
    mcmc: dict | None = None  # emcee settings and sampling diagnostics
    init_params: dict[str, float] | None = None  # {name: initial_value_before_fit}
    errorbars: bool = True  # False when the fit couldn't estimate uncertainties
    xerr_data: np.ndarray | None = None
    # The weights w_i the fit actually multiplied the residuals by (None for
    # an unweighted fit). This is what "weighted residuals" must divide by;
    # yerr alone is wrong for the 1/y and effective-variance modes.
    fit_weights: np.ndarray | None = None
    # How the minimizer ended. success is False when it stopped without
    # converging (max_nfev reached, abort, tolerance failure); message is the
    # minimizer's own description; nfev the number of function evaluations.
    success: bool = True
    message: str = ""
    nfev: int | None = None
    # Points dropped by nan_policy="omit" because the model was not finite
    # there; chi-square and AIC/BIC refer to the points that remained.
    n_omitted: int = 0
    # Parameter pairs with |correlation| above STRONG_CORRELATION, as
    # [name_a, name_b, rho]: their individual values are poorly determined,
    # only a combination is.
    strong_correlations: list = field(default_factory=list)
    # How the fit was run (method, weight_mode, objective, scale_covar), in a
    # JSON-safe form, so a workspace reload can reproduce it rather than
    # refit with defaults. None for results from before this was recorded.
    fit_options: dict | None = None

    def comparison_key(self) -> tuple:
        """What has to agree between two results for AIC/BIC or an F-test
        to compare them: the same points (count and x range), the same
        weights, and the same objective. Different data or a different
        objective make chi-square, and everything derived from it,
        incommensurable."""
        opts = self.fit_options or {}
        x = np.asarray(self.x_data, dtype=float)
        xr = (round(float(x.min()), 12), round(float(x.max()), 12)) if x.size else (None, None)
        objective = opts.get("objective") or {}
        return (
            int(x.size), xr,
            opts.get("weight_mode"),
            self.yerr_data is not None,
            tuple(sorted((k, str(v)) for k, v in objective.items())),
        )

    def residuals(self) -> np.ndarray:
        """Raw residuals (data minus fit) at the data points."""
        return self.y_data - self.y_fit_data

    def is_weighted(self) -> bool:
        """Whether the fit used weights, i.e. whether weighted_residuals()
        differs from residuals()."""
        return self.fit_weights is not None or (
            self.fit_options is None and self.yerr_data is not None)

    def weighted_residuals(self) -> np.ndarray:
        """Residuals as the fit saw them: multiplied by the fit weights
        (r_i / sigma_i for the default 1/yerr mode). Raw residuals for an
        unweighted fit. Results saved before the weights were recorded fall
        back to dividing by yerr, floored at MIN_ERROR."""
        if self.fit_weights is not None:
            return self.residuals() * np.asarray(self.fit_weights, dtype=float)
        if self.fit_options is not None or self.yerr_data is None:
            return self.residuals()
        return self.residuals() / np.maximum(np.abs(self.yerr_data), MIN_ERROR)


@runtime_checkable
class FitManagerProtocol(Protocol):
    @property
    def params(self) -> Any: ...
    def set_param(self, name: str, **kwargs) -> None: ...
    def clear_components(self) -> None: ...


# Lazy import to avoid circular dependency at module level
def _fit_manager_factory():
    from .fit_manager import FitManager
    return FitManager()

FIT_COLORS = [
    "xkcd:red", "xkcd:bright blue", "xkcd:green", "xkcd:purple",
    "xkcd:orange", "xkcd:magenta", "xkcd:teal", "xkcd:gold",
    "xkcd:coral", "xkcd:navy blue", "xkcd:lime green", "xkcd:lavender",
    "xkcd:hot pink", "xkcd:olive", "xkcd:sky blue", "xkcd:salmon",
]

COMPONENT_COLORS = [
    "xkcd:dusty rose", "xkcd:sea green", "xkcd:periwinkle", "xkcd:mustard",
    "xkcd:light brown", "xkcd:slate blue", "xkcd:burnt orange", "xkcd:sage",
    "xkcd:mauve", "xkcd:dark cyan", "xkcd:peach", "xkcd:steel blue",
]


@dataclass
class ParamEdit:
    """Record of a single parameter edit for undo/redo."""

    param_name: str
    field: str  # "value", "min", "max", "vary"
    old_value: Any
    new_value: Any


@dataclass
class FitSession:
    """One named fit attempt attached to a series."""

    name: str
    fit_manager: FitManagerProtocol = field(default_factory=_fit_manager_factory)
    result: FitResult | None = None
    color: str = ""
    visible: bool = True
    undo_stack: list[ParamEdit] = field(default_factory=list)
    redo_stack: list[ParamEdit] = field(default_factory=list)


@dataclass
class SeriesRecord:
    """Cached data and fit sessions for a plotted series."""

    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None = None
    xerr: np.ndarray | None = None
    style: dict = field(default_factory=dict)
    dataset_name: str = ""
    visible: bool = True
    mask: np.ndarray | None = None  # True = included, None = all included
    fit_sessions: dict[str, FitSession] = field(default_factory=dict)
    active_session_name: str | None = None

    @property
    def active_session(self) -> FitSession | None:
        if self.active_session_name is None:
            return None
        return self.fit_sessions.get(self.active_session_name)

    def ensure_session(self, name: str) -> FitSession:
        """Return the named session, creating it (with the next fit color)
        if absent. A newly created session becomes active only when no
        session was active before."""
        if name not in self.fit_sessions:
            color = FIT_COLORS[len(self.fit_sessions) % len(FIT_COLORS)]
            self.fit_sessions[name] = FitSession(name=name, color=color)
            if self.active_session_name is None:
                self.active_session_name = name
        return self.fit_sessions[name]
