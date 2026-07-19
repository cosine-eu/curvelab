"""Per-series fit session dataclasses (no GUI imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


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
    init_params: dict[str, float] | None = None  # {name: initial_value_before_fit}


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
