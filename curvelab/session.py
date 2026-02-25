"""Per-series fit session dataclasses (no GUI imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fit_manager import FitManager, FitResult

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
    fit_manager: FitManager = field(default_factory=FitManager)
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
    fit_sessions: dict[str, FitSession] = field(default_factory=dict)
    active_session_name: str | None = None

    @property
    def active_session(self) -> FitSession | None:
        if self.active_session_name is None:
            return None
        return self.fit_sessions.get(self.active_session_name)
