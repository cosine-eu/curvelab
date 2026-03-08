"""Pure data preprocessing functions for fit data preparation."""

from __future__ import annotations

import numpy as np

from .session import SeriesRecord


def prepare_fit_data(
    rec: SeriesRecord,
    x_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    """Prepare data for fitting: mask, range-filter, clean NaN/inf, sort.

    Parameters
    ----------
    rec : SeriesRecord
        The data series with optional mask.
    x_range : tuple of (xmin, xmax), optional
        If provided, restrict data to this x range.

    Returns
    -------
    x, y, yerr, xerr : arrays
        Cleaned, sorted data arrays.
    warnings : list of str
        Warning messages (NaN removal count, duplicate x count).
    """
    x, y = rec.x.copy(), rec.y.copy()
    yerr = rec.yerr.copy() if rec.yerr is not None else None
    xerr = rec.xerr.copy() if rec.xerr is not None else None
    warnings: list[str] = []

    # 1. Apply point exclusion mask
    if rec.mask is not None:
        x, y = x[rec.mask], y[rec.mask]
        yerr = yerr[rec.mask] if yerr is not None else None
        xerr = xerr[rec.mask] if xerr is not None else None

    # 2. Apply x range
    if x_range is not None:
        xmin, xmax = x_range
        mask = (x >= xmin) & (x <= xmax)
        x, y = x[mask], y[mask]
        yerr = yerr[mask] if yerr is not None else None
        xerr = xerr[mask] if xerr is not None else None

    # 3. Filter NaN / inf
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        finite_mask &= np.isfinite(yerr)
    if xerr is not None:
        finite_mask &= np.isfinite(xerr)
    n_dropped = int((~finite_mask).sum())
    if n_dropped > 0:
        x, y = x[finite_mask], y[finite_mask]
        yerr = yerr[finite_mask] if yerr is not None else None
        xerr = xerr[finite_mask] if xerr is not None else None
        warnings.append(f"Removed {n_dropped} point(s) with NaN/inf values.")

    # 4. Sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]
    yerr = yerr[order] if yerr is not None else None
    xerr = xerr[order] if xerr is not None else None

    # 5. Check for duplicate x values
    if len(x) > 0:
        n_dup = len(x) - len(np.unique(x))
        if n_dup > 0:
            warnings.append(
                f"{n_dup} duplicate x-value(s) detected. "
                "This may cause issues with some models."
            )

    return x, y, yerr, xerr, warnings
