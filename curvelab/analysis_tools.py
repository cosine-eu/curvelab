"""Pure numeric analysis helpers (no GUI imports).

These back the data-tool and analysis dialogs — smoothing, outlier
rejection, peak detection, F-tests, x-spec parsing, and residual
diagnostics — and are unit-testable without a display.
"""

import numpy as np

SMOOTH_METHODS = ["Savitzky-Golay", "Moving Average", "Median Filter", "Gaussian Filter"]


def smooth_data(y: np.ndarray, method: str, window: int = 11, order: int = 3,
                sigma: float = 5) -> np.ndarray:
    """Smooth y with the named method.

    window is forced odd and clipped to the data length; order applies to
    Savitzky-Golay, sigma to the Gaussian filter. Arrays shorter than 3
    points are returned unchanged.
    """
    from scipy.signal import savgol_filter, medfilt
    from scipy.ndimage import uniform_filter1d, gaussian_filter1d

    if len(y) < 3:
        return y  # Too few points for any windowed smoothing method

    if window % 2 == 0:
        window += 1
    # len(y) >= 3 here, so this max is always >= 3 and the floor below
    # never has to push window past the data length.
    window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    if window < 3:
        window = 3

    if method == "Savitzky-Golay":
        return savgol_filter(y, window, min(order, window - 1))
    elif method == "Moving Average":
        return uniform_filter1d(y, size=window)
    elif method == "Median Filter":
        return medfilt(y, kernel_size=window)
    elif method == "Gaussian Filter":
        return gaussian_filter1d(y, sigma=max(1, sigma))
    return y


def detect_outliers(x: np.ndarray, y: np.ndarray, y_smooth: np.ndarray,
                    smooth_func, threshold: float = 3.0, n_iter: int = 1) -> np.ndarray:
    """Iterative MAD-based outlier rejection against a smooth baseline.

    smooth_func(y) -> smoothed y is used to re-estimate the baseline from
    inliers between iterations. Returns a boolean inlier mask (True = keep).
    """
    inlier = np.ones(len(y), dtype=bool)

    for _ in range(n_iter):
        residuals = y - y_smooth
        med = np.median(residuals[inlier]) if inlier.any() else 0.0
        mad = np.median(np.abs(residuals[inlier] - med)) if inlier.any() else 1.0
        sigma_est = 1.4826 * mad if mad > 0 else 1.0
        inlier = np.abs(residuals - med) < threshold * sigma_est
        # Re-smooth on inliers for next iteration
        if not inlier.all() and inlier.sum() >= 3:
            from scipy.interpolate import interp1d
            f = interp1d(x[inlier], y[inlier], kind="linear",
                         fill_value="extrapolate")
            y_smooth = smooth_func(f(x))
    return inlier


def find_peaks_with_widths(x: np.ndarray, y: np.ndarray,
                           prominence: float | None = None,
                           distance: int | None = None) -> list[tuple[float, float, float]]:
    """Detect peaks and estimate their widths in x units.

    Without an explicit prominence, 10% of the y range is used. Returns
    a list of (center_x, height_y, width_x) tuples.
    """
    from scipy.signal import find_peaks, peak_widths

    kwargs = {}
    if prominence is not None:
        kwargs["prominence"] = prominence
    else:
        # Auto-prominence: 10% of data range
        yrange = np.ptp(y)
        if yrange > 0:
            kwargs["prominence"] = yrange * 0.1
    if distance is not None:
        kwargs["distance"] = distance

    indices, _ = find_peaks(y, **kwargs)
    if len(indices) > 0:
        widths_pts = peak_widths(y, indices, rel_height=0.5)[0]
        dx = np.median(np.diff(x)) if len(x) > 1 else 1.0
        widths_x = widths_pts * abs(dx)
    else:
        widths_x = []

    return [
        (float(x[idx]), float(y[idx]),
         float(widths_x[i]) if i < len(widths_x) else 0.0)
        for i, idx in enumerate(indices)
    ]


def f_test(chi_reduced: float, n_params_reduced: int,
           chi_full: float, n_params_full: int,
           n_data: int) -> tuple[float, float, int, int]:
    """F-test for nested models: does the full model significantly improve
    on the reduced one?

    Returns (f_stat, p_value, df1, df2). Raises ValueError when the models
    are not comparable (reduced not smaller, full not better, or no
    residual degrees of freedom).
    """
    import scipy.stats as stats

    if n_params_reduced >= n_params_full:
        raise ValueError("Reduced model must have fewer parameters than the full model.")
    if chi_full >= chi_reduced:
        raise ValueError("Full model has equal or worse chi-squared than the reduced model.")

    df1 = n_params_full - n_params_reduced  # extra parameters
    df2 = n_data - n_params_full            # residual DOF of full model
    if df2 <= 0:
        raise ValueError("Not enough data points for this comparison.")

    f_stat = ((chi_reduced - chi_full) / df1) / (chi_full / df2)
    p_value = stats.f.sf(f_stat, df1, df2)
    return f_stat, p_value, df1, df2


def parse_x_spec(text: str) -> np.ndarray | None:
    """Parse an x-values spec: 'start:stop:npoints' or comma/space-separated
    numbers. Returns None for empty input."""
    text = text.strip()
    if not text:
        return None
    # Range syntax: start:stop:npoints
    if text.count(":") == 2:
        parts = text.split(":")
        return np.linspace(float(parts[0]), float(parts[1]), int(parts[2]))
    # Comma or space separated
    text = text.replace(",", " ")
    return np.array([float(v) for v in text.split()])


def compute_diagnostic_stats(residuals) -> list[str]:
    """Residual-randomness statistics (Durbin-Watson, runs test) as display lines."""
    import scipy.stats as stats

    lines = []

    # Durbin-Watson statistic
    diff_resid = np.diff(residuals)
    ss_resid = np.sum(residuals ** 2)
    if ss_resid > 0:
        dw = np.sum(diff_resid ** 2) / ss_resid
        if dw < 1.5:
            dw_interp = "positive autocorrelation (model may be systematically wrong)"
        elif dw > 2.5:
            dw_interp = "negative autocorrelation"
        else:
            dw_interp = "no significant autocorrelation"
        lines.append(f"Durbin-Watson: {dw:.4f} — {dw_interp}")

    # Runs test (sign changes in residuals)
    signs = np.sign(residuals)
    signs = signs[signs != 0]  # drop zeros
    if len(signs) >= 10:
        n_pos = int(np.sum(signs > 0))
        n_neg = int(np.sum(signs < 0))
        n_total = n_pos + n_neg
        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
        # Expected runs and variance under H0 (random sequence)
        expected = 1 + 2 * n_pos * n_neg / n_total
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n_total)) / (
            n_total ** 2 * (n_total - 1)
        )
        if var_runs > 0:
            z_runs = (runs - expected) / np.sqrt(var_runs)
            p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            if p_runs < 0.05:
                runs_interp = "non-random pattern (systematic misfit)"
            else:
                runs_interp = "consistent with random residuals"
            lines.append(
                f"Runs test: {runs} runs (expected {expected:.1f}), "
                f"z = {z_runs:.3f}, p = {p_runs:.4f} — {runs_interp}"
            )

    return lines
