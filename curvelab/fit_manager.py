"""Curve fitting logic using lmfit."""

from dataclasses import dataclass

import numpy as np
from lmfit import CompositeModel, Model, Parameters
from lmfit.model import ModelResult
from lmfit.models import SplineModel

from .models import create_expression_model, create_model, create_spline_model
# Re-exports; canonical location is session.py
from .session import FitResult, MIN_ERROR


def _reduce_negentropy(r):
    """Neg-entropy reduce function for robust fitting."""
    return -np.sum(r * np.log(np.maximum(np.abs(r), MIN_ERROR)))


def _reduce_cauchylogpdf(r):
    """Cauchy log-pdf reduce function for outlier-tolerant fitting."""
    return np.sum(np.log1p(r * r))


REDUCE_FUNCTIONS = {
    "Chi-square (default)": None,
    "Neg. entropy": _reduce_negentropy,
    "Cauchy log-pdf": _reduce_cauchylogpdf,
}

WEIGHT_MODES = [
    "1/yerr (default)",
    "1/yerr\u00b2",
    "1/y",
    "No weights",
    "yerr as weights",
    "Effective variance",
]


@dataclass
class FitComponent:
    """A single model component in a composite fit."""

    name: str  # Registry name, e.g. "Gaussian"
    prefix: str  # e.g. "gauss1_"
    operator: str = "+"  # "+" (sum) or "*" (multiply); ignored for first component
    expression: str = ""  # Only used when name == "Expression"


class FitManager:
    """Builds composite lmfit models, auto-guesses parameters, and runs fits."""

    def __init__(self):
        self.components: list[FitComponent] = []
        self._model: Model | None = None
        self._params: Parameters | None = None
        self._last_result: ModelResult | None = None
        self._param_hints: dict[str, dict] = {}
        self._name_counters: dict[str, int] = {}

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
                # Retroactively add prefix to first component
                old = self.components[0]
                old.prefix = f"{old.name.lower()}1_"
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
        self._drop_param_hints(removed.prefix)
        if len(self.components) == 1:
            # Single component: no prefix for cleaner parameter names
            old_prefix = self.components[0].prefix
            self.components[0].prefix = ""
            self._reprefix_param_hints(old_prefix, "")
        self._rebuild_model()

    def _drop_param_hints(self, prefix: str):
        """Forget hints belonging to a component prefix."""
        for name in [n for n in self._param_hints if n.startswith(prefix)]:
            del self._param_hints[name]

    def _reprefix_param_hints(self, old_prefix: str, new_prefix: str):
        """Re-key hints from one component prefix to another."""
        if old_prefix == new_prefix:
            return
        for name in [n for n in self._param_hints if n.startswith(old_prefix)]:
            hints = self._param_hints.pop(name)
            self._param_hints[new_prefix + name[len(old_prefix):]] = hints

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
                    if pname in self._params:
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
        for name, hints in self._param_hints.items():
            for old_prefix, new_prefix in prefix_pairs:
                if name.startswith(old_prefix):
                    target.set_param_hint(new_prefix + name[len(old_prefix):], **hints)
                    break

    def set_param_hint(self, name: str, **kwargs):
        """Store a parameter hint that survives model rebuilds."""
        self._param_hints[name] = {**self._param_hints.get(name, {}), **kwargs}
        if self._model is not None:
            self._model.set_param_hint(name, **kwargs)
        if self._params is not None and name in self._params:
            self._params[name].set(**kwargs)

    def set_param(self, name: str, **kwargs):
        """Set parameter attributes (value, min, max, vary)."""
        if self._params is not None and name in self._params:
            self._params[name].set(**kwargs)

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
            name: {
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
                "expr": par.expr or "",
            }
            for name, par in params.items()
        }

    @staticmethod
    def _compute_weights(y, yerr, weight_mode):
        """Compute weights array from y, yerr, and the selected weight mode."""
        if weight_mode == "No weights":
            return None
        if weight_mode == "1/yerr\u00b2" and yerr is not None:
            safe_yerr = np.maximum(np.abs(yerr), MIN_ERROR)
            return 1.0 / (safe_yerr * safe_yerr)
        if weight_mode == "1/y":
            return 1.0 / np.maximum(np.abs(y), MIN_ERROR)
        if weight_mode == "yerr as weights" and yerr is not None:
            return yerr
        # Default: "1/yerr (default)"
        if yerr is not None:
            return 1.0 / np.maximum(np.abs(yerr), MIN_ERROR)
        return None

    def run_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        xerr: np.ndarray | None = None,
        n_dense: int = 500,
        method: str = "least_squares",
        iter_cb=None,
        fit_kws: dict | None = None,
        reduce_fcn=None,
        weight_mode: str = "1/yerr (default)",
        max_nfev: int | None = None,
        band_sigma: int = 1,
        scale_covar: bool = True,
    ) -> FitResult:
        """Run the fit and return results."""
        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        # Rebuild with real x-data if model contains a Spline component
        if self._has_spline:
            self._rebuild_model_with_data(x)

        # Snapshot initial parameter values before fitting
        init_values = {name: par.value for name, par in self._params.items()}

        weights = self._compute_weights(y, yerr, weight_mode)

        # Effective variance: w = 1/sqrt(yerr² + (df/dx)² · xerr²)
        if weight_mode == "Effective variance":
            if xerr is not None and yerr is not None:
                h = np.maximum(np.abs(x) * 1e-8, 1e-10)
                y_plus = self._model.eval(self._params, x=x + h)
                y_minus = self._model.eval(self._params, x=x - h)
                dfdx = (y_plus - y_minus) / (2 * h)
                denom = np.maximum(np.sqrt(yerr**2 + (dfdx * xerr) ** 2), MIN_ERROR)
                weights = 1.0 / denom
            elif yerr is not None:
                weights = 1.0 / np.maximum(np.abs(yerr), MIN_ERROR)
            else:
                weights = None

        kws = dict(fit_kws or {})
        if reduce_fcn is not None:
            kws["reduce_fcn"] = reduce_fcn

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

        # Update internal params with fitted values
        self._params = self._last_result.params

        gof = {
            "chi-squared": self._last_result.chisqr,
            "reduced chi-squared": self._last_result.redchi,
            "R-squared": self._last_result.rsquared,
            "AIC": self._last_result.aic,
            "BIC": self._last_result.bic,
        }

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

        # Capture emcee flatchain
        flatchain = None
        if hasattr(self._last_result, "flatchain") and self._last_result.flatchain is not None:
            flatchain = self._last_result.flatchain

        return FitResult(
            x_dense=x_dense,
            y_fit_dense=y_fit_dense,
            x_data=x,
            y_data=y,
            y_fit_data=self._last_result.best_fit,
            yerr_data=yerr,
            params=params_info,
            report=self._last_result.fit_report(),
            gof=gof,
            component_curves=component_curves,
            y_uncertainty=y_uncertainty,
            candidates=candidates,
            flatchain=flatchain,
            init_params=init_values,
        )

    def refit_from_result(self, result: FitResult) -> None:
        """Re-run the fit using stored result data to reconstruct _last_result.

        This is used after loading a workspace to make analysis tools
        (CIs, contours, profiles, etc.) available without manual re-fitting.
        """
        if self._model is None or self._params is None:
            return
        x, y = result.x_data, result.y_data
        yerr = result.yerr_data
        weights = self._compute_weights(y, yerr, "1/yerr (default)")
        if self._has_spline:
            self._rebuild_model_with_data(x)
        self._last_result = self._model.fit(
            y, self._params, x=x, weights=weights,
            method="least_squares", nan_policy="omit",
        )
        self._params = self._last_result.params

    def run_global_fit(
        self,
        datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]],
        shared_params: set[str],
        method: str = "least_squares",
        max_nfev: int | None = None,
        weight_mode: str = "1/yerr (default)",
    ) -> list[FitResult]:
        """Run a global fit across multiple datasets with shared parameters.

        Each dataset is a tuple of (x, y, yerr, xerr).
        Parameters that are in *shared_params* get one entry in the combined
        Parameters object.  Non-shared params get per-dataset prefixed entries
        (s0_, s1_, ...).  Returns one FitResult per dataset.
        """
        from lmfit import Parameters, minimize

        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        n = len(datasets)
        base_params = self._params

        # Build combined Parameters
        combined = Parameters()
        base_names = list(base_params.keys())

        for name in base_names:
            bp = base_params[name]
            if name in shared_params:
                combined.add(name, value=bp.value, min=bp.min, max=bp.max,
                             vary=bp.vary, expr=bp.expr or None)
            else:
                for i in range(n):
                    pname = f"s{i}_{name}"
                    combined.add(pname, value=bp.value, min=bp.min, max=bp.max,
                                 vary=bp.vary, expr=bp.expr or None)

        # Pre-compute constant weights (everything except Effective variance
        # with xerr, which depends on the model derivative per iteration)
        use_effective_variance = weight_mode == "Effective variance"
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
                    h = np.maximum(np.abs(x) * 1e-8, 1e-10)
                    y_plus = self._model.eval(x=x + h, **kw)
                    y_minus = self._model.eval(x=x - h, **kw)
                    dfdx = (y_plus - y_minus) / (2 * h)
                    denom = np.maximum(np.sqrt(yerr**2 + (dfdx * xerr) ** 2), MIN_ERROR)
                    weights = 1.0 / denom
                if weights is not None:
                    resid = resid * weights
                all_resid.append(resid)
            return np.concatenate(all_resid)

        kws = {}
        if max_nfev is not None:
            kws["max_nfev"] = max_nfev

        mini_result = minimize(objective, combined, method=method,
                               nan_policy="omit", **kws)

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
                params_info[name] = {
                    "value": p.value,
                    "stderr": p.stderr,
                    "min": p.min,
                    "max": p.max,
                    "vary": p.vary,
                    "expr": p.expr or "",
                }
                init_values[name] = bp.value

            # Evaluate model for this dataset
            eval_kw = {name: params_info[name]["value"] for name in base_names}
            x_dense = np.linspace(x.min(), x.max(), 500)
            y_fit_data = self._model.eval(x=x, **eval_kw)
            y_fit_dense = self._model.eval(x=x_dense, **eval_kw)

            gof = {
                "chi-squared": mini_result.chisqr,
                "reduced chi-squared": mini_result.redchi,
                "R-squared": 1 - np.sum((y - y_fit_data) ** 2) / np.sum((y - y.mean()) ** 2),
                "AIC": mini_result.aic,
                "BIC": mini_result.bic,
            }

            # Generate a report string
            lines = [f"Global Fit — Dataset {i + 1}/{n}"]
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
            ))

        return results

    def compute_confidence_intervals(self, sigmas=None) -> str:
        """Compute confidence intervals and return a CI report string."""
        if self._last_result is None:
            raise ValueError("No fit result available")
        from lmfit import conf_interval, ci_report
        ci = conf_interval(self._last_result, self._last_result, sigmas=sigmas)
        return ci_report(ci)

    def compute_ci_profiles(self, sigmas=None) -> dict[str, list[tuple[float, float]]]:
        """Compute profile likelihood traces for each parameter.

        Returns {param_name: [(param_value, chi_squared), ...]}.
        """
        if self._last_result is None:
            raise ValueError("No fit result available")
        from lmfit import conf_interval
        from scipy.stats import chi2

        ci_result, trace = conf_interval(
            self._last_result, self._last_result,
            sigmas=sigmas, trace=True,
        )
        best_chi2 = self._last_result.chisqr

        profiles = {}
        for pname, tdata in trace.items():
            if not isinstance(tdata, dict) or pname not in tdata or "prob" not in tdata:
                continue
            param_vals = np.asarray(tdata[pname])
            probs = np.asarray(tdata["prob"])
            # Convert probability to delta-chi2, then to absolute chi2
            # prob=0 -> delta=0 (best fit), prob=0.68 -> delta~1 (1-sigma)
            delta_chi2 = np.where(probs > 0, chi2.ppf(probs, 1), 0.0)
            chi2_vals = best_chi2 + delta_chi2
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
        n_dense: int = 500,
        band_sigma: int = 1,
    ) -> FitResult:
        """Run orthogonal distance regression using the odrpack package.

        Requires both xerr and yerr for proper ODR weighting.
        """
        try:
            from odrpack import odr_fit
        except ImportError:
            raise ImportError(
                "ODR requires the 'odrpack' package. Install with: pip install odrpack"
            )

        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        if self._has_spline:
            self._rebuild_model_with_data(x)

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
        # Replace -inf/inf with None-compatible values
        lower = np.where(np.isfinite(lower), lower, -1e308)
        upper = np.where(np.isfinite(upper), upper, 1e308)

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

        # Confidence band via error propagation (numerical)
        y_uncertainty = None
        if any(result.sd_beta[i] > 0 for i in range(len(param_names)) if vary_mask[i]):
            try:
                # Simple numerical uncertainty propagation
                var_y = np.zeros(n_dense)
                for i, name in enumerate(param_names):
                    if not vary_mask[i] or result.sd_beta[i] <= 0:
                        continue
                    h = max(abs(result.beta[i]) * 1e-6, 1e-10)
                    kw_plus = dict(best_kw)
                    kw_plus[name] = result.beta[i] + h
                    kw_minus = dict(best_kw)
                    kw_minus[name] = result.beta[i] - h
                    dy = (model.eval(x=x_dense, **kw_plus) -
                          model.eval(x=x_dense, **kw_minus)) / (2 * h)
                    var_y += (dy * result.sd_beta[i]) ** 2
                y_uncertainty = band_sigma * np.sqrt(var_y)
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
            params_info[name] = {
                "value": result.beta[i],
                "stderr": result.sd_beta[i] if vary_mask[i] else None,
                "min": self._params[name].min,
                "max": self._params[name].max,
                "vary": vary_mask[i],
                "expr": "",
            }

        # GOF
        n_data = len(x)
        n_vary = sum(vary_mask)
        dof = n_data - n_vary
        chisqr = result.sum_square
        redchi = chisqr / dof if dof > 0 else float("inf")
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - np.sum((y - y_fit_data) ** 2) / ss_tot if ss_tot > 0 else 0

        gof = {
            "chi-squared": chisqr,
            "reduced chi-squared": redchi,
            "R-squared": r_squared,
        }

        # Report
        lines = ["Orthogonal Distance Regression (odrpack)"]
        lines.append(f"  Stop reason: {result.stopreason}")
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
        )

    def run_bootstrap(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        n_boot: int = 200,
        method: str = "least_squares",
        boot_type: str = "residual",
        weight_mode: str = "1/yerr (default)",
    ) -> tuple[dict[str, np.ndarray], int]:
        """Run bootstrap resampling and return parameter distributions.

        boot_type:
          - "residual": Resample residuals and add to fitted values
          - "case": Resample (x, y, yerr) rows with replacement

        Returns ({param_name: array of successful resample values}, n_failed),
        where n_failed is the number of resamples whose fit didn't converge
        or raised (dropped from the distributions -- a high count means the
        reported confidence intervals are based on fewer samples than
        n_boot and may be unreliable).
        """
        if self._last_result is None or self._model is None:
            raise ValueError("Run a fit first")

        best_params = self._last_result.params
        y_fit = self._last_result.best_fit
        residuals = y - y_fit

        param_names = [n for n, p in best_params.items() if p.vary]
        distributions: dict[str, list[float]] = {n: [] for n in param_names}
        n_failed = 0

        for _ in range(n_boot):
            if boot_type == "case":
                idx = np.random.randint(0, len(x), size=len(x))
                x_b, y_b = x[idx], y[idx]
                yerr_b = yerr[idx] if yerr is not None else None
            else:
                idx = np.random.randint(0, len(residuals), size=len(residuals))
                x_b, y_b = x, y_fit + residuals[idx]
                yerr_b = yerr

            weights = self._compute_weights(y_b, yerr_b, weight_mode)
            try:
                params_copy = best_params.copy()
                result = self._model.fit(
                    y_b, params_copy, x=x_b, weights=weights,
                    method=method, nan_policy="omit",
                )
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
        return fm
