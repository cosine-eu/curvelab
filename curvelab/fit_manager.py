"""Curve fitting logic using lmfit."""

from dataclasses import dataclass

import numpy as np
from lmfit import CompositeModel, Model, Parameters
from lmfit.model import ModelResult
from lmfit.models import SplineModel

from .models import create_expression_model, create_model, create_spline_model
from .session import FitResult  # re-export; canonical location is session.py


def _reduce_negentropy(r):
    """Neg-entropy reduce function for robust fitting."""
    return -np.sum(r * np.log(np.maximum(np.abs(r), 1e-12)))


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

    def add_component(
        self, model_name: str, operator: str = "+", expression: str = ""
    ) -> FitComponent:
        """Add a model component. operator is '+' or '*'; ignored for first component."""
        # Count existing components with this base name to generate prefix
        count = sum(1 for c in self.components if c.name == model_name) + 1
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
        """Remove a component by index and rebuild."""
        if 0 <= index < len(self.components):
            self.components.pop(index)
            # Reset prefixes
            if len(self.components) == 1:
                self.components[0].prefix = ""
            self._rebuild_model()

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

        Only updates value/min/max from guess; preserves user-edited vary,
        expr, and param_hints.
        """
        if self._model is None:
            raise ValueError("No model defined")

        if self._has_spline:
            self._rebuild_model_with_data(x)

        for comp in self.components:
            m = self._build_component_model(comp, x_data=x)
            try:
                guessed = m.guess(y, x=x)
                for pname, par in guessed.items():
                    if pname in self._params:
                        self._params[pname].set(
                            value=par.value, min=par.min, max=par.max
                        )
            except NotImplementedError:
                pass  # Model doesn't implement guess()

        return self._params

    def clone_components_to(self, target: "FitManager"):
        """Copy this manager's component list into target, rebuilding its model."""
        target.clear_components()
        for comp in self.components:
            target.add_component(comp.name, operator=comp.operator, expression=comp.expression)

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

    @staticmethod
    def _compute_weights(y, yerr, weight_mode):
        """Compute weights array from y, yerr, and the selected weight mode."""
        if weight_mode == "No weights":
            return None
        if weight_mode == "1/yerr\u00b2" and yerr is not None:
            safe_yerr = np.maximum(np.abs(yerr), 1e-12)
            return 1.0 / (safe_yerr * safe_yerr)
        if weight_mode == "1/y":
            return 1.0 / np.maximum(np.abs(y), 1e-12)
        if weight_mode == "yerr as weights" and yerr is not None:
            return yerr
        # Default: "1/yerr (default)"
        if yerr is not None:
            return 1.0 / np.maximum(np.abs(yerr), 1e-12)
        return None

    def run_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        xerr: np.ndarray | None = None,
        n_dense: int = 500,
        method: str = "leastsq",
        iter_cb=None,
        fit_kws: dict | None = None,
        reduce_fcn=None,
        weight_mode: str = "1/yerr (default)",
        max_nfev: int | None = None,
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
                weights = 1.0 / np.sqrt(yerr**2 + (dfdx * xerr) ** 2)
            elif yerr is not None:
                weights = 1.0 / yerr
            else:
                weights = None

        kws = dict(fit_kws or {})
        if reduce_fcn is not None:
            kws["reduce_fcn"] = reduce_fcn

        fit_kwargs = dict(
            method=method, nan_policy="omit",
            iter_cb=iter_cb, fit_kws=kws,
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

        # 1-sigma confidence band on dense grid
        y_uncertainty = None
        try:
            y_uncertainty = self._last_result.eval_uncertainty(x=x_dense, sigma=1)
        except Exception:
            pass  # Covariance matrix not available

        # Component curves on dense grid
        component_curves = {}
        if len(self.components) > 1:
            comps = self._last_result.eval_components(x=x_dense)
            for key, vals in comps.items():
                component_curves[key] = vals

        # Extract parameter info
        params_info = {}
        for name, par in self._last_result.params.items():
            params_info[name] = {
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
                "expr": par.expr or "",
            }

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

    def run_global_fit(
        self,
        datasets: list[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]],
        shared_params: set[str],
        method: str = "leastsq",
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
                    weights = 1.0 / np.sqrt(yerr**2 + (dfdx * xerr) ** 2)
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
