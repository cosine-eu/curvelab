"""Curve fitting logic using lmfit."""

from dataclasses import dataclass

import numpy as np
from lmfit import CompositeModel, Model, Parameters
from lmfit.model import ModelResult

from .models import create_expression_model, create_model
from .session import FitResult  # re-export; canonical location is session.py


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

    def _rebuild_model(self):
        """Rebuild the composite model from current components."""
        if not self.components:
            self._model = None
            self._params = None
            return

        models = []
        for comp in self.components:
            if comp.name == "Expression":
                m = create_expression_model(comp.expression, prefix=comp.prefix)
            else:
                m = create_model(comp.name, prefix=comp.prefix)
            models.append(m)

        self._model = models[0]
        for comp, m in zip(self.components[1:], models[1:]):
            if comp.operator == "*":
                self._model = self._model * m
            elif comp.operator == "-":
                self._model = self._model - m
            elif comp.operator == "/":
                self._model = self._model / m
            else:
                self._model = self._model + m
        self._params = self._model.make_params()

        # ExpressionModel params default to -inf; set to 1.0 so fits don't NaN
        for par in self._params.values():
            if par.value == float("-inf"):
                par.set(value=1.0)

    @property
    def model(self) -> Model | None:
        return self._model

    @property
    def params(self) -> Parameters | None:
        return self._params

    def auto_guess(self, x: np.ndarray, y: np.ndarray) -> Parameters:
        """Auto-guess parameters for each component using lmfit's guess()."""
        if self._model is None:
            raise ValueError("No model defined")

        self._params = self._model.make_params()

        # ExpressionModel params default to -inf; set to 1.0 so fits don't NaN
        for par in self._params.values():
            if par.value == float("-inf"):
                par.set(value=1.0)

        for comp in self.components:
            if comp.name == "Expression":
                m = create_expression_model(comp.expression, prefix=comp.prefix)
            else:
                m = create_model(comp.name, prefix=comp.prefix)
            try:
                guessed = m.guess(y, x=x)
                for pname, par in guessed.items():
                    if pname in self._params:
                        self._params[pname].set(
                            value=par.value, min=par.min, max=par.max
                        )
            except (NotImplementedError, Exception):
                # Some models don't implement guess(); skip
                pass

        return self._params

    def clone_components_to(self, target: "FitManager"):
        """Copy this manager's component list into target, rebuilding its model."""
        target.clear_components()
        for comp in self.components:
            target.add_component(comp.name, operator=comp.operator, expression=comp.expression)

    def set_param(self, name: str, **kwargs):
        """Set parameter attributes (value, min, max, vary)."""
        if self._params is not None and name in self._params:
            self._params[name].set(**kwargs)

    def run_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray | None = None,
        n_dense: int = 500,
        method: str = "leastsq",
    ) -> FitResult:
        """Run the fit and return results."""
        if self._model is None or self._params is None:
            raise ValueError("No model defined")

        weights = 1.0 / yerr if yerr is not None else None
        self._last_result = self._model.fit(
            y, self._params, x=x, weights=weights,
            method=method, nan_policy="omit",
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
        )

    def serialize(self) -> dict:
        """Serialize component list for workspace persistence."""
        return {
            "components": [
                {
                    "name": c.name,
                    "prefix": c.prefix,
                    "operator": c.operator,
                    "expression": c.expression,
                }
                for c in self.components
            ]
        }

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
        return fm
