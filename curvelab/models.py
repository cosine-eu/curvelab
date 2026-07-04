"""lmfit 1D model registry."""

from functools import partial

import numpy as np
from lmfit.models import (
    BreitWignerModel,
    ConstantModel,
    DampedHarmonicOscillatorModel,
    DampedOscillatorModel,
    DoniachModel,
    ExponentialGaussianModel,
    ExponentialModel,
    ExpressionModel,
    GaussianModel,
    LinearModel,
    LognormalModel,
    LorentzianModel,
    MoffatModel,
    Pearson4Model,
    Pearson7Model,
    PolynomialModel,
    PowerLawModel,
    PseudoVoigtModel,
    QuadraticModel,
    RectangleModel,
    SineModel,
    SkewedGaussianModel,
    SkewedVoigtModel,
    SplineModel,
    SplitLorentzianModel,
    StepModel,
    StudentsTModel,
    ThermalDistributionModel,
    VoigtModel,
)

# lmfit >= 1.3 models (optional)
try:
    from lmfit.models import BoseModel
except ImportError:
    BoseModel = None
try:
    from lmfit.models import FermiModel
except ImportError:
    FermiModel = None

# Display name -> factory callable (no-arg returns a fresh Model instance)
MODEL_REGISTRY: dict[str, type] = {
    "Gaussian": GaussianModel,
    "Lorentzian": LorentzianModel,
    "Voigt": VoigtModel,
    "PseudoVoigt": PseudoVoigtModel,
    "Moffat": MoffatModel,
    "Pearson4": Pearson4Model,
    "Pearson7": Pearson7Model,
    "StudentT": StudentsTModel,
    "BreitWigner": BreitWignerModel,
    "DampedOscillator": DampedOscillatorModel,
    "DampedHarmonicOscillator": DampedHarmonicOscillatorModel,
    "Exponential": ExponentialModel,
    "PowerLaw": PowerLawModel,
    "Lognormal": LognormalModel,
    "Linear": LinearModel,
    "Quadratic": QuadraticModel,
    "Step": StepModel,
    "Rectangle": RectangleModel,
    "SplitLorentzian": SplitLorentzianModel,
    "SkewedGaussian": SkewedGaussianModel,
    "SkewedVoigt": SkewedVoigtModel,
    "ThermalDistribution": ThermalDistributionModel,
    "Doniach": DoniachModel,
    "Sine": SineModel,
    "Constant": ConstantModel,
    "ExponentialGaussian": ExponentialGaussianModel,
    "Expression": None,  # Sentinel: requires expression string at creation time
    "Spline": None,  # Sentinel: requires knot count at creation time
}

for _degree in range(2, 8):
    MODEL_REGISTRY[f"Polynomial{_degree}"] = partial(PolynomialModel, degree=_degree)

# Add lmfit >= 1.3 models when available
if BoseModel is not None:
    MODEL_REGISTRY["Bose"] = BoseModel
if FermiModel is not None:
    MODEL_REGISTRY["Fermi"] = FermiModel

MODEL_NAMES: list[str] = sorted(MODEL_REGISTRY.keys())


def create_expression_model(expr: str, prefix: str = ""):
    """Create an ExpressionModel from a math expression string.

    ExpressionModel does not support the prefix kwarg, so we bake the prefix
    into the expression by renaming parameter names (e.g. a -> expr1_a).
    """
    if not prefix:
        return ExpressionModel(expr, independent_vars=["x"])

    import re

    # Discover parameter names from an unprefixed model
    tmp = ExpressionModel(expr, independent_vars=["x"])
    params = set(tmp.param_names)

    # Replace each param with prefixed version (longest first to avoid partial matches)
    prefixed_expr = expr
    for p in sorted(params, key=len, reverse=True):
        prefixed_expr = re.sub(r"\b" + re.escape(p) + r"\b", prefix + p, prefixed_expr)

    return ExpressionModel(prefixed_expr, independent_vars=["x"])


def create_spline_model(n_knots: int, x_data: np.ndarray, prefix: str = ""):
    """Create a SplineModel with evenly-spaced knots over the data range."""
    xknots = np.linspace(x_data.min(), x_data.max(), n_knots)
    return SplineModel(xknots=xknots, prefix=prefix)


def create_model(name: str, prefix: str = ""):
    """Create a fresh model instance by registry name, with optional prefix."""
    factory = MODEL_REGISTRY[name]
    return factory(prefix=prefix)
