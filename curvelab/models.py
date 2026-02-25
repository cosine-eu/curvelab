"""lmfit 1D model registry."""

from lmfit.models import (
    BreitWignerModel,
    ConstantModel,
    DampedHarmonicOscillatorModel,
    DampedOscillatorModel,
    DoniachModel,
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
    SplitLorentzianModel,
    StepModel,
    StudentsTModel,
    ThermalDistributionModel,
    VoigtModel,
)

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
    "Polynomial2": lambda prefix="": PolynomialModel(degree=2, prefix=prefix),
    "Polynomial3": lambda prefix="": PolynomialModel(degree=3, prefix=prefix),
    "Polynomial4": lambda prefix="": PolynomialModel(degree=4, prefix=prefix),
    "Polynomial5": lambda prefix="": PolynomialModel(degree=5, prefix=prefix),
    "Polynomial6": lambda prefix="": PolynomialModel(degree=6, prefix=prefix),
    "Polynomial7": lambda prefix="": PolynomialModel(degree=7, prefix=prefix),
    "Step": StepModel,
    "Rectangle": RectangleModel,
    "SplitLorentzian": SplitLorentzianModel,
    "SkewedGaussian": SkewedGaussianModel,
    "SkewedVoigt": SkewedVoigtModel,
    "ThermalDistribution": ThermalDistributionModel,
    "Doniach": DoniachModel,
    "Sine": SineModel,
    "Constant": ConstantModel,
    "Expression": None,  # Sentinel: requires expression string at creation time
}

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


def create_model(name: str, prefix: str = ""):
    """Create a fresh model instance by registry name, with optional prefix."""
    factory = MODEL_REGISTRY[name]
    # Polynomial lambdas accept prefix kwarg; lmfit Model classes accept it too
    if callable(factory) and isinstance(factory, type):
        return factory(prefix=prefix)
    else:
        # Lambda factory for PolynomialModel
        return factory(prefix=prefix)
