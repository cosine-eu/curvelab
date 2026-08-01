"""CurveLab - Data Plotter & Curve Fitter."""

from .app import CurveLabApp

__all__ = ["CurveLabApp"]
__version__ = "0.11.1"


def notebook_widget(**kwargs):
    """Create and display an interactive CurveLab widget in a Jupyter notebook.

    Requires ``%matplotlib widget`` (ipympl) to be activated first.
    """
    from .notebook import CurveLabWidget

    return CurveLabWidget(**kwargs)
