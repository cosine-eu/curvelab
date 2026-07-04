import re
import unittest
from pathlib import Path


class FitManagerDeserializeTests(unittest.TestCase):
    def test_deserialize_preserves_component_metadata(self):
        try:
            from curvelab.fit_manager import FitManager
        except Exception as exc:  # pragma: no cover - environment dependency fallback
            raise unittest.SkipTest(f"curvelab/lmfit unavailable: {exc}")

        payload = {
            "components": [
                {
                    "name": "Gaussian",
                    "prefix": "",
                    "operator": "+",
                    "expression": "",
                },
                {
                    "name": "Expression",
                    "prefix": "expression1_",
                    "operator": "*",
                    "expression": "a*exp(-b*x)+c",
                },
            ]
        }

        fm = FitManager.deserialize(payload)

        self.assertEqual(len(fm.components), 2)
        self.assertEqual(fm.components[0].name, "Gaussian")
        self.assertEqual(fm.components[0].prefix, "gaussian1_")
        self.assertEqual(fm.components[0].operator, "+")

        self.assertEqual(fm.components[1].name, "Expression")
        self.assertEqual(fm.components[1].prefix, "expression1_")
        self.assertEqual(fm.components[1].operator, "*")
        self.assertEqual(fm.components[1].expression, "a*exp(-b*x)+c")


class WorkspaceContractTests(unittest.TestCase):
    def test_app_workspace_includes_reduce_and_weight_in_save_and_load(self):
        ws_path = Path(__file__).resolve().parents[1] / "curvelab" / "app_workspace.py"
        source = ws_path.read_text(encoding="utf-8")

        # Save path
        self.assertRegex(source, r'"reduce_fcn"')
        self.assertRegex(source, r'"weight_mode"')

        # Load path
        self.assertRegex(source, r'self\.fit_panel\.reduce_var\.set\(')
        self.assertRegex(source, r'self\.fit_panel\.weight_var\.set\(')

    def test_diagnostic_plots_dialog_is_wired(self):
        curvelab_dir = Path(__file__).resolve().parents[1] / "curvelab"
        app_source = (curvelab_dir / "app.py").read_text(encoding="utf-8")
        handlers_source = (curvelab_dir / "app_analysis_handlers.py").read_text(encoding="utf-8")
        dlg_source = (curvelab_dir / "ui_dialogs_analysis.py").read_text(encoding="utf-8")

        self.assertRegex(handlers_source, r"def _show_diagnostic_plots\(self\)")
        self.assertIn("DiagnosticPlotsDialog", dlg_source)
        self.assertRegex(
            app_source,
            r'label="Diagnostic Plots\.\.\.", command=self\._show_diagnostic_plots',
        )

    def test_confidence_contour_dialog_is_wired(self):
        curvelab_dir = Path(__file__).resolve().parents[1] / "curvelab"
        app_source = (curvelab_dir / "app.py").read_text(encoding="utf-8")
        handlers_source = (curvelab_dir / "app_analysis_handlers.py").read_text(encoding="utf-8")
        dlg_source = (curvelab_dir / "ui_dialogs_analysis.py").read_text(encoding="utf-8")

        self.assertRegex(handlers_source, r"def _show_confidence_contours\(self\)")
        self.assertIn("ConfidenceContourDialog", dlg_source)
        self.assertRegex(
            app_source,
            r'label="2D Confidence Contours\.\.\.", command=self\._show_confidence_contours',
        )


class ModelRegistryTests(unittest.TestCase):
    def test_exponential_gaussian_in_registry(self):
        try:
            from curvelab.models import MODEL_REGISTRY
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        self.assertIn("ExponentialGaussian", MODEL_REGISTRY)

    def test_spline_in_registry(self):
        try:
            from curvelab.models import MODEL_REGISTRY
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        self.assertIn("Spline", MODEL_REGISTRY)
        # Spline is a sentinel (None) — requires knot count
        self.assertIsNone(MODEL_REGISTRY["Spline"])

    def test_create_spline_model(self):
        try:
            import numpy as np
            from curvelab.models import create_spline_model
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        x = np.linspace(0, 10, 100)
        model = create_spline_model(8, x)
        self.assertIsNotNone(model)


class FitManagerReduceWeightTests(unittest.TestCase):
    def test_reduce_functions_dict(self):
        try:
            from curvelab.fit_manager import REDUCE_FUNCTIONS
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        self.assertIn("Chi-square (default)", REDUCE_FUNCTIONS)
        self.assertIsNone(REDUCE_FUNCTIONS["Chi-square (default)"])
        self.assertIn("Neg. entropy", REDUCE_FUNCTIONS)
        self.assertIn("Cauchy log-pdf", REDUCE_FUNCTIONS)

    def test_weight_modes_list(self):
        try:
            from curvelab.fit_manager import WEIGHT_MODES
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        self.assertIn("1/yerr (default)", WEIGHT_MODES)
        self.assertIn("No weights", WEIGHT_MODES)
        self.assertIn("1/y", WEIGHT_MODES)

    def test_compute_weights_default(self):
        try:
            import numpy as np
            from curvelab.fit_manager import FitManager
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        y = np.array([1.0, 2.0, 3.0])
        yerr = np.array([0.1, 0.2, 0.3])
        w = FitManager._compute_weights(y, yerr, "1/yerr (default)")
        np.testing.assert_allclose(w, 1.0 / yerr)

    def test_compute_weights_no_weights(self):
        try:
            import numpy as np
            from curvelab.fit_manager import FitManager
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        y = np.array([1.0, 2.0, 3.0])
        yerr = np.array([0.1, 0.2, 0.3])
        w = FitManager._compute_weights(y, yerr, "No weights")
        self.assertIsNone(w)

    def test_compute_weights_one_over_y(self):
        try:
            import numpy as np
            from curvelab.fit_manager import FitManager
        except Exception as exc:
            raise unittest.SkipTest(f"curvelab unavailable: {exc}")
        y = np.array([1.0, 2.0, 4.0])
        w = FitManager._compute_weights(y, None, "1/y")
        np.testing.assert_allclose(w, 1.0 / np.abs(y))


if __name__ == "__main__":
    unittest.main()
