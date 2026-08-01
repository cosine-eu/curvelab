"""Regression tests for bugs fixed on the code-review branch.

Each test reproduces a specific defect and asserts the fixed behavior, so the
bug can't silently return. GUI-layer fixes (Tk widget lockout, TclError on
double-commit, etc.) are not covered here — this file is pure logic only.
"""

import json
import math
import unittest

import numpy as np


class ComponentPrefixCollisionTests(unittest.TestCase):
    """6ba7037: add_component reused a freed suffix after remove_component."""

    def test_readd_after_remove_gives_fresh_prefix(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Gaussian")
        fm.add_component("Gaussian")
        # Remove the middle component, freeing suffix "2".
        fm.remove_component(1)
        fm.add_component("Gaussian")

        prefixes = [c.prefix for c in fm.components]
        self.assertEqual(len(set(prefixes)), len(prefixes),
                         f"prefix collision: {prefixes}")
        # The re-added component must not reuse the freed "gaussian2_".
        self.assertNotIn("gaussian2_", prefixes)

    def test_clear_components_resets_counters(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Gaussian")
        fm.clear_components()
        fm.add_component("Gaussian")
        # After a clear, numbering starts fresh (single component: no prefix).
        self.assertEqual(fm.components[0].prefix, "")


class WorkspaceInfinityRoundTripTests(unittest.TestCase):
    """e0365e7: decode_workspace only restored inf under min/max keys."""

    def test_generic_infinity_field_restored(self):
        from curvelab.workspace import WorkspaceEncoder, decode_workspace, encode_value
        # gof["reduced chi-squared"] can be inf when dof <= 0.
        payload = encode_value({
            "gof": {"reduced chi-squared": float("inf"), "chi-squared": 3.5},
            "params": {"a": {"min": float("-inf"), "max": float("inf")}},
        })
        s = json.dumps({"result": payload}, cls=WorkspaceEncoder)
        back = json.loads(s, object_hook=decode_workspace)

        self.assertEqual(back["result"]["gof"]["reduced chi-squared"], float("inf"))
        self.assertEqual(back["result"]["params"]["a"]["min"], float("-inf"))
        self.assertEqual(back["result"]["params"]["a"]["max"], float("inf"))
        # A real string value must not be coerced to a float.
        back2 = json.loads(json.dumps({"note": "hello"}), object_hook=decode_workspace)
        self.assertEqual(back2["note"], "hello")


class NumericTokenTests(unittest.TestCase):
    """7fe6af5: negative scientific notation was misread as a header token."""

    def test_negative_scientific_notation_is_numeric(self):
        from curvelab.data_manager import _is_numeric_token
        for tok in ("-1.5e-3", "1.5e-3", "-42", "3.14", "+2.0E5", "-1e-10"):
            self.assertTrue(_is_numeric_token(tok), tok)
        for tok in ("foo", "x1", "1.2.3", ""):
            self.assertFalse(_is_numeric_token(tok), tok)

    def test_clipboard_negative_exponent_first_row_not_dropped(self):
        from curvelab.data_manager import DataManager
        dm = DataManager()
        text = "-1.5e-3\t2.0e-4\n3.1e-5\t4.2e-6\n5.5e-7\t6.6e-8\n"
        name, cols = dm.load_from_text(text)
        # All three rows must survive; the first must not be consumed as a header.
        self.assertEqual(dm.datasets[name].shape[0], 3)


class RefitWeightGuardTests(unittest.TestCase):
    """cd499e2: refit_from_result inlined an unguarded 1.0/yerr."""

    def test_zero_yerr_does_not_produce_inf_weights(self):
        from curvelab.fit_manager import FitManager
        from curvelab.session import FitResult
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        yerr = np.full_like(y, 0.5)
        yerr[3] = 0.0  # would give inf weight if unguarded

        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        result = FitResult(
            x_dense=x, y_fit_dense=y, x_data=x, y_data=y, y_fit_data=y,
            yerr_data=yerr, y_uncertainty=None, component_curves={},
            params={}, gof={}, report="", candidates=None, init_params=None,
        )
        fm.refit_from_result(result)
        self.assertIsNotNone(fm._last_result)
        self.assertTrue(math.isfinite(fm._last_result.redchi))


class EffectiveVarianceGuardTests(unittest.TestCase):
    """1df7ad8: Effective variance weights divided by an unguarded denominator."""

    def _linear_fm(self, x, y):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        return fm

    def test_yerr_only_fallback_stays_finite(self):
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        yerr = np.full_like(y, 0.5)
        yerr[3] = 0.0
        fm = self._linear_fm(x, y)
        result = fm.run_fit(x, y, yerr=yerr, weight_mode="Effective variance")
        self.assertTrue(math.isfinite(result.gof["reduced chi-squared"]))

    def test_xerr_yerr_branch_stays_finite(self):
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        yerr = np.full_like(y, 0.5)
        yerr[3] = 0.0
        xerr = np.full_like(x, 0.2)
        xerr[3] = 0.0
        fm = self._linear_fm(x, y)
        result = fm.run_fit(x, y, yerr=yerr, xerr=xerr, weight_mode="Effective variance")
        self.assertTrue(math.isfinite(result.gof["reduced chi-squared"]))


class AutoGuessMultiPeakTests(unittest.TestCase):
    """e57e455: every component guessed against raw y, collapsing onto one peak."""

    def test_three_gaussians_guess_distinct_centers(self):
        from curvelab.fit_manager import FitManager

        def gauss(x, amp, cen, wid):
            return amp * np.exp(-((x - cen) ** 2) / (2 * wid ** 2))

        x = np.linspace(0, 30, 600)
        y = gauss(x, 10, 5, 1) + gauss(x, 6, 15, 1) + gauss(x, 3, 25, 1)

        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Gaussian")
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)

        centers = [round(fm.params[f"{c.prefix}center"].value) for c in fm.components]
        # Sequential residual guessing must not put all three on the same peak.
        self.assertEqual(len(set(centers)), 3, f"centers collapsed: {centers}")


class CloneComponentsHintsTests(unittest.TestCase):
    """1e2bdb8: clone_components_to dropped fixed/bounded parameters."""

    def test_fixed_and_bounded_params_survive_clone(self):
        from curvelab.fit_manager import FitManager
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1

        source = FitManager()
        source.add_component("Linear")
        source.auto_guess(x, y)
        source.set_param_hint("slope", value=2.0, vary=False)     # fixed
        source.set_param_hint("intercept", min=0.0, max=5.0)      # bounded

        target = FitManager()
        source.clone_components_to(target)

        self.assertFalse(target.params["slope"].vary)
        self.assertEqual(target.params["intercept"].min, 0.0)
        self.assertEqual(target.params["intercept"].max, 5.0)


class CloneComponentsSelfTests(unittest.TestCase):
    """Batch fit clones the source model onto every series, the source
    included. Cloning onto itself used to clear the component list it was
    about to copy, wiping the model and failing every later series."""

    def test_self_clone_preserves_model(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        fm.add_component("Gaussian")
        fm.set_param_hint("linear1_slope", value=2.0, vary=False)

        fm.clone_components_to(fm)

        self.assertEqual([c.name for c in fm.components], ["Linear", "Gaussian"])
        self.assertIsNotNone(fm.model)
        self.assertFalse(fm.params["linear1_slope"].vary)


class ParamHintPersistenceTests(unittest.TestCase):
    """69a37b0 (core mechanism): set_param_hint must survive a model rebuild.

    Undo/redo routes through set_param_hint so undone edits don't reappear
    when a later rebuild re-applies the hint dict. Uses an already-prefixed
    (multi-component) model so param names stay stable across the rebuild
    (the 1->2 component transition renames the first component's params).
    """

    def test_hint_reapplied_after_rebuild(self):
        from curvelab.fit_manager import FitManager
        x = np.linspace(0, 10, 40)
        y = 2 * x + 1
        fm = FitManager()
        fm.add_component("Linear")
        fm.add_component("Constant")   # now params are prefixed and stable
        fm.auto_guess(x, y)
        fm.set_param_hint("linear1_slope", min=-5.0)
        self.assertEqual(fm.params["linear1_slope"].min, -5.0)

        # Adding a third component rebuilds the model; the hint must persist.
        fm.add_component("Gaussian")
        self.assertEqual(fm.params["linear1_slope"].min, -5.0)


class RemoveComponentHintsTests(unittest.TestCase):
    """Parameter hints are keyed by prefixed name, so removing a component
    must drop its hints and re-key a lone survivor's when it loses its
    prefix -- otherwise fixed values and bounds are silently lost."""

    def test_survivor_hints_follow_prefix_reset(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Linear")
        fm.set_param_hint("gaussian1_center", value=5.0, min=0.0)

        fm.remove_component(1)   # Linear; Gaussian loses its prefix

        self.assertEqual(fm.components[0].prefix, "")
        self.assertEqual(fm.params["center"].value, 5.0)
        self.assertEqual(fm.params["center"].min, 0.0)

    def test_removed_component_hints_are_dropped(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Linear")
        fm.add_component("Constant")
        fm.set_param_hint("linear1_slope", value=3.0, vary=False)
        fm.set_param_hint("gaussian1_center", value=5.0)

        fm.remove_component(1)   # Linear

        self.assertNotIn("linear1_slope", fm._param_hints)
        # Prefixes of the survivors are unchanged, so their hints still apply.
        self.assertEqual(fm.params["gaussian1_center"].value, 5.0)

    def test_similar_prefixes_are_not_confused(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        for _ in range(11):
            fm.add_component("Gaussian")
        fm.set_param_hint("gaussian11_center", value=7.0)

        fm.remove_component(0)   # gaussian1_, not gaussian11_

        self.assertEqual(fm.params["gaussian11_center"].value, 7.0)


class RefitFromGuessTests(unittest.TestCase):
    """Each fit starts from the remembered guess/user values, not the
    previous result, so repeated fits are reproducible and don't drift
    along a degenerate direction."""

    def _fit_setup(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 80)
        y = 3 * np.exp(-(x - 5) ** 2 / (2 * 1.0 ** 2)) + rng.normal(0, 0.02, x.size)
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        return fm, x, y

    def test_repeated_fits_start_from_same_values(self):
        fm, x, y = self._fit_setup()
        r1 = fm.run_fit(x, y, weight_mode="No weights")
        r2 = fm.run_fit(x, y, weight_mode="No weights")
        # Both fits start from the guess, so their init snapshots match
        # (under the old chaining, r2 would have started from r1's result).
        self.assertEqual(r1.init_params, r2.init_params)
        for name in r1.params:
            self.assertAlmostEqual(r1.params[name]["value"],
                                   r2.params[name]["value"], places=8)

    def test_manual_start_value_is_honored(self):
        fm, x, y = self._fit_setup()
        fm.run_fit(x, y, weight_mode="No weights")   # move _params off the guess
        fm.set_start_value("center", 7.5)
        r = fm.run_fit(x, y, weight_mode="No weights")
        self.assertAlmostEqual(r.init_params["center"], 7.5)

    def test_fixed_parameter_is_not_reset(self):
        # A pinned (vary=False) value must survive the reset-to-start, not be
        # clobbered by a stale guess.
        fm, x, y = self._fit_setup()
        fm.set_param_hint("center", value=4.0, vary=False)
        r = fm.run_fit(x, y, weight_mode="No weights")
        self.assertAlmostEqual(r.params["center"]["value"], 4.0, places=6)

    def test_first_fit_without_guess_is_reproducible(self):
        from curvelab.fit_manager import FitManager
        x = np.linspace(0, 10, 40)
        y = 2.0 * x + 1.0
        fm = FitManager()
        fm.add_component("Linear")
        # No auto_guess: the first fit records its own starting values.
        r1 = fm.run_fit(x, y, weight_mode="No weights")
        r2 = fm.run_fit(x, y, weight_mode="No weights")
        self.assertEqual(r1.init_params, r2.init_params)


class CaptureStartValuesTests(unittest.TestCase):
    """capture_start_values adopts the current (fitted) values as the next
    fit's start -- the 'Use as Start' action -- restoring opt-in chaining."""

    def test_capture_makes_next_fit_start_from_result(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 80)
        y = 3 * np.exp(-(x - 5) ** 2 / (2 * 0.8 ** 2)) + rng.normal(0, 0.02, x.size)
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        guess_center = fm.params["center"].value

        fm.run_fit(x, y, weight_mode="No weights")
        fitted_center = fm.params["center"].value

        fm.capture_start_values()
        r = fm.run_fit(x, y, weight_mode="No weights")
        # Now the fit starts from the fitted values, not the original guess.
        self.assertAlmostEqual(r.init_params["center"], fitted_center)
        self.assertNotAlmostEqual(r.init_params["center"], guess_center, places=6)


class FitErrorbarsFlagTests(unittest.TestCase):
    """FitResult.errorbars reports whether the fit could estimate
    uncertainties -- False signals unidentifiable parameters."""

    def test_good_fit_has_errorbars(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 60)
        y = 3 * np.exp(-(x - 5) ** 2 / (2 * 0.8 ** 2)) + rng.normal(0, 0.02, x.size)
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        self.assertTrue(fm.run_fit(x, y, weight_mode="No weights").errorbars)

    def test_degenerate_fit_reports_no_errorbars(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(2)
        x = np.linspace(0, 10, 50)
        y = 5.0 + rng.normal(0, 0.05, x.size)
        fm = FitManager()
        fm.add_component("Constant")   # two constants -> perfectly degenerate
        fm.add_component("Constant")
        fm.auto_guess(x, y)
        self.assertFalse(fm.run_fit(x, y, weight_mode="No weights").errorbars)


class OdsEngineTests(unittest.TestCase):
    """07c55a5: .ods loading used the package name instead of pandas engine id."""

    def test_ods_round_trip(self):
        import pandas as pd
        try:
            import odf  # noqa: F401
        except ImportError:
            self.skipTest("odfpy not installed")
        import tempfile, os
        from curvelab.data_manager import DataManager

        d = tempfile.mkdtemp()
        path = os.path.join(d, "t.ods")
        pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}).to_excel(
            path, engine="odf", index=False)
        try:
            dm = DataManager()
            name, cols = dm.load(path)
            self.assertEqual(cols, ["x", "y"])
            np.testing.assert_allclose(dm.get_column(name, "y"), [4.0, 5.0, 6.0])
        finally:
            os.unlink(path)


class EncodingFallbackTests(unittest.TestCase):
    """cbe7c8a: non-UTF-8 CSV/text files raised UnicodeDecodeError."""

    def test_cp1252_csv_loads(self):
        import tempfile, os
        from curvelab.data_manager import DataManager
        # Header with degree/micro signs, encoded cp1252 (Excel-on-Windows style).
        content = "temp_°C,value_µm\n1,2\n3,4\n"
        d = tempfile.mkdtemp()
        path = os.path.join(d, "t.csv")
        with open(path, "wb") as f:
            f.write(content.encode("cp1252"))
        try:
            dm = DataManager()
            name, cols = dm.load(path)
            self.assertEqual(cols, ["temp_°C", "value_µm"])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
