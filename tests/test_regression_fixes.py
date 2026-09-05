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


class EffectiveVarianceIterationTests(unittest.TestCase):
    """The effective-variance weights depend on df/dx and were evaluated once
    at the starting parameters, so the fitted slope and its stderr depended
    on the initial guess. They are now iterated to self-consistency."""

    def _fit_from(self, start_slope):
        from curvelab.fit_manager import FitManager, WEIGHT_EFFECTIVE_VARIANCE
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 30)
        xerr = np.full(30, 0.5)
        yerr = np.full(30, 0.05)
        y = 3.0 * (x + rng.normal(0, 0.5, 30)) + 1.0 + rng.normal(0, 0.05, 30)
        fm = FitManager()
        fm.add_component("Linear")
        fm.set_param_hint("slope", value=start_slope)
        fm.set_param_hint("intercept", value=1.0)
        return fm.run_fit(x, y, yerr=yerr, xerr=xerr, weight_mode=WEIGHT_EFFECTIVE_VARIANCE)

    def test_result_does_not_depend_on_the_start(self):
        a = self._fit_from(0.0)
        b = self._fit_from(10.0)
        self.assertAlmostEqual(a.params["slope"]["value"], b.params["slope"]["value"], places=5)
        self.assertAlmostEqual(a.params["slope"]["stderr"], b.params["slope"]["stderr"], places=5)
        self.assertAlmostEqual(a.gof["reduced chi-squared"], b.gof["reduced chi-squared"], places=5)

    def test_weights_are_consistent_with_the_fit(self):
        # With sigma_x = 0.5 on slope 3 the effective sigma is ~1.5, so the
        # weighted chi2/nu must be of order 1, not the ~900 that sigma_y alone
        # (the weights a start at slope 0 would give) implies.
        r = self._fit_from(0.0)
        self.assertLess(r.gof["reduced chi-squared"], 3.0)
        self.assertGreater(r.gof["reduced chi-squared"], 0.2)


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


class AutoGuessDerivedParamTests(unittest.TestCase):
    """Auto-guess used Parameter.set(value=...) on every guessed parameter,
    which clears the expression on derived ones (fwhm, height): they froze
    at the guess and no longer tracked the fit."""

    def test_fwhm_keeps_its_expression_and_tracks_sigma(self):
        from curvelab.fit_manager import FitManager

        sigma_true = 2.5
        x = np.linspace(-10, 10, 200)
        y = 3.0 * np.exp(-x ** 2 / (2 * sigma_true ** 2))

        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        self.assertEqual(fm.params["fwhm"].expr, "2.3548200*sigma")

        fm.run_fit(x, y)
        sigma = fm.params["sigma"].value
        self.assertAlmostEqual(sigma, sigma_true, places=3)
        self.assertAlmostEqual(fm.params["fwhm"].value, 2.3548200 * sigma, places=6)
        self.assertAlmostEqual(fm.params["height"].value, 3.0, places=3)


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


class StartValuePersistenceTests(unittest.TestCase):
    """The auto-guessed start values were neither serialized nor cloned, so
    after a workspace load (or a batch-fit clone) the first fit began from
    the model defaults and captured its own output as the 'guess'."""

    def _guessed_source(self):
        from curvelab.fit_manager import FitManager
        x = np.linspace(-5, 5, 100)
        y = 4.0 * np.exp(-(x - 1.0) ** 2 / (2 * 0.8 ** 2))
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        return fm, x, y

    def test_start_values_survive_serialize_round_trip(self):
        from curvelab.fit_manager import FitManager
        source, x, y = self._guessed_source()
        restored = FitManager.deserialize(source.serialize())
        self.assertEqual(restored._start_values, source._start_values)
        r = restored.run_fit(x, y, weight_mode="No weights")
        self.assertAlmostEqual(r.init_params["center"], source._start_values["center"])

    def test_old_payload_without_start_values_still_loads(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager.deserialize({"components": [
            {"name": "Linear", "prefix": "", "operator": "+", "expression": ""}]})
        self.assertIsNotNone(fm.model)

    def test_start_values_follow_clone(self):
        from curvelab.fit_manager import FitManager
        source, x, y = self._guessed_source()
        target = FitManager()
        source.clone_components_to(target)
        self.assertEqual(target._start_values["center"], source._start_values["center"])
        self.assertEqual(target._start_values["sigma"], source._start_values["sigma"])


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


class AddComponentHintsTests(unittest.TestCase):
    """Adding a second component gives the first one a prefix. Its hints and
    start values were keyed by the bare names and stopped applying, so a
    user's fixed value or bound silently reverted to the model default."""

    def test_first_component_hints_follow_new_prefix(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.set_param_hint("center", value=4.0, vary=False)
        fm.set_param_hint("sigma", min=0.5, max=3.0)

        fm.add_component("Linear")   # Gaussian becomes gaussian1_

        self.assertEqual(fm.components[0].prefix, "gaussian1_")
        self.assertNotIn("center", fm._param_hints)
        self.assertEqual(fm.params["gaussian1_center"].value, 4.0)
        self.assertFalse(fm.params["gaussian1_center"].vary)
        self.assertEqual(fm.params["gaussian1_sigma"].min, 0.5)
        self.assertEqual(fm.params["gaussian1_sigma"].max, 3.0)

    def test_start_values_follow_prefix_and_do_not_resurrect(self):
        from curvelab.fit_manager import FitManager
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1

        fm = FitManager()
        fm.add_component("Linear")
        fm.set_start_value("slope", 7.0)

        fm.add_component("Constant")   # Linear becomes linear1_
        self.assertNotIn("slope", fm._start_values)
        self.assertEqual(fm._start_values["linear1_slope"], 7.0)

        # Remove the Constant again: the start value must be moved back, and
        # a stale bare-name entry must not be resurrected over a fresh edit.
        fm.remove_component(1)
        self.assertNotIn("linear1_slope", fm._start_values)
        self.assertEqual(fm._start_values["slope"], 7.0)
        fm.set_param("slope", value=1.5)
        fm.set_start_value("slope", 1.5)
        result = fm.run_fit(x, y)
        self.assertAlmostEqual(result.init_params["slope"], 1.5)


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

    def test_set_param_value_becomes_the_start(self):
        # set_param(value=...) is how the notebook and the Find Peaks dialog
        # edit values; they did not call set_start_value, so run_fit's reset
        # threw the edit away. The setter now records the start itself.
        fm, x, y = self._fit_setup()
        fm.run_fit(x, y, weight_mode="No weights")
        fm.set_param("center", value=6.5)
        r = fm.run_fit(x, y, weight_mode="No weights")
        self.assertAlmostEqual(r.init_params["center"], 6.5)

    def test_set_param_hint_value_becomes_the_start(self):
        fm, x, y = self._fit_setup()
        fm.run_fit(x, y, weight_mode="No weights")
        fm.set_param_hint("center", value=6.0)
        r = fm.run_fit(x, y, weight_mode="No weights")
        self.assertAlmostEqual(r.init_params["center"], 6.0)

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


class FitOutcomeTests(unittest.TestCase):
    """How a fit ended used to reach the user only as a generic 'no
    uncertainties' warning. The result now carries success/message/nfev and
    marks parameters that sit on a bound, and the post-fit notes name the
    actual cause."""

    def _peak_data(self):
        rng = np.random.default_rng(0)
        x = np.linspace(-10, 10, 200)
        y = 3.0 * np.exp(-(x - 1.0) ** 2 / (2 * 2.0 ** 2)) + rng.normal(0, 0.05, 200)
        return x, y

    def test_max_nfev_abort_is_reported_as_not_converged(self):
        from curvelab.fit_manager import FitManager
        from curvelab.app_fit_handlers import fit_outcome_notes
        x, y = self._peak_data()
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.set_param_hint("center", value=-5.0)
        fm.set_param_hint("sigma", value=0.5)
        fm.set_param_hint("amplitude", value=1.0)
        r = fm.run_fit(x, y, weight_mode="No weights", max_nfev=4)
        self.assertFalse(r.success)
        self.assertEqual(r.nfev, 4)
        self.assertIn("aborted", r.message.lower())
        self.assertTrue(r.report.startswith("[[Convergence]]\n    NOT converged"))
        notes = fit_outcome_notes(r, "least_squares")
        self.assertTrue(any(n.startswith("The fit did not converge after 4") for n in notes))
        # The abort is not blamed on identifiability.
        self.assertFalse(any("identifiable" in n for n in notes))

    def test_converged_fit_reports_success(self):
        from curvelab.fit_manager import FitManager
        from curvelab.app_fit_handlers import fit_outcome_notes
        x, y = self._peak_data()
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        r = fm.run_fit(x, y, weight_mode="No weights")
        self.assertTrue(r.success)
        self.assertGreater(r.nfev, 0)
        self.assertTrue(r.report.startswith("[[Convergence]]\n    converged"))
        self.assertEqual(fit_outcome_notes(r, "least_squares"), [])

    def test_parameter_at_bound_is_flagged(self):
        from curvelab.fit_manager import FitManager
        from curvelab.app_fit_handlers import fit_outcome_notes
        x = np.linspace(0, 10, 40)
        fm = FitManager()
        fm.add_component("Linear")
        fm.set_param_hint("slope", value=1.0, min=0.0, max=1.5)   # truth is 2
        fm.set_param_hint("intercept", value=1.0)
        r = fm.run_fit(x, 2.0 * x + 1.0, weight_mode="No weights")
        self.assertEqual(r.params["slope"]["at_bound"], "max")
        self.assertIsNone(r.params["intercept"]["at_bound"])
        self.assertIn("at a bound, stderr not meaningful: slope (at max)", r.report)
        notes = fit_outcome_notes(r, "least_squares")
        self.assertEqual(len(notes), 1)
        self.assertIn("slope (at max)", notes[0])

    def test_scalar_method_without_numdifftools_is_explained(self):
        from unittest import mock
        from curvelab.session import FitResult
        from curvelab.app_fit_handlers import fit_outcome_notes
        x = np.linspace(0, 1, 5)
        r = FitResult(x_dense=x, y_fit_dense=x, x_data=x, y_data=x, y_fit_data=x,
                      yerr_data=None, params={}, report="", errorbars=False)
        with mock.patch("importlib.util.find_spec", return_value=None):
            notes = fit_outcome_notes(r, "nelder")
        self.assertIn("numdifftools", notes[0])
        # With the package present (or a covariance method) it is identifiability.
        notes = fit_outcome_notes(r, "least_squares")
        self.assertIn("identifiable", notes[0])


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
