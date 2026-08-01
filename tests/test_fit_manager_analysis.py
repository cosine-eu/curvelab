"""Tests for FitManager's analysis and advanced-fit methods.

Covers ODR, global fit, confidence intervals / profiles, covariance and
correlation extraction, the composite-operator (* - /) fit paths, and the
reduce-function passthrough — all previously untested.
"""

import unittest

import numpy as np


def _gauss(x, amp, cen, wid):
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid ** 2))


class LinearFitBase(unittest.TestCase):
    """A completed linear fit whose _last_result feeds the analysis methods."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(0)
        self.x = np.linspace(0, 10, 60)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 0.2, 60)
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)
        self.fm.run_fit(self.x, self.y, weight_mode="No weights")


class CovarianceTests(LinearFitBase):
    def test_covariance_matrix_shape_and_names(self):
        names, cov = self.fm.get_covariance_matrix()
        self.assertEqual(names, ["slope", "intercept"])
        self.assertEqual(cov.shape, (2, 2))
        # Covariance matrix is symmetric.
        np.testing.assert_allclose(cov, cov.T, rtol=1e-6)

    def test_covariance_none_before_fit(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        self.assertIsNone(fm.get_covariance_matrix())


class CorrelationTests(LinearFitBase):
    def test_correlations_present_for_varied_params(self):
        cors = self.fm.get_correlations()
        self.assertIn("slope", cors)
        self.assertIn("intercept", cors["slope"])
        # slope/intercept of a line are strongly (negatively) correlated.
        self.assertLess(cors["slope"]["intercept"], 0)
        self.assertGreaterEqual(cors["slope"]["intercept"], -1.0)

    def test_correlations_raise_before_fit(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        with self.assertRaises(ValueError):
            fm.get_correlations()


class ConfidenceIntervalTests(LinearFitBase):
    def test_ci_report_is_string_mentioning_params(self):
        text = self.fm.compute_confidence_intervals()
        self.assertIsInstance(text, str)
        self.assertIn("slope", text)
        self.assertIn("intercept", text)

    def test_ci_raises_before_fit(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        with self.assertRaises(ValueError):
            fm.compute_confidence_intervals()


class ProfileLikelihoodTests(LinearFitBase):
    def test_profiles_shape_and_minimum(self):
        profiles = self.fm.compute_ci_profiles()
        self.assertIn("slope", profiles)
        pts = profiles["slope"]
        self.assertGreater(len(pts), 1)
        # Each point is (param_value, chi_squared).
        self.assertEqual(len(pts[0]), 2)
        # Parameter values are sorted ascending.
        vals = [v for v, _ in pts]
        self.assertEqual(vals, sorted(vals))
        # chi-squared floor equals the best-fit chi-squared.
        chi2s = [c for _, c in pts]
        self.assertAlmostEqual(min(chi2s), self.fm._last_result.chisqr, places=6)

    def test_profiles_raise_before_fit(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        with self.assertRaises(ValueError):
            fm.compute_ci_profiles()


class OdrTests(unittest.TestCase):
    def setUp(self):
        try:
            import odrpack  # noqa: F401
        except ImportError:
            self.skipTest("odrpack not installed")
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(2)
        self.x = np.linspace(0, 10, 50)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 0.2, 50)
        self.xerr = np.full_like(self.x, 0.1)
        self.yerr = np.full_like(self.y, 0.2)
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)

    def test_odr_recovers_slope(self):
        result = self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=self.xerr)
        self.assertAlmostEqual(result.params["slope"]["value"], 2.0, delta=0.1)
        self.assertAlmostEqual(result.params["intercept"]["value"], 1.0, delta=0.3)
        self.assertIsNotNone(result.params["slope"]["stderr"])

    def test_odr_gof_and_dense_curve(self):
        result = self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=self.xerr)
        self.assertIn("reduced chi-squared", result.gof)
        self.assertGreater(result.gof["R-squared"], 0.99)
        self.assertEqual(len(result.x_dense), 500)

    def test_odr_respects_fixed_parameter(self):
        self.fm.set_param_hint("intercept", value=1.0, vary=False)
        result = self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=self.xerr)
        # A fixed parameter stays put and reports no stderr.
        self.assertAlmostEqual(result.params["intercept"]["value"], 1.0, places=6)
        self.assertIsNone(result.params["intercept"]["stderr"])

    def test_odr_refits_from_the_guess(self):
        # Like run_fit, repeated ODR fits start from the same values instead
        # of chaining, so they are reproducible.
        r1 = self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=self.xerr)
        r2 = self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=self.xerr)
        self.assertEqual(r1.init_params, r2.init_params)
        for name in r1.params:
            self.assertAlmostEqual(r1.params[name]["value"],
                                   r2.params[name]["value"], places=8)


class GlobalFitTests(unittest.TestCase):
    def setUp(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(3)
        self.x1 = np.linspace(0, 10, 40)
        self.y1 = 3.0 * self.x1 + 2.0 + rng.normal(0, 0.1, 40)
        self.x2 = np.linspace(0, 10, 40)
        self.y2 = 3.0 * self.x2 + 5.0 + rng.normal(0, 0.1, 40)
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x1, self.y1)

    def test_shared_slope_is_identical_across_datasets(self):
        results = self.fm.run_global_fit(
            [(self.x1, self.y1, None, None), (self.x2, self.y2, None, None)],
            shared_params={"slope"},
        )
        self.assertEqual(len(results), 2)
        s1 = results[0].params["slope"]["value"]
        s2 = results[1].params["slope"]["value"]
        self.assertEqual(s1, s2)  # shared -> one underlying parameter
        self.assertAlmostEqual(s1, 3.0, delta=0.1)

    def test_unshared_intercepts_differ(self):
        results = self.fm.run_global_fit(
            [(self.x1, self.y1, None, None), (self.x2, self.y2, None, None)],
            shared_params={"slope"},
        )
        b1 = results[0].params["intercept"]["value"]
        b2 = results[1].params["intercept"]["value"]
        self.assertAlmostEqual(b1, 2.0, delta=0.2)
        self.assertAlmostEqual(b2, 5.0, delta=0.2)
        self.assertGreater(abs(b1 - b2), 1.0)

    def test_report_and_gof_present(self):
        results = self.fm.run_global_fit(
            [(self.x1, self.y1, None, None), (self.x2, self.y2, None, None)],
            shared_params={"slope"},
        )
        self.assertIn("Global Fit", results[0].report)
        self.assertIn("[shared]", results[0].report)
        self.assertIn("reduced chi-squared", results[0].gof)


class CompositeOperatorFitTests(unittest.TestCase):
    """The +, *, -, / composition operators in a real fit."""

    def _fit(self, operator, make_y):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(4)
        x = np.linspace(0, 20, 300)
        y = make_y(x) + rng.normal(0, 0.02, 300)
        fm = FitManager()
        fm.add_component("Constant")
        fm.add_component("Gaussian", operator=operator)
        fm.auto_guess(x, y)
        return fm.run_fit(x, y, weight_mode="No weights"), x, y

    def test_additive(self):
        result, x, y = self._fit("+", lambda x: 2.0 + _gauss(x, 5, 10, 1.5))
        self.assertGreater(result.gof["R-squared"], 0.98)

    def test_multiplicative(self):
        result, x, y = self._fit("*", lambda x: 2.0 * _gauss(x, 5, 10, 1.5))
        self.assertGreater(result.gof["R-squared"], 0.9)

    def test_subtractive(self):
        result, x, y = self._fit("-", lambda x: 5.0 - _gauss(x, 3, 10, 1.5))
        self.assertGreater(result.gof["R-squared"], 0.98)

    def test_divisive(self):
        # Gaussian / Constant is a well-behaved scaled Gaussian (the reverse,
        # Constant / Gaussian, divides by a value that decays to zero).
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(4)
        x = np.linspace(0, 20, 300)
        y = _gauss(x, 6.0, 10.0, 1.5) / 2.0 + rng.normal(0, 0.02, 300)
        fm = FitManager()
        fm.add_component("Gaussian")
        fm.add_component("Constant", operator="/")
        fm.auto_guess(x, y)
        result = fm.run_fit(x, y, weight_mode="No weights")
        self.assertGreater(result.gof["R-squared"], 0.98)


class ReduceFunctionTests(unittest.TestCase):
    """The robust reduce functions are passed through run_fit."""

    def _linear_with_outlier(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(5)
        x = np.linspace(0, 10, 80)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 80)
        y[40] += 50.0  # gross outlier
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        return fm, x, y

    def test_negentropy_reduce_runs_and_fits(self):
        from curvelab.fit_manager import REDUCE_FUNCTIONS
        fm, x, y = self._linear_with_outlier()
        result = fm.run_fit(
            x, y, weight_mode="No weights",
            reduce_fcn=REDUCE_FUNCTIONS["Neg. entropy"],
        )
        self.assertAlmostEqual(result.params["slope"]["value"], 2.0, delta=0.5)

    def test_cauchy_reduce_is_outlier_robust(self):
        from curvelab.fit_manager import REDUCE_FUNCTIONS
        fm, x, y = self._linear_with_outlier()
        result = fm.run_fit(
            x, y, weight_mode="No weights",
            reduce_fcn=REDUCE_FUNCTIONS["Cauchy log-pdf"],
        )
        # Robust loss should keep the slope near truth despite the outlier.
        self.assertAlmostEqual(result.params["slope"]["value"], 2.0, delta=0.3)


class WeightModeCoverageTests(unittest.TestCase):
    """The _compute_weights modes not covered by the existing tests."""

    def test_one_over_yerr_squared(self):
        from curvelab.fit_manager import FitManager
        yerr = np.array([0.5, 0.25, 1.0])
        w = FitManager._compute_weights(np.array([1.0, 2, 3]), yerr, "1/yerr²")
        np.testing.assert_allclose(w, 1.0 / (yerr ** 2))

    def test_yerr_as_weights(self):
        from curvelab.fit_manager import FitManager
        yerr = np.array([0.5, 0.25, 1.0])
        w = FitManager._compute_weights(np.array([1.0, 2, 3]), yerr, "yerr as weights")
        np.testing.assert_allclose(w, yerr)

    def test_default_mode_without_yerr_returns_none(self):
        from curvelab.fit_manager import FitManager
        w = FitManager._compute_weights(np.array([1.0, 2, 3]), None, "1/yerr (default)")
        self.assertIsNone(w)


if __name__ == "__main__":
    unittest.main()
