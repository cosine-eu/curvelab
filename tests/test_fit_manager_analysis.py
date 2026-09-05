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


class ConfidenceAfterOtherObjectivesTests(unittest.TestCase):
    """lmfit's CI tools re-minimize with leastsq and forward the stored
    fit keywords to scipy. After a robust-loss or emcee fit those keywords
    (loss, f_scale, is_weighted, steps, ...) made leastsq raise, and the
    raise left parameters with vary=False. ci_minimizer hands the tools a
    copy with the keywords cleared."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(1)
        self.x = np.linspace(0, 10, 60)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 0.1, 60)
        self.y[30] += 20.0
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)

    def _fit_cauchy(self):
        from curvelab.fit_manager import objective_kwargs
        self.fm.run_fit(self.x, self.y, weight_mode="No weights", method="least_squares",
                        objective_kws=objective_kwargs("least_squares", "Cauchy", 1.0))

    def test_ci_and_profiles_after_robust_loss(self):
        self._fit_cauchy()
        text = self.fm.compute_confidence_intervals()
        self.assertIn("slope", text)
        profiles = self.fm.compute_ci_profiles()
        self.assertEqual(set(profiles), {"slope", "intercept"})
        # The tools must not disturb the parameters or the stored result.
        self.assertTrue(all(p.vary for p in self.fm.params.values()))
        self.assertEqual(self.fm._last_result.kws, {"loss": "cauchy", "f_scale": 1.0})

    def test_contour_grid_after_robust_loss(self):
        self._fit_cauchy()
        _, _, dchi2, levels = self.fm.compute_ci_contour("slope", "intercept", nx=5, ny=5)
        self.assertEqual(dchi2.shape, (5, 5))
        self.assertAlmostEqual(float(dchi2.min()), 0.0, places=6)
        self.assertEqual(list(levels), ["1\u03c3", "2\u03c3", "3\u03c3"])

    def test_joint_thresholds_are_two_parameter_levels(self):
        # 1-sigma for two parameters jointly is delta-chi2 = 2.30, not 1;
        # with scale_covar on the levels shrink by about chi2/nu.
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.1, 30)
        yerr = np.full(30, 0.1)
        self.fm.run_fit(x, y, yerr=yerr, scale_covar=False)
        lv = self.fm.joint_ci_thresholds()
        self.assertAlmostEqual(lv["1\u03c3"], 2.2957, places=3)
        self.assertAlmostEqual(lv["2\u03c3"], 6.1801, places=3)
        self.assertAlmostEqual(lv["3\u03c3"], 11.829, places=2)
        self.fm.run_fit(x, y, yerr=yerr, scale_covar=True)
        redchi = self.fm._last_result.redchi
        scaled = self.fm.joint_ci_thresholds()
        self.assertAlmostEqual(scaled["1\u03c3"] / 2.2957, redchi, delta=0.15 * redchi)

    def test_ci_after_emcee(self):
        try:
            import emcee  # noqa: F401
        except ImportError:
            self.skipTest("emcee not installed")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fm.run_fit(self.x, self.y, weight_mode="No weights", method="emcee",
                            fit_kws={"is_weighted": False, "steps": 60,
                                     "nwalkers": 12, "burn": 10})
        # A posterior sample has no chi-square minimum to profile from; the
        # user is pointed at the posterior errors instead of a cryptic
        # root-finding failure.
        with self.assertRaises(ValueError) as cm:
            self.fm.compute_confidence_intervals()
        self.assertIn("emcee", str(cm.exception))
        self.assertTrue(all(p.vary for p in self.fm.params.values() if p.name != "__lnsigma"))

    def test_ci_minimizer_raises_before_fit(self):
        with self.assertRaises(ValueError):
            self.fm.ci_minimizer()


class ConfidenceIntervalScalingTests(unittest.TestCase):
    """The profile chi-square axis was synthesized with chi2.ppf from lmfit's
    F-test probability, so the 'chi2_min + 1' line met the CI by
    construction whatever chi2/nu was; and the CIs were always F-test
    (reduced-chi-square scaled) even with Scale covariance off, so CI and
    StdErr disagreed. Both now follow the fit's scale_covar and the profile
    chi-square is the real re-minimized value."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = np.linspace(0, 10, 8)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 0.05, 8)
        self.yerr = np.full(8, 0.3)                 # overestimated: chi2/nu ~ 0.01

    def _fit(self, scale_covar):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(self.x, self.y)
        fm.run_fit(self.x, self.y, yerr=self.yerr, scale_covar=scale_covar)
        return fm

    def _ci_halfwidth(self, fm):
        text = fm.compute_confidence_intervals(sigmas=[1])
        line = next(l for l in text.splitlines() if l.strip().startswith("slope"))
        lo, best, hi = (float(v) for v in line.split(":")[1].split())
        return -lo, hi

    def test_profile_chi2_equals_a_direct_refit(self):
        from curvelab.fit_manager import FitManager
        for scale_covar in (True, False):
            fm = self._fit(scale_covar)
            pts = fm.compute_ci_profiles(sigmas=[1])["slope"]
            value, chi2_profile = max(pts, key=lambda t: t[1])
            direct = FitManager()
            direct.add_component("Linear")
            direct.set_param_hint("slope", value=value, vary=False)
            direct.set_param_hint("intercept", value=1.0)
            chi2_direct = direct.run_fit(self.x, self.y, yerr=self.yerr).gof["chi-squared"]
            self.assertAlmostEqual(chi2_profile, chi2_direct, places=6)

    def test_unscaled_ci_matches_stderr(self):
        fm = self._fit(scale_covar=False)
        se = fm._last_result.params["slope"].stderr
        lo, hi = self._ci_halfwidth(fm)
        self.assertAlmostEqual(lo, se, places=4)
        self.assertAlmostEqual(hi, se, places=4)
        self.assertAlmostEqual(fm.ci_threshold(1.0), 1.0, places=9)
        self.assertIn("unscaled", fm.compute_confidence_intervals(sigmas=[1]))

    def test_scaled_ci_is_reduced_chi2_scaled(self):
        fm = self._fit(scale_covar=True)
        r = fm._last_result
        lo, hi = self._ci_halfwidth(fm)
        # ~ stderr (which is chi2_nu-scaled), and far below the unscaled width.
        self.assertAlmostEqual(lo, r.params["slope"].stderr, delta=0.15 * r.params["slope"].stderr)
        self.assertLess(hi, 0.02)
        thr = fm.ci_threshold(1.0)
        self.assertLess(thr, 0.05)                  # ~ chi2_nu, not 1
        self.assertGreater(thr, 0.0)
        self.assertIn("scaled by", fm.compute_confidence_intervals(sigmas=[1]))


class EmceeTests(unittest.TestCase):
    """emcee ran with is_weighted taken from the yerr column rather than the
    weights in use, started every walker at value*(1+1e-4 N) so a zero
    parameter could not move, reset to the start values instead of sampling
    around the least-squares optimum, and reported none of its diagnostics."""

    def setUp(self):
        try:
            import emcee  # noqa: F401
        except ImportError:
            self.skipTest("emcee not installed")
        import warnings
        warnings.simplefilter("ignore")
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(1)
        self.x = np.linspace(-5, 5, 60)
        self.y = 2.0 * self.x + rng.normal(0, 0.1, 60)          # intercept is 0
        self.yerr = np.full(60, 0.1)
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)
        self.fm.set_param("intercept", value=0.0)
        self.kws = {"steps": 200, "nwalkers": 16, "burn": 40, "seed": 0}

    def _emcee(self, **run_kw):
        return self.fm.run_fit(self.x, self.y, method="emcee", fit_kws=dict(self.kws), **run_kw)

    def test_is_weighted_follows_the_weights_not_the_yerr_column(self):
        from curvelab.fit_manager import WEIGHT_NONE
        r = self._emcee(yerr=self.yerr, weight_mode=WEIGHT_NONE)
        self.assertFalse(r.mcmc["is_weighted"])
        self.assertIn("__lnsigma", r.flatchain.columns)
        r = self._emcee(yerr=self.yerr)
        self.assertTrue(r.mcmc["is_weighted"])
        self.assertNotIn("__lnsigma", r.flatchain.columns)

    def test_zero_valued_parameter_is_sampled(self):
        r = self._emcee(weight_mode="No weights")
        self.assertGreater(r.flatchain["intercept"].std(), 0.01)

    def test_seed_reproduces_the_chain(self):
        # emcee samples around the current values, so put the parameters
        # at the same (least-squares) point before each run.
        self.fm.run_fit(self.x, self.y, weight_mode="No weights")
        a = self._emcee(weight_mode="No weights")
        self.fm.run_fit(self.x, self.y, weight_mode="No weights")
        b = self._emcee(weight_mode="No weights")
        np.testing.assert_array_equal(a.flatchain["slope"].values, b.flatchain["slope"].values)

    def test_emcee_starts_from_the_current_values_not_the_start(self):
        # After a least-squares fit the parameters sit at the optimum; emcee
        # must sample around that, not reset to the auto-guess.
        ls = self.fm.run_fit(self.x, self.y, weight_mode="No weights")
        r = self._emcee(weight_mode="No weights")
        self.assertAlmostEqual(r.init_params["slope"], ls.params["slope"]["value"], places=6)

    def test_diagnostics_reported(self):
        r = self._emcee(weight_mode="No weights")
        m = r.mcmc
        self.assertEqual((m["steps"], m["nwalkers"], m["burn"], m["seed"]), (200, 16, 40, 0))
        self.assertIsNotNone(m["acceptance_fraction"])
        self.assertIn("[[MCMC (emcee)]]", r.report)
        self.assertIn("posterior medians", r.report)
        self.assertIn("acceptance fraction", r.report)
        self.assertIn("__lnsigma is the fitted log of the noise scale", r.report)


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

    def test_odr_refuses_to_run_without_xerr(self):
        # odrpack would silently take sigma_x = 1 and absorb the residuals
        # into x (chi2 far below the least-squares value, meaningless errors).
        with self.assertRaises(ValueError) as cm:
            self.fm.run_odr(self.x, self.y, yerr=self.yerr, xerr=None)
        self.assertIn("x-errors", str(cm.exception))

    def test_odr_band_uses_full_parameter_covariance(self):
        # For a line fitted away from x = 0, slope and intercept are almost
        # perfectly anticorrelated. Propagating only the diagonal (sd_beta)
        # ignores that and gives a band several times too wide.
        from curvelab.fit_manager import FitManager
        from odrpack import odr_fit
        rng = np.random.default_rng(0)
        x = np.linspace(10, 20, 20)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.5, 20)
        yerr = np.full(20, 0.5)
        xerr = np.full(20, 0.1)
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        result = fm.run_odr(x, y, yerr=yerr, xerr=xerr)

        ref = odr_fit(lambda xx, b: b[0] * xx + b[1], x, y, np.array([2.0, 1.0]),
                      weight_y=1 / yerr ** 2, weight_x=1 / xerr ** 2)
        cov = ref.cov_beta * ref.res_var
        x0 = 15.0
        expected = np.sqrt(cov[0, 0] * x0 ** 2 + 2 * cov[0, 1] * x0 + cov[1, 1])
        diagonal_only = np.sqrt(cov[0, 0] * x0 ** 2 + cov[1, 1])

        i = np.argmin(np.abs(result.x_dense - x0))
        self.assertAlmostEqual(result.y_uncertainty[i], expected, places=4)
        self.assertLess(result.y_uncertainty[i], diagonal_only / 3)

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

    def test_objective_and_scale_covar_are_honored(self):
        # The Objective and Scale-covariance controls used to be dropped on
        # the way into run_global_fit while the panel showed them enabled.
        from curvelab.fit_manager import objective_kwargs, WEIGHT_NONE
        y1 = self.y1.copy()
        y1[20] += 30.0                                # gross outlier
        ds = [(self.x1, y1, None, None), (self.x2, self.y2, None, None)]
        plain = self.fm.run_global_fit(
            ds, {"slope"}, method="least_squares", weight_mode=WEIGHT_NONE,
            objective_kws=objective_kwargs("least_squares", "Least squares"))
        robust = self.fm.run_global_fit(
            ds, {"slope"}, method="least_squares", weight_mode=WEIGHT_NONE,
            objective_kws=objective_kwargs("least_squares", "Cauchy", 1.0))
        # The outlier drags the plain intercept up; the Cauchy loss ignores it.
        self.assertGreater(plain[0].params["intercept"]["value"], 2.5)
        self.assertAlmostEqual(robust[0].params["intercept"]["value"], 2.0, delta=0.2)

        yerr = np.full_like(self.y1, 0.1)
        ds = [(self.x1, self.y1, yerr, None), (self.x2, self.y2, yerr, None)]
        scaled = self.fm.run_global_fit(ds, {"slope"}, scale_covar=True)
        raw = self.fm.run_global_fit(ds, {"slope"}, scale_covar=False)
        self.assertNotAlmostEqual(scaled[0].params["slope"]["stderr"],
                                  raw[0].params["slope"]["stderr"], places=6)

    def test_global_fit_starts_from_the_start_values(self):
        ds = [(self.x1, self.y1, None, None), (self.x2, self.y2, None, None)]
        r1 = self.fm.run_global_fit(ds, {"slope"})
        r2 = self.fm.run_global_fit(ds, {"slope"})
        # Not chained: both fits report the same initial values.
        self.assertEqual(r1[0].init_params, r2[0].init_params)

    def test_mixed_yerr_is_refused_for_yerr_weight_modes(self):
        # One dataset with sigma = 0.01 and one without yerr: the unweighted
        # residuals are in y units next to 1/sigma ones, the first dataset
        # dominates completely and chi2 mixes units. Refuse, don't guess.
        yerr1 = np.full_like(self.y1, 0.01)
        with self.assertRaises(ValueError) as cm:
            self.fm.run_global_fit(
                [(self.x1, self.y1, yerr1, None), (self.x2, self.y2, None, None)],
                shared_params={"slope"},
            )
        self.assertIn("dataset(s) [2] of 2", str(cm.exception))

    def test_mixed_yerr_is_fine_without_weights(self):
        from curvelab.fit_manager import WEIGHT_NONE
        yerr1 = np.full_like(self.y1, 0.01)
        results = self.fm.run_global_fit(
            [(self.x1, self.y1, yerr1, None), (self.x2, self.y2, None, None)],
            shared_params={"slope"}, weight_mode=WEIGHT_NONE,
        )
        self.assertEqual(len(results), 2)

    def test_report_and_gof_present(self):
        results = self.fm.run_global_fit(
            [(self.x1, self.y1, None, None), (self.x2, self.y2, None, None)],
            shared_params={"slope"},
        )
        self.assertIn("Global Fit", results[0].report)
        self.assertIn("[shared]", results[0].report)
        self.assertIn("reduced chi-squared", results[0].gof)


class GlobalFitDerivedParamTests(unittest.TestCase):
    """Derived parameters (fwhm = 2.3548*sigma) were copied into the
    per-dataset s{i}_ namespace with their expression verbatim, so any peak
    model whose sigma was not shared raised NameError('sigma'). A derived
    parameter now follows its dependencies: shared when they all are,
    otherwise per dataset with a rewritten expression."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        self.x = np.linspace(-10, 10, 200)
        self.y1 = 3.0 * np.exp(-self.x ** 2 / (2 * 2.0 ** 2))
        self.y2 = 5.0 * np.exp(-self.x ** 2 / (2 * 1.0 ** 2))
        self.fm = FitManager()
        self.fm.add_component("Gaussian")
        self.fm.auto_guess(self.x, self.y1)
        self.datasets = [(self.x, self.y1, None, None), (self.x, self.y2, None, None)]

    def test_unshared_sigma_gives_per_dataset_fwhm(self):
        # The dialog ticks every parameter by default, derived ones included.
        results = self.fm.run_global_fit(
            self.datasets, shared_params={"center", "fwhm", "height"},
        )
        for res, sigma_true, amp_true in zip(results, (2.0, 1.0), (3.0, 5.0)):
            sigma = res.params["sigma"]["value"]
            self.assertAlmostEqual(sigma, sigma_true, places=3)
            self.assertAlmostEqual(res.params["fwhm"]["value"], 2.3548200 * sigma, places=5)
            self.assertAlmostEqual(res.params["height"]["value"], amp_true, places=3)
            # The report shows the user's expression, not the s0_/s1_ one.
            self.assertEqual(res.params["fwhm"]["expr"], "2.3548200*sigma")

    def test_shared_sigma_gives_shared_fwhm(self):
        results = self.fm.run_global_fit(
            self.datasets, shared_params={"center", "sigma"},
        )
        f1 = results[0].params["fwhm"]["value"]
        f2 = results[1].params["fwhm"]["value"]
        self.assertEqual(f1, f2)
        self.assertAlmostEqual(f1, 2.3548200 * results[0].params["sigma"]["value"], places=6)


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

    def test_cauchy_reduce_is_outlier_robust(self):
        from curvelab.fit_manager import REDUCE_FUNCTIONS
        fm, x, y = self._linear_with_outlier()
        result = fm.run_fit(
            x, y, weight_mode="No weights",
            reduce_fcn=REDUCE_FUNCTIONS["Cauchy log-pdf"],
        )
        # Robust loss should keep the slope near truth despite the outlier.
        self.assertAlmostEqual(result.params["slope"]["value"], 2.0, delta=0.3)


class NonChiSquareObjectiveGofTests(unittest.TestCase):
    """lmfit's chisqr/aic/bic are the plain sum of squares whatever the
    objective. After a robust or reduce-function fit they are not what was
    minimized, and AIC/BIC would rank the outlier-dominated chi-square fit
    above the robust one. AIC/BIC are left out and the minimized value is
    reported as 'objective'."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(1)
        self.x = np.linspace(0, 10, 60)
        self.y = 2.0 * self.x + 1.0 + rng.normal(0, 0.1, 60)
        self.y[30] += 20.0
        self.fm = FitManager()
        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)

    def test_robust_loss_reports_objective_not_aic(self):
        from scipy.optimize import least_squares
        from curvelab.fit_manager import objective_kwargs, WEIGHT_NONE
        r = self.fm.run_fit(self.x, self.y, weight_mode=WEIGHT_NONE, method="least_squares",
                            objective_kws=objective_kwargs("least_squares", "Cauchy", 1.0))
        self.assertNotIn("AIC", r.gof)
        self.assertNotIn("BIC", r.gof)
        self.assertIn("chi-squared", r.gof)          # descriptive, still shown
        ref = least_squares(lambda b: b[0] * self.x + b[1] - self.y, [2.0, 1.0],
                            loss="cauchy", f_scale=1.0)
        self.assertAlmostEqual(r.gof["objective"], ref.cost, places=3)

    def test_reduce_fcn_reports_its_value(self):
        from curvelab.fit_manager import objective_kwargs, WEIGHT_NONE, _reduce_cauchylogpdf
        r = self.fm.run_fit(self.x, self.y, weight_mode=WEIGHT_NONE, method="nelder",
                            objective_kws=objective_kwargs("nelder", "Cauchy log-pdf"))
        self.assertNotIn("AIC", r.gof)
        resid = r.y_data - r.y_fit_data
        self.assertAlmostEqual(r.gof["objective"], _reduce_cauchylogpdf(resid), places=6)

    def test_chi_square_fit_keeps_aic_and_has_no_objective_key(self):
        from curvelab.fit_manager import WEIGHT_NONE
        r = self.fm.run_fit(self.x, self.y, weight_mode=WEIGHT_NONE)
        self.assertIn("AIC", r.gof)
        self.assertNotIn("objective", r.gof)


class WeightedResidualTests(unittest.TestCase):
    """'Weighted residuals' divided by yerr whatever weight mode had been
    used, and the diagnostics used raw residuals for weighted fits. The
    result now records the weights the fit applied, and weighted residuals
    are residual * weight for every mode."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = np.linspace(1, 10, 40)
        self.y = 5.0 * self.x + rng.normal(0, 0.05, 40) * self.x    # sigma ~ y
        self.yerr = 0.05 * self.y

    def _fm(self):
        from curvelab.fit_manager import FitManager
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(self.x, self.y)
        return fm

    def test_default_mode_matches_residual_over_yerr(self):
        r = self._fm().run_fit(self.x, self.y, yerr=self.yerr)
        self.assertTrue(r.is_weighted())
        np.testing.assert_allclose(r.weighted_residuals(), r.residuals() / self.yerr)

    def test_inverse_y_mode_divides_by_y_not_yerr(self):
        from curvelab.fit_manager import WEIGHT_INV_Y
        r = self._fm().run_fit(self.x, self.y, yerr=self.yerr, weight_mode=WEIGHT_INV_Y)
        np.testing.assert_allclose(r.weighted_residuals(), r.residuals() / np.abs(self.y))
        # Not residual / yerr, which is what the old helper would have given.
        self.assertFalse(np.allclose(r.weighted_residuals(), r.residuals() / self.yerr))

    def test_unweighted_fit_is_not_weighted(self):
        from curvelab.fit_manager import WEIGHT_NONE
        r = self._fm().run_fit(self.x, self.y, yerr=self.yerr, weight_mode=WEIGHT_NONE)
        self.assertFalse(r.is_weighted())
        np.testing.assert_allclose(r.weighted_residuals(), r.residuals())

    def test_weighted_residuals_are_homoscedastic(self):
        # The point of the change: with sigma proportional to y, raw
        # residuals grow with x while weighted ones do not.
        r = self._fm().run_fit(self.x, self.y, yerr=self.yerr)
        raw, w = np.abs(r.residuals()), np.abs(r.weighted_residuals())
        half = len(self.x) // 2
        self.assertGreater(raw[half:].mean(), 1.5 * raw[:half].mean())
        self.assertLess(w[half:].mean(), 1.5 * w[:half].mean())

    def test_old_result_without_weights_falls_back_to_yerr(self):
        from curvelab.session import FitResult
        x = np.linspace(0, 1, 5)
        r = FitResult(x_dense=x, y_fit_dense=x, x_data=x, y_data=x + 0.1, y_fit_data=x,
                      yerr_data=np.full(5, 0.5), params={}, report="")
        self.assertTrue(r.is_weighted())
        np.testing.assert_allclose(r.weighted_residuals(), 0.2)


class KnownSigmaStderrTests(unittest.TestCase):
    """Pins the weight convention end to end: with weights = 1/sigma and
    scale_covar off, the reported stderr must be the analytic one."""

    def test_slope_stderr_matches_the_analytic_value(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 50)
        sigma = 0.3
        y = 2.0 * x + 1.0 + rng.normal(0, sigma, 50)
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        r = fm.run_fit(x, y, yerr=np.full(50, sigma), scale_covar=False)
        # For a straight line with equal sigma: var(slope) = sigma^2 / sum((x - xbar)^2)
        analytic = sigma / np.sqrt(np.sum((x - x.mean()) ** 2))
        self.assertAlmostEqual(r.params["slope"]["stderr"], analytic, places=8)
        # With scaling on, the stderr is multiplied by sqrt(chi2/nu) instead.
        rs = fm.run_fit(x, y, yerr=np.full(50, sigma), scale_covar=True)
        self.assertAlmostEqual(rs.params["slope"]["stderr"],
                               analytic * np.sqrt(rs.gof["reduced chi-squared"]), places=8)


class WeightedRSquaredTests(unittest.TestCase):
    """lmfit 1.2.x computed R^2 from the weighted residuals over the
    unweighted total sum of squares, so a perfect fit with small sigma gave a
    large negative R^2. lmfit 1.3 uses raw residuals; this pins that."""

    def test_weighted_perfect_fit_has_r_squared_near_one(self):
        from curvelab.fit_manager import FitManager
        rng = np.random.default_rng(0)
        x = np.linspace(0, 10, 30)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.005, 30)
        fm = FitManager()
        fm.add_component("Linear")
        fm.auto_guess(x, y)
        r = fm.run_fit(x, y, yerr=np.full(30, 0.01))
        self.assertGreater(r.gof["R-squared"], 0.999)


class WeightModeCoverageTests(unittest.TestCase):
    """The _compute_weights modes not covered by the existing tests."""

    def test_only_chi_square_consistent_modes_remain(self):
        # "1/yerr^2" and "yerr as weights" minimized sum r^2/sigma^4 and
        # sum (sigma r)^2; stderr, chi2 and AIC from them were meaningless.
        from curvelab.fit_manager import WEIGHT_MODES
        self.assertEqual(WEIGHT_MODES, ["1/yerr (default)", "1/y", "No weights",
                                        "Effective variance"])

    def test_default_mode_without_yerr_returns_none(self):
        from curvelab.fit_manager import FitManager
        w = FitManager._compute_weights(np.array([1.0, 2, 3]), None, "1/yerr (default)")
        self.assertIsNone(w)


if __name__ == "__main__":
    unittest.main()
