"""Tests for FitManager fitting, preprocessing, workspace round-trip, and bootstrap."""

import json
import unittest

import numpy as np


class FitManagerRunFitTests(unittest.TestCase):
    """Gap 1: Test that run_fit recovers known parameters."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        self.fm = FitManager()

    def test_gaussian_fit_recovers_params(self):
        rng = np.random.default_rng(42)
        x = np.linspace(-5, 5, 200)
        y = 3.0 * np.exp(-0.5 * ((x - 0.5) / 0.8) ** 2) + rng.normal(0, 0.05, 200)

        self.fm.add_component("Gaussian")
        self.fm.auto_guess(x, y)
        result = self.fm.run_fit(x, y, weight_mode="No weights")

        center = result.params["center"]["value"]
        self.assertAlmostEqual(center, 0.5, delta=0.15)
        self.assertGreater(result.gof["R-squared"], 0.99)

    def test_linear_fit_recovers_slope_intercept(self):
        rng = np.random.default_rng(123)
        x = np.linspace(0, 10, 50)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.1, 50)

        self.fm.add_component("Linear")
        self.fm.auto_guess(x, y)
        result = self.fm.run_fit(x, y, weight_mode="No weights")

        slope = result.params["slope"]["value"]
        intercept = result.params["intercept"]["value"]
        self.assertAlmostEqual(slope, 2.5, delta=0.1)
        self.assertAlmostEqual(intercept, 1.0, delta=0.3)

    def test_fit_with_yerr_weights(self):
        rng = np.random.default_rng(99)
        x = np.linspace(-3, 3, 100)
        y = 5.0 * np.exp(-0.5 * ((x - 0.0) / 1.0) ** 2) + rng.normal(0, 0.1, 100)
        yerr = np.full(100, 0.1)

        self.fm.add_component("Gaussian")
        self.fm.auto_guess(x, y)
        result = self.fm.run_fit(x, y, yerr=yerr, weight_mode="1/yerr (default)")

        self.assertIsNotNone(result.params["center"]["stderr"])
        self.assertAlmostEqual(result.params["center"]["value"], 0.0, delta=0.2)

    def test_fit_result_has_gof_fields(self):
        x = np.linspace(0, 5, 50)
        y = 2.0 * x + 1.0

        self.fm.add_component("Linear")
        self.fm.auto_guess(x, y)
        result = self.fm.run_fit(x, y, weight_mode="No weights")

        for key in ("chi-squared", "reduced chi-squared", "R-squared", "AIC", "BIC"):
            self.assertIn(key, result.gof)
            self.assertIsNotNone(result.gof[key])

    def test_fit_result_has_dense_curves(self):
        x = np.linspace(0, 5, 30)
        y = np.exp(-x)

        self.fm.add_component("Exponential")
        self.fm.auto_guess(x, y)
        result = self.fm.run_fit(x, y, weight_mode="No weights", n_dense=200)

        self.assertEqual(len(result.x_dense), 200)
        self.assertEqual(len(result.y_fit_dense), 200)
        self.assertEqual(len(result.y_fit_data), len(x))

    def test_evaluate_after_fit(self):
        x = np.linspace(0, 10, 50)
        y = 3.0 * x + 2.0

        self.fm.add_component("Linear")
        self.fm.auto_guess(x, y)
        self.fm.run_fit(x, y, weight_mode="No weights")

        x_eval = np.array([0.0, 5.0, 10.0])
        y_eval = self.fm.evaluate(x_eval)

        np.testing.assert_allclose(y_eval, 3.0 * x_eval + 2.0, atol=0.1)

    def test_composite_model_fit(self):
        """Test fitting a Gaussian + Linear composite model."""
        rng = np.random.default_rng(7)
        x = np.linspace(0, 10, 200)
        y = (2.0 * np.exp(-0.5 * ((x - 5.0) / 0.5) ** 2)
             + 0.3 * x + 1.0
             + rng.normal(0, 0.05, 200))

        self.fm.add_component("Gaussian")
        self.fm.add_component("Linear")
        self.fm.auto_guess(x, y)
        # Provide reasonable initial guesses for the composite model
        self.fm.set_param("gaussian1_center", value=5.0)
        self.fm.set_param("gaussian1_sigma", value=0.5)
        self.fm.set_param("gaussian1_amplitude", value=2.0)
        self.fm.set_param("linear1_slope", value=0.3)
        self.fm.set_param("linear1_intercept", value=1.0)
        result = self.fm.run_fit(x, y, weight_mode="No weights")

        self.assertGreater(result.gof["R-squared"], 0.99)
        self.assertIn("component_curves", dir(result))
        self.assertGreater(len(result.component_curves), 0)


class WorkspaceRoundTripTests(unittest.TestCase):
    """Gap 2: Test serialize -> deserialize round-trip."""

    def test_series_record_round_trip_no_fit(self):
        from curvelab.session import SeriesRecord
        from curvelab.workspace import (
            WorkspaceEncoder, serialize_series_records,
            deserialize_series_record, decode_workspace,
        )

        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        yerr = np.full(50, 0.1)
        mask = np.ones(50, dtype=bool)
        mask[5] = False
        mask[10] = False

        rec = SeriesRecord(
            x=x, y=y, yerr=yerr,
            style={"x": "x", "y": "y", "yerr": "yerr"},
            dataset_name="test_ds",
            mask=mask,
        )

        series = serialize_series_records({"sid1": rec})

        # Round-trip through JSON
        json_str = json.dumps(series, cls=WorkspaceEncoder)
        loaded = json.loads(json_str, object_hook=decode_workspace)

        sdata = loaded["sid1"]
        rec2 = deserialize_series_record(sdata, x, y, yerr, None, "test_ds")

        np.testing.assert_array_equal(rec2.x, x)
        np.testing.assert_array_equal(rec2.y, y)
        np.testing.assert_array_equal(rec2.yerr, yerr)
        np.testing.assert_array_equal(rec2.mask, mask)
        self.assertEqual(rec2.dataset_name, "test_ds")

    def test_series_record_round_trip_with_fit(self):
        from curvelab.fit_manager import FitManager
        from curvelab.session import SeriesRecord, FitSession
        from curvelab.workspace import (
            WorkspaceEncoder, serialize_series_records,
            deserialize_series_record, decode_workspace,
        )

        x = np.linspace(-3, 3, 100)
        y = 2.0 * np.exp(-0.5 * (x / 1.0) ** 2)

        fm = FitManager()
        fm.add_component("Gaussian")
        fm.auto_guess(x, y)
        result = fm.run_fit(x, y, weight_mode="No weights")

        rec = SeriesRecord(
            x=x, y=y,
            style={"x": "x", "y": "y"},
            dataset_name="test",
        )
        sess = FitSession(name="fit1", fit_manager=fm, result=result)
        rec.fit_sessions["fit1"] = sess

        series = serialize_series_records({"sid": rec})
        json_str = json.dumps(series, cls=WorkspaceEncoder)
        loaded = json.loads(json_str, object_hook=decode_workspace)

        rec2 = deserialize_series_record(loaded["sid"], x, y, None, None, "test")

        self.assertIn("fit1", rec2.fit_sessions)
        sess2 = rec2.fit_sessions["fit1"]
        self.assertIsNotNone(sess2.result)
        self.assertEqual(len(sess2.result.x_dense), len(result.x_dense))
        np.testing.assert_allclose(sess2.result.y_fit_dense, result.y_fit_dense, atol=1e-10)
        self.assertAlmostEqual(
            sess2.result.gof["R-squared"],
            result.gof["R-squared"],
            places=6,
        )

    def test_special_float_round_trip(self):
        from curvelab.workspace import encode_value, decode_workspace

        data = {
            "min": float("-inf"),
            "max": float("inf"),
            "value": 3.14,
        }
        encoded = encode_value(data)
        json_str = json.dumps(encoded)
        decoded = json.loads(json_str, object_hook=decode_workspace)

        self.assertEqual(decoded["min"], float("-inf"))
        self.assertEqual(decoded["max"], float("inf"))
        self.assertAlmostEqual(decoded["value"], 3.14)

    def test_ndarray_round_trip(self):
        from curvelab.workspace import WorkspaceEncoder, decode_workspace

        arr = np.array([1.0, 2.0, 3.0])
        json_str = json.dumps(arr, cls=WorkspaceEncoder)
        loaded = json.loads(json_str, object_hook=decode_workspace)

        np.testing.assert_array_equal(loaded, arr)


class BootstrapTests(unittest.TestCase):
    """Gap 3: Test bootstrap CI produces reasonable distributions."""

    def setUp(self):
        from curvelab.fit_manager import FitManager
        self.fm = FitManager()
        self.rng = np.random.default_rng(42)
        self.x = np.linspace(0, 10, 50)
        self.y = 2.0 * self.x + 1.0 + self.rng.normal(0, 0.3, 50)

        self.fm.add_component("Linear")
        self.fm.auto_guess(self.x, self.y)
        self.fm.run_fit(self.x, self.y, weight_mode="No weights")

    def test_residual_bootstrap_returns_distributions(self):
        dists, n_failed = self.fm.run_bootstrap(
            self.x, self.y, n_boot=50, boot_type="residual",
            weight_mode="No weights",
        )
        self.assertIn("slope", dists)
        self.assertIn("intercept", dists)
        self.assertGreater(len(dists["slope"]), 30)  # most should succeed
        self.assertEqual(len(dists["slope"]), 50 - n_failed)

    def test_case_bootstrap_returns_distributions(self):
        dists, n_failed = self.fm.run_bootstrap(
            self.x, self.y, n_boot=50, boot_type="case",
            weight_mode="No weights",
        )
        self.assertIn("slope", dists)
        self.assertGreater(len(dists["slope"]), 30)
        self.assertEqual(len(dists["slope"]), 50 - n_failed)

    def test_bootstrap_slope_near_true_value(self):
        dists, _n_failed = self.fm.run_bootstrap(
            self.x, self.y, n_boot=100, boot_type="residual",
            weight_mode="No weights",
        )
        mean_slope = np.mean(dists["slope"])
        self.assertAlmostEqual(mean_slope, 2.0, delta=0.3)


class PreprocessingTests(unittest.TestCase):
    """Gap 4: Test data preprocessing guard-rails."""

    def _make_rec(self, x, y, yerr=None, xerr=None, mask=None):
        from curvelab.session import SeriesRecord
        return SeriesRecord(x=x, y=y, yerr=yerr, xerr=xerr, mask=mask)

    def test_nan_removal(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, np.nan, 3.0, 4.0])
        rec = self._make_rec(x, y)

        xo, yo, _, _, warnings = prepare_fit_data(rec)
        self.assertEqual(len(xo), 3)
        self.assertEqual(len(warnings), 1)
        self.assertIn("1 point", warnings[0])

    def test_inf_removal(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, np.inf, 3.0])
        rec = self._make_rec(x, y)

        xo, yo, _, _, warnings = prepare_fit_data(rec)
        self.assertEqual(len(xo), 2)

    def test_sorting_by_x(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        rec = self._make_rec(x, y)

        xo, yo, _, _, _ = prepare_fit_data(rec)
        np.testing.assert_array_equal(xo, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(yo, [10.0, 20.0, 30.0])

    def test_x_range_filtering(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        rec = self._make_rec(x, y)

        xo, yo, _, _, _ = prepare_fit_data(rec, x_range=(2.0, 4.0))
        np.testing.assert_array_equal(xo, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(yo, [20.0, 30.0, 40.0])

    def test_mask_excludes_points(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mask = np.array([True, False, True, True])
        rec = self._make_rec(x, y, mask=mask)

        xo, yo, _, _, _ = prepare_fit_data(rec)
        np.testing.assert_array_equal(xo, [1.0, 3.0, 4.0])
        np.testing.assert_array_equal(yo, [10.0, 30.0, 40.0])

    def test_mask_and_range_combined(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mask = np.array([True, False, True, True, True])
        rec = self._make_rec(x, y, mask=mask)

        xo, yo, _, _, _ = prepare_fit_data(rec, x_range=(2.5, 4.5))
        np.testing.assert_array_equal(xo, [3.0, 4.0])

    def test_duplicate_x_warning(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 21.0, 30.0])
        rec = self._make_rec(x, y)

        _, _, _, _, warnings = prepare_fit_data(rec)
        self.assertEqual(len(warnings), 1)
        self.assertIn("duplicate", warnings[0].lower())

    def test_clean_data_no_warnings(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        rec = self._make_rec(x, y)

        _, _, _, _, warnings = prepare_fit_data(rec)
        self.assertEqual(len(warnings), 0)

    def test_yerr_xerr_follow_filtering(self):
        from curvelab.preprocessing import prepare_fit_data
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        yerr = np.array([0.3, 0.1, 0.2])
        xerr = np.array([0.03, 0.01, 0.02])
        rec = self._make_rec(x, y, yerr=yerr, xerr=xerr)

        xo, yo, yerro, xerro, _ = prepare_fit_data(rec)
        # Should be sorted by x
        np.testing.assert_array_equal(yerro, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(xerro, [0.01, 0.02, 0.03])


class SmoothOutlierLogicTests(unittest.TestCase):
    """Gap 5: Test smoothing and outlier detection numerical logic."""

    def test_savgol_window_clamped_odd(self):
        from scipy.signal import savgol_filter
        y = np.sin(np.linspace(0, 4 * np.pi, 50))

        # Window must be odd for savgol
        window = 10  # even
        if window % 2 == 0:
            window += 1
        result = savgol_filter(y, window, 3)
        self.assertEqual(len(result), len(y))

    def test_sigma_clipping_detects_outlier(self):
        """MAD-based sigma clipping should detect injected outliers."""
        rng = np.random.default_rng(42)
        y = np.zeros(100) + rng.normal(0, 0.1, 100)
        y_smooth = np.zeros(100)

        # Inject 2 obvious outliers
        y[20] = 10.0
        y[60] = -8.0

        residuals = y - y_smooth
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma_est = 1.4826 * mad
        threshold = 3.0
        inlier = np.abs(residuals - med) < threshold * sigma_est

        self.assertFalse(inlier[20])
        self.assertFalse(inlier[60])
        # Most points should be inliers
        self.assertGreater(inlier.sum(), 95)

    def test_mad_scaling_factor(self):
        """1.4826 * MAD should approximate std for Gaussian data."""
        rng = np.random.default_rng(99)
        data = rng.normal(0, 1.0, 10000)
        mad = np.median(np.abs(data - np.median(data)))
        sigma_est = 1.4826 * mad
        self.assertAlmostEqual(sigma_est, 1.0, delta=0.05)

    def test_all_smooth_methods_produce_output(self):
        """All four smoothing methods should return arrays of correct length."""
        from scipy.signal import savgol_filter, medfilt
        from scipy.ndimage import uniform_filter1d, gaussian_filter1d

        y = np.sin(np.linspace(0, 4 * np.pi, 100))

        r1 = savgol_filter(y, 11, 3)
        r2 = uniform_filter1d(y, size=11)
        r3 = medfilt(y, kernel_size=11)
        r4 = gaussian_filter1d(y, sigma=5)

        for r in (r1, r2, r3, r4):
            self.assertEqual(len(r), 100)


class DiagnosticStatsTests(unittest.TestCase):
    """compute_diagnostic_stats is a pure function extracted from the diagnostics dialog."""

    def test_random_residuals(self):
        from curvelab.ui_dialogs_analysis import compute_diagnostic_stats
        rng = np.random.default_rng(1)
        lines = compute_diagnostic_stats(rng.normal(0, 1, 200))
        self.assertTrue(any("no significant autocorrelation" in l for l in lines))
        self.assertTrue(any("consistent with random residuals" in l for l in lines))

    def test_systematic_misfit(self):
        from curvelab.ui_dialogs_analysis import compute_diagnostic_stats
        rng = np.random.default_rng(1)
        r = np.sin(np.linspace(0, 6 * np.pi, 200)) + rng.normal(0, 0.05, 200)
        lines = compute_diagnostic_stats(r)
        self.assertTrue(any("positive autocorrelation" in l for l in lines))
        self.assertTrue(any("non-random pattern" in l for l in lines))

    def test_degenerate_zero_residuals(self):
        from curvelab.ui_dialogs_analysis import compute_diagnostic_stats
        self.assertEqual(compute_diagnostic_stats(np.zeros(50)), [])


if __name__ == "__main__":
    unittest.main()
