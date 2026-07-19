"""Tests for the pure numeric helpers in curvelab.analysis_tools.

These functions were extracted from the Tk dialogs (SmoothOutlierDialog,
FindPeaksDialog, FTestDialog, EvaluateModelDialog) so they could be tested
without a display.
"""

import unittest

import numpy as np

from curvelab.analysis_tools import (
    SMOOTH_METHODS,
    detect_outliers,
    f_test,
    find_peaks_with_widths,
    parse_x_spec,
    smooth_data,
)


class SmoothDataTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = np.linspace(0, 10, 101)
        self.clean = np.sin(self.x)
        self.noisy = self.clean + rng.normal(0, 0.2, len(self.x))

    def test_all_methods_reduce_noise(self):
        for method in SMOOTH_METHODS:
            smoothed = smooth_data(self.noisy, method, window=11, order=3, sigma=3)
            self.assertEqual(len(smoothed), len(self.noisy), method)
            resid_before = np.std(self.noisy - self.clean)
            resid_after = np.std(smoothed - self.clean)
            self.assertLess(resid_after, resid_before, method)

    def test_unknown_method_returns_input(self):
        out = smooth_data(self.noisy, "No Such Method")
        np.testing.assert_array_equal(out, self.noisy)

    def test_short_array_returned_unchanged(self):
        y = np.array([1.0, 2.0])
        np.testing.assert_array_equal(smooth_data(y, "Savitzky-Golay"), y)

    def test_even_window_and_oversized_window_are_corrected(self):
        # Even window is bumped to odd; window larger than the data is clipped.
        y = np.sin(np.linspace(0, 3, 9))
        out = smooth_data(y, "Savitzky-Golay", window=100, order=3)
        self.assertEqual(len(out), len(y))
        out = smooth_data(y, "Median Filter", window=4)
        self.assertEqual(len(out), len(y))


class DetectOutliersTests(unittest.TestCase):
    def test_flags_planted_outliers(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 10, 200)
        y = np.sin(x) + rng.normal(0, 0.05, len(x))
        y[50] += 3.0
        y[150] -= 3.0
        smooth = lambda yy: smooth_data(yy, "Savitzky-Golay", window=11, order=3)
        inlier = detect_outliers(x, y, smooth(y), smooth, threshold=4.0, n_iter=2)
        self.assertFalse(inlier[50])
        self.assertFalse(inlier[150])
        # The overwhelming majority of points survive.
        self.assertGreater(inlier.sum(), 190)

    def test_clean_data_keeps_everything_at_loose_threshold(self):
        # Constant baseline so boundary smoothing artifacts can't inflate
        # edge residuals (a moving average flattens at the ends of a ramp).
        rng = np.random.default_rng(2)
        x = np.linspace(0, 10, 100)
        y = 5.0 + rng.normal(0, 0.01, len(x))
        smooth = lambda yy: smooth_data(yy, "Moving Average", window=11)
        inlier = detect_outliers(x, y, smooth(y), smooth, threshold=10.0, n_iter=1)
        self.assertTrue(inlier.all())


class FindPeaksTests(unittest.TestCase):
    def test_finds_two_gaussians_with_reasonable_widths(self):
        x = np.linspace(0, 10, 500)
        y = (np.exp(-((x - 3) ** 2) / (2 * 0.2 ** 2))
             + 0.8 * np.exp(-((x - 7) ** 2) / (2 * 0.3 ** 2)))
        peaks = find_peaks_with_widths(x, y)
        self.assertEqual(len(peaks), 2)
        centers = sorted(p[0] for p in peaks)
        self.assertAlmostEqual(centers[0], 3.0, delta=0.05)
        self.assertAlmostEqual(centers[1], 7.0, delta=0.05)
        # FWHM = 2.355 sigma
        widths = sorted(p[2] for p in peaks)
        self.assertAlmostEqual(widths[0], 2.355 * 0.2, delta=0.1)
        self.assertAlmostEqual(widths[1], 2.355 * 0.3, delta=0.1)

    def test_explicit_prominence_filters_small_peaks(self):
        x = np.linspace(0, 10, 500)
        y = (np.exp(-((x - 3) ** 2) / (2 * 0.2 ** 2))
             + 0.05 * np.exp(-((x - 7) ** 2) / (2 * 0.3 ** 2)))
        peaks = find_peaks_with_widths(x, y, prominence=0.5)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0][0], 3.0, delta=0.05)

    def test_flat_data_has_no_peaks(self):
        x = np.linspace(0, 10, 50)
        self.assertEqual(find_peaks_with_widths(x, np.ones(50)), [])


class FTestTests(unittest.TestCase):
    def test_known_value(self):
        # chi1=100 (p=2), chi2=50 (p=4), n=54: F = (50/2)/(50/50) = 25
        f_stat, p_value, df1, df2 = f_test(100.0, 2, 50.0, 4, 54)
        self.assertAlmostEqual(f_stat, 25.0)
        self.assertEqual((df1, df2), (2, 50))
        self.assertLess(p_value, 1e-6)

    def test_insignificant_improvement(self):
        f_stat, p_value, _, _ = f_test(100.0, 2, 99.0, 4, 54)
        self.assertGreater(p_value, 0.5)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            f_test(100.0, 4, 50.0, 4, 54)     # reduced not smaller
        with self.assertRaises(ValueError):
            f_test(50.0, 2, 100.0, 4, 54)     # full model worse
        with self.assertRaises(ValueError):
            f_test(100.0, 2, 50.0, 4, 4)      # no residual DOF


class ParseXSpecTests(unittest.TestCase):
    def test_range_syntax(self):
        x = parse_x_spec("0:10:11")
        np.testing.assert_allclose(x, np.linspace(0, 10, 11))

    def test_comma_and_space_separated(self):
        np.testing.assert_allclose(parse_x_spec("1, 2.5, 4"), [1.0, 2.5, 4.0])
        np.testing.assert_allclose(parse_x_spec("1 2.5 4"), [1.0, 2.5, 4.0])

    def test_empty_returns_none(self):
        self.assertIsNone(parse_x_spec(""))
        self.assertIsNone(parse_x_spec("   "))

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            parse_x_spec("1, banana, 3")


if __name__ == "__main__":
    unittest.main()
