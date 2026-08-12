"""The PAPR of a waveform, and the law it follows.

Two things moved into the library because every script that draws a PAPR
figure was reimplementing them: reducing an array of OFDM symbols to one
value per symbol (and then to one number), and the closed-form CCDF whose
effective-sample count is a fitted constant with a domain of validity.
A constant with a domain belongs where its domain can be documented and
checked, not in a tutorial.
"""
import logging
import unittest

import numpy as np

from comnumpy.ofdm.metrics import compute_papr, compute_papr_ccdf_theo


class TestReducingSeveralWaveforms(unittest.TestCase):

    def setUp(self):
        # two OFDM symbols, one peakier than the other
        self.blocks = np.array([[1.0, 1.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 2.0]])

    def test_one_value_per_waveform(self):
        papr = compute_papr(self.blocks, unit="natural", axis=-1)
        np.testing.assert_allclose(papr, [1.0, 2.0])

    def test_a_flat_waveform_has_unit_papr(self):
        self.assertAlmostEqual(
            float(compute_papr(self.blocks[0], unit="natural")), 1.0)

    def test_the_mean_is_the_mean_of_what_was_asked_for(self):
        """In dB it is the mean of the decibels, not the dB of the mean."""
        in_dB = compute_papr(self.blocks, unit="dB", axis=-1)
        reduced = compute_papr(self.blocks, unit="dB", axis=-1,
                               reduction="mean")
        self.assertAlmostEqual(float(reduced), float(np.mean(in_dB)))
        self.assertNotAlmostEqual(
            float(reduced),
            10 * np.log10(np.mean(10 ** (in_dB / 10))))

    def test_max_and_min_pick_the_waveforms(self):
        self.assertAlmostEqual(
            float(compute_papr(self.blocks, axis=-1, reduction="max")), 2.0)
        self.assertAlmostEqual(
            float(compute_papr(self.blocks, axis=-1, reduction="min")), 1.0)

    def test_an_unknown_reduction_is_refused(self):
        with self.assertRaises(ValueError):
            compute_papr(self.blocks, axis=-1, reduction="median")


class TestTheClosedFormCCDF(unittest.TestCase):

    def test_it_is_a_probability_and_it_decreases(self):
        thresholds = np.arange(2.0, 14.0, 0.5)
        ccdf = compute_papr_ccdf_theo(thresholds, 256, unit="dB")
        self.assertTrue(np.all((ccdf >= 0) & (ccdf <= 1)))
        self.assertTrue(np.all(np.diff(ccdf) <= 0))

    def test_more_subcarriers_can_only_peak_more_often(self):
        small = compute_papr_ccdf_theo(10.0, 256, unit="dB")
        large = compute_papr_ccdf_theo(10.0, 1024, unit="dB")
        self.assertGreater(large, small)

    def test_oversampling_raises_the_curve(self):
        """More effective samples, so more chances to exceed."""
        nyquist = compute_papr_ccdf_theo(10.0, 256, unit="dB")
        oversampled = compute_papr_ccdf_theo(10.0, 256, oversampling=4,
                                             unit="dB")
        self.assertGreater(oversampled, nyquist)

    def test_the_two_units_are_the_same_threshold(self):
        """natural is the amplitude ratio, dB the power ratio: dB = 20 log."""
        amplitude = 3.0
        np.testing.assert_allclose(
            compute_papr_ccdf_theo(amplitude, 256),
            compute_papr_ccdf_theo(20 * np.log10(amplitude), 256, unit="dB"))

    def test_it_matches_a_simulated_exponential_population(self):
        """The model is 'N independent exponential samples', so check it.

        Drawing the powers directly rather than synthesizing OFDM keeps
        this a test of the formula and not of the transmitter.
        """
        rng = np.random.default_rng(0)
        n_sub, trials = 64, 20000
        peak = np.max(rng.exponential(size=(trials, n_sub)), axis=-1)
        for threshold in (6.0, 8.0, 10.0):
            measured = float(np.mean(peak > 10 ** (threshold / 10)))
            predicted = float(compute_papr_ccdf_theo(threshold, n_sub,
                                                     unit="dB"))
            self.assertAlmostEqual(measured, predicted, delta=0.02)

    def test_it_warns_when_the_fit_is_extrapolated(self):
        """alpha = 2.8 is reported for an oversampling of 4 or more."""
        with self.assertLogs("comnumpy.ofdm.metrics", logging.WARNING) as logs:
            compute_papr_ccdf_theo(10.0, 256, oversampling=2, unit="dB")
        self.assertIn("extrapolation", logs.output[0])

    def test_it_refuses_impossible_arguments(self):
        for kwargs in ({"n_sub": 0}, {"n_sub": 256, "oversampling": 0},
                       {"n_sub": 256, "unit": "watt"}):
            with self.assertRaises(ValueError):
                compute_papr_ccdf_theo(10.0, **kwargs)


if __name__ == "__main__":
    unittest.main()
