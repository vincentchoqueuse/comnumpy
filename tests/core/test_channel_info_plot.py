"""What a channel says about itself, and how it draws itself.

A tutorial that shows a multipath channel spends a page recomputing the
same four things by hand: how many taps the sampling rate resolves, how
much of the energy is in the first path, how deep the worst fade is, and
what the frequency response looks like. Those are properties of the
channel, so the channel should answer them.

The tests below pin the two entry points that replaced that page:
``info()``, whose keys examples read by name and whose numbers have to be
the physical ones, and ``plot()``, which must label its axes according to
what it was asked to draw and refuse anything else.
"""
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from comnumpy.core.channels import (FIRChannel,  # noqa: E402
                                    TappedDelayLineChannel)
from comnumpy.core.fading import get_delay_profile  # noqa: E402
from comnumpy.core.visualizers import plot_channel_response  # noqa: E402


class TestWhatAFIRChannelSaysAboutItself(unittest.TestCase):

    def setUp(self):
        # |H(f)| swings between |1 + 0.5| and |1 - 0.5|, so the peak to
        # notch ratio in power is (1.5/0.5)^2 = 9, i.e. 9.54 dB
        self.channel = FIRChannel(np.array([1.0, 0.0, 0.5]))

    def test_it_counts_its_taps_and_its_energy(self):
        info = self.channel.info()
        self.assertEqual(info["kind"], "FIR")
        self.assertEqual(info["n_taps"], 3)
        self.assertAlmostEqual(info["energy"], 1.25)

    def test_the_first_tap_fraction_is_a_fraction_of_the_energy(self):
        info = self.channel.info()
        self.assertAlmostEqual(info["first_tap_fraction"], 1.0 / 1.25)

    def test_the_peak_to_notch_is_the_swing_of_the_frequency_response(self):
        info = self.channel.info()
        self.assertAlmostEqual(info["peak_to_notch_dB"],
                               20 * np.log10(1.5 / 0.5), places=2)

    def test_a_flat_channel_has_no_swing_at_all(self):
        info = FIRChannel(np.array([1.0])).info()
        self.assertAlmostEqual(info["peak_to_notch_dB"], 0.0)
        self.assertAlmostEqual(info["first_tap_fraction"], 1.0)


class TestWhatATappedDelayLineSaysAboutItself(unittest.TestCase):

    def setUp(self):
        self.fs = 7.68e6
        self.profile = get_delay_profile("EPA")
        self.channel = TappedDelayLineChannel(self.profile, fs=self.fs)

    def test_it_reports_the_profile_it_was_built_from(self):
        info = self.channel.info()
        self.assertEqual(info["standard"], "EPA")
        self.assertEqual(info["n_paths"], self.profile.n_taps)
        self.assertEqual(info["delays_ns"], self.profile.delays_ns.tolist())
        self.assertEqual(info["powers_dB"], self.profile.powers_dB.tolist())

    def test_the_resolvable_taps_are_what_this_sampling_rate_makes_of_it(self):
        info = self.channel.info()
        delays, powers = self.profile.to_taps(self.fs)
        self.assertEqual(info["resolvable_delays_samples"], delays.tolist())
        self.assertEqual(info["n_taps"], int(delays[-1]) + 1)
        # seven arrivals within 410 ns, sampled at 7.68 MHz, land on
        # fewer bins than there are paths
        self.assertLess(info["n_taps"], info["n_paths"])
        self.assertAlmostEqual(sum(powers), 1.0, places=6)

    def test_sampling_faster_resolves_more_taps(self):
        fast = TappedDelayLineChannel(self.profile, fs=4 * self.fs).info()
        self.assertGreater(fast["n_taps"], self.channel.info()["n_taps"])
        # the physics does not depend on how we sample it
        self.assertAlmostEqual(fast["rms_delay_spread_ns"],
                               self.channel.info()["rms_delay_spread_ns"])

    def test_the_coherence_bandwidth_is_the_one_the_profile_computes(self):
        info = self.channel.info()
        self.assertAlmostEqual(info["coherence_bandwidth_Hz"],
                               self.profile.coherence_bandwidth_hz)
        self.assertAlmostEqual(info["fs_Hz"], self.fs)


class TestDrawingAChannel(unittest.TestCase):

    def setUp(self):
        self.taps = np.array([1.0, 0.0, 0.5])
        self.addCleanup(plt.close, "all")

    def test_the_delay_domain_is_a_stem_per_tap(self):
        ax = plot_channel_response(self.taps)
        self.assertEqual(ax.get_xlabel(), "tap index")
        self.assertEqual(ax.get_ylabel(), "$|h[l]|$")

    def test_a_sampling_rate_turns_the_abscissa_into_a_delay(self):
        ax = plot_channel_response(self.taps, fs=7.68e6)
        self.assertEqual(ax.get_xlabel(), "delay [us]")

    def test_the_frequency_domain_is_a_curve(self):
        ax = plot_channel_response(self.taps, domain="frequency")
        self.assertEqual(ax.get_ylabel(), "$|H(f)|$")
        drawn = ax.lines[0].get_ydata()
        # the swing of |1 + 0.5 e^{-2j pi 2 f}| over the band
        self.assertAlmostEqual(float(np.max(drawn)), 1.5, places=3)
        self.assertAlmostEqual(float(np.min(drawn)), 0.5, places=3)

    def test_decibels_are_twenty_log_of_the_same_curve(self):
        ax = plot_channel_response(self.taps, domain="frequency", scale="dB")
        self.assertEqual(ax.get_ylabel(), "$|H(f)|$ [dB]")
        drawn = ax.lines[0].get_ydata()
        self.assertAlmostEqual(float(np.max(drawn)),
                               20 * np.log10(1.5), places=3)

    def test_a_zero_tap_does_not_send_the_axis_to_minus_infinity(self):
        ax = plot_channel_response(np.array([1.0, 0.0]), scale="dB")
        self.assertTrue(np.all(np.isfinite(ax.collections[0].get_offsets())))

    def test_it_refuses_a_domain_it_cannot_draw(self):
        with self.assertRaises(ValueError):
            plot_channel_response(self.taps, domain="phase")

    def test_it_refuses_a_scale_it_cannot_draw(self):
        with self.assertRaises(ValueError):
            plot_channel_response(self.taps, scale="log")

    def test_a_channel_draws_itself_on_the_axis_it_is_given(self):
        _, ax = plt.subplots()
        returned = FIRChannel(self.taps).plot("frequency", scale="dB", ax=ax)
        self.assertIs(returned, ax)
        self.assertEqual(ax.get_ylabel(), "$|H(f)|$ [dB]")


class TestDrawingAFadingChannel(unittest.TestCase):

    def setUp(self):
        self.channel = TappedDelayLineChannel(get_delay_profile("EPA"),
                                              fs=7.68e6, seed=0)
        self.addCleanup(plt.close, "all")

    def test_it_draws_a_realization_before_it_has_ever_been_run(self):
        ax = self.channel.plot()
        self.assertEqual(ax.get_xlabel(), "delay [us]")
        self.assertGreater(len(ax.collections), 0)

    def test_after_a_run_it_draws_the_realization_that_was_used(self):
        x = np.ones(4096, dtype=complex)
        self.channel(x)
        ax = self.channel.plot("frequency", scale="dB")
        expected = np.zeros(int(self.channel.delays_[-1]) + 1, dtype=complex)
        expected[self.channel.delays_] = self.channel.h_[:, 0]
        drawn = ax.lines[0].get_ydata()
        reference = 20 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(expected, 512))))
        np.testing.assert_allclose(drawn, reference, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
