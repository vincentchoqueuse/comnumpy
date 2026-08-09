"""Multi-pump, multi-signal Raman: every pair of waves is coupled (D45).

The single-pair solver could not describe what a Raman amplifier is
actually designed against: the tilt a WDM comb picks up, and the second
pump placed to cancel it. The generalization keeps one equation per
wave and one coupling coefficient per pair, so the same code covers
pump-to-signal gain, pump-to-pump transfer, and the comb tilting
itself -- the blue channels pumping the red ones.

What is checked here is what the formulation cannot fake: photon-number
conservation over the whole set in the lossless limit, exact agreement
with the old single-pair solve, superposition in the small-signal
limit, and the ordering results a designer relies on.
"""
import unittest

import numpy as np

from comnumpy.optical.raman import (SPEED_OF_LIGHT, get_gain_spectrum,
                                    solve_raman)

L_KM = 80.0
GAIN = 0.4
SPECTRUM = get_gain_spectrum("blow-wood")
COMB = [1530.0, 1540.0, 1550.0, 1560.0]


def frequencies(wavelengths_nm):
    return SPEED_OF_LIGHT / (np.asarray(wavelengths_nm, dtype=float) * 1e-9)


def solve(**kwargs):
    kwargs.setdefault("spectrum", SPECTRUM)
    kwargs.setdefault("bandwidth_Hz", 0.0)
    return solve_raman(length_km=L_KM, gain_peak_W_km=GAIN, **kwargs)


class TestShapes(unittest.TestCase):
    """One row per wave -- and the plain 1-D curve when there is one."""

    def test_a_single_wave_keeps_the_one_dimensional_profiles(self):
        solution = solve(pump_backward_W=0.3, spectrum=None)
        self.assertEqual(solution.signal_W.ndim, 1)
        self.assertEqual(solution.pump_forward_W.ndim, 1)
        self.assertIsInstance(solution.on_off_gain_dB, float)
        self.assertEqual((solution.n_signals, solution.n_pumps), (1, 1))

    def test_several_waves_give_one_row_each(self):
        solution = solve(wavelength_signal_nm=COMB,
                         pump_backward_W=[0.2, 0.2],
                         wavelength_pump_nm=[1425.0, 1455.0])
        n_z = solution.z_km.size
        self.assertEqual(solution.signal_W.shape, (4, n_z))
        self.assertEqual(solution.ase_W.shape, (4, n_z))
        self.assertEqual(solution.pump_forward_W.shape, (2, n_z))
        self.assertEqual(solution.pump_backward_W.shape, (2, n_z))
        self.assertEqual(solution.gain_profile_dB.shape, (4, n_z))
        self.assertEqual(np.shape(solution.on_off_gain_dB), (4,))

    def test_a_scalar_is_shared_by_every_wave_of_its_group(self):
        """One launch power, one loss, four channels."""
        solution = solve(wavelength_signal_nm=COMB, signal_W=1e-3,
                         pump_backward_W=0.3, wavelength_pump_nm=1450.0)
        np.testing.assert_allclose(solution.signal_W[:, 0], 1e-3)

    def test_a_pump_off_in_one_direction_is_a_zero_row(self):
        solution = solve(pump_backward_W=[0.2, 0.1],
                         wavelength_pump_nm=[1425.0, 1455.0])
        np.testing.assert_array_equal(solution.pump_forward_W,
                                      np.zeros_like(solution.pump_forward_W))
        np.testing.assert_allclose(solution.pump_backward_W[:, -1], [0.2, 0.1])


class TestReductionToTheSinglePair(unittest.TestCase):
    """The generalization must not have moved the single-channel answer."""

    def test_one_signal_one_pump_matches_the_closed_form(self):
        """The undepleted closed form, unchanged by the rewrite."""
        alpha = 0.25 / (10 / np.log(10))
        effective = (1 - np.exp(-alpha * L_KM)) / alpha
        for pump in (0.05, 0.2):
            with self.subTest(pump_W=pump):
                solution = solve(pump_forward_W=pump, signal_W=1e-9,
                                 spectrum=None)
                expected = 10 * np.log10(np.exp(GAIN * pump * effective))
                self.assertAlmostEqual(solution.on_off_gain_dB, expected,
                                       places=4)

    def test_a_duplicated_channel_gains_exactly_like_a_lone_one(self):
        """Two co-located channels see the same pump, hence the same gain.

        They are at the same frequency, so they do not exchange with
        each other: any difference would be a spurious self-coupling.
        """
        lone = solve(wavelength_signal_nm=1550.0, signal_W=1e-9,
                     pump_backward_W=0.3, spectrum=SPECTRUM)
        pair = solve(wavelength_signal_nm=[1550.0, 1550.0], signal_W=1e-9,
                     pump_backward_W=0.3)
        np.testing.assert_allclose(pair.on_off_gain_dB,
                                   [lone.on_off_gain_dB] * 2, rtol=1e-6)

    def test_two_pumps_at_the_same_wavelength_add_their_power(self):
        single = solve(pump_backward_W=0.4, wavelength_pump_nm=1450.0,
                       signal_W=1e-9, spectrum=SPECTRUM)
        split = solve(pump_backward_W=[0.25, 0.15],
                      wavelength_pump_nm=[1450.0, 1450.0], signal_W=1e-9)
        np.testing.assert_allclose(split.on_off_gain_dB,
                                   single.on_off_gain_dB, rtol=1e-6)


class TestPhotonConservation(unittest.TestCase):
    r"""One photon in, one photon out -- over the whole set of waves.

    Without loss the total photon flux :math:`\sum_i P_i / \nu_i`,
    counted with the sign of each wave's direction, is constant. This
    holds under arbitrary depletion and for any number of waves, which
    is exactly what a pairwise coupling matrix can get wrong.
    """

    def flux(self, solution, signal_nm, pump_nm, backward=False):
        photons = np.sum(np.atleast_2d(solution.signal_W)
                         / frequencies(signal_nm)[:, None], axis=0)
        pumps = np.atleast_2d(solution.pump_backward_W if backward
                              else solution.pump_forward_W)
        sign = -1.0 if backward else 1.0
        return photons + sign * np.sum(pumps / frequencies(pump_nm)[:, None],
                                       axis=0)

    def test_conserved_with_a_comb_and_two_co_propagating_pumps(self):
        pump_nm = [1425.0, 1455.0]
        solution = solve(wavelength_signal_nm=COMB, signal_W=0.05,
                         pump_forward_W=[0.5, 0.5], wavelength_pump_nm=pump_nm,
                         alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                         tol=1e-12)
        flux = self.flux(solution, COMB, pump_nm)
        self.assertLess(float(np.ptp(flux) / flux[0]), 1e-7)

    def test_the_pumps_really_deplete_so_the_check_is_not_vacuous(self):
        solution = solve(wavelength_signal_nm=COMB, signal_W=0.05,
                         pump_forward_W=[0.5, 0.5],
                         wavelength_pump_nm=[1425.0, 1455.0],
                         alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0)
        self.assertGreater(solution.pump_depletion, 0.3)

    def test_conserved_with_a_counter_propagating_pump(self):
        """The backward pump enters the balance with the opposite sign."""
        solution = solve(wavelength_signal_nm=COMB, signal_W=5e-3,
                         pump_backward_W=0.3, wavelength_pump_nm=1450.0,
                         alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                         tol=1e-10)
        flux = self.flux(solution, COMB, [1450.0], backward=True)
        self.assertLess(float(np.ptp(flux) / abs(flux[0])), 1e-6)


class TestInterChannelTilt(unittest.TestCase):
    """The comb pumps itself: blue channels feed red ones (SRS tilt)."""

    def unpumped_tilt(self, power_W):
        """A comb with no pump at all still tilts, if it is strong enough."""
        return solve(wavelength_signal_nm=COMB, signal_W=power_W,
                     pump_backward_W=1e-12, wavelength_pump_nm=1450.0,
                     alpha_signal_dB_km=0.2)

    def test_the_red_channel_gains_what_the_blue_one_loses(self):
        solution = self.unpumped_tilt(0.05)
        gains = np.asarray(solution.on_off_gain_dB)
        self.assertLess(gains[0], 0.0)        # 1530 nm, the bluest, is drained
        self.assertGreater(gains[-1], 0.0)    # 1560 nm, the reddest, is fed
        self.assertTrue(np.all(np.diff(gains) > 0), f"not monotonic: {gains}")

    def test_the_tilt_grows_with_the_comb_power(self):
        weak = self.unpumped_tilt(1e-4).tilt_dB
        strong = self.unpumped_tilt(0.05).tilt_dB
        self.assertGreater(strong, 10 * weak)

    def test_a_negligible_comb_power_does_not_tilt(self):
        self.assertLess(self.unpumped_tilt(1e-9).tilt_dB, 1e-6)

    def test_the_transfer_conserves_photons_between_channels(self):
        """Nothing is created: the comb only moves power towards the red."""
        solution = solve(wavelength_signal_nm=COMB, signal_W=0.05,
                         pump_backward_W=1e-12, wavelength_pump_nm=1450.0,
                         alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                         tol=1e-12)
        photons = np.sum(solution.signal_W / frequencies(COMB)[:, None], axis=0)
        self.assertLess(float(np.ptp(photons) / photons[0]), 1e-8)
        # power itself is *not* conserved: the phonons take their share
        self.assertLess(float(np.sum(solution.signal_W[:, -1])),
                        float(np.sum(solution.signal_W[:, 0])))


class TestAgainstZirngibl(unittest.TestCase):
    r"""The published closed form for the tilt of a WDM comb.

    With a triangular gain and the photon factor
    :math:`\nu_i/\nu_j \simeq 1`, the coupled equations of a comb have
    an analytical solution: every common term factors out and what is
    left is a Boltzmann-like reweighting of the channels,

    .. math::

        P_i(L) = P_\mathrm{tot} e^{-\alpha L}\,
            \frac{P_i(0) e^{-C_R \nu_i P_\mathrm{tot} L_\mathrm{eff}}}
                 {\sum_k P_k(0) e^{-C_R \nu_k P_\mathrm{tot} L_\mathrm{eff}}}

    ``validation/optical_raman.py`` runs this over a range of powers;
    this keeps one point of it in the fast suite.

    References
    ----------
    M. Zirngibl, "Analytical model of Raman gain effects in massive
    wavelength division multiplexed transmission systems", Electron.
    Lett., vol. 34, no. 8, pp. 789-790, 1998.
    """

    PEAK_THZ = 13.2

    def test_the_comb_tilt_matches_the_closed_form(self):
        channel_W, alpha_dB = 1.5e-2, 0.2
        alpha = alpha_dB / (10 / np.log(10))
        effective = (1 - np.exp(-alpha * L_KM)) / alpha
        solution = solve(
            wavelength_signal_nm=COMB, signal_W=channel_W,
            pump_backward_W=1e-14, wavelength_pump_nm=1455.0,
            alpha_signal_dB_km=alpha_dB,
            spectrum=get_gain_spectrum("triangular",
                                       peak_shift_THz=self.PEAK_THZ),
            tol=1e-10)

        slope = GAIN / (self.PEAK_THZ * 1e12)
        total = len(COMB) * channel_W
        weight = np.exp(-slope * frequencies(COMB) * total * effective)
        closed = (total * np.exp(-alpha * L_KM) * channel_W * weight
                  / np.sum(channel_W * weight))

        error_dB = np.abs(10 * np.log10(solution.signal_W[:, -1] / closed))
        tilt_dB = float(np.ptp(10 * np.log10(closed)))
        self.assertGreater(tilt_dB, 0.3)          # there is a tilt to match
        # the model drops nu_i/nu_j ~ 1; what is left must be a fraction
        # of a percent of the tilt, not of the tilt itself
        self.assertLess(float(np.max(error_dB)), 0.01 * tilt_dB)


class TestPumpToPumpTransfer(unittest.TestCase):
    """A short-wavelength pump also amplifies the longer-wavelength one."""

    def test_the_blue_pump_feeds_the_red_pump(self):
        together = solve(pump_forward_W=[1.0, 0.2],
                         wavelength_pump_nm=[1365.0, 1455.0], signal_W=1e-9)
        alone = solve(pump_forward_W=[1e-12, 0.2],
                      wavelength_pump_nm=[1365.0, 1455.0], signal_W=1e-9)
        self.assertGreater(together.pump_forward_W[1, -1],
                           1.5 * alone.pump_forward_W[1, -1])

    def test_the_second_order_pump_raises_the_signal_gain(self):
        """The whole point of second-order pumping."""
        first_order = solve(pump_backward_W=[1e-12, 0.2],
                            wavelength_pump_nm=[1365.0, 1455.0], signal_W=1e-9)
        second_order = solve(pump_backward_W=[1.0, 0.2],
                             wavelength_pump_nm=[1365.0, 1455.0], signal_W=1e-9)
        self.assertGreater(second_order.on_off_gain_dB,
                           first_order.on_off_gain_dB + 1.0)


class TestMultiPumpDesign(unittest.TestCase):
    """The ordering results a Raman design relies on."""

    def gains(self, pump_W, pump_nm):
        return solve(wavelength_signal_nm=COMB, signal_W=1e-6,
                     pump_backward_W=pump_W, wavelength_pump_nm=pump_nm)

    def test_splitting_the_power_over_two_pumps_flattens_the_gain(self):
        one = self.gains(0.4, 1450.0)
        two = self.gains([0.15, 0.25], [1425.0, 1455.0])
        self.assertLess(two.tilt_dB, 0.6 * one.tilt_dB)

    def test_the_noise_figure_is_reported_per_channel(self):
        solution = solve(wavelength_signal_nm=COMB, signal_W=1e-6,
                         pump_backward_W=[0.15, 0.25],
                         wavelength_pump_nm=[1425.0, 1455.0],
                         bandwidth_Hz=12.5e9)
        noise = np.asarray(solution.noise_figure_dB)
        self.assertEqual(noise.shape, (4,))
        self.assertTrue(np.all(np.isfinite(noise)))

    def test_the_repr_states_the_tilt_rather_than_the_arrays(self):
        text = repr(self.gains([0.15, 0.25], [1425.0, 1455.0]))
        self.assertIn("4 signal(s), 2 pump(s)", text)
        self.assertIn("tilt", text)
        self.assertNotIn("\n", text)


class TestGuards(unittest.TestCase):

    def test_a_multi_wave_solve_requires_a_spectrum(self):
        with self.assertRaises(ValueError) as ctx:
            solve(wavelength_signal_nm=COMB, pump_backward_W=0.3,
                  spectrum=None)
        self.assertIn("multi-wave", str(ctx.exception))
        self.assertIn("blow-wood", str(ctx.exception))

    def test_mismatched_lengths_inside_a_group_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            solve(wavelength_signal_nm=COMB, signal_W=[1e-3, 2e-3],
                  pump_backward_W=0.3)
        self.assertIn("2 entries", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

    def test_a_pump_below_a_signal_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            solve(wavelength_signal_nm=COMB, pump_backward_W=[0.2, 0.2],
                  wavelength_pump_nm=[1425.0, 1600.0])
        self.assertIn("above every signal", str(ctx.exception))

    def test_pumps_far_outside_the_gain_band_are_rejected(self):
        """Only a spectrum that really reaches zero can be caught here.

        The damped-oscillator tail never quite vanishes -- 5e-5 of the
        peak at a 300 THz shift -- so the guard is written as "no gain
        at all" rather than around an invented threshold, and it is the
        triangular model that exercises it.
        """
        with self.assertRaises(ValueError) as ctx:
            solve(wavelength_signal_nm=COMB, pump_backward_W=[0.2, 0.2],
                  wavelength_pump_nm=[600.0, 610.0],
                  spectrum=get_gain_spectrum("triangular"))
        self.assertIn("no gain at all", str(ctx.exception))

    def test_every_pump_off_is_rejected_whatever_the_count(self):
        with self.assertRaises(ValueError) as ctx:
            solve(wavelength_signal_nm=COMB, pump_backward_W=[0.0, 0.0],
                  wavelength_pump_nm=[1425.0, 1455.0])
        self.assertIn("at least one pump", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
