"""Raman amplification: the gain spectra and the coupled power equations (D45).

The solver is checked against things it cannot fake: the closed-form
gain in the undepleted limit, photon-number conservation in the
lossless limit -- which holds *under depletion*, where no closed form
exists -- and the ordering between the pumping schemes.
"""
import unittest

import numpy as np

from comnumpy.optical import raman
from comnumpy.optical.raman import (PLANCK, RamanGainSpectrum,
                                    available_gain_spectra, get_gain_spectrum,
                                    register_gain_spectrum, solve_raman)

_COUPLING_MATRIX = raman._coupling_matrix

L_KM = 80.0
GAIN = 0.4                 # 1/(W km), a typical SMF value
ALPHA_P_DB = 0.25
ALPHA_S_DB = 0.2


def effective_length(alpha_dB_km, length_km):
    alpha = alpha_dB_km / (10 / np.log(10))
    return (1 - np.exp(-alpha * length_km)) / alpha


def undepleted_gain_dB(pump_W):
    """The closed form the solver must reproduce when nothing depletes."""
    return 10 * np.log10(np.exp(GAIN * pump_W
                                * effective_length(ALPHA_P_DB, L_KM)))


class TestGainSpectrum(unittest.TestCase):
    """The catalog is process-global state, so restore it around each test."""

    def setUp(self):
        self._registry = dict(raman._SPECTRUM_REGISTRY)

    def tearDown(self):
        raman._SPECTRUM_REGISTRY.clear()
        raman._SPECTRUM_REGISTRY.update(self._registry)

    def test_blow_wood_reproduces_the_published_peak(self):
        """13.08 THz against the 13.2 THz published for silica."""
        spectrum = get_gain_spectrum("blow-wood")
        self.assertAlmostEqual(spectrum.peak_shift_THz, 13.2, delta=0.2)

    def test_blow_wood_is_too_broad_and_says_so(self):
        """The known failure of a single oscillator, pinned as a fact.

        Measured silica is 5 to 6 THz wide; this model gives 9.55. The
        test exists so nobody 'fixes' the width by tuning the time
        constants, which would break the peak the source does specify.
        """
        self.assertAlmostEqual(get_gain_spectrum("blow-wood").fwhm_THz,
                               9.55, delta=0.1)

    def test_the_shape_is_normalized_to_one_at_its_peak(self):
        for name in available_gain_spectra():
            with self.subTest(spectrum=name):
                spectrum = get_gain_spectrum(name)
                peak = float(spectrum.shape(spectrum.peak_shift_THz * 1e12))
                self.assertAlmostEqual(peak, 1.0, places=6)

    def test_no_gain_above_the_pump(self):
        """A signal above the pump in frequency gets nothing."""
        for name in available_gain_spectra():
            with self.subTest(spectrum=name):
                self.assertEqual(
                    float(get_gain_spectrum(name).shape(-6e12)), 0.0)

    def test_the_triangular_model_is_linear_below_its_peak(self):
        spectrum = get_gain_spectrum("triangular", peak_shift_THz=13.2)
        shifts = np.array([1e12, 2e12, 4e12, 8e12])
        shape = spectrum.shape(shifts)
        np.testing.assert_allclose(shape, shifts / 13.2e12, rtol=1e-12)

    def test_the_self_check_catches_a_wrong_model(self):
        """A model that misses its published peak must fail at construction."""
        @register_gain_spectrum("_mistyped")
        def _entry():
            return raman._check_expect(
                RamanGainSpectrum(lorentzian=(30.0, 32.0),
                                  standard="_mistyped", reference="none"),
                {"peak_shift_THz": 13.2})
        with self.assertRaises(ValueError) as ctx:
            get_gain_spectrum("_mistyped")
        self.assertIn("mistyped", str(ctx.exception))

    def test_rejects_zero_or_two_parameterizations(self):
        with self.assertRaises(ValueError):
            RamanGainSpectrum()
        with self.assertRaises(ValueError):
            RamanGainSpectrum(lorentzian=(12.2, 32.0), triangular=13.2)

    def test_rejects_nonsense_parameters(self):
        with self.assertRaises(ValueError):
            RamanGainSpectrum(lorentzian=(-12.2, 32.0))
        with self.assertRaises(ValueError):
            RamanGainSpectrum(triangular=0.0)

    def test_is_frozen_and_hashable(self):
        spectrum = get_gain_spectrum("blow-wood")
        self.assertIsInstance(hash(spectrum), int)
        self.assertEqual(spectrum, get_gain_spectrum("blow-wood"))

    def test_repr_shows_the_two_figures_that_matter(self):
        text = repr(get_gain_spectrum("blow-wood"))
        self.assertIn("13.08", text)
        self.assertIn("9.55", text)


NU_S = 2.99792458e8 / 1550e-9
NU_P = 2.99792458e8 / 1455e-9


def exact_copumped_signal(z_km, signal_W, pump_W, alpha_dB_km=0.0):
    r"""Exact co-pumped profile, valid under *arbitrary* depletion.

    Photon-number conservation eliminates the pump from the coupled
    pair, and a logistic equation is what remains:

    .. math::

        \frac{\mathrm{d}P_s}{\mathrm{d}z} = r P_s
            \left(1 - \frac{P_s}{P_s^\infty}\right)

    with :math:`r = g(P_p(0) + \frac{\nu_p}{\nu_s} P_s(0))` and
    :math:`P_s^\infty = P_s(0) + \frac{\nu_s}{\nu_p} P_p(0)`, the
    power reached once every pump photon has been converted. With
    *equal* losses the same solution holds in the effective length,
    because :math:`Q = P e^{\alpha z}` maps the pair back to the
    lossless one.
    """
    alpha = alpha_dB_km / (10 / np.log(10))
    zeta = z_km if alpha == 0 else (1 - np.exp(-alpha * z_km)) / alpha
    limit = signal_W + (NU_S / NU_P) * pump_W
    rate = GAIN * (pump_W + (NU_P / NU_S) * signal_W)
    profile = limit / (1 + (limit / signal_W - 1) * np.exp(-rate * zeta))
    return profile * np.exp(-alpha * z_km)


class TestExactSolution(unittest.TestCase):
    """The closed form of the simple case, over the whole profile.

    Stronger than the undepleted formula in every way: it pins every
    point of the curve rather than one output number, and it stays
    exact when the pump is fully consumed -- the regime the
    approximation cannot describe at all.
    """

    def test_lossless_profile_matches_the_logistic(self):
        for pump in (0.2, 0.5, 1.0, 2.0):
            with self.subTest(pump_W=pump):
                solution = solve_raman(
                    length_km=L_KM, gain_peak_W_km=GAIN, pump_forward_W=pump,
                    alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                    bandwidth_Hz=0.0, tol=1e-12)
                exact = exact_copumped_signal(solution.z_km, 1e-3, pump)
                error = float(np.max(np.abs(solution.signal_W - exact) / exact))
                self.assertLess(error, 1e-9)

    def test_equal_losses_match_the_logistic_in_effective_length(self):
        for pump in (0.2, 1.0, 2.0):
            with self.subTest(pump_W=pump):
                solution = solve_raman(
                    length_km=L_KM, gain_peak_W_km=GAIN, pump_forward_W=pump,
                    alpha_signal_dB_km=0.22, alpha_pump_dB_km=0.22,
                    bandwidth_Hz=0.0, tol=1e-12)
                exact = exact_copumped_signal(solution.z_km, 1e-3, pump, 0.22)
                error = float(np.max(np.abs(solution.signal_W - exact) / exact))
                self.assertLess(error, 1e-9)

    def test_the_regime_tested_is_actually_depleted(self):
        """Otherwise the two tests above only exercise the easy case."""
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=0.5, alpha_signal_dB_km=0.0,
                               alpha_pump_dB_km=0.0, bandwidth_Hz=0.0)
        self.assertGreater(1 - float(solution.pump_forward_W[-1] / 0.5), 0.9)

    def test_the_signal_saturates_at_the_converted_photon_limit(self):
        """Every pump photon becomes a signal photon, and no more."""
        pump = 2.0
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=pump, alpha_signal_dB_km=0.0,
                               alpha_pump_dB_km=0.0, bandwidth_Hz=0.0,
                               tol=1e-12)
        limit = 1e-3 + (NU_S / NU_P) * pump
        self.assertLessEqual(float(solution.signal_W[-1]), limit * (1 + 1e-9))
        self.assertGreater(float(solution.signal_W[-1]), 0.99 * limit)


class TestUndepletedLimit(unittest.TestCase):
    """Where an exact answer exists, the solver must land on it."""

    def test_every_scheme_matches_the_closed_form_at_low_pump(self):
        pump = 0.05
        expected = undepleted_gain_dB(pump)
        schemes = {
            "co": dict(pump_forward_W=pump),
            "counter": dict(pump_backward_W=pump),
            "bidirectional": dict(pump_forward_W=pump / 2,
                                  pump_backward_W=pump / 2),
        }
        for name, pumping in schemes.items():
            with self.subTest(scheme=name):
                solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                                       **pumping)
                self.assertAlmostEqual(solution.on_off_gain_dB, expected,
                                       delta=0.01)

    def test_the_closed_form_becomes_optimistic_under_depletion(self):
        """It ignores the pump the signal consumes, so it can only overstate."""
        for pump in (0.2, 0.5, 1.0):
            with self.subTest(pump_W=pump):
                solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                                       pump_forward_W=pump)
                self.assertLess(solution.on_off_gain_dB,
                                undepleted_gain_dB(pump))

    def test_net_gain_is_the_on_off_gain_minus_the_span_loss(self):
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_backward_W=0.2)
        self.assertAlmostEqual(solution.net_gain_dB,
                               solution.on_off_gain_dB - ALPHA_S_DB * L_KM,
                               places=9)


class TestPhotonConservation(unittest.TestCase):
    r"""The invariant that survives depletion, where no closed form does.

    Without loss, one pump photon becomes exactly one signal photon, so
    :math:`P_s/\nu_s + P_p/\nu_p` is constant along the fibre. This is
    the check that covers the strongly-depleted regime.
    """

    def test_photon_number_is_conserved_without_loss(self):
        for pump in (0.2, 1.0):
            with self.subTest(pump_W=pump):
                solution = solve_raman(
                    length_km=L_KM, gain_peak_W_km=GAIN, pump_forward_W=pump,
                    alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                    bandwidth_Hz=0.0, tol=1e-12)
                nu_s = solution.frequency_signal_Hz
                nu_p = 2.99792458e8 / 1455e-9      # the solver's default pump
                photons = (solution.signal_W / nu_s
                           + solution.pump_forward_W / nu_p)
                relative = np.ptp(photons) / photons[0]
                self.assertLess(relative, 1e-8, f"photon flux drifts by {relative:.2e}")

    def test_the_pump_really_does_deplete(self):
        """Otherwise the conservation test above would be vacuous."""
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=1.0, alpha_signal_dB_km=0.0,
                               alpha_pump_dB_km=0.0, bandwidth_Hz=0.0)
        self.assertLess(solution.pump_forward_W[-1], 0.1 * 1.0)


class TestTheConventionIsNotArbitrary(unittest.TestCase):
    r"""The closed form rejects the other way of writing the pair.

    :math:`C_{ji} = -\frac{\nu_j}{\nu_i} C_{ij}` conserves photon number:
    a pump photon annihilated gives a Stokes photon *and* an optical
    phonon, so the pump loses :math:`\nu_p/\nu_s` times what the signal
    gains and the difference leaves the optical field. Dropping that
    factor gives an antisymmetric matrix, which conserves total power
    instead -- the phonon is never paid for. Both are self-consistent
    and both pass photon- or energy-accounting on their own terms, so
    neither is refuted by an invariant.

    The logistic of :class:`TestExactSolution` does refute one of them,
    and by a predictable amount: the two conventions saturate at
    :math:`P_{s0} + \frac{\nu_s}{\nu_p} P_{p0}` and at
    :math:`P_{s0} + P_{p0}`, whose ratio is 6.5% here.

    This test exists because that check is the only thing standing
    behind the factor. Without it the choice reads as a detail of the
    implementation rather than the physics it is.
    """

    @staticmethod
    def _power_conserving(frequency_Hz, gain_peak_W_km, spectrum):
        # the captured original, not the module attribute: that one is the
        # patch itself while this runs
        full = _COUPLING_MATRIX(frequency_Hz, gain_peak_W_km, spectrum)
        gain = np.where(full > 0, full, 0.0)
        return gain - gain.T

    def _profile(self, pump_W, coupling=None):
        original = raman._coupling_matrix
        if coupling is not None:
            raman._coupling_matrix = coupling
        try:
            return solve_raman(
                length_km=L_KM, gain_peak_W_km=GAIN, pump_forward_W=pump_W,
                alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
                bandwidth_Hz=0.0, tol=1e-12)
        finally:
            raman._coupling_matrix = original

    def test_dropping_the_photon_factor_misses_the_closed_form(self):
        for pump in (0.5, 2.0):
            with self.subTest(pump_W=pump):
                exact = exact_copumped_signal(
                    self._profile(pump).z_km, 1e-3, pump)
                theirs = self._profile(pump, self._power_conserving).signal_W
                error = float(np.max(np.abs(theirs - exact) / exact))
                self.assertGreater(error, 0.04, "the two conventions are "
                                   "indistinguishable here, so this test "
                                   "proves nothing about the factor")

    def test_the_two_conventions_differ_by_the_saturation_ratio(self):
        """The size of the miss is predicted, not merely observed."""
        pump = 2.0
        signal = 1e-3
        photon_limit = signal + (NU_S / NU_P) * pump
        power_limit = signal + pump
        theirs = self._profile(pump, self._power_conserving).signal_W
        # driven to saturation, the wrong convention lands on the wrong limit
        self.assertAlmostEqual(float(theirs[-1]) / power_limit, 1.0, places=2)
        self.assertAlmostEqual(power_limit / photon_limit, 1.0653, places=3)


class TestPumpingSchemes(unittest.TestCase):

    def test_counter_pumping_keeps_more_gain_under_depletion(self):
        """Non-obvious, and it catches a sign error on the direction flip.

        The counter-propagating pump is strongest where the signal is
        weakest, so it depletes less. The two schemes agree at low pump
        and separate as it rises -- measured 0.006 dB apart at 50 mW,
        2.9 dB at 1 W.
        """
        gaps = []
        for pump in (0.05, 0.2, 0.5, 1.0):
            co = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                             pump_forward_W=pump)
            counter = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                                  pump_backward_W=pump)
            with self.subTest(pump_W=pump):
                self.assertGreater(counter.on_off_gain_dB, co.on_off_gain_dB)
            gaps.append(counter.on_off_gain_dB - co.on_off_gain_dB)
        self.assertTrue(np.all(np.diff(gaps) > 0),
                        f"the gap must widen with the pump, got {gaps}")
        self.assertLess(gaps[0], 0.01)
        self.assertGreater(gaps[-1], 2.0)

    def test_bidirectional_sits_between_the_two(self):
        pump = 1.0
        co = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                         pump_forward_W=pump)
        counter = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                              pump_backward_W=pump)
        both = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                           pump_forward_W=pump / 2, pump_backward_W=pump / 2)
        self.assertLess(co.on_off_gain_dB, both.on_off_gain_dB)
        self.assertLess(both.on_off_gain_dB, counter.on_off_gain_dB)

    def test_co_pumping_has_the_lower_noise_figure(self):
        """Gain delivered while the signal is still strong costs less OSNR.

        This power model does not include pump RIN transfer, which is
        the reason counter-pumping is preferred in practice despite
        this; the docstring of the module says so.
        """
        co = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                         pump_forward_W=0.5)
        counter = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                              pump_backward_W=0.5)
        self.assertLess(co.noise_figure_dB, counter.noise_figure_dB)


class TestAse(unittest.TestCase):

    def test_no_ase_in_zero_bandwidth(self):
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=0.2, bandwidth_Hz=0.0)
        self.assertEqual(float(solution.ase_W[-1]), 0.0)

    def test_ase_grows_with_bandwidth_and_with_temperature(self):
        base = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                           pump_forward_W=0.2, bandwidth_Hz=12.5e9,
                           temperature_K=300.0)
        wider = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                            pump_forward_W=0.2, bandwidth_Hz=25e9,
                            temperature_K=300.0)
        hotter = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                             pump_forward_W=0.2, bandwidth_Hz=12.5e9,
                             temperature_K=400.0)
        self.assertAlmostEqual(float(wider.ase_W[-1] / base.ase_W[-1]), 2.0,
                               places=6)
        self.assertGreater(float(hotter.ase_W[-1]), float(base.ase_W[-1]))

    def test_the_cold_limit_approaches_one_spontaneous_photon_per_mode(self):
        """The phonon occupancy vanishes as T -> 0, leaving the 1 of 1 + eta."""
        self.assertAlmostEqual(raman._photon_occupancy(13.2e12, 1.0), 1.0,
                               places=12)
        self.assertGreater(raman._photon_occupancy(13.2e12, 300.0), 1.0)

    def test_noise_figure_uses_the_photon_energy(self):
        """A sanity anchor on the units: hv B at 1550 nm over 12.5 GHz.

        1.602 nW, computed rather than recalled -- an error of a factor
        of a thousand here would move the noise figure by 30 dB and
        every other test would still pass.
        """
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=0.2)
        photon = PLANCK * solution.frequency_signal_Hz * solution.bandwidth_Hz
        self.assertAlmostEqual(photon, 1.602e-9, delta=0.002e-9)


class TestGuards(unittest.TestCase):

    def test_rejects_having_no_pump_on(self):
        with self.assertRaises(ValueError) as ctx:
            solve_raman(length_km=L_KM, gain_peak_W_km=GAIN)
        self.assertIn("pump_forward_W", str(ctx.exception))

    def test_rejects_negative_powers_and_lengths(self):
        for kwargs in (dict(length_km=-1.0, pump_forward_W=0.2),
                       dict(length_km=L_KM, pump_forward_W=-0.2),
                       dict(length_km=L_KM, pump_forward_W=0.2, signal_W=0.0)):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    solve_raman(gain_peak_W_km=GAIN, **kwargs)

    def test_rejects_a_pump_below_the_signal(self):
        with self.assertRaises(ValueError) as ctx:
            solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                        pump_forward_W=0.2, wavelength_pump_nm=1600.0)
        self.assertIn("Stokes shift", str(ctx.exception))

    def test_rejects_a_pump_outside_the_gain_band(self):
        """A pump past the peak gets nothing from the triangular model.

        1400 nm against a 1550 nm signal is a 20.7 THz shift, well above
        the 13.2 THz peak where the model drops to zero -- which is
        exactly the region it is not allowed to be trusted in, so the
        solver refuses rather than returning a transparent fibre.
        """
        with self.assertRaises(ValueError) as ctx:
            solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                        pump_forward_W=0.2, wavelength_pump_nm=1400.0,
                        spectrum=get_gain_spectrum("triangular"))
        self.assertIn("no gain", str(ctx.exception))

    def test_the_spectrum_scales_the_peak_coefficient(self):
        """Off-peak pumping must give strictly less gain than on-peak."""
        on_peak = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                              pump_forward_W=0.2)
        off_peak = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_forward_W=0.2,
                               wavelength_pump_nm=1500.0,
                               spectrum=get_gain_spectrum("triangular"))
        self.assertLess(off_peak.on_off_gain_dB, on_peak.on_off_gain_dB)
        self.assertGreater(off_peak.on_off_gain_dB, 0.0)


class TestGainProfile(unittest.TestCase):

    def test_the_profile_starts_at_zero_and_ends_at_the_on_off_gain(self):
        solution = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                               pump_backward_W=0.5)
        profile = solution.gain_profile_dB
        self.assertAlmostEqual(float(profile[0]), 0.0, places=9)
        self.assertAlmostEqual(float(profile[-1]), solution.on_off_gain_dB,
                               places=9)

    def test_co_and_counter_pumping_shape_the_profile_differently(self):
        """Where the gain happens is the whole point of distributed pumping."""
        pump = 0.5
        co = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                         pump_forward_W=pump).gain_profile_dB
        counter = solve_raman(length_km=L_KM, gain_peak_W_km=GAIN,
                              pump_backward_W=pump).gain_profile_dB
        half = len(co) // 2
        # co-pumping has delivered most of its gain by mid-span, counter
        # pumping almost none of it
        self.assertGreater(co[half] / co[-1], 0.7)
        self.assertLess(counter[half] / counter[-1], 0.3)


if __name__ == "__main__":
    unittest.main()
