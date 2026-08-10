"""The GN model against the reference implementation of the same paper.

Decision D7 asks that a physical model be pinned to something outside
this library. For the GN model the outside thing is GNPy (Telecom Infra
Project), the open-source planning tool whose ``NliSolver`` implements
the same two equations, is used by network operators to dimension real
links, and is validated against measurements.

The first class here does not *import* GNPy -- adding a heavyweight
dependency to run four lines of arithmetic would be a poor trade. It
re-transcribes those four lines in **SI units** (metres, s^2/m), which
is what GNPy works in, while the library under test works in kilometres
and ps^2/km. Agreement to twelve digits therefore tests two independent
things at once: that the formula is the same, and that the unit
conversions on the way in and out of it cancel.

The remaining classes pin the properties the closed form exists for --
the cubic growth, the logarithmic bandwidth dependence, the position of
the optimum -- since a formula can be transcribed correctly and still be
wired up wrongly.
"""
import unittest

import numpy as np

from comnumpy.optical.fiber import FiberSpec, get_fiber
from comnumpy.optical.gn_model import (gn_model_nli_power, gn_model_psi,
                                       gn_model_snr, optimal_launch_power)

# -- The reference, re-transcribed in SI units ------------------------
#
# gnpy/core/science_utils.py, NliSolver._psi and NliSolver._gn_analytic
# (eq. 123 and eq. 120 of arXiv:1209.0394), with the loops written out
# rather than vectorized so that the correspondence with the source is
# readable. Everything below is in metres, seconds and watts.


def gnpy_psi(df_Hz, cut_baud_Hz, pump_baud_Hz, beta2_s2_m,
             effective_length_m, asymptotic_length_m):
    """GNPy's ``_psi``, one (cut, pump) pair at a time, in SI units."""
    beta2 = abs(beta2_s2_m)
    right = df_Hz + pump_baud_Hz / 2
    left = df_Hz - pump_baud_Hz / 2
    psi = (np.arcsinh(np.pi ** 2 * asymptotic_length_m * beta2
                      * cut_baud_Hz * right)
           - np.arcsinh(np.pi ** 2 * asymptotic_length_m * beta2
                        * cut_baud_Hz * left)) / 2
    return psi * effective_length_m ** 2 / (
        2 * np.pi * beta2 * asymptotic_length_m)


def gnpy_nli(alpha_per_m, beta2_s2_m, gamma_per_W_m, length_m,
             powers_W, frequencies_Hz, bauds_Hz):
    """GNPy's ``_gn_analytic`` plus ``compute_nli``, in SI units.

    ``nli_matrix = cut_power * pump_power ** 2 * eta``, summed over the
    pump axis, with ``eta = gamma ** 2 * weight * psi / pump_baud ** 2``.
    """
    effective_length = (1 - np.exp(-alpha_per_m * length_m)) / alpha_per_m
    asymptotic_length = 1 / alpha_per_m
    n_channels = len(powers_W)
    nli = np.zeros(n_channels)
    for cut in range(n_channels):
        total = 0.0
        for pump in range(n_channels):
            weight = 16 / 27 if cut == pump else 32 / 27
            psi = gnpy_psi(frequencies_Hz[pump] - frequencies_Hz[cut],
                           bauds_Hz[cut], bauds_Hz[pump], beta2_s2_m,
                           effective_length, asymptotic_length)
            eta = gamma_per_W_m ** 2 * weight * psi / bauds_Hz[pump] ** 2
            total += eta * powers_W[cut] * powers_W[pump] ** 2
        nli[cut] = total
    return nli


def to_si(fiber):
    """The fibre's coefficients in the units GNPy uses."""
    return (fiber.alpha_per_km / 1e3,       # 1/km -> 1/m
            fiber.beta2 * 1e-24 * 1e-3,     # ps^2/km -> s^2/m
            fiber.gamma / 1e3)              # 1/W/km -> 1/W/m


class TestAgainstGNPy(unittest.TestCase):
    """Twelve digits against an independent transcription (D7)."""

    def test_psi_agrees_for_every_spacing(self):
        fiber = get_fiber("SMF")
        alpha_si, beta2_si, _ = to_si(fiber)
        length_km = 80.0
        for spacing_GHz in (0.0, 25.0, 50.0, 100.0, 375.0, 1000.0):
            for baud_GBd in (10.0, 32.0, 64.0, 96.0):
                with self.subTest(spacing=spacing_GHz, baud=baud_GBd):
                    got = gn_model_psi(
                        spacing_GHz * 1e9, baud_GBd * 1e9, baud_GBd * 1e9,
                        fiber.beta2, fiber.effective_length_km(length_km),
                        1 / fiber.alpha_per_km)
                    want = gnpy_psi(
                        spacing_GHz * 1e9, baud_GBd * 1e9, baud_GBd * 1e9,
                        beta2_si, (1 - np.exp(-alpha_si * length_km * 1e3))
                        / alpha_si, 1 / alpha_si)
                    # psi is the one quantity here that is *not* unit
                    # invariant: its bracket is dimensionless but its
                    # prefactor goes as a squared length, so the SI
                    # value is 1e6 times the kilometre one. The factor
                    # cancels against gamma^2 in the NLI power, which
                    # the next tests check without any conversion.
                    self.assertAlmostEqual(
                        float(got) / (want * 1e-6), 1.0, places=12)

    def test_psi_is_even_in_the_spacing(self):
        """Which is why the implementation may take an absolute value."""
        fiber = get_fiber("SMF")
        arguments = (32e9, 32e9, fiber.beta2,
                     fiber.effective_length_km(80.0), 1 / fiber.alpha_per_km)
        for spacing in (12.5e9, 50e9, 200e9):
            with self.subTest(spacing=spacing):
                self.assertAlmostEqual(
                    float(gn_model_psi(spacing, *arguments))
                    / float(gn_model_psi(-spacing, *arguments)), 1.0,
                    places=12)

    def test_nli_agrees_on_a_realistic_comb(self):
        for fiber_name in ("SMF", "NZDSF", "DCF"):
            fiber = get_fiber(fiber_name)
            alpha_si, beta2_si, gamma_si = to_si(fiber)
            n_channels = 11
            frequencies = 193.4e12 + 50e9 * np.arange(n_channels)
            bauds = np.full(n_channels, 32e9)
            # Deliberately unequal powers: an equal-power comb would
            # hide a swapped cut/pump index.
            powers = 1e-3 * np.linspace(0.5, 1.5, n_channels)
            length_km = 80.0
            with self.subTest(fiber=fiber_name):
                got = gn_model_nli_power(
                    fiber, span_length_km=length_km, n_spans=1,
                    powers_W=powers, frequencies_Hz=frequencies,
                    baud_rates_Hz=bauds)
                want = gnpy_nli(alpha_si, beta2_si, gamma_si,
                                length_km * 1e3, powers, frequencies, bauds)
                np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_nli_agrees_with_mixed_symbol_rates(self):
        """A flex-grid comb: the cut and pump rates enter differently."""
        fiber = get_fiber("SMF")
        alpha_si, beta2_si, gamma_si = to_si(fiber)
        frequencies = np.array([193.4e12, 193.475e12, 193.6e12])
        bauds = np.array([32e9, 64e9, 96e9])
        powers = np.array([1e-3, 2e-3, 3e-3])
        got = gn_model_nli_power(fiber, span_length_km=100.0, n_spans=1,
                                 powers_W=powers, frequencies_Hz=frequencies,
                                 baud_rates_Hz=bauds)
        want = gnpy_nli(alpha_si, beta2_si, gamma_si, 100e3, powers,
                        frequencies, bauds)
        np.testing.assert_allclose(got, want, rtol=1e-12)


class TestAgainstPublishedBenchmark(unittest.TestCase):
    """The number a different group published, from a different method.

    P. Serena and A. Bononi, "A time-domain extended Gaussian noise
    model", J. Lightwave Technol. 33(7), pp. 1459-1472, 2015,
    Section III: 15 channels at 32 GBd on a 37.5 GHz grid, 5 x 100 km
    of SMF (0.2 dB/km, D = 17 ps/nm/km, gamma = 1.3 /W/km), -4 dBm per
    channel, sinc pulses, dual polarization. They report the normalized
    NLI coefficient a_NL, defined by sigma^2_NLI = a_NL * P^3 with
    powers in mW, measured by split-step Monte Carlo:

        Gaussian  -23.5 dB      16QAM  -25.1 dB      QPSK  -26.3 dB

    The Gaussian row is the one the GN model claims to predict; the
    other two are the modulation-format effect the GN model is blind to
    by construction, and are quoted here only so the sign and size of
    that blindness stay on the record.
    """

    PUBLISHED_GAUSSIAN_dB = -23.5
    SMF = FiberSpec(0.2, gamma=1.3, cd_coefficient=17.0, wavelength_nm=1550.0)
    N_CHANNELS = 15
    SPACING_Hz = 37.5e9
    BAUD_Hz = 32e9
    POWER_W = 1e-3 * 10 ** (-4 / 10)

    def a_nl_dB(self, **kwargs):
        """10*log10(a_NL) in mW^-2 for the centre channel."""
        offsets = self.SPACING_Hz * (np.arange(self.N_CHANNELS)
                                     - (self.N_CHANNELS - 1) / 2)
        nli = gn_model_nli_power(
            self.SMF, span_length_km=100.0, n_spans=5,
            powers_W=np.full(self.N_CHANNELS, self.POWER_W),
            frequencies_Hz=self.SMF.carrier_frequency_Hz + offsets,
            baud_rates_Hz=np.full(self.N_CHANNELS, self.BAUD_Hz), **kwargs)
        centre = nli[self.N_CHANNELS // 2]
        # a_NL = sigma^2/P^3; in mW^-2 that is the SI value over 1e6.
        return 10 * np.log10(centre / self.POWER_W ** 3 / 1e6)

    def test_within_half_a_dB_of_the_published_value(self):
        self.assertAlmostEqual(self.a_nl_dB(), self.PUBLISHED_GAUSSIAN_dB,
                               delta=0.5)

    def test_the_value_does_not_drift(self):
        """Pinned to what this implementation returns today."""
        self.assertAlmostEqual(self.a_nl_dB(), -23.30, places=2)

    def test_the_gn_model_over_estimates_every_real_format(self):
        """Both published modulated values sit below the GN prediction.

        This is the EGN effect, and its sign is not negotiable: a real
        constellation has a smaller fourth moment than a Gaussian, so it
        generates less interference. A GN implementation that came out
        *below* a measured 16QAM point would be wrong.
        """
        predicted = self.a_nl_dB()
        for published in (-25.1, -26.3):
            with self.subTest(published=published):
                self.assertGreater(predicted, published)


class TestThePolarizationWeights(unittest.TestCase):
    """Table I of Serena and Bononi (2015).

    The dual-polarization row carries 16/27 and 32/27; the scalar row
    carries (8/9)^2 times 2 and 4, and since a scalar NLSE applies gamma
    rather than 8*gamma/9 the (8/9)^2 cancels, leaving 2 and 4. The ratio
    is 3.375, and it was confirmed to 0.04 dB by simulating the same
    single-channel link twice, once with a one-dimensional field and once
    with a polarization pair (validation/optical_gn_model.py).
    """

    def nli(self, polarizations, n_channels=1):
        fiber = get_fiber("SMF")
        offsets = 50e9 * (np.arange(n_channels) - (n_channels - 1) / 2)
        return gn_model_nli_power(
            fiber, span_length_km=100.0, n_spans=5,
            powers_W=np.full(n_channels, 5e-4),
            frequencies_Hz=fiber.carrier_frequency_Hz + offsets,
            baud_rates_Hz=np.full(n_channels, 32e9),
            polarizations=polarizations)[n_channels // 2]

    def test_the_scalar_case_is_exactly_27_over_8_times_the_manakov_one(self):
        for n_channels in (1, 5, 11):
            with self.subTest(n_channels=n_channels):
                self.assertAlmostEqual(
                    self.nli(1, n_channels) / self.nli(2, n_channels),
                    27 / 8, places=12)

    def test_the_ratio_is_5_28_dB(self):
        self.assertAlmostEqual(10 * np.log10(self.nli(1) / self.nli(2)),
                               5.28, places=2)

    def test_an_impossible_polarization_count_is_refused(self):
        for count in (0, 3, -1):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    self.nli(count)


class TestTheScalingLaws(unittest.TestCase):
    """The three properties the model is quoted for."""

    def setUp(self):
        self.fiber = get_fiber("SMF")
        self.frequencies = np.array([193.4e12])
        self.bauds = np.array([32e9])

    def nli(self, power_W, n_spans=1, coherence_exponent=0.0):
        return gn_model_nli_power(
            self.fiber, span_length_km=80.0, n_spans=n_spans,
            powers_W=np.array([power_W]), frequencies_Hz=self.frequencies,
            baud_rates_Hz=self.bauds,
            coherence_exponent=coherence_exponent)[0]

    def test_the_nli_is_cubic_in_the_launch_power(self):
        """P^3 is the model's headline claim: +1 dB in, +3 dB out."""
        reference = self.nli(1e-3)
        for factor in (0.1, 0.5, 2.0, 10.0):
            with self.subTest(factor=factor):
                self.assertAlmostEqual(
                    self.nli(factor * 1e-3) / (reference * factor ** 3),
                    1.0, places=12)

    def test_spans_accumulate_incoherently_by_default(self):
        for n_spans in (1, 5, 20, 50):
            with self.subTest(n_spans=n_spans):
                self.assertAlmostEqual(
                    self.nli(1e-3, n_spans=n_spans)
                    / (n_spans * self.nli(1e-3)), 1.0, places=12)

    def test_the_coherence_exponent_adds_a_power_of_the_span_count(self):
        for epsilon in (0.05, 0.1):
            with self.subTest(epsilon=epsilon):
                self.assertAlmostEqual(
                    self.nli(1e-3, n_spans=20, coherence_exponent=epsilon)
                    / (20 ** epsilon * self.nli(1e-3, n_spans=20)),
                    1.0, places=12)

    def test_the_bandwidth_dependence_is_logarithmic(self):
        """The asinh: each doubling of the comb costs a bounded amount.

        If every neighbour interfered as much as the channel does with
        itself, N channels would cost 10*log10(2N - 1) dB. They cost far
        less, and the increment shrinks as the comb widens -- that
        shrinking is the whole point of the asinh.
        """
        increments = []
        previous = None
        for n_channels in (1, 3, 9, 27, 81):
            comb = 193.4e12 + 50e9 * (np.arange(n_channels)
                                      - (n_channels - 1) / 2)
            nli = gn_model_nli_power(
                self.fiber, span_length_km=80.0, n_spans=1,
                powers_W=np.full(n_channels, 1e-3), frequencies_Hz=comb,
                baud_rates_Hz=np.full(n_channels, 32e9))
            centre = nli[n_channels // 2]
            naive_dB = 10 * np.log10(2 * n_channels - 1)
            if previous is not None:
                increment = 10 * np.log10(centre / previous)
                self.assertLess(increment, naive_dB)
                increments.append(increment)
            previous = centre
        # Tripling the comb costs less each time it is tripled.
        for smaller, larger in zip(increments[1:], increments[:-1],
                                  strict=True):
            self.assertLess(smaller, larger)


class TestTheOptimum(unittest.TestCase):

    def test_the_optimum_is_where_the_nli_is_half_the_ase(self):
        for ase in (1e-7, 1e-6, 1e-5):
            for eta in (1e2, 5e3, 1e5):
                with self.subTest(ase=ase, eta=eta):
                    power, _ = optimal_launch_power(ase, eta)
                    self.assertAlmostEqual(eta * power ** 3 / (ase / 2),
                                           1.0, places=12)

    def test_the_optimum_really_is_the_maximum(self):
        """Checked against a fine sweep, not against the derivation."""
        ase, eta = 2e-6, 3e3
        power, snr = optimal_launch_power(ase, eta)
        sweep = np.logspace(np.log10(power) - 1, np.log10(power) + 1, 20001)
        self.assertAlmostEqual(
            float(np.max(gn_model_snr(ase, eta, sweep))) / snr, 1.0, places=6)

    def test_the_optimum_moves_as_the_cube_root_of_the_ase(self):
        """Ten times the ASE, 3.33 dB more power -- not 10 dB."""
        reference, _ = optimal_launch_power(1e-6, 5e3)
        moved, _ = optimal_launch_power(1e-5, 5e3)
        self.assertAlmostEqual(10 * np.log10(moved / reference), 10 / 3,
                               places=9)

    def test_missing_the_optimum_upwards_costs_more_than_downwards(self):
        """The asymmetry that makes operators run under the optimum.

        Linear on one side, cubic on the other. The four numbers are
        the ones quoted in the docstring of ``optimal_launch_power``,
        and they depend on nothing but the shape of P/(A + eta P^3): the
        two constants cancel, so any (ase, eta) gives the same table.
        """
        expected = {0.5: (0.0596, 0.0552), 1.0: (0.2442, 0.2102),
                    2.0: (0.9966, 0.7529), 3.0: (2.2041, 1.5042)}
        for ase, eta in ((1e-6, 5e3), (3e-7, 1.2e4)):
            power, snr = optimal_launch_power(ase, eta)
            for offset_dB, (up, down) in expected.items():
                with self.subTest(ase=ase, eta=eta, offset=offset_dB):
                    above = float(gn_model_snr(
                        ase, eta, power * 10 ** (offset_dB / 10)))
                    below = float(gn_model_snr(
                        ase, eta, power * 10 ** (-offset_dB / 10)))
                    self.assertAlmostEqual(10 * np.log10(snr / above), up,
                                           places=4)
                    self.assertAlmostEqual(10 * np.log10(snr / below), down,
                                           places=4)
                    self.assertGreater(10 * np.log10(snr / above),
                                       10 * np.log10(snr / below))

    def test_the_snr_reduces_to_the_linear_regime(self):
        """At vanishing power the fibre term disappears, leaving P/ASE."""
        snr = gn_model_snr(1e-6, 5e3, 1e-9)
        self.assertAlmostEqual(float(snr) / (1e-9 / 1e-6), 1.0, places=6)


class TestTheGuards(unittest.TestCase):

    def test_a_dispersionless_fibre_is_refused(self):
        fiber = FiberSpec(0.2, cd_coefficient=0.0)
        with self.assertRaises(ValueError) as caught:
            gn_model_psi(0.0, 32e9, 32e9, fiber.beta2, 21.0, 21.7)
        self.assertIn("beta2 = 0", str(caught.exception))

    def test_the_three_comb_arrays_must_agree(self):
        with self.assertRaises(ValueError):
            gn_model_nli_power(get_fiber("SMF"), span_length_km=80.0,
                               n_spans=1, powers_W=np.ones(3) * 1e-3,
                               frequencies_Hz=193.4e12 + 50e9 * np.arange(2),
                               baud_rates_Hz=np.full(3, 32e9))

    def test_a_link_has_at_least_one_span(self):
        with self.assertRaises(ValueError):
            gn_model_nli_power(get_fiber("SMF"), span_length_km=80.0,
                               n_spans=0, powers_W=np.array([1e-3]),
                               frequencies_Hz=np.array([193.4e12]),
                               baud_rates_Hz=np.array([32e9]))

    def test_the_optimum_needs_both_noises(self):
        for ase, eta in ((0.0, 5e3), (1e-6, 0.0), (-1e-6, 5e3)):
            with self.subTest(ase=ase, eta=eta):
                with self.assertRaises(ValueError):
                    optimal_launch_power(ase, eta)


if __name__ == "__main__":
    unittest.main()
