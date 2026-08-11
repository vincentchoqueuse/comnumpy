"""Capacity: the reference axis, checked against what theory guarantees.

Every assertion here is a property information theory proves, not a
number this implementation happened to produce. That is what makes the
module usable as a reference for the rest of the library.
"""
import unittest

import numpy as np

from comnumpy.core.capacity import (_noise_quadrature, awgn_capacity,
                                    bicm_capacity, constellation_capacity,
                                    mimo_ergodic_capacity, outage_capacity,
                                    rayleigh_ergodic_capacity, waterfilling)
from comnumpy.core.information import compute_mi
from comnumpy.core.shaping import distribution_entropy, maxwell_boltzmann
from comnumpy.core.utils import get_alphabet

SNR = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])


class TestShannon(unittest.TestCase):

    def test_matches_the_closed_form(self):
        np.testing.assert_allclose(awgn_capacity(np.array([0.0, 1.0, 3.0])),
                                   [0.0, 1.0, 2.0])

    def test_grows_by_one_bit_per_3dB_at_high_snr(self):
        """The 'one bit per 3 dB' rule: C(2 rho) - C(rho) -> 1."""
        self.assertAlmostEqual(float(awgn_capacity(2e6) - awgn_capacity(1e6)),
                               1.0, places=4)


class TestConstellationCapacity(unittest.TestCase):

    def test_saturates_at_log2_M(self):
        """A finite constellation cannot carry more than its own entropy."""
        for mod, order in (("PSK", 4), ("QAM", 16), ("QAM", 64)):
            with self.subTest(modulation=f"{mod}-{order}"):
                capacity = constellation_capacity(get_alphabet(mod, order), 1e8)
                self.assertAlmostEqual(float(capacity), np.log2(order), places=5)

    def test_meets_shannon_at_low_snr(self):
        """A known result: the shaping loss vanishes as rho -> 0."""
        for mod, order in (("PSK", 4), ("QAM", 16)):
            with self.subTest(modulation=f"{mod}-{order}"):
                low = 0.01
                self.assertAlmostEqual(
                    float(constellation_capacity(get_alphabet(mod, order), low)),
                    float(awgn_capacity(low)), places=5)

    def test_never_exceeds_shannon(self):
        """A constrained input cannot beat the Gaussian one."""
        for mod, order in (("PSK", 4), ("QAM", 16), ("QAM", 64)):
            with self.subTest(modulation=f"{mod}-{order}"):
                capacity = constellation_capacity(get_alphabet(mod, order), SNR)
                self.assertTrue(np.all(capacity <= awgn_capacity(SNR) + 1e-9))

    def test_is_monotone_in_snr(self):
        capacity = constellation_capacity(get_alphabet("QAM", 16), SNR)
        self.assertTrue(np.all(np.diff(capacity) > 0))

    def test_bigger_constellation_is_never_worse(self):
        """More points can only help, at equal SNR."""
        small = constellation_capacity(get_alphabet("QAM", 4), SNR)
        large = constellation_capacity(get_alphabet("QAM", 16), SNR)
        self.assertTrue(np.all(large >= small - 1e-9))

    def test_quadrature_meets_its_documented_accuracy(self):
        """The module states a per-constellation accuracy; hold it to it.

        Convergence is not uniform in the constellation size -- a denser
        constellation makes the integrand sharper -- so the claim is a
        table, not a single number, and this is what pins it.
        """
        for mod, order, promised in (("PSK", 4, 1.6e-7), ("QAM", 16, 1.2e-6),
                                     ("QAM", 64, 2.1e-5)):
            with self.subTest(modulation=f"{mod}-{order}"):
                alphabet = get_alphabet(mod, order)
                reference = constellation_capacity(alphabet, SNR, n_nodes=100)
                default = constellation_capacity(alphabet, SNR)
                error = float(np.max(np.abs(default - reference)))
                self.assertLess(error, 1.5 * promised,
                                f"{mod}-{order}: {error:.2e} against a "
                                f"documented {promised:.1e}")


class TestANonUniformInput(unittest.TestCase):
    """``px=``: the same integral, with the input law put back in.

    Shaping makes :math:`P_X` non-uniform, and then the :math:`1/M`
    factors of the classical expression are no longer constants to pull
    out of the sum. These tests pin the three things that must hold: the
    uniform law reproduces the default exactly, the rate saturates at
    the *entropy* rather than at :math:`\\log_2 M`, and the quadrature
    agrees with the Monte-Carlo estimator that reads the same quantity
    off samples.
    """

    PAM16 = np.real(get_alphabet("PAM", 16))

    def test_a_uniform_px_is_the_default(self):
        for snr in SNR:
            with self.subTest(snr=snr):
                self.assertAlmostEqual(
                    float(constellation_capacity(self.PAM16, snr,
                                                 px=np.full(16, 1 / 16))),
                    float(constellation_capacity(self.PAM16, snr)),
                    places=12)

    def test_it_saturates_at_the_entropy_not_at_log2_M(self):
        law = maxwell_boltzmann(self.PAM16, entropy=3.2)
        self.assertAlmostEqual(
            float(constellation_capacity(self.PAM16, 1e9, px=law)),
            distribution_entropy(law), places=6)

    def test_it_agrees_with_the_estimator_read_off_samples(self):
        """Quadrature against Monte-Carlo, which share no code path."""
        law = maxwell_boltzmann(self.PAM16, entropy=3.2)
        energy = float(np.sum(law * self.PAM16 ** 2))
        rng = np.random.default_rng(4)
        for snr_dB in (8.0, 14.0, 20.0):
            sigma2 = energy / 10 ** (snr_dB / 10)
            with self.subTest(snr_dB=snr_dB):
                symbols = rng.choice(16, size=400000, p=law)
                received = (self.PAM16[symbols]
                            + np.sqrt(sigma2) * rng.normal(size=400000))
                # a real channel of variance sigma2 is the complex
                # convention's rho = 1 / (2 sigma2), on both sides
                measured = compute_mi(received, symbols, self.PAM16,
                                      snr=1 / (2 * sigma2), px=law)
                exact = float(constellation_capacity(
                    self.PAM16, 1 / (2 * sigma2), px=law))
                self.assertAlmostEqual(measured, exact, delta=0.01)

    def test_the_refusals(self):
        for bad in (np.full(8, 1 / 8),            # wrong length
                    np.full(16, 1 / 8),           # does not sum to one
                    np.concatenate([[-0.1, 0.2], np.full(14, 0.9 / 14)])):
            with self.subTest(px=bad[:3]):
                with self.assertRaises(ValueError):
                    constellation_capacity(self.PAM16, 10.0, px=bad)


class TestBicmCapacity(unittest.TestCase):

    def test_never_exceeds_the_constellation_capacity(self):
        """The bit-wise interface costs something -- it never pays."""
        for mod, order in (("PSK", 4), ("QAM", 16), ("QAM", 64)):
            with self.subTest(modulation=f"{mod}-{order}"):
                alphabet = get_alphabet(mod, order)
                self.assertTrue(np.all(bicm_capacity(alphabet, SNR)
                                       <= constellation_capacity(alphabet, SNR) + 1e-9))

    def test_gray_qpsk_has_no_bicm_loss(self):
        """A known exact result, and a strong check on the implementation.

        With Gray-labelled QPSK the two bits ride independent quadratures,
        so the bit-wise decomposition loses nothing at all.
        """
        qpsk = get_alphabet("PSK", 4)
        np.testing.assert_allclose(bicm_capacity(qpsk, SNR),
                                   constellation_capacity(qpsk, SNR), atol=1e-9)

    def test_16qam_does_lose_something(self):
        """...and the loss is real for a denser constellation."""
        alphabet = get_alphabet("QAM", 16)
        gap = constellation_capacity(alphabet, 1.0) - bicm_capacity(alphabet, 1.0)
        self.assertGreater(float(gap), 0.05)

    def test_saturates_at_log2_M(self):
        for order in (4, 16):
            with self.subTest(order=order):
                self.assertAlmostEqual(
                    float(bicm_capacity(get_alphabet("QAM", order), 1e8)),
                    np.log2(order), places=5)

    def test_rejects_non_binary_constellations(self):
        with self.assertRaises(ValueError):
            bicm_capacity(np.array([1.0, -0.5, -0.5]), 1.0)


class TestIntegrationMethod(unittest.TestCase):
    """The quadrature is a choice, and the choice must not show.

    Both capacity functions integrate against a Gaussian weight. The
    default rule is matched to that weight; ``method="simpson"`` is the
    textbook composite rule on a truncated grid. They share nothing but
    the integrand, so agreeing is evidence about the integrand and not
    about either rule.
    """

    def test_the_two_rules_agree_on_both_quantities(self):
        for mod, order in (("PSK", 2), ("PSK", 8), ("PAM", 8),
                           ("QAM", 16), ("QAM", 64)):
            alphabet = get_alphabet(mod, order)
            for snr in (1.0, 10 ** 0.8, 10 ** 1.6):
                with self.subTest(modulation=f"{mod}-{order}", snr=snr):
                    self.assertAlmostEqual(
                        float(constellation_capacity(alphabet, snr)),
                        float(constellation_capacity(alphabet, snr,
                                                     method="simpson")),
                        delta=1e-3)
                    self.assertAlmostEqual(
                        float(bicm_capacity(alphabet, snr)),
                        float(bicm_capacity(alphabet, snr, method="simpson")),
                        delta=1e-3)

    def test_the_classical_rule_converges_on_the_default_one(self):
        """Refining it walks towards the answer, it does not wander."""
        alphabet = get_alphabet("QAM", 16)
        exact = float(constellation_capacity(alphabet, 10.0, n_nodes=200))
        errors = [abs(float(constellation_capacity(alphabet, 10.0,
                                                   n_nodes=nodes,
                                                   method="simpson")) - exact)
                  for nodes in (20, 40, 80)]
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:],
                                                  strict=False)), errors)
        self.assertLess(errors[-1], 1e-6)

    def test_both_rules_integrate_the_constant(self):
        """A rule whose weights miss one cannot integrate anything else.

        Gauss-Hermite is exact here whatever the node count; Simpson is
        only asymptotically so, and the defect measures its truncation.
        """
        for method in ("gauss-hermite", "simpson"):
            for n_nodes in (20, 40, 80):
                with self.subTest(method=method, n_nodes=n_nodes):
                    _, weights = _noise_quadrature(method, n_nodes)
                    tolerance = 1e-12 if method == "gauss-hermite" else 1e-3
                    self.assertAlmostEqual(float(np.sum(weights)), 1.0,
                                           delta=tolerance)

    def test_the_default_rule_reproduces_the_first_two_moments(self):
        """It is a standard normal it is integrating, so check that."""
        nodes, weights = _noise_quadrature("gauss-hermite", 20)
        self.assertAlmostEqual(float(weights @ nodes), 0.0, places=12)
        self.assertAlmostEqual(float(weights @ nodes ** 2), 1.0, places=12)

    def test_simpson_is_given_the_odd_node_count_it_needs(self):
        nodes, _ = _noise_quadrature("simpson", 40)
        self.assertEqual(nodes.size, 41)

    def test_an_unknown_method_names_the_ones_that_exist(self):
        with self.assertRaises(ValueError) as ctx:
            constellation_capacity(get_alphabet("QAM", 4), 1.0,
                                   method="romberg")
        message = str(ctx.exception)
        self.assertIn("romberg", message)
        self.assertIn("gauss-hermite", message)
        self.assertIn("simpson", message)

    def test_a_degenerate_node_count_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            bicm_capacity(get_alphabet("QAM", 4), 1.0, n_nodes=1)
        self.assertIn("at least 2 nodes", str(ctx.exception))


class TestFadingCapacity(unittest.TestCase):

    def test_rayleigh_closed_form_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        gains = rng.exponential(size=500_000)
        for rho in (1.0, 10.0, 100.0):
            with self.subTest(snr=rho):
                monte_carlo = float(np.mean(np.log2(1 + rho * gains)))
                self.assertAlmostEqual(float(rayleigh_ergodic_capacity(rho)),
                                       monte_carlo, delta=0.01)

    def test_fading_always_costs(self):
        """Jensen: averaging the log is worse than the log of the average."""
        self.assertTrue(np.all(rayleigh_ergodic_capacity(SNR)
                               < awgn_capacity(SNR)))

    def test_outage_is_far_below_ergodic(self):
        """A slow channel cannot average its own fading out."""
        for rho in (10.0, 100.0):
            with self.subTest(snr=rho):
                self.assertLess(outage_capacity(rho, outage=0.01),
                                float(rayleigh_ergodic_capacity(rho)))

    def test_outage_grows_with_the_tolerated_outage(self):
        rho = 100.0
        self.assertLess(outage_capacity(rho, 0.001), outage_capacity(rho, 0.1))

    def test_outage_rejects_impossible_probabilities(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(outage=bad):
                with self.assertRaises(ValueError):
                    outage_capacity(10.0, outage=bad)


class TestMimoCapacity(unittest.TestCase):

    def test_1x1_reduces_to_the_rayleigh_closed_form(self):
        rng = np.random.default_rng(1)
        siso = mimo_ergodic_capacity(1, 1, 10.0, n_realizations=40_000, rng=rng)
        self.assertAlmostEqual(float(siso),
                               float(rayleigh_ergodic_capacity(10.0)), delta=0.05)

    def test_multiplexing_gain_is_min_nt_nr(self):
        """Telatar: capacity grows as min(Nt, Nr) bits per 3 dB at high SNR."""
        rng = np.random.default_rng(2)
        for n_tx, n_rx in ((2, 2), (2, 4), (4, 2)):
            with self.subTest(antennas=f"{n_tx}x{n_rx}"):
                low = mimo_ergodic_capacity(n_tx, n_rx, 1e5,
                                            n_realizations=4000, rng=rng)
                high = mimo_ergodic_capacity(n_tx, n_rx, 2e5,
                                             n_realizations=4000, rng=rng)
                slope = float(high - low)
                self.assertAlmostEqual(slope, min(n_tx, n_rx), delta=0.15)

    def test_more_antennas_never_hurt(self):
        rng = np.random.default_rng(3)
        base = mimo_ergodic_capacity(2, 2, 10.0, n_realizations=8000, rng=rng)
        more = mimo_ergodic_capacity(2, 4, 10.0, n_realizations=8000, rng=rng)
        self.assertGreater(float(more), float(base))


class TestWaterfilling(unittest.TestCase):

    def test_conserves_the_power_budget(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            gains = rng.exponential(size=16)
            power, _ = waterfilling(gains, snr=10.0)
            self.assertAlmostEqual(float(power.sum()), 16.0, places=9)

    def test_satisfies_the_kkt_water_level(self):
        """Every active channel sits at the same water level."""
        rng = np.random.default_rng(5)
        gains = rng.exponential(size=32)
        snr = 5.0
        power, _ = waterfilling(gains, snr)
        active = power > 1e-12
        level = power[active] + 1.0 / (snr * gains[active])
        np.testing.assert_allclose(level, level[0], rtol=1e-9)
        # and every inactive channel is below that level
        inactive = ~active
        if np.any(inactive):
            self.assertTrue(np.all(1.0 / (snr * gains[inactive]) >= level[0] - 1e-9))

    def test_beats_uniform_allocation(self):
        rng = np.random.default_rng(6)
        for _ in range(20):
            gains = rng.exponential(size=8)
            _, optimal = waterfilling(gains, snr=10.0)
            uniform = float(np.mean(np.log2(1 + 10.0 * gains)))
            self.assertGreaterEqual(optimal, uniform - 1e-12)

    def test_equal_gains_give_equal_power(self):
        power, capacity = waterfilling(np.ones(5), snr=10.0)
        np.testing.assert_allclose(power, 1.0)
        self.assertAlmostEqual(capacity, float(awgn_capacity(10.0)))

    def test_drops_channels_below_the_water_level(self):
        power, _ = waterfilling(np.array([4.0, 1.0, 1e-6]), snr=10.0)
        self.assertEqual(float(power[-1]), 0.0)
        self.assertGreater(float(power[0]), float(power[1]))

    def test_rejects_invalid_gains(self):
        with self.assertRaises(ValueError):
            waterfilling(np.array([1.0, -1.0]), snr=10.0)
        with self.assertRaises(ValueError):
            waterfilling(np.zeros(3), snr=10.0)


if __name__ == "__main__":
    unittest.main()
