"""Golden tests for the distance-spectrum enumerator and the union bound.

Two kinds of checks live here. The *structural* ones are exact: they
compare the enumerator against the closed-form transfer function of the
(5, 7) code, against the free distances the literature quotes, and
against invariants any weight enumerator must satisfy. The *numerical*
one is the fast counterpart of ``validation/fec_union_bound.py``: a short
soft-decision Viterbi run that must stay under the bound at 4 dB.
"""
import unittest

import numpy as np

from comnumpy.fec import ConvolutionalEncoder, ViterbiDecoder
from comnumpy.fec.analysis import (DistanceSpectrum, distance_spectrum,
                                   union_bound_ber)


class TestDistanceSpectrum(unittest.TestCase):

    def test_free_distance_of_k3_code(self):
        """(5, 7) K=3 has free distance 5 (classical value)."""
        self.assertEqual(distance_spectrum((0o5, 0o7)).d_free, 5)

    def test_free_distance_of_the_nasa_code(self):
        """(133, 171) K=7 has free distance 10, derived from its trellis."""
        self.assertEqual(distance_spectrum((0o133, 0o171)).d_free, 10)
        self.assertEqual(distance_spectrum().d_free, 10)   # same code by default

    def test_k3_spectrum_matches_its_transfer_function(self):
        r"""(5, 7): T(D, N) = D^5 N / (1 - 2 D N), so a_d = 2^(d-5).

        Differentiating with respect to N at N = 1 gives
        D^5 / (1 - 2D)^2, hence beta_d = (d - 4) 2^(d - 5). The
        enumerator walks the trellis and must land on the same series --
        this is the check that it counts events, not something else.
        """
        spectrum = distance_spectrum((0o5, 0o7), d_max=15)
        d = spectrum.distances
        np.testing.assert_array_equal(spectrum.a_d, 2 ** (d - 5))
        np.testing.assert_array_equal(spectrum.beta_d, (d - 4) * 2 ** (d - 5))

    def test_spectrum_of_a_weak_code_differs(self):
        """(4, 6) is a different, much weaker code than (5, 7).

        Its first generator is the polynomial 1 (a bare copy of the input
        bit), which cripples the code: d_free drops from 5 to 3. It is
        *not* catastrophic though -- gcd(1, 1 + D) = 1 -- so the
        enumerator returns a spectrum rather than refusing.
        """
        weak = distance_spectrum((0o4, 0o6), n_terms=6)
        good = distance_spectrum((0o5, 0o7), n_terms=6)
        self.assertEqual(weak.d_free, 3)
        self.assertLess(weak.d_free, good.d_free)
        self.assertFalse(np.array_equal(weak.a_d, good.a_d))
        # far fewer competing paths at each weight than the (5,7) code
        self.assertLess(int(weak.a_d.sum()), int(good.a_d.sum()))

    def test_catastrophic_code_is_refused(self):
        """(6, 5) = (1+D, 1+D^2) shares the factor 1+D: a_d is infinite.

        The trellis has a zero-output-weight self-loop on state 3, so an
        error event can accumulate input weight for free. Enumerating it
        would never terminate; the enumerator must say so instead.
        """
        with self.assertRaises(ValueError) as ctx:
            distance_spectrum((0o6, 0o5))
        self.assertIn("catastrophic", str(ctx.exception))

    def test_coefficients_are_counts(self):
        """a_d and beta_d are counts: non-negative integers, beta_d >= a_d.

        The inequality is exact and cheap to see: every error event
        diverges on an input bit 1, so it flips at least one information
        bit, hence its contribution to beta_d is at least its
        contribution to a_d.
        """
        for g in [(0o5, 0o7), (0o133, 0o171), (0o4, 0o6),
                  (0o23, 0o35), (0o133, 0o171, 0o165)]:
            spectrum = distance_spectrum(g, n_terms=6)
            with self.subTest(g=g):
                self.assertTrue(np.issubdtype(spectrum.a_d.dtype, np.integer))
                self.assertTrue(np.issubdtype(spectrum.beta_d.dtype, np.integer))
                self.assertTrue(np.all(spectrum.a_d >= 0))
                self.assertGreater(int(spectrum.a_d[0]), 0)   # d_free is populated
                populated = spectrum.a_d > 0
                self.assertTrue(np.all(spectrum.beta_d[populated]
                                       >= spectrum.a_d[populated]))
                self.assertTrue(np.all(spectrum.beta_d[~populated] == 0))

    def test_rate_one_half_counts_grow_with_d(self):
        """For these rate-1/2 codes, a_d increases along its support.

        More weight budget buys more diverging paths -- but see
        :meth:`test_counts_are_not_monotonic_in_general`: this is an
        observed property of the codes below, not a theorem.
        """
        for g in [(0o5, 0o7), (0o15, 0o17), (0o23, 0o35), (0o133, 0o171)]:
            spectrum = distance_spectrum(g, n_terms=6)
            counts = spectrum.a_d[spectrum.a_d > 0]
            with self.subTest(g=g):
                self.assertTrue(np.all(np.diff(counts) > 0),
                                f"a_d not increasing: {counts.tolist()}")

    def test_counts_are_not_monotonic_in_general(self):
        """The rate-1/3 (133, 171, 165) code has a_19 < a_18: 4 against 9.

        Pinned deliberately. "a_d grows with d" is a habit picked up from
        rate-1/2 tables, not a property of weight enumerators; adding a
        third generator redistributes the weights and breaks it. If a
        future change makes this test fail, the enumerator changed --
        the code did not.
        """
        spectrum = distance_spectrum((0o133, 0o171, 0o165), n_terms=6)
        self.assertEqual(spectrum.d_free, 15)
        np.testing.assert_array_equal(spectrum.a_d[:6], [3, 3, 6, 9, 4, 18])
        self.assertLess(int(spectrum.a_d[4]), int(spectrum.a_d[3]))

    def test_spectrum_does_not_depend_on_generator_order(self):
        """Swapping g_1 and g_2 permutes the coded stream, not the weights."""
        direct = distance_spectrum((0o133, 0o171), n_terms=5)
        swapped = distance_spectrum((0o171, 0o133), n_terms=5)
        np.testing.assert_array_equal(direct.a_d, swapped.a_d)
        np.testing.assert_array_equal(direct.beta_d, swapped.beta_d)

    def test_n_terms_and_d_max_agree(self):
        """The two ways of sizing the enumeration return the same numbers."""
        by_terms = distance_spectrum((0o5, 0o7), n_terms=4)
        by_d_max = distance_spectrum((0o5, 0o7), d_max=8)
        self.assertEqual(by_terms.d_free, by_d_max.d_free)
        np.testing.assert_array_equal(by_terms.a_d, by_d_max.a_d)
        np.testing.assert_array_equal(by_terms.beta_d, by_d_max.beta_d)
        self.assertIsInstance(by_terms, DistanceSpectrum)
        self.assertEqual((by_terms.K, by_terms.k, by_terms.rate), (3, 1, 0.5))

    def test_encoder_agrees_with_the_enumerated_free_distance(self):
        """An input of weight 1 produces a codeword of weight >= d_free.

        The encoder is the independent witness: whatever the enumerator
        claims about the trellis has to show up in the coded stream.
        """
        for g in [(0o5, 0o7), (0o133, 0o171)]:
            d_free = distance_spectrum(g).d_free
            encoder = ConvolutionalEncoder(g)
            bits = np.zeros((1, 40), dtype=int)
            weights = []
            for position in range(20):
                bits[:] = 0
                bits[0, position] = 1
                weights.append(int(encoder(bits).sum()))
            with self.subTest(g=g):
                self.assertEqual(min(weights), d_free)


class TestUnionBound(unittest.TestCase):

    def test_bound_decreases_with_snr(self):
        spectrum = distance_spectrum((0o5, 0o7), d_max=26)
        bound = union_bound_ber(spectrum, np.arange(2.0, 9.0))
        self.assertEqual(bound.shape, (7,))
        self.assertTrue(np.all(np.diff(bound) < 0))

    def test_bound_tends_to_its_leading_term(self):
        """At high SNR only the d_free term survives: check it to 1 %."""
        from scipy.stats import norm
        spectrum = distance_spectrum((0o5, 0o7), d_max=26)
        gamma = 10 ** (12.0 / 10)
        leading = spectrum.beta_d[0] * norm.sf(
            np.sqrt(2 * spectrum.d_free * spectrum.rate * gamma))
        bound = float(union_bound_ber(spectrum, 12.0))
        self.assertLess(abs(bound - leading) / bound, 0.01)

    def test_scalar_and_array_inputs_agree(self):
        spectrum = distance_spectrum((0o5, 0o7), n_terms=8)
        array = union_bound_ber(spectrum, np.array([4.0, 6.0]))
        self.assertAlmostEqual(float(union_bound_ber(spectrum, 4.0)),
                               float(array[0]), places=12)

    def test_simulated_ber_stays_under_the_bound(self):
        """Fast golden version of validation/fec_union_bound.py, at 4 dB.

        Measured with this seed: BER 6.37e-4 (955 errors in 1.5 million
        bits) against a bound of 9.04e-4, i.e. the simulation sits at
        0.70 of the bound -- under it, but of
        the same order, which is the whole content of the comparison. The
        full curve (1 to 5 dB, 35 million bits) lives in the validation
        script.
        """
        g, rate, ebn0_dB = (0o5, 0o7), 0.5, 4.0
        bound = float(union_bound_ber(distance_spectrum(g, d_max=26), ebn0_dB))

        rng = np.random.default_rng(4242)
        sigma2 = 1.0 / (2 * rate * 10 ** (ebn0_dB / 10))
        bits = rng.integers(0, 2, (750, 2000))
        tx = 1.0 - 2.0 * ConvolutionalEncoder(g)(bits)
        rx = tx + rng.normal(scale=np.sqrt(sigma2), size=tx.shape)
        decoded = ViterbiDecoder(g, soft=True)(2 * rx / sigma2)
        ber = np.count_nonzero(decoded != bits) / bits.size

        self.assertLess(ber, bound, f"BER {ber:.3e} exceeds the bound {bound:.3e}")
        self.assertGreater(ber, bound / 5,
                           f"BER {ber:.3e} far below the bound {bound:.3e}: "
                           f"the comparison would be vacuous")


if __name__ == "__main__":
    unittest.main()
