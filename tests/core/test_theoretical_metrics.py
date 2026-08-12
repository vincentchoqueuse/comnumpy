"""The closed-form performance front-ends return a dictionary.

A tutorial that draws a theory curve wants the symbol error rate *and*
the bit error rate, and often the rate the same constellation carries.
Returning one number and taking a `type="ser"` string to say which one
meant the caller wrote the same call twice, and the string was the only
thing that said which. One call, one entry per metric, is what the pages
now use -- these tests hold that shape and check that the entries agree
with the functions they front.
"""
import unittest

import numpy as np

from comnumpy.core.capacity import bicm_capacity, constellation_capacity
from comnumpy.core.metrics import (compute_metric_awgn_theo,
                                   compute_metric_rayleigh_theo,
                                   compute_ser_awgn_psk, compute_ser_awgn_qam)
from comnumpy.core.utils import get_alphabet

SNR = 10 ** (np.arange(0.0, 21.0, 5.0) / 10)


class TestAWGNTheory(unittest.TestCase):

    def test_the_default_is_the_two_closed_forms(self):
        theory = compute_metric_awgn_theo("QAM", 16, SNR)
        self.assertEqual(list(theory), ["ser", "ber"])

    def test_the_entries_match_the_functions_behind_them(self):
        for family, order, reference in (("QAM", 16, compute_ser_awgn_qam),
                                         ("PSK", 8, compute_ser_awgn_psk)):
            with self.subTest(modulation=f"{family}-{order}"):
                theory = compute_metric_awgn_theo(family, order, SNR)
                expected = reference(order, SNR)
                np.testing.assert_allclose(theory["ser"], expected)
                np.testing.assert_allclose(theory["ber"],
                                           expected / np.log2(order))

    def test_the_order_asked_for_is_the_order_returned(self):
        theory = compute_metric_awgn_theo("QAM", 4, SNR, metrics=("ber", "ser"))
        self.assertEqual(list(theory), ["ber", "ser"])

    def test_the_rates_are_the_capacity_functions_at_the_symbol_snr(self):
        """The closed forms take E_b/N_0, the rates take the symbol SNR."""
        alphabet = get_alphabet("QAM", 16)
        theory = compute_metric_awgn_theo("QAM", 16, SNR, metrics=("mi", "gmi"))
        np.testing.assert_allclose(theory["mi"],
                                   constellation_capacity(alphabet, 4 * SNR))
        np.testing.assert_allclose(theory["gmi"],
                                   bicm_capacity(alphabet, 4 * SNR))

    def test_a_scalar_in_gives_scalars_out(self):
        theory = compute_metric_awgn_theo("QAM", 16, 10.0)
        self.assertIsInstance(float(theory["ser"]), float)
        self.assertAlmostEqual(float(theory["ser"]) / 4,
                               float(theory["ber"]), places=12)

    def test_an_unknown_metric_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            compute_metric_awgn_theo("QAM", 16, SNR, metrics=("ser", "evm"))
        self.assertIn("evm", str(ctx.exception))

    def test_an_unknown_modulation_names_the_two_families(self):
        with self.assertRaises(ValueError) as ctx:
            compute_metric_awgn_theo("FSK", 4, SNR)
        self.assertIn("'PSK' or 'QAM'", str(ctx.exception))


class TestRayleighTheory(unittest.TestCase):

    def test_the_two_front_ends_answer_in_the_same_shape(self):
        awgn = compute_metric_awgn_theo("QAM", 16, SNR)
        fading = compute_metric_rayleigh_theo("QAM", 16, SNR)
        self.assertEqual(list(awgn), list(fading))

    def test_fading_is_never_better_than_awgn(self):
        awgn = compute_metric_awgn_theo("PSK", 4, SNR)
        fading = compute_metric_rayleigh_theo("PSK", 4, SNR)
        self.assertTrue(np.all(fading["ser"] >= awgn["ser"]))

    def test_the_bit_error_rate_is_the_gray_approximation(self):
        theory = compute_metric_rayleigh_theo("QAM", 64, SNR, diversity=2)
        np.testing.assert_allclose(theory["ber"], theory["ser"] / 6)


if __name__ == "__main__":
    unittest.main()
