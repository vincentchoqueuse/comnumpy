"""A constellation is an object, not two arguments carried side by side.

`get_alphabet("QAM", 16)` gives the array; the bits per symbol, the
energy, the minimum distance and the closed-form error rate were asked
for elsewhere, with the family and the order passed a second time.
Nothing tied the two together, so a page could draw the theory of one
modulation under the measurement of another and look right. These tests
hold the object's promises: the derived facts agree with the alphabet,
the object goes wherever the array went, and `metrics` converts between
the two SNR conventions instead of leaving that to the caller.
"""
import dataclasses
import unittest

import matplotlib
import numpy as np

from comnumpy import Constellation
from comnumpy.core.capacity import bicm_capacity, constellation_capacity
from comnumpy.core.compensators import BlindPhaseCompensation
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import (compute_metric_awgn_theo,
                                   compute_metric_rayleigh_theo)
from comnumpy.core.shaping import maxwell_boltzmann
from comnumpy.core.utils import get_alphabet

matplotlib.use("Agg")

FAMILIES = (("PAM", 4), ("PSK", 8), ("QAM", 16), ("QAM", 64))
SNR_DB = np.array([0.0, 6.0, 12.0, 18.0])


class TestConstruction(unittest.TestCase):

    def test_the_alphabet_is_the_one_get_alphabet_returns(self):
        for family, order in FAMILIES:
            with self.subTest(modulation=f"{family}-{order}"):
                np.testing.assert_array_equal(
                    Constellation(family, order).alphabet,
                    get_alphabet(family, order))

    def test_the_derived_facts_agree_with_the_alphabet(self):
        for family, order in FAMILIES:
            with self.subTest(modulation=f"{family}-{order}"):
                constellation = Constellation(family, order)
                alphabet = constellation.alphabet
                self.assertEqual(constellation.bits_per_symbol,
                                 int(np.log2(order)))
                self.assertAlmostEqual(constellation.energy, 1.0, places=12)
                distances = np.abs(alphabet[:, None] - alphabet[None, :])
                np.fill_diagonal(distances, np.inf)
                self.assertAlmostEqual(constellation.min_distance,
                                       float(np.min(distances)), places=12)

    def test_norm_false_leaves_the_raw_grid(self):
        raw = Constellation("QAM", 16, norm=False)
        self.assertAlmostEqual(raw.energy, 10.0, places=12)

    def test_an_unknown_family_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            Constellation("FSK", 4)

    def test_it_is_frozen(self):
        constellation = Constellation("PSK", 4)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            constellation.order = 8       # type: ignore[misc]


class TestDropIn(unittest.TestCase):
    """The object must go wherever the array went."""

    def test_asarray_returns_the_alphabet(self):
        constellation = Constellation("QAM", 16)
        np.testing.assert_array_equal(np.asarray(constellation),
                                      constellation.alphabet)
        self.assertEqual(len(constellation), 16)

    def test_asarray_honours_dtype_and_copy(self):
        constellation = Constellation("PAM", 4)
        self.assertEqual(np.asarray(constellation, dtype=complex).dtype,
                         np.dtype(complex))
        copied = np.array(constellation, copy=True)
        self.assertFalse(np.shares_memory(copied, constellation.alphabet))

    def test_a_mapper_and_a_demapper_take_it(self):
        constellation = Constellation("QAM", 16)
        symbols = np.arange(16)
        mapped = SymbolMapper(constellation)(symbols)
        np.testing.assert_array_equal(mapped, constellation.alphabet)
        np.testing.assert_array_equal(SymbolDemapper(constellation)(mapped),
                                      symbols)

    def test_a_blind_compensator_takes_it(self):
        constellation = Constellation("PSK", 4)
        received = constellation.alphabet * np.exp(-0.3j)
        compensator = BlindPhaseCompensation(constellation)
        compensator(received)
        self.assertAlmostEqual(float(compensator.theta_), 0.3, places=3)


class TestInfoAndPlot(unittest.TestCase):

    def test_info_reports_what_the_object_holds(self):
        info = Constellation("QAM", 16).info()
        self.assertEqual(info["family"], "QAM")
        self.assertEqual(info["order"], 16)
        self.assertEqual(info["bits_per_symbol"], 4)
        self.assertAlmostEqual(info["energy"], 1.0, places=12)

    def test_a_psk_constellation_has_no_peak_to_average_ratio(self):
        """Every PSK symbol has the same power, so its PAPR is 0 dB."""
        self.assertAlmostEqual(Constellation("PSK", 8).info()["papr_dB"], 0.0)
        self.assertGreater(Constellation("QAM", 64).info()["papr_dB"], 3.0)

    def test_plot_titles_itself_and_returns_the_axis(self):
        ax = Constellation("QAM", 16).plot()
        self.assertEqual(ax.get_title(), "16-QAM")
        self.assertEqual(ax.get_xlabel(), "real part")


class TestMetrics(unittest.TestCase):

    def test_the_default_is_the_two_closed_forms(self):
        self.assertEqual(list(Constellation("QAM", 16).metrics(SNR_DB)),
                         ["ser", "ber"])

    def test_the_error_rates_are_the_functions_behind_them(self):
        for family, order in FAMILIES[1:]:
            with self.subTest(modulation=f"{family}-{order}"):
                constellation = Constellation(family, order)
                measured = constellation.metrics(SNR_DB)
                expected = compute_metric_awgn_theo(family, order,
                                                    10 ** (SNR_DB / 10))
                np.testing.assert_allclose(measured["ser"], expected["ser"])
                np.testing.assert_allclose(measured["ber"], expected["ber"])

    def test_fading_reaches_the_rayleigh_front_end(self):
        constellation = Constellation("PSK", 8)
        measured = constellation.metrics(SNR_DB, channel="rayleigh",
                                         diversity=2)
        expected = compute_metric_rayleigh_theo("PSK", 8, 10 ** (SNR_DB / 10),
                                                diversity=2)
        np.testing.assert_allclose(measured["ser"], expected["ser"])

    def test_per_bit_and_per_symbol_differ_by_ten_log_k(self):
        """The conversion the object exists to make: k times, not once."""
        constellation = Constellation("QAM", 16)
        shift = 10 * np.log10(constellation.bits_per_symbol)
        per_bit = constellation.metrics(SNR_DB)["ser"]
        per_symbol = constellation.metrics(SNR_DB + shift,
                                           per="symbol")["ser"]
        np.testing.assert_allclose(per_bit, per_symbol, rtol=1e-12)

    def test_the_rates_are_the_capacity_functions_at_the_symbol_snr(self):
        constellation = Constellation("QAM", 16)
        rates = constellation.metrics(SNR_DB, metrics=("mi", "gmi"))
        per_symbol = 10 ** (SNR_DB / 10) * 4
        np.testing.assert_allclose(
            rates["mi"], constellation_capacity(constellation.alphabet,
                                                per_symbol))
        np.testing.assert_allclose(
            rates["gmi"], bicm_capacity(constellation.alphabet, per_symbol))

    def test_the_gmi_never_exceeds_the_mi(self):
        """No bit-wise receiver can beat the symbol-wise channel."""
        for family, order in FAMILIES:
            with self.subTest(modulation=f"{family}-{order}"):
                rates = Constellation(family, order).metrics(
                    SNR_DB, metrics=("mi", "gmi"))
                self.assertTrue(np.all(rates["gmi"] <= rates["mi"] + 1e-9))

    def test_the_requested_order_is_the_returned_order(self):
        rates = Constellation("PSK", 4).metrics(6.0,
                                                metrics=("gmi", "ber", "mi"))
        self.assertEqual(list(rates), ["gmi", "ber", "mi"])

    def test_a_shaped_law_is_compared_at_equal_transmit_power(self):
        """px rescales the constellation, or the shaped input is just quieter."""
        constellation = Constellation("QAM", 64)
        law = maxwell_boltzmann(constellation.alphabet, lam=0.64)
        shaped = constellation.metrics(12.0, metrics=("mi",), px=law)["mi"]
        scaled = constellation.alphabet / np.sqrt(
            float(law @ np.abs(constellation.alphabet) ** 2))
        expected = constellation_capacity(scaled, 10 ** 1.2 * 6, px=law)
        np.testing.assert_allclose(shaped, expected)

    def test_a_scalar_in_gives_scalars_out(self):
        theory = Constellation("QAM", 16).metrics(10.0)
        self.assertAlmostEqual(float(theory["ser"]) / 4,
                               float(theory["ber"]), places=12)


class TestMetricsRefusals(unittest.TestCase):

    def test_an_unknown_metric_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            Constellation("QAM", 16).metrics(SNR_DB, metrics=("ser", "evm"))
        self.assertIn("evm", str(ctx.exception))

    def test_an_unknown_snr_convention_says_what_it_costs(self):
        with self.assertRaises(ValueError) as ctx:
            Constellation("QAM", 16).metrics(SNR_DB, per="chip")
        self.assertIn("Eb/N0", str(ctx.exception))
        self.assertIn("6.02 dB", str(ctx.exception))

    def test_a_rate_over_fading_is_refused_rather_than_approximated(self):
        with self.assertRaises(ValueError) as ctx:
            Constellation("QAM", 16).metrics(SNR_DB, channel="rayleigh",
                                             metrics=("mi",))
        self.assertIn("averaged over the fading law", str(ctx.exception))

    def test_a_shaped_error_rate_is_refused(self):
        constellation = Constellation("QAM", 16)
        law = maxwell_boltzmann(constellation.alphabet, lam=0.5)
        with self.assertRaises(ValueError) as ctx:
            constellation.metrics(SNR_DB, px=law)
        self.assertIn("equiprobable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
