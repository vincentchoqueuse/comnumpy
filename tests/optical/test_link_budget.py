"""The link budgets its own amplifier noise.

Three tutorials used to write the same product by hand -- span count,
spectral density, bandwidth, and a factor for the polarizations -- with
three different spellings of the last term. That factor is the one whose
convention made two pages of this documentation appear to disagree by
several decibels, so the formula now lives next to the block that
generates the noise, and these tests hold it there.
"""
import unittest

import numpy as np

from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import (compute_erbium_doped_fiber_N_ase,
                                    dbm_to_watt, launch_amplitude,
                                    watt_to_dbm)


class TestBudget(unittest.TestCase):

    def link(self, **kwargs):
        params = dict(L_span=100.0, NF_dB=5.0, fs=192e9)
        params.update(kwargs)
        n_spans = params.pop("N_spans", 10)
        return FiberLink(n_spans, **params)

    def test_matches_the_formula_it_replaces(self):
        """The three examples wrote this product; it must not move."""
        link = self.link()
        density = compute_erbium_doped_fiber_N_ase(
            link.fiber.alpha_dB, link.L_span, link.NF_dB,
            nu=link.fiber.carrier_frequency_Hz)
        by_hand = link.N_spans * density * 32e9
        self.assertAlmostEqual(link.budget(32e9)["ase_power_W"], by_hand,
                               delta=1e-18)

    def test_the_second_polarization_doubles_it(self):
        """The 3 dB that made two pages look like they disagreed."""
        link = self.link()
        single = link.budget(32e9)["ase_power_dBm"]
        pair = link.budget(32e9, polarizations=2)["ase_power_dBm"]
        self.assertAlmostEqual(pair - single, 3.0103, places=3)

    def test_published_values_of_the_two_tutorials(self):
        """Both pages quote a budget; both come from here now."""
        # the back-propagation page: 10 x 100 km, NF 5 dB, one polarization
        self.assertAlmostEqual(self.link().budget(32e9)["ase_power_dBm"],
                               -21.88, places=2)
        # the GN-model page: 5 x 100 km, NF 6 dB, both polarizations
        smf = FiberSpec(0.2, gamma=1.3, cd_coefficient=17.0,
                        wavelength_nm=1550.0)
        page = self.link(N_spans=5, NF_dB=6.0, fiber=smf)
        self.assertAlmostEqual(page.budget(32e9, polarizations=2)["ase_power_dBm"],
                               -20.88, places=2)

    def test_it_scales_with_the_spans_and_the_bandwidth(self):
        one = self.link(N_spans=1).budget(32e9)["ase_power_W"]
        self.assertAlmostEqual(self.link(N_spans=10).budget(32e9)["ase_power_W"],
                               10 * one, delta=1e-18)
        self.assertAlmostEqual(self.link(N_spans=1).budget(64e9)["ase_power_W"],
                               2 * one, delta=1e-18)

    def test_it_agrees_with_what_the_link_actually_adds(self):
        """`prepare` and `budget` read one formula, so they cannot drift.

        The block stores its per-span variance over the *simulated*
        bandwidth; the budget quotes the accumulated power over the
        bandwidth a receiver keeps. Same density, two bandwidths.
        """
        link = self.link()
        link(np.zeros(64, dtype=complex))       # prepare() runs on the call
        density = link.budget(32e9)["ase_density_W_per_Hz"]
        self.assertAlmostEqual(link.edfa_N_ase, link.fs * density, delta=1e-18)

    def test_noise_scaling_is_honoured(self):
        """A link with its noise off budgets nothing, and says so."""
        budget = self.link(noise_scaling=0.0).budget(32e9)
        self.assertEqual(budget["ase_power_W"], 0.0)
        self.assertEqual(budget["ase_power_dBm"], -np.inf)

    def test_the_link_really_adds_what_it_budgets(self):
        """The claim the tutorials make: this predicts the simulation.

        A transparent span returns the power it was given, so whatever
        power the output carries above the input is the noise the
        amplifiers put there -- measured over the whole simulated band,
        which is the bandwidth to ask the budget for. Comparing powers
        rather than a sample-wise difference keeps the dispersion out of
        it: it reshapes the waveform and conserves the power.
        """
        from comnumpy.core import Sequential
        from comnumpy.core.generators import GaussianGenerator

        link = FiberLink(5, L_span=80.0, NF_dB=5.0, fs=64e9,
                         use_only_linear=True, name="link")
        chain = Sequential([GaussianGenerator(dbm_to_watt(0.0), name="tx"),
                            link], taps=["tx"]).seed(0)
        y = chain(400000)
        measured = float(np.var(y) - np.var(chain.tap("tx")))
        predicted = link.budget(link.fs)["ase_power_W"]
        self.assertAlmostEqual(10 * np.log10(measured / predicted), 0.0,
                               delta=0.1)

    def test_refusals(self):
        link = self.link()
        for bandwidth in (0.0, -1.0):
            with self.subTest(bandwidth=bandwidth):
                with self.assertRaises(ValueError) as ctx:
                    link.budget(bandwidth)
                self.assertIn("symbol rate", str(ctx.exception))
        with self.assertRaises(ValueError):
            link.budget(32e9, polarizations=3)


class TestLaunchAmplitude(unittest.TestCase):

    def test_it_is_the_square_root_of_the_power(self):
        self.assertAlmostEqual(launch_amplitude(1e-4) ** 2, 1e-4)

    def test_the_pair_splits_the_channel_power(self):
        """Each polarization carries half, i.e. 3 dB less."""
        pair = launch_amplitude(1e-3, polarizations=2)
        self.assertAlmostEqual(watt_to_dbm(pair ** 2), watt_to_dbm(5e-4))

    def test_an_array_of_powers_gives_an_array(self):
        """A launch-power sweep passes the whole axis at once."""
        powers = np.array([1e-4, 1e-3, 1e-2])
        amplitudes = launch_amplitude(powers)
        self.assertEqual(amplitudes.shape, powers.shape)
        np.testing.assert_allclose(amplitudes ** 2, powers)

    def test_a_scalar_stays_a_scalar(self):
        """Scalar in, scalar out -- the library's rule everywhere else."""
        self.assertIsInstance(launch_amplitude(1e-3), float)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            launch_amplitude(-1e-3)
        with self.assertRaises(ValueError):
            launch_amplitude(1e-3, polarizations=0)


if __name__ == "__main__":
    unittest.main()
