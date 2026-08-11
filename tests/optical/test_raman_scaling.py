"""The waveguide correction of the Raman gain, and its off switch.

``shape()`` is a property of the glass: it depends on the Stokes shift
alone, which is what makes the user's instinct -- "g only depends on the
difference" -- correct about the *material*. The coefficient that
multiplies P_i P_j in a power equation is not that shape but
g_R / A_eff, and the effective area belongs to the *waveguide*, so it
does depend on the absolute frequency. Ignoring it costs a few per cent
across one band and a wrong tilt across C+L+S.

The tests below pin the three things a caller depends on: a spectrum
that does not say where it was quoted changes nothing at all (so every
existing result is untouched), the law has the right sign and the right
anchor, and the three numbers that define it travel together.
"""
import unittest

import numpy as np

from comnumpy.optical.raman import RamanGainSpectrum, get_gain_spectrum

SPEED_OF_LIGHT = 2.99792458e8
# The SSMF numbers the measured profile of D'Amico et al. is quoted at.
REFERENCE = (SPEED_OF_LIGHT / 206184634112792.0 * 1e9, 75.74659443542413, 4.2)


def spectrum(quoted_at=REFERENCE):
    return RamanGainSpectrum(triangular=13.2, quoted_at=quoted_at)


class TestTheCorrectionIsOptional(unittest.TestCase):

    def test_a_spectrum_without_provenance_scales_by_one(self):
        """Every catalogue entry and every existing call is untouched."""
        for name in ("blow-wood", "triangular"):
            with self.subTest(spectrum=name):
                factors = get_gain_spectrum(name).pair_scaling(
                    np.array([191e12, 196e12, 205e12]))
                np.testing.assert_allclose(factors, 1.0)

    def test_the_shape_itself_never_moves(self):
        """The correction multiplies the shape, it does not redefine it."""
        plain, scaled = spectrum(None), spectrum()
        shifts = np.array([1e12, 6e12, 13.2e12, 20e12])
        np.testing.assert_allclose(plain.shape(shifts), scaled.shape(shifts))


class TestTheLaw(unittest.TestCase):

    def test_it_is_one_at_the_frequency_it_was_quoted_at(self):
        """A_eff(nu_0) = A_0 by construction, so the anchor is exact."""
        reference_Hz = SPEED_OF_LIGHT / (REFERENCE[0] * 1e-9)
        factor = spectrum().pair_scaling(np.array([reference_Hz]))
        self.assertAlmostEqual(float(factor[0, 0]), 1.0, places=12)

    def test_the_gain_falls_towards_longer_wavelengths(self):
        """The mode spreads, the area grows, the same shift buys less.

        This is the sign that matters: it is what flattens a wideband
        Raman tilt, and getting it backwards would steepen one.
        """
        frequencies = np.array([185e12, 191e12, 196e12, 206.18e12])
        diagonal = np.diag(spectrum().pair_scaling(frequencies))
        self.assertTrue(np.all(np.diff(diagonal) > 0),
                        f"expected the factor to grow with frequency, got "
                        f"{diagonal}")
        self.assertLess(diagonal[0], 1.0)

    def test_a_pair_is_charged_the_mean_of_the_two_areas(self):
        """Symmetric in the area, so the matrix is not lopsided by it."""
        frequencies = np.array([191e12, 205e12])
        factors = spectrum().pair_scaling(frequencies)
        # strip the partner-frequency factor to leave the area term
        area_term = factors / (frequencies[None, :]
                               / (SPEED_OF_LIGHT / (REFERENCE[0] * 1e-9)))
        self.assertAlmostEqual(float(area_term[0, 1]), float(area_term[1, 0]),
                               places=12)

    def test_the_correction_stays_a_few_per_cent_over_one_band(self):
        """Small in the C band -- which is why it went unnoticed.

        Measured at 5.3 % across 191.3-196.05 THz. Small enough that no
        single-band result moves much, large enough that it was worth
        0.8 dB of the tilt against GNPy's reference case.
        """
        band = 191.3e12 + 50e9 * np.arange(96)
        factors = np.diag(spectrum().pair_scaling(band))
        spread = abs(float(np.max(factors) - np.min(factors)))
        self.assertAlmostEqual(spread, 0.053, places=3)


class TestTheGuards(unittest.TestCase):

    def test_the_three_numbers_travel_together(self):
        for broken in ((1550.0, 80.0), (1550.0,), (1550.0, 80.0, 4.2, 1.0)):
            with self.subTest(quoted_at=broken):
                with self.assertRaises(ValueError):
                    spectrum(broken)

    def test_none_of_them_may_be_zero_or_negative(self):
        for broken in ((0.0, 80.0, 4.2), (1550.0, -1.0, 4.2),
                       (1550.0, 80.0, 0.0)):
            with self.subTest(quoted_at=broken):
                with self.assertRaises(ValueError):
                    spectrum(broken)


if __name__ == "__main__":
    unittest.main()
