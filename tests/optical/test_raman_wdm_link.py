"""A multi-signal Raman solution driving a WDM link (D44 + D45).

The multiplexer sums the channels into one field before the fibre, so a
per-channel Raman gain is not a per-row multiplication: it is a
*transfer function* over the simulated band. ``FiberLink`` interpolates
the solved channels onto the FFT grid and applies the result inside the
split-step loop, half-step by half-step, exactly where the flat gain
used to go.

The properties checked here are the ones a tilt can silently break: the
right channel gets the right gain, the mean stays transparent, and the
single-channel path is untouched.
"""
import unittest

import numpy as np

from comnumpy.optical import FiberLink, WDMGrid, solve_raman
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.raman import SPEED_OF_LIGHT, get_gain_spectrum
from comnumpy.optical.wdm import WDMMultiplexer

SPAN_KM = 80.0
# A Raman tilt is a *terahertz*-scale effect: a four-channel 100 GHz comb
# spans 0.3 THz and tilts by hundredths of a dB. The C band is 4.4 THz
# wide, so a simulation that shows the tilt at all has to carry the whole
# band -- hence a 5 THz sampling rate and 500 GHz spacing here.
FS = 5e12
# 2000 samples at 5 THz put the 500 GHz spacing on exact DFT bins,
# which keeps the multiplex free of the leakage the WDM guard warns about
N = 2000
CENTRE_HZ = 193.4e12
SPACING_HZ = 500e9
# low enough that Kerr four-wave mixing between the tones stays far below
# the tilt being measured: what is under test is the Raman gain, not FWM
TONE_W = 1e-6
SPECTRUM = get_gain_spectrum("blow-wood")


def grid(n_channels=8, spacing_Hz=SPACING_HZ):
    return WDMGrid.uniform(n_channels, spacing_Hz=spacing_Hz,
                           bandwidth_Hz=50e9, center_Hz=CENTRE_HZ)


def wavelengths_nm(frequencies_Hz):
    return 1e9 * SPEED_OF_LIGHT / np.asarray(frequencies_Hz, dtype=float)


def solution(n_channels=8, pump_W=(0.15, 0.25), pump_nm=(1425.0, 1455.0),
             signal_W=TONE_W, bandwidth_Hz=0.0):
    return solve_raman(
        length_km=SPAN_KM, gain_peak_W_km=0.4,
        wavelength_signal_nm=wavelengths_nm(grid(n_channels).frequencies_Hz),
        signal_W=signal_W, pump_backward_W=list(pump_W),
        wavelength_pump_nm=list(pump_nm), spectrum=SPECTRUM,
        bandwidth_Hz=bandwidth_Hz)


def fiber():
    """The FiberSpec carrier must be the comb centre the solve used."""
    return FiberSpec(0.2, wavelength_nm=float(wavelengths_nm(CENTRE_HZ)))


def link(**kwargs):
    kwargs.setdefault("noise_scaling", 0)
    kwargs.setdefault("fiber", fiber())
    return FiberLink(1, L_span=SPAN_KM, StPS=20, fs=FS, **kwargs)


def tones(n_channels=8, power_W=TONE_W):
    """One unmodulated tone per channel, multiplexed into a single field."""
    channels = np.sqrt(power_W) * np.ones((n_channels, N), dtype=complex)
    return WDMMultiplexer(grid(n_channels), fs=FS)(channels)


def channel_power(y, n_channels=8):
    """Power in each channel slot, read off the spectrum."""
    spectrum = np.fft.fft(y) / N
    frequencies = CENTRE_HZ + np.fft.fftfreq(N, d=1 / FS)
    powers = []
    for centre in grid(n_channels).frequencies_Hz:
        mask = np.abs(frequencies - centre) < 25e9
        powers.append(float(np.sum(np.abs(spectrum[mask]) ** 2)))
    return np.array(powers)


class TestTheTiltReachesTheField(unittest.TestCase):

    def test_each_channel_gets_its_own_solved_gain(self):
        """The tilt measured on the field must be the solved tilt."""
        raman = solution()
        chain = link(raman=raman)
        x = tones()
        gains_dB = 10 * np.log10(channel_power(chain(x)) / channel_power(x))
        # the EDFA is flat and makes up the mean, so what survives on the
        # field is the solved gain minus its own mean
        solved = np.asarray(raman.on_off_gain_dB)
        np.testing.assert_allclose(gains_dB, solved - solved.mean(),
                                   atol=0.05)

    def test_the_mean_channel_power_is_transparent(self):
        x = tones()
        y = link(raman=solution())(x)
        ratio = float(np.mean(np.abs(y) ** 2) / np.mean(np.abs(x) ** 2))
        self.assertAlmostEqual(10 * np.log10(ratio), 0.0, delta=0.05)

    def test_the_tilt_survives_the_linear_only_mode(self):
        raman = solution()
        x = tones()
        gains_dB = 10 * np.log10(
            channel_power(link(raman=raman, use_only_linear=True)(x))
            / channel_power(x))
        solved = np.asarray(raman.on_off_gain_dB)
        np.testing.assert_allclose(gains_dB, solved - solved.mean(), atol=0.05)

    def test_two_pumps_leave_a_flatter_field_than_one(self):
        """Same total pump power, one wavelength against two.

        Measured on the field, not on the solution: this is the check
        that the tilt filter carries the design result all the way to
        the samples.
        """
        x = tones()
        one = link(raman=solution(pump_W=(0.4,), pump_nm=(1450.0,)))(x)
        two = link(raman=solution(pump_W=(0.2, 0.2),
                                  pump_nm=(1420.0, 1460.0)))(x)

        def spread(y):
            return float(np.ptp(10 * np.log10(channel_power(y))))

        self.assertLess(spread(two), 0.5 * spread(one))


class TestShapeOfTheState(unittest.TestCase):

    def test_a_multi_signal_solution_builds_a_transfer_function(self):
        chain = link(raman=solution())
        chain.prepare(tones())
        self.assertEqual(chain.raman_step_gain_.shape, (8, 40))
        self.assertEqual(chain.raman_tilt_.shape, (40, N))

    def test_a_single_signal_solution_keeps_the_scalar_path(self):
        chain = link(raman=solve_raman(length_km=SPAN_KM, gain_peak_W_km=0.4,
                                       pump_backward_W=0.3, bandwidth_Hz=0.0))
        chain.prepare(tones(1))
        self.assertEqual(chain.raman_step_gain_.shape, (40,))
        self.assertIsNone(chain.raman_tilt_)

    def test_the_half_step_filters_telescope_to_the_solved_gain(self):
        raman = solution()
        chain = link(raman=raman)
        chain.prepare(tones())
        total = np.prod(chain.raman_tilt_, axis=0)
        frequencies = CENTRE_HZ + np.fft.fftfreq(N, d=1 / FS)
        for centre, expected in zip(grid().frequencies_Hz,
                                    np.asarray(raman.on_off_gain_dB),
                                    strict=True):
            bin_index = int(np.argmin(np.abs(frequencies - centre)))
            self.assertAlmostEqual(20 * np.log10(total[bin_index]), expected,
                                   places=6)


class TestGuards(unittest.TestCase):

    def test_a_band_that_misses_the_comb_is_rejected(self):
        """The FiberSpec wavelength is what centres the simulated band."""
        chain = link(raman=solution(), fiber=FiberSpec(0.2,
                                                       wavelength_nm=1310.0))
        with self.assertRaises(ValueError) as ctx:
            chain.prepare(tones())
        self.assertIn("does not overlap", str(ctx.exception))

    def test_the_span_length_guard_still_applies(self):
        with self.assertRaises(ValueError):
            link(raman=solve_raman(
                length_km=50.0, gain_peak_W_km=0.4,
                wavelength_signal_nm=wavelengths_nm(grid().frequencies_Hz),
                pump_backward_W=0.3, spectrum=SPECTRUM,
                bandwidth_Hz=0.0)).prepare(tones())


if __name__ == "__main__":
    unittest.main()
