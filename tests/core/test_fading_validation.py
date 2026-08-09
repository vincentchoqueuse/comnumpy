"""Golden test: the analytic properties pinned by validation/fading_doppler.py.

Fast version of the validation script (a few seconds instead of ~15 s).
The discriminating property is the first one: the tap autocorrelation must
follow J0(2 pi f_D tau), which fixes the *shape* of the Doppler spectrum
and not merely its support. The remaining checks are cheap enough to keep
alongside it.
"""
import unittest

import numpy as np
from scipy.special import j0

from comnumpy.core.channels import TappedDelayLineChannel
from comnumpy.core.fading import DopplerSpectrum, get_delay_profile, rayleigh_process

FS_TAP = 2000.0
F_DOPPLER = 100.0
N_TAP = 2048
N_RUNS = 600            # 0.2 s; the validation script uses 3000
MAX_LAG = 80            # f_D * tau up to 4

FS_RF = 30.72e6         # LTE 20 MHz rate: delay quantization below 1%
N_RUNS_PDP = 1500       # 1.6 s; the validation script uses 6000
TABLE_SPREAD_NS = {"EVA": 357.0, "ETU": 991.0}


def doppler_statistics(n_runs, *, spectrum=DopplerSpectrum.CLASSICAL, seed=1):
    """Ensemble autocorrelation (normalized by R(0)) and averaged periodogram."""
    rng = np.random.default_rng(seed)
    lags = np.arange(MAX_LAG + 1)
    acf = np.zeros(MAX_LAG + 1, dtype=complex)
    psd = np.zeros(N_TAP)
    for _ in range(n_runs):
        h = rayleigh_process(N_TAP, FS_TAP, F_DOPPLER, spectrum=spectrum, rng=rng)
        psd += np.abs(np.fft.fft(h)) ** 2
        full = np.fft.ifft(np.abs(np.fft.fft(h, 2 * N_TAP)) ** 2)
        acf += full[:MAX_LAG + 1] / (N_TAP - lags)
    return acf / acf[0].real, psd / psd.sum()


class TestDopplerProcess(unittest.TestCase):

    def test_autocorrelation_follows_bessel_j0(self):
        """R(tau)/R(0) = J0(2 pi f_D tau) -- the discriminating property."""
        acf, _ = doppler_statistics(N_RUNS)
        reference = j0(2 * np.pi * F_DOPPLER * np.arange(MAX_LAG + 1) / FS_TAP)
        error = np.abs(acf - reference).max()
        # Measured 0.0099 with 600 runs (0.0026 of it deterministic, from
        # the FFT bin grid); the full validation script reaches 0.0048
        # with 3000. Threshold 0.03 = 3x; a flat Doppler spectrum misses
        # J0 by 0.30, which the next test uses as a live control.
        self.assertLess(error, 0.03, f"autocorrelation departs from J0 by {error:.4f}")

    def test_a_flat_spectrum_is_rejected_by_the_same_check(self):
        """Control: the J0 check is discriminating, not vacuous."""
        acf, _ = doppler_statistics(200, spectrum=DopplerSpectrum.FLAT)
        reference = j0(2 * np.pi * F_DOPPLER * np.arange(MAX_LAG + 1) / FS_TAP)
        # Measured 0.3047 -- ten times the tolerance of the test above.
        self.assertGreater(np.abs(acf - reference).max(), 0.1)

    def test_spectrum_is_band_limited_and_u_shaped(self):
        """Cumulative Doppler power follows arcsin(f/f_D)/pi + 1/2."""
        _, psd = doppler_statistics(N_RUNS)
        freq = np.fft.fftfreq(N_TAP, d=1 / FS_TAP)
        order = np.argsort(freq)
        freq, psd = freq[order], psd[order]
        # exact by construction: out-of-band bins are never filled
        self.assertLess(psd[np.abs(freq) > F_DOPPLER].sum(), 1e-12)
        probe = np.linspace(-0.95, 0.95, 39) * F_DOPPLER
        measured = np.interp(probe, freq, np.cumsum(psd))
        theory = np.arcsin(probe / F_DOPPLER) / np.pi + 0.5
        error = np.abs(measured - theory).max()
        # Measured 0.0054 with 600 runs; a flat spectrum deviates by 0.105.
        self.assertLess(error, 0.03, f"cumulative power off by {error:.4f}")


class TestDelayProfile(unittest.TestCase):

    def channel_powers(self, name, seed=10):
        profile = get_delay_profile(name)
        delays, powers = profile.to_taps(FS_RF)
        channel = TappedDelayLineChannel(profile, fs=FS_RF, f_doppler=0.0, seed=seed)
        x = np.ones(int(delays[-1]) + 64, dtype=complex)
        measured = np.zeros(delays.size)
        for _ in range(N_RUNS_PDP):
            channel(x)
            measured += np.abs(channel.h_[:, 0]) ** 2
        return profile, delays, powers, measured / N_RUNS_PDP

    def test_realized_powers_match_the_profile(self):
        for name in ("EPA", "EVA", "ETU"):
            with self.subTest(profile=name):
                _, _, reference, measured = self.channel_powers(name)
                error = np.abs(measured / reference - 1).max()
                # Measured 3.0% to 6.6% on the worst path, over three
                # seeds and three profiles, with 1500 runs (relative std
                # per path 1/sqrt(1500) = 2.6%). Threshold 15% = ~6 sigma.
                self.assertLess(error, 0.15, f"{name}: path powers off by {error:.1%}")

    def test_rms_delay_spread_matches_the_3gpp_table(self):
        for name, table in TABLE_SPREAD_NS.items():
            with self.subTest(profile=name):
                _, delays, _, measured = self.channel_powers(name)
                tau = delays * 1e9 / FS_RF
                weights = measured / measured.sum()
                mean = float(np.sum(weights * tau))
                spread = float(np.sqrt(np.sum(weights * tau ** 2) - mean ** 2))
                relative = abs(spread / table - 1)
                # Measured 0.2% to 1.4% over three seeds with 1500 runs;
                # delay-grid quantization alone accounts for ~1%
                # (EVA -0.93%, ETU +0.14% at 30.72 MHz). Threshold 5%.
                self.assertLess(relative, 0.05,
                                f"{name}: RMS delay spread {spread:.1f} ns vs "
                                f"{table:.0f} ns published ({relative:.1%})")


if __name__ == "__main__":
    unittest.main()
