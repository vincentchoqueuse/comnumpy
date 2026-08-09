"""Delay profiles, Doppler processes and the fading channel (decision D43)."""
import unittest

import numpy as np

from comnumpy.core.channels import TappedDelayLineChannel
from comnumpy.core.fading import (DopplerSpectrum, PowerDelayProfile,
                                  available_delay_profiles, get_delay_profile,
                                  rayleigh_process, register_delay_profile)
from comnumpy.exceptions import ShapeError

FS = 15.36e6          # LTE 15.36 MHz, the rate the 3GPP profiles are used at


class TestPowerDelayProfile(unittest.TestCase):

    def test_published_delay_spreads_are_reproduced(self):
        """The self-check of D20, applied to the delay axis.

        EVA and ETU quote an RMS delay spread next to their table; the
        transcription reproduces both, which is what makes the entry
        verifiable rather than merely typed.
        """
        self.assertAlmostEqual(get_delay_profile("EVA").rms_delay_spread_ns,
                               357, delta=0.5)
        self.assertAlmostEqual(get_delay_profile("ETU").rms_delay_spread_ns,
                               991, delta=0.5)

    def test_epa_is_only_checked_on_what_we_can_source(self):
        """EPA's commonly quoted 45 ns is not this table's RMS spread.

        It is much closer to its mean delay. The entry therefore pins the
        tap count only; pinning an unsourced figure would be worse than
        pinning none.
        """
        epa = get_delay_profile("EPA")
        self.assertEqual(epa.n_taps, 7)
        self.assertAlmostEqual(epa.rms_delay_spread_ns, 43.1, delta=0.1)
        self.assertAlmostEqual(epa.mean_delay_ns, 44.2, delta=0.1)

    def test_powers_are_normalized(self):
        for name in available_delay_profiles():
            with self.subTest(profile=name):
                self.assertAlmostEqual(
                    float(get_delay_profile(name).powers_lin.sum()), 1.0)

    def test_self_check_catches_a_mistyped_table(self):
        """A wrong entry must fail at construction, not in a curve."""
        @register_delay_profile("_mistyped")
        def _entry(**kwargs):
            from comnumpy.core.fading import _check_expect
            return _check_expect(
                PowerDelayProfile([0, 100], [0.0, -3.0], standard="_mistyped"),
                {"rms_delay_spread_ns": 991})
        with self.assertRaises(ValueError) as ctx:
            get_delay_profile("_mistyped")
        self.assertIn("mistyped", str(ctx.exception))

    def test_rejects_inconsistent_tables(self):
        with self.assertRaises(ValueError):
            PowerDelayProfile([0, 100], [0.0])              # length mismatch
        with self.assertRaises(ValueError):
            PowerDelayProfile([0, 100, 50], [0, -1, -2])    # not ascending
        with self.assertRaises(ValueError):
            PowerDelayProfile([], [])                       # empty

    def test_to_taps_merges_paths_landing_on_one_sample(self):
        """Below a minimum rate a model loses resolution -- in power."""
        profile = PowerDelayProfile([0, 10, 20], [0.0, 0.0, 0.0])
        delays, powers = profile.to_taps(fs=1e6)     # 1 us grid: all at 0
        np.testing.assert_array_equal(delays, [0])
        self.assertAlmostEqual(float(powers.sum()), 1.0)
        delays, _ = profile.to_taps(fs=1e9)          # 1 ns grid: resolved
        np.testing.assert_array_equal(delays, [0, 10, 20])

    def test_repr_shows_provenance_and_the_profile(self):
        text = repr(get_delay_profile("ETU"))
        self.assertIn("3GPP TS 36.101", text)        # clause, per D20
        self.assertIn("990.9 ns", text)              # the delay spread
        self.assertIn("#", text)                     # the ASCII profile

    def test_unknown_name_lists_the_catalog(self):
        with self.assertRaises(KeyError) as ctx:
            get_delay_profile("NOPE")
        self.assertIn("EVA", str(ctx.exception))


class TestRayleighProcess(unittest.TestCase):

    def test_ensemble_power_is_unity(self):
        rng = np.random.default_rng(4)
        power = [np.mean(np.abs(rayleigh_process(1 << 14, 1e5, 200.0, rng=rng)) ** 2)
                 for _ in range(200)]
        self.assertAlmostEqual(float(np.mean(power)), 1.0, delta=0.05)

    def test_autocorrelation_follows_bessel(self):
        """The discriminating check: R(tau) = J0(2 pi f_D tau)."""
        from scipy.special import j0
        rng = np.random.default_rng(5)
        fs, f_d, n = 1e5, 200.0, 1 << 15
        h = np.stack([rayleigh_process(n, fs, f_d, rng=rng) for _ in range(60)])
        lags = np.arange(0, 400, 40)
        r = np.array([np.mean(h[:, :n - lag] * np.conj(h[:, lag:]))
                      if lag else np.mean(np.abs(h) ** 2) for lag in lags])
        r = r.real / r.real[0]                     # normalize out the power
        np.testing.assert_allclose(r, j0(2 * np.pi * f_d * lags / fs),
                                   atol=0.06)

    def test_bandlimited_to_the_doppler_frequency(self):
        rng = np.random.default_rng(6)
        fs, f_d, n = 1e5, 200.0, 1 << 15
        spectrum = np.abs(np.fft.fft(rayleigh_process(n, fs, f_d, rng=rng))) ** 2
        freq = np.fft.fftfreq(n, d=1 / fs)
        out_of_band = spectrum[np.abs(freq) > 1.05 * f_d]
        # exactly zero up to the IFFT's own round-off
        self.assertLess(float(out_of_band.max()) / float(spectrum.max()), 1e-20)

    def test_static_when_window_is_shorter_than_a_doppler_period(self):
        rng = np.random.default_rng(7)
        h = rayleigh_process(100, fs=1e6, f_doppler=1.0, rng=rng)
        np.testing.assert_allclose(h, h[0])

    def test_zero_doppler_is_block_fading(self):
        rng = np.random.default_rng(8)
        h = rayleigh_process(500, fs=1e6, f_doppler=0.0, rng=rng)
        np.testing.assert_allclose(h, h[0])
        # but a second call draws a different block
        self.assertNotAlmostEqual(
            abs(h[0] - rayleigh_process(500, 1e6, 0.0, rng=rng)[0]), 0.0)

    def test_rejects_invalid_parameters(self):
        for kwargs in ({"n_samples": 0}, {"fs": 0.0}, {"f_doppler": -1.0}):
            base = {"n_samples": 10, "fs": 1e6, "f_doppler": 10.0}
            base.update(kwargs)
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    rayleigh_process(**base)


class TestTappedDelayLineChannel(unittest.TestCase):

    def test_realized_profile_matches_the_table(self):
        profile = get_delay_profile("EVA")
        channel = TappedDelayLineChannel(profile, fs=FS, f_doppler=70.0, seed=3)
        measured = 0.0
        for _ in range(150):
            channel(np.ones(4096, dtype=complex))
            measured = measured + np.mean(np.abs(channel.h_) ** 2, axis=1)
        measured /= 150
        _, target = profile.to_taps(FS)
        np.testing.assert_allclose(measured, target, rtol=0.15)

    def test_output_length_and_causality(self):
        channel = TappedDelayLineChannel(get_delay_profile("EPA"), fs=FS, seed=1)
        x = np.zeros(500, dtype=complex)
        x[0] = 1.0
        y = channel(x)
        self.assertEqual(y.shape, x.shape)
        # a unit impulse comes out as the impulse response, nothing before 0
        support = np.nonzero(np.abs(y) > 1e-12)[0]
        self.assertEqual(int(support[0]), 0)
        self.assertEqual(int(support[-1]), int(channel.delays_[-1]))

    def test_block_fading_is_constant_within_a_call(self):
        channel = TappedDelayLineChannel(get_delay_profile("EPA"), fs=FS,
                                         f_doppler=0.0, seed=2)
        channel(np.ones(2048, dtype=complex))
        for tap in channel.h_:
            np.testing.assert_allclose(tap, tap[0])

    def test_doppler_makes_the_taps_vary(self):
        channel = TappedDelayLineChannel(get_delay_profile("EVA"), fs=FS,
                                         f_doppler=300.0, seed=2)
        channel(np.ones(1 << 16, dtype=complex))
        self.assertGreater(float(np.std(np.abs(channel.h_[0]))), 0.05)

    def test_seed_makes_it_reproducible(self):
        x = np.ones(2048, dtype=complex)
        a = TappedDelayLineChannel(get_delay_profile("ETU"), fs=FS, seed=11)(x)
        b = TappedDelayLineChannel(get_delay_profile("ETU"), fs=FS, seed=11)(x)
        np.testing.assert_array_equal(a, b)

    def test_rejects_2d_and_too_short_input(self):
        channel = TappedDelayLineChannel(get_delay_profile("ETU"), fs=FS, seed=0)
        with self.assertRaises(ShapeError):
            channel(np.ones((2, 500), dtype=complex))
        with self.assertRaises(ShapeError) as ctx:
            channel(np.ones(30, dtype=complex))
        self.assertIn("longest path", str(ctx.exception))

    def test_rice_factor_adds_a_specular_component(self):
        # the window must span several Doppler periods, or every tap is
        # static and a Rayleigh one would also show a non-zero mean
        profile = PowerDelayProfile([0, 3000], [0.0, -3.0], rice_k_dB=13.0,
                                    doppler=DopplerSpectrum.CLASSICAL)
        channel = TappedDelayLineChannel(profile, fs=1e6, f_doppler=1000.0,
                                         seed=9)
        channel(np.ones(1 << 16, dtype=complex))
        # a Rician tap has a non-zero mean; a Rayleigh one does not
        self.assertGreater(abs(np.mean(channel.h_[0])), 0.4)
        self.assertLess(abs(np.mean(channel.h_[1])), 0.2)


if __name__ == "__main__":
    unittest.main()
