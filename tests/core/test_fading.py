"""Delay profiles, Doppler processes and the fading channel (decision D43)."""
import unittest

import numpy as np

from comnumpy.core import fading
from comnumpy.core.channels import TappedDelayLineChannel
from comnumpy.core.fading import (DopplerSpectrum, PowerDelayProfile,
                                  available_delay_profiles, get_delay_profile,
                                  rayleigh_process, register_delay_profile)
from comnumpy.exceptions import ShapeError

FS = 15.36e6          # LTE 15.36 MHz, the rate the 3GPP profiles are used at


class TestPowerDelayProfile(unittest.TestCase):
    """The catalog is process-global state, so restore it around each test.

    Without this, a test that registers a deliberately broken entry
    leaves it visible to every later test that iterates the catalog --
    and the suite only passes because unittest happens to run the names
    in a favourable alphabetical order.
    """

    def setUp(self):
        self._registry = dict(fading._PROFILE_REGISTRY)

    def tearDown(self):
        fading._PROFILE_REGISTRY.clear()
        fading._PROFILE_REGISTRY.update(self._registry)

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

    def test_max_excess_delays_are_reproduced(self):
        """The third published figure, and the one that confirms EPA's delays."""
        for name, span in (("EPA", 410), ("EVA", 2510), ("ETU", 5000)):
            with self.subTest(profile=name):
                self.assertEqual(float(get_delay_profile(name).delays_ns[-1]),
                                 span)

    def test_epa_is_only_checked_on_what_we_can_source(self):
        """EPA matches two of its three published figures, not the third.

        Tap count and maximum excess delay come out exactly, so the
        *delays* are confirmed. The RMS spread does not: 43.13 ns against
        a published 45 ns. The entry therefore leaves that one unasserted
        -- see the catalog comment for why no definition and no plausible
        typo bridges the gap.
        """
        epa = get_delay_profile("EPA")
        self.assertEqual(epa.n_taps, 7)
        self.assertEqual(float(epa.delays_ns[-1]), 410)
        self.assertAlmostEqual(epa.rms_delay_spread_ns, 43.13, delta=0.02)

    def test_the_delay_spread_definition_is_the_one_3gpp_uses(self):
        """Power-weighted and central -- confirmed by two models at once.

        This is what makes EPA's gap a finding rather than a suspicion:
        the same formula reproduces EVA to 0.4 ns and ETU to 0.1 ns, so
        it cannot be the formula that is wrong for EPA.
        """
        for name, published in (("EVA", 357), ("ETU", 991)):
            with self.subTest(profile=name):
                profile = get_delay_profile(name)
                weights = profile.powers_lin
                delays = profile.delays_ns
                mean = float(np.sum(weights * delays))
                central = float(np.sqrt(np.sum(weights * delays ** 2) - mean ** 2))
                self.assertAlmostEqual(central, published, delta=0.5)
                # the alternatives are nowhere near, so the choice is not
                # a coincidence of this profile
                amplitude = np.sqrt(profile.powers_lin)
                amplitude = amplitude / amplitude.sum()
                mean_a = float(np.sum(amplitude * delays))
                spread_a = float(np.sqrt(np.sum(amplitude * delays ** 2) - mean_a ** 2))
                self.assertGreater(abs(spread_a - published), 0.1 * published)

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


class TestTDLProfiles(unittest.TestCase):
    r"""The TR 38.901 entries, and what makes them different.

    Unlike the LTE profiles, a TDL model is a *shape*: the standard
    tabulates :math:`\tau_l / \sigma_\tau`, so the caller supplies the
    RMS delay spread it is stretched to. That is what the tests below
    pin -- the scaling, the merging of paths that share a delay, and
    the Rice factor of the two line-of-sight models. The confrontation
    with an independent transcription lives in
    ``validation/fading_tdl_3gpp.py`` (D7).
    """

    NAMES = ("TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E")

    def test_the_spread_is_what_was_asked_for(self):
        """The normalization invariant, at three scenario spreads."""
        for name in self.NAMES:
            for spread in (30.0, 100.0, 1000.0):
                profile = get_delay_profile(name, delay_spread_ns=spread)
                with self.subTest(profile=name, spread=spread):
                    self.assertAlmostEqual(
                        profile.rms_delay_spread_ns / spread, 1.0,
                        delta=7e-3)

    def test_the_delays_scale_and_nothing_else_does(self):
        """Doubling the spread doubles the delays, powers untouched."""
        one = get_delay_profile("TDL-C", delay_spread_ns=50.0)
        two = get_delay_profile("TDL-C", delay_spread_ns=100.0)
        np.testing.assert_allclose(two.delays_ns, 2 * one.delays_ns)
        np.testing.assert_allclose(two.powers_dB, one.powers_dB)

    def test_paths_sharing_a_delay_become_one_tap(self):
        """TDL-E lists two clusters at 0.544; a tap is a delay."""
        profile = get_delay_profile("TDL-E")
        self.assertEqual(profile.n_taps, 13)          # 15 table entries
        self.assertTrue(np.all(np.diff(profile.delays_ns) > 0))
        # the merged tap carries the sum of the two powers, which the
        # ratio to a neighbouring tap shows without needing the
        # normalization constant
        merged = int(np.argmin(np.abs(profile.delays_ns - 54.40)))
        alone = int(np.argmin(np.abs(profile.delays_ns - 51.33)))
        self.assertAlmostEqual(
            profile.powers_lin[merged] / profile.powers_lin[alone],
            (10 ** (-18.1 / 10) + 10 ** (-22.9 / 10)) / 10 ** (-15.8 / 10),
            places=9)

    def test_the_line_of_sight_models_carry_their_rice_factor(self):
        """K_1 is printed in the standard: 13.3 dB and 22.0 dB."""
        self.assertAlmostEqual(get_delay_profile("TDL-D").rice_k_dB, 13.3,
                               places=6)
        self.assertAlmostEqual(get_delay_profile("TDL-E").rice_k_dB, 22.0,
                               places=6)
        for name in ("TDL-A", "TDL-B", "TDL-C"):
            with self.subTest(profile=name):
                self.assertIsNone(get_delay_profile(name).rice_k_dB)

    def test_a_non_positive_spread_names_the_normalization(self):
        with self.assertRaises(ValueError) as ctx:
            get_delay_profile("TDL-A", delay_spread_ns=0.0)
        self.assertIn("normalized", str(ctx.exception))

    def test_the_powers_are_normalized_like_every_other_profile(self):
        for name in self.NAMES:
            with self.subTest(profile=name):
                self.assertAlmostEqual(
                    float(np.sum(get_delay_profile(name).powers_lin)), 1.0,
                    places=12)


if __name__ == "__main__":
    unittest.main()
