"""WDM: the frequency plan and the multiplex/demultiplex pair (D44).

The decisive test is the round trip. A multiplexer and a demultiplexer
that agree on a grid must reconstruct every channel exactly -- not
approximately -- when the grid lands on DFT bins, and must leak nothing
between channels. Everything else here guards a way that can silently
stop being true.
"""
import logging
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from comnumpy.optical.wdm import WDMDemultiplexer, WDMGrid, WDMMultiplexer

FS = 250e9            # composite rate
N = 1280              # a multiple of fs / 50 GHz, so offsets are integer bins
SPACING = 50e9
BANDWIDTH = 32e9


def bandlimited(n_channels, n_samples=N, fs=FS, bandwidth=BANDWIDTH, seed=0):
    """Channels strictly inside their own band, which the model assumes."""
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n_channels, n_samples))
         + 1j * rng.normal(size=(n_channels, n_samples)))
    keep = np.abs(np.fft.fftfreq(n_samples, d=1 / fs)) <= bandwidth / 2
    return np.fft.ifft(np.fft.fft(x, axis=-1) * keep, axis=-1)


class TestWDMGrid(unittest.TestCase):

    def test_uniform_comb_is_centred(self):
        grid = WDMGrid.uniform(4, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        np.testing.assert_allclose(grid.offsets_Hz,
                                   [-75e9, -25e9, 25e9, 75e9])
        self.assertAlmostEqual(float(np.sum(grid.offsets_Hz)), 0.0)

    def test_odd_comb_puts_one_channel_on_the_centre(self):
        grid = WDMGrid.uniform(5, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.assertEqual(float(grid.offsets_Hz[2]), 0.0)

    def test_min_fs_is_the_edge_of_the_outermost_channel(self):
        grid = WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.assertAlmostEqual(grid.min_fs, 2 * (SPACING + BANDWIDTH / 2))

    def test_spacing_and_guard(self):
        grid = WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.assertAlmostEqual(grid.spacing_Hz, SPACING)
        self.assertAlmostEqual(grid.guard_Hz, SPACING - BANDWIDTH)

    def test_a_single_channel_has_no_spacing(self):
        grid = WDMGrid.uniform(1, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.assertIsNone(grid.spacing_Hz)
        self.assertIsNone(grid.guard_Hz)
        self.assertAlmostEqual(grid.min_fs, BANDWIDTH)

    def test_itu_indices_land_on_the_published_grid(self):
        """n multiple of 8 with m=4 is the fixed 50 GHz grid."""
        grid = WDMGrid.itu([-8, 0, 8], bandwidth_Hz=BANDWIDTH)
        np.testing.assert_allclose(grid.frequencies_Hz,
                                   [193.05e12, 193.1e12, 193.15e12])
        self.assertAlmostEqual(grid.spacing_Hz, 50e9)

    def test_rejects_overlapping_channels(self):
        with self.assertRaises(ValueError) as ctx:
            WDMGrid.uniform(3, spacing_Hz=25e9, bandwidth_Hz=BANDWIDTH)
        self.assertIn("overlapping", str(ctx.exception))

    def test_accepts_the_nyquist_wdm_limit(self):
        """Channels touching exactly is legal; it is what Nyquist WDM does."""
        grid = WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=SPACING)
        self.assertAlmostEqual(grid.guard_Hz, 0.0)

    def test_rejects_a_malformed_plan(self):
        cases = {
            "no channel": ((), 1e9),
            "duplicate frequency": ((1e12, 1e12), 1e9),
            "descending": ((2e12, 1e12), 1e9),
            "zero bandwidth": ((1e12, 2e12), 0.0),
        }
        for label, (frequencies, bandwidth) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    WDMGrid(frequencies, bandwidth_Hz=bandwidth)

    def test_itu_rejects_a_channel_wider_than_its_slot(self):
        with self.assertRaises(ValueError):
            WDMGrid.itu([0, 8], bandwidth_Hz=60e9)     # 50 GHz slot

    def test_validate_fs_names_the_rate_it_wants(self):
        grid = WDMGrid.uniform(5, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        grid.validate_fs(grid.min_fs)                  # the equality case passes
        with self.assertRaises(ValueError) as ctx:
            grid.validate_fs(0.99 * grid.min_fs)
        self.assertIn("232", str(ctx.exception))       # the required rate, in GHz

    def test_repr_draws_the_channels(self):
        text = repr(WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH))
        self.assertEqual(text.count("["), 3)
        self.assertIn("guard 18 GHz", text)

    def test_is_hashable_and_shareable(self):
        """Frozen, so one grid object can be shared by mux and demux."""
        grid = WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.assertEqual(grid, WDMGrid.uniform(3, spacing_Hz=SPACING,
                                               bandwidth_Hz=BANDWIDTH))
        self.assertIsInstance(hash(grid), int)


class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        self.grid = WDMGrid.uniform(5, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        self.x = bandlimited(5)
        self.y = WDMMultiplexer(self.grid, fs=FS)(self.x)

    def test_reconstructs_every_channel_exactly(self):
        """Exact, not approximate: the offsets are integer DFT bins here."""
        recovered = WDMDemultiplexer(self.grid, fs=FS)(self.y)
        error = np.max(np.abs(recovered - self.x)) / np.max(np.abs(self.x))
        self.assertLess(error, 1e-10)

    def test_no_crosstalk_between_channels(self):
        """One channel lit; every other output must be numerically dark."""
        lit = np.zeros_like(self.x)
        lit[1] = self.x[1]
        out = WDMDemultiplexer(self.grid, fs=FS)(WDMMultiplexer(self.grid, fs=FS)(lit))
        leak = np.max(np.abs(np.delete(out, 1, axis=0)))
        self.assertLess(leak / np.max(np.abs(self.x[1])), 1e-10)

    def test_multiplexing_conserves_power(self):
        """Non-overlapping slots are orthogonal, so the powers add."""
        self.assertAlmostEqual(float(np.mean(np.abs(self.y) ** 2)),
                               float(np.sum(np.mean(np.abs(self.x) ** 2, axis=-1))),
                               places=9)

    def test_selecting_one_channel_matches_selecting_all(self):
        every = WDMDemultiplexer(self.grid, fs=FS)(self.y)
        for index in range(self.grid.n_channels):
            with self.subTest(channel=index):
                one = WDMDemultiplexer(self.grid, fs=FS, channel=index)(self.y)
                self.assertEqual(one.shape, self.y.shape)
                np.testing.assert_allclose(one, every[index], atol=1e-12)

    def test_batch_axes_are_carried_through(self):
        """(..., C, N) in, (..., N) out: the leading axes are untouched."""
        batch = np.stack([self.x, 2 * self.x])          # (2, 5, N)
        out = WDMMultiplexer(self.grid, fs=FS)(batch)
        self.assertEqual(out.shape, (2, N))
        np.testing.assert_allclose(out[1], 2 * out[0], atol=1e-12)
        back = WDMDemultiplexer(self.grid, fs=FS)(out)
        self.assertEqual(back.shape, (2, 5, N))
        np.testing.assert_allclose(back[0], self.x, atol=1e-10)

    def test_a_single_channel_grid_is_a_pass_through(self):
        grid = WDMGrid.uniform(1, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)
        x = bandlimited(1)
        y = WDMMultiplexer(grid, fs=FS)(x)
        np.testing.assert_allclose(y, x[0], atol=1e-12)


class TestGuards(unittest.TestCase):

    def setUp(self):
        self.grid = WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH)

    def test_multiplexer_rejects_a_serial_signal(self):
        with self.assertRaises(ShapeError) as ctx:
            WDMMultiplexer(self.grid, fs=FS)(np.ones(N, dtype=complex))
        self.assertIn("C=3", str(ctx.exception))

    def test_multiplexer_rejects_the_wrong_channel_count(self):
        with self.assertRaises(ShapeError):
            WDMMultiplexer(self.grid, fs=FS)(np.ones((4, N), dtype=complex))

    def test_both_blocks_reject_an_undersampled_grid(self):
        low = 0.5 * self.grid.min_fs
        with self.assertRaises(ValueError):
            WDMMultiplexer(self.grid, fs=low)(np.ones((3, N), dtype=complex))
        with self.assertRaises(ValueError):
            WDMDemultiplexer(self.grid, fs=low)(np.ones(N, dtype=complex))

    def test_demultiplexer_rejects_an_out_of_range_channel(self):
        with self.assertRaises(ValueError):
            WDMDemultiplexer(self.grid, fs=FS, channel=3)(np.ones(N, dtype=complex))

    def test_warns_when_the_offsets_miss_the_dft_bins(self):
        """The one case where the round trip is not exact must announce itself.

        At fs = 250 GHz the 50 GHz offsets land on bin N/5, so a block
        length that is not a multiple of 5 shifts by a fraction of a bin.
        """
        with self.assertLogs("comnumpy.optical.wdm", level=logging.WARNING) as logs:
            WDMMultiplexer(self.grid, fs=FS)(np.ones((3, 1001), dtype=complex))
        self.assertIn("integer DFT bins", logs.output[0])

    def test_stays_quiet_on_an_aligned_block_length(self):
        with self.assertNoLogs("comnumpy.optical.wdm", level=logging.WARNING):
            WDMMultiplexer(self.grid, fs=FS)(np.ones((3, N), dtype=complex))


class TestChannelAxisPipeline(unittest.TestCase):
    """The channel axis is created once and every per-channel block rides it.

    This is the contract that makes the multiplexer trivial: the whole
    transmitter runs on ``(C, N)``, shaping and oversampling included,
    and multiplexing is then nothing but a weighted sum over axis −2.
    If a per-channel block ever stops broadcasting over leading axes,
    this test is what says so.
    """

    C, NS, SPS, RS = 5, 256, 8, 32e9

    def setUp(self):
        from comnumpy.core import (Sequential, SRRCFilter, SymbolGenerator,
                                   SymbolMapper, Upsampler, get_alphabet)
        self.fs = self.SPS * self.RS
        self.grid = WDMGrid.uniform(self.C, spacing_Hz=SPACING,
                                    bandwidth_Hz=1.2 * self.RS)
        self.tx = Sequential([
            SymbolGenerator(16, seed=0),
            SymbolMapper(get_alphabet("QAM", 16)),
            Upsampler(self.SPS),
            SRRCFilter(0.2, self.SPS, N_h=10),
        ])

    def test_the_transmitter_carries_the_channel_axis_end_to_end(self):
        x = self.tx((self.C, self.NS))
        self.assertEqual(x.shape, (self.C, self.NS * self.SPS))
        y = WDMMultiplexer(self.grid, fs=self.fs)(x)
        self.assertEqual(y.shape, (self.NS * self.SPS,))

    def test_multiplexing_is_a_weighted_sum_over_the_channel_axis(self):
        """State it as an identity, so the block cannot drift from it."""
        x = self.tx((self.C, self.NS))
        mux = WDMMultiplexer(self.grid, fs=self.fs)
        y = mux(x)
        n = np.arange(x.shape[-1])
        by_hand = sum(x[c] * np.exp(2j * np.pi * self.grid.offsets_Hz[c] * n / self.fs)
                      for c in range(self.C))
        np.testing.assert_allclose(y, by_hand, atol=1e-12)

    def test_a_shaped_channel_survives_the_round_trip_untouched(self):
        """One channel lit: whatever comes back is the block's own doing."""
        x = self.tx((self.C, self.NS))
        lit = np.zeros_like(x)
        lit[2] = x[2]
        out = WDMDemultiplexer(self.grid, fs=self.fs)(
            WDMMultiplexer(self.grid, fs=self.fs)(lit))
        keep = np.abs(np.fft.fftfreq(x.shape[-1], d=1 / self.fs)) <= self.grid.bandwidth_Hz / 2
        reference = np.fft.ifft(np.fft.fft(x[2]) * keep)
        error = np.max(np.abs(out[2] - reference)) / np.max(np.abs(reference))
        self.assertLess(error, 1e-12)

    def test_the_composite_rate_must_cover_the_comb(self):
        """The real design constraint, and it must be stated, not guessed."""
        self.assertGreaterEqual(self.fs, self.grid.min_fs)
        with self.assertRaises(ValueError):
            WDMMultiplexer(self.grid, fs=4 * self.RS)(self.tx((self.C, self.NS)))


class TestChainIntegration(unittest.TestCase):

    def test_the_fibre_guard_points_at_the_multiplexer(self):
        """A rejection is only useful when it names the way out (D38)."""
        from comnumpy.optical import FiberLink
        with self.assertRaises(ShapeError) as ctx:
            FiberLink(1, fs=FS)(np.ones((3, 64), dtype=complex))
        self.assertIn("WDMMultiplexer", str(ctx.exception))

    def test_survives_a_json_round_trip(self):
        from comnumpy.core import Sequential
        from comnumpy.serialization import from_json, to_json
        chain = Sequential([WDMDemultiplexer(
            WDMGrid.uniform(3, spacing_Hz=SPACING, bandwidth_Hz=BANDWIDTH),
            fs=FS, channel=1)])
        restored = from_json(to_json(chain))
        self.assertEqual(restored[0].grid, chain[0].grid)
        self.assertEqual(restored[0].channel, 1)


if __name__ == "__main__":
    unittest.main()
