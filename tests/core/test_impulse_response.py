"""``TappedDelayLineChannel.impulse_response``, and the dBm conversions.

Both were extracted from the tutorials rather than invented. Three
scripts were sounding a fading channel by hand -- build an impulse,
call the block, slice off the leading taps -- to get the tap vector an
equalizer needs, and sixteen places were writing ``1e-3 * 10 ** (dBm /
10)`` or its inverse inline. Neither is hard; both are easy to get
subtly wrong, and neither belongs in a tutorial whose subject is
something else.

What is tested here is what the callers rely on: the sounding really is
the realized channel and not an approximation of it, the vector is dense
so it can be convolved directly, and the two power conversions invert
each other and refuse the input that would silently produce -inf.
"""
import unittest

import numpy as np

from comnumpy.core.channels import TappedDelayLineChannel
from comnumpy.core.fading import get_delay_profile
from comnumpy.optical.utils import dbm_to_watt, watt_to_dbm


class TestImpulseResponse(unittest.TestCase):

    def channel(self, name="EPA", fs=7.68e6, seed=18):
        return TappedDelayLineChannel(get_delay_profile(name), fs=fs, seed=seed)

    def test_the_taps_are_the_realized_ones(self):
        """Not an approximation of the channel -- the channel itself."""
        for name, fs in (("EPA", 7.68e6), ("EVA", 15.36e6), ("TDL-C", 10e6)):
            with self.subTest(profile=name):
                channel = self.channel(name, fs)
                h = channel.impulse_response()
                np.testing.assert_allclose(h[channel.delays_],
                                           channel.h_[:, 0], atol=1e-12)

    def test_the_vector_is_dense_and_ends_on_the_longest_path(self):
        for name, fs in (("EPA", 7.68e6), ("EVA", 15.36e6)):
            with self.subTest(profile=name):
                channel = self.channel(name, fs)
                h = channel.impulse_response()
                self.assertEqual(h.size, int(channel.delays_[-1]) + 1)
                # unresolved positions are exactly zero, so the vector
                # can be handed straight to a convolution
                unresolved = set(range(h.size)) - set(channel.delays_.tolist())
                for index in unresolved:
                    self.assertEqual(h[index], 0)

    def test_it_is_reproducible_under_a_seed(self):
        np.testing.assert_allclose(self.channel().impulse_response(),
                                   self.channel().impulse_response())

    def test_a_new_call_draws_a_new_realization(self):
        """Documented behaviour: it sounds, it does not memoize."""
        channel = self.channel()
        first, second = channel.impulse_response(), channel.impulse_response()
        self.assertFalse(np.allclose(first, second))

    def test_the_record_length_does_not_change_the_first_tap_count(self):
        for n_samples in (64, 256, 4096):
            with self.subTest(n_samples=n_samples):
                self.assertEqual(
                    self.channel().impulse_response(n_samples).size,
                    self.channel().impulse_response().size)

    def test_the_sounded_response_is_the_filter_the_channel_applies(self):
        """The claim that makes the method worth having.

        Two channels with the same seed draw the same realization, so
        sounding one and running a signal through the other must agree
        with a plain convolution. Asking for the same record length
        matters: the fading process is drawn per sample, so a different
        length is a different draw.
        """
        rng = np.random.default_rng(0)
        for length in (256, 1024):
            with self.subTest(length=length):
                x = (rng.standard_normal(length)
                     + 1j * rng.standard_normal(length))
                h = self.channel().impulse_response(length)
                np.testing.assert_allclose(self.channel()(x),
                                           np.convolve(x, h)[:length],
                                           atol=1e-12)


class TestThePowerConversions(unittest.TestCase):

    def test_they_invert_each_other(self):
        for value in (-30.0, -10.0, -3.0, 0.0, 7.0, 20.0):
            with self.subTest(dBm=value):
                self.assertAlmostEqual(watt_to_dbm(dbm_to_watt(value)),
                                       value, places=12)

    def test_the_anchor_points(self):
        self.assertAlmostEqual(dbm_to_watt(0.0), 1e-3, places=15)
        self.assertAlmostEqual(dbm_to_watt(30.0), 1.0, places=12)
        self.assertAlmostEqual(watt_to_dbm(1.0), 30.0, places=12)

    def test_ten_dB_is_a_factor_ten(self):
        self.assertAlmostEqual(dbm_to_watt(10.0) / dbm_to_watt(0.0), 10.0,
                               places=12)

    def test_arrays_go_through_element_wise(self):
        values = np.array([-20.0, -6.0, 0.0, 13.0])
        np.testing.assert_allclose(watt_to_dbm(dbm_to_watt(values)), values,
                                   atol=1e-12)
        self.assertEqual(dbm_to_watt(values).shape, values.shape)

    def test_a_scalar_stays_a_scalar(self):
        self.assertIsInstance(dbm_to_watt(3.0), float)
        self.assertIsInstance(watt_to_dbm(2e-3), float)

    def test_a_non_positive_power_is_refused(self):
        """Silent -inf in a link budget is worse than a stop."""
        for value in (0.0, -1e-6, np.array([1e-3, 0.0])):
            with self.subTest(power=value):
                with self.assertRaises(ValueError):
                    watt_to_dbm(value)


if __name__ == "__main__":
    unittest.main()
