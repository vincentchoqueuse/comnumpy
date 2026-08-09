"""The cyclic prefix, including the length nobody tries: zero.

``N_cp = 0`` is explicitly accepted by ``__post_init__``, so it must
work. It did not: the mask was built with ``[-N_cp:]``, and ``[-0:]``
is ``[0:]`` -- the whole block. The prefixer then tried to copy N
samples into a zero-length slot and raised a shape error, which is the
kind of defect that only ever shows up in a sweep over N_cp.
"""
import unittest

import numpy as np

from comnumpy.ofdm.processors import CyclicPrefixer, CyclicPrefixRemover


class TestCyclicPrefix(unittest.TestCase):

    def test_round_trip_for_every_prefix_length(self):
        """Including 0 and the degenerate N_cp == N."""
        block = np.arange(12).reshape(2, 6)
        for n_cp in (0, 1, 3, 5, 6):
            with self.subTest(N_cp=n_cp):
                prefixed = CyclicPrefixer(N_cp=n_cp)(block)
                self.assertEqual(prefixed.shape, (2, 6 + n_cp))
                np.testing.assert_array_equal(
                    CyclicPrefixRemover(N_cp=n_cp)(prefixed), block)

    def test_zero_prefix_is_the_identity(self):
        block = np.arange(5)
        np.testing.assert_array_equal(CyclicPrefixer(N_cp=0)(block), block)
        np.testing.assert_array_equal(CyclicPrefixRemover(N_cp=0)(block), block)

    def test_the_prefix_is_the_tail_of_the_block(self):
        """The defining property, stated on the array itself."""
        block = np.arange(12).reshape(2, 6)
        for n_cp in (1, 3, 6):
            with self.subTest(N_cp=n_cp):
                prefixed = CyclicPrefixer(N_cp=n_cp)(block)
                np.testing.assert_array_equal(prefixed[..., :n_cp],
                                              block[..., -n_cp:])
                np.testing.assert_array_equal(prefixed[..., n_cp:], block)

    def test_accepts_a_numpy_integer(self):
        """N_cp usually comes out of a shape computation, i.e. as np.int64."""
        block = np.arange(8)
        np.testing.assert_array_equal(
            CyclicPrefixer(N_cp=np.int64(2))(block),
            CyclicPrefixer(N_cp=2)(block))

    def test_rejects_a_length_that_is_not_a_count(self):
        for bad in (-1, 2.5, "3", None):
            with self.subTest(N_cp=bad):
                with self.assertRaises(ValueError):
                    CyclicPrefixer(N_cp=bad)
                with self.assertRaises(ValueError):
                    CyclicPrefixRemover(N_cp=bad)


if __name__ == "__main__":
    unittest.main()
