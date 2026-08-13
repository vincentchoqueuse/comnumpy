"""The stacked MIMO channel: one draw per trial, detectors aligned.

The MIMO Monte-Carlo redraws the channel every frame, and every
detector needs that frame's matrix. The batch contract (D51) makes the
draw a *stack*: ``rayleigh_channel(size=K)`` returns ``(K, N_r, N_t)``,
``FlatMIMOChannel`` applies channel k to frame k, and a detector built
with the same stack decides frame k against channel k. The linear
detectors batch through numpy's stacked linalg; the search detectors
(ML, OSIC, sphere) loop per draw internally -- convenience and
correctness, not a speedup, and the docstrings say so.
"""
import unittest

import numpy as np

from comnumpy.core.utils import Constellation
from comnumpy.exceptions import ShapeError
from comnumpy.mimo.channels import FlatMIMOChannel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector, SphereDecoder)
from comnumpy.mimo.utils import rayleigh_channel

CONSTELLATION = Constellation("PSK", 4)
K, N_R, N_T, N = 4, 3, 2, 60


def factories():
    return {
        "ZF": lambda H: LinearDetector(CONSTELLATION, H=H, method="zf"),
        "MMSE": lambda H: LinearDetector(CONSTELLATION, H=H,
                                         method="mmse", sigma2=0.02),
        "ML": lambda H: MaximumLikelihoodDetector(CONSTELLATION, H=H),
        "OSIC": lambda H: OrderedSuccessiveInterferenceCancellationDetector(
            CONSTELLATION, osic_type="sinr", H=H, sigma2=0.02),
        "SD": lambda H: SphereDecoder(CONSTELLATION, H=H),
    }


class TestStackedDraws(unittest.TestCase):

    def test_size_grows_a_leading_axis(self):
        self.assertEqual(rayleigh_channel(N_R, N_T, seed=0, size=K).shape,
                         (K, N_R, N_T))

    def test_size_equals_the_python_loop_draw_for_draw(self):
        """Moving a study to the batched form keeps its channels."""
        first = np.random.default_rng(7)
        second = np.random.default_rng(7)
        looped = []
        for _ in range(5):
            looped.append(rayleigh_channel(N_R, N_T, rng=first))
        stacked = rayleigh_channel(N_R, N_T, rng=second, size=5)
        np.testing.assert_allclose(np.stack(looped), stacked)


class TestStackedChannelAndDetectors(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        alphabet = np.asarray(CONSTELLATION.alphabet)
        self.H = rayleigh_channel(N_R, N_T, seed=1, size=K)
        self.sent = rng.integers(0, 4, (K, N_T, N))
        noise = 0.1 * (rng.standard_normal((K, N_R, N))
                       + 1j * rng.standard_normal((K, N_R, N)))
        self.Y = np.matmul(self.H, alphabet[self.sent]) + noise

    def test_channel_k_is_applied_to_frame_k(self):
        x = np.asarray(CONSTELLATION.alphabet)[self.sent]
        batched = FlatMIMOChannel(self.H)(x)
        single = FlatMIMOChannel(self.H[2])(x[2])
        np.testing.assert_allclose(batched[2], single)

    def test_every_detector_decides_frame_k_against_channel_k(self):
        for name, make in factories().items():
            with self.subTest(detector=name):
                stacked = make(self.H)(self.Y)
                single = make(self.H[2])(self.Y[2])
                self.assertEqual(stacked.shape, (K, N_T, N))
                np.testing.assert_array_equal(stacked[2], single)

    def test_a_mismatched_batch_is_refused(self):
        """K channels cannot decide K+1 frames: no silent broadcast."""
        detector = MaximumLikelihoodDetector(CONSTELLATION, H=self.H)
        with self.assertRaises(ShapeError):
            detector(np.ones((K + 1, N_R, N), dtype=complex))

    def test_the_sphere_decoder_still_counts_its_nodes(self):
        decoder = SphereDecoder(CONSTELLATION, H=self.H)
        decoder(self.Y)
        self.assertGreater(decoder.nodes_, 0.0)


if __name__ == "__main__":
    unittest.main()
