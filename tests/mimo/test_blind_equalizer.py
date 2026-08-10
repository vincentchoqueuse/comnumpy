r"""The blind 2x2 adaptive equalizer: what it needs to converge.

``BlindDualMIMOCompensator`` is a butterfly equalizer updated by
stochastic gradient on one of three blind losses. Two things about it
are easy to get wrong from the outside, and both cost an afternoon:

* the equalizer has a **group delay of L input samples** -- its centre
  tap sits L samples back -- so comparing its output with the
  transmitted symbols index by index reads as pure noise, whatever the
  equalizer is doing;
* **only CMA converges from a cold start**. RDE and DD need decisions
  that are already roughly right, which is why the literature writes
  schedules like ``['cma', 'rde']``.

Both are pinned here rather than left to the docstring, and the staged
CMA -> RDE -> DD sequence is checked to reach the noise floor.
"""
import unittest

import numpy as np

from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.compensators import BlindDualMIMOCompensator

N = 60000
ORDER = 16
L = 5
NOISE = 0.002                  # per quadrature, so 0.004 total
CEILING_DB = -10 * np.log10(2 * NOISE)


def alphabet():
    points = get_alphabet("QAM", ORDER)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def rotated_channel(seed=0, theta=0.7, phi=1.3, noise=NOISE):
    """A static Jones rotation of a PDM 16-QAM signal, plus noise."""
    rng = np.random.default_rng(seed)
    points = alphabet()
    sent = points[rng.integers(0, ORDER, size=(2, N))]
    jones = np.array([[np.cos(theta), np.sin(theta) * np.exp(1j * phi)],
                      [-np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]])
    received = jones @ sent + np.sqrt(noise) * (
        rng.normal(size=(2, N)) + 1j * rng.normal(size=(2, N)))
    return sent, received


def snr_dB(estimated, reference):
    """SNR after removing the one complex gain the equalizer leaves free."""
    gain = np.vdot(estimated, reference) / np.vdot(estimated, estimated)
    return float(-10 * np.log10(
        np.mean(np.abs(gain * estimated - reference) ** 2)
        / np.mean(np.abs(reference) ** 2)))


def scored(output, sent, lag=L, tail=slice(N // 2, N - L - 1)):
    """SNR of each output polarization against its own transmitted one."""
    return [snr_dB(output[i][tail], np.roll(sent[i], lag)[tail])
            for i in (0, 1)]


class TestDelay(unittest.TestCase):
    """The trap: the answer is right, at the wrong index."""

    def test_the_output_lags_the_input_by_L_samples(self):
        sent, received = rotated_channel()
        equalizer = BlindDualMIMOCompensator(L=L, alphabet=alphabet(),
                                             mu=1e-3, mode="cma")
        output = equalizer(received)
        aligned = scored(output, sent, lag=L)
        misaligned = scored(output, sent, lag=0)
        self.assertGreater(min(aligned), 20.0)
        self.assertLess(max(misaligned), 1.0)

    def test_the_untouched_head_of_the_output_is_zero(self):
        """The filter has no history yet, and says so with zeros."""
        _, received = rotated_channel()
        output = BlindDualMIMOCompensator(L=L, alphabet=alphabet())(received)
        np.testing.assert_array_equal(output[:, :2 * L + 1],
                                      np.zeros((2, 2 * L + 1)))


class TestStagedAdaptation(unittest.TestCase):

    def test_cma_converges_from_a_cold_start(self):
        sent, received = rotated_channel()
        equalizer = BlindDualMIMOCompensator(L=L, alphabet=alphabet(),
                                             mu=1e-3, mode="cma")
        self.assertGreater(min(scored(equalizer(received), sent)), 20.0)

    def test_rde_alone_does_not(self):
        """Not a defect: it is why the staged schedule exists."""
        sent, received = rotated_channel()
        equalizer = BlindDualMIMOCompensator(L=L, alphabet=alphabet(),
                                             mu=3e-4, mode="rde")
        self.assertLess(max(scored(equalizer(received), sent)), 10.0)

    def test_the_three_stages_reach_the_noise_floor(self):
        sent, received = rotated_channel()
        equalizer = BlindDualMIMOCompensator(L=L, alphabet=alphabet(),
                                             mu=1e-3, mode="cma")
        equalizer.partial_fit(received)
        after_cma = min(scored(equalizer(received), sent))

        equalizer.mode, equalizer.mu = "rde", 3e-4
        after_rde = min(scored(equalizer(received), sent))

        equalizer.mode, equalizer.mu = "dd", 1e-4
        after_dd = min(scored(equalizer(received), sent))

        self.assertLess(after_cma, after_rde)
        self.assertLess(after_rde, after_dd + 1e-9)
        self.assertGreater(after_dd, CEILING_DB - 0.3)
        self.assertLess(after_dd, CEILING_DB + 0.3)


class TestIterationHook(unittest.TestCase):
    """``process_after_iteration`` is what schedules the stages in-pass."""

    def test_the_hook_sees_a_bounded_block_of_recent_outputs(self):
        seen = []

        class Watching(BlindDualMIMOCompensator):
            def process_after_iteration(self, n, Y_sub):
                seen.append(Y_sub.shape)

        _, received = rotated_channel()
        equalizer = Watching(L=L, alphabet=alphabet(), sub_block_length=7)
        equalizer(received[:, :400])
        widths = {shape[1] for shape in seen}
        self.assertEqual({shape[0] for shape in seen}, {2})
        # it grows to sub_block_length and stops there: the block used to
        # be a stride over the whole history, so this width grew with n
        self.assertLessEqual(max(widths), 7)
        self.assertEqual(max(widths), 7)

    def test_switching_mode_inside_the_pass_works(self):
        sent, received = rotated_channel()

        class Scheduled(BlindDualMIMOCompensator):
            def process_after_iteration(self, n, Y_sub):
                if self.mode == "cma" and n >= N // 3:
                    self.mode, self.mu = "rde", 3e-4

        equalizer = Scheduled(L=L, alphabet=alphabet(), mu=1e-3, mode="cma")
        output = equalizer(received)
        self.assertEqual(equalizer.mode, "rde")
        self.assertGreater(min(scored(output, sent)), 22.0)


class TestGuards(unittest.TestCase):

    def test_a_single_polarization_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            BlindDualMIMOCompensator(alphabet=alphabet())(np.zeros((1, 100),
                                                                   dtype=complex))
        self.assertIn("dual polarization", str(ctx.exception))

    def test_an_unknown_mode_says_which_ones_exist(self):
        equalizer = BlindDualMIMOCompensator(L=2, alphabet=alphabet())
        equalizer.mode = "lms"       # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            equalizer(np.ones((2, 50), dtype=complex))
        self.assertIn("'cma', 'rde' or 'dd'", str(ctx.exception))


class TestFrozenFastPath(unittest.TestCase):
    r"""``mu = 0`` must be the same equalizer, only faster.

    A stochastic gradient cannot be vectorized -- the update from
    :math:`y[n]` is needed to compute :math:`y[n+1]`. But when the step
    size is zero the map is the same for every sample, so the pass is a
    matrix product, and the only thing worth asserting is that it is
    *exactly* the loop: same output, bit for bit, and an equalizer left
    where it was.
    """

    class Hooked(BlindDualMIMOCompensator):
        """Overriding the hook forces the per-sample loop back on."""

        def process_after_iteration(self, n, Y_sub):
            pass

    def signal(self, n_samples=3000, seed=3):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((2, n_samples))
                + 1j * rng.standard_normal((2, n_samples))) / np.sqrt(2)

    def test_the_fast_path_reproduces_the_loop_bit_for_bit(self):
        X = self.signal()
        for taps, oversampling in ((5, 1), (5, 2), (10, 2), (3, 4)):
            with self.subTest(L=taps, oversampling=oversampling):
                looped = self.Hooked(taps, alphabet=alphabet(), mu=0.0,
                                     oversampling=oversampling)
                fast = BlindDualMIMOCompensator(taps, alphabet=alphabet(),
                                                mu=0.0,
                                                oversampling=oversampling)
                np.testing.assert_array_equal(fast(X), looped(X))

    def test_it_spans_the_block_boundary(self):
        """The gather is chunked; the seam must not be visible."""
        original = BlindDualMIMOCompensator._FROZEN_BLOCK
        BlindDualMIMOCompensator._FROZEN_BLOCK = 97
        try:
            X = self.signal()
            fast = BlindDualMIMOCompensator(5, alphabet=alphabet(), mu=0.0,
                                            oversampling=2)
            looped = self.Hooked(5, alphabet=alphabet(), mu=0.0,
                                 oversampling=2)
            np.testing.assert_array_equal(fast(X), looped(X))
        finally:
            BlindDualMIMOCompensator._FROZEN_BLOCK = original

    def test_a_frozen_equalizer_does_not_move(self):
        X = self.signal()
        fast = BlindDualMIMOCompensator(5, alphabet=alphabet(), mu=0.0)
        before = fast.H_.copy()
        fast(X)
        np.testing.assert_array_equal(fast.H_, before)

    def test_fit_then_apply_is_the_point(self):
        """Adapt on a preamble, freeze, apply: the state carries over."""
        sent, received = rotated_channel(seed=1)
        compensator = BlindDualMIMOCompensator(L, alphabet=alphabet(),
                                               mu=1e-3, oversampling=1)
        compensator.partial_fit(received[:, :N // 2])
        trained = compensator.H_.copy()
        compensator.mu = 0.0
        payload = compensator(received)
        np.testing.assert_array_equal(compensator.H_, trained)
        self.assertGreater(min(scored(payload, sent)), 15.0)


if __name__ == "__main__":
    unittest.main()
