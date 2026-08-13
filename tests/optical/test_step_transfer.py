"""The linear step, split into what it is and what it does.

`apply_chromatic_dispersion` built its transfer function on every call
and threw it away: an `fftfreq`, a `w**2` and a length-N complex
exponential, a thousand times per propagation, for something that
depends only on `(n, dz, beta2, fs, alpha_dB, direction)`. Splitting it
into `linear_step_transfer` (what the step is) and
`apply_frequency_response` (what it does to a signal) lets the blocks
build the first once and call the second in the loop.

The composed function keeps its name and its behaviour, so nothing
downstream had to move; these tests hold both halves and their
composition, and pin the property that makes the cache legitimate --
that a linear schedule has exactly one distinct step length.
"""
import unittest

import numpy as np

from comnumpy.optical.utils import (apply_chromatic_dispersion,
                                    apply_frequency_response,
                                    linear_step_transfer, step_transfers)

N = 256
BETA2, FS, ALPHA = -21.7, 64e9, 0.2


class TestLinearStepTransfer(unittest.TestCase):

    def test_lossless_dispersion_moves_phase_not_energy(self):
        H = linear_step_transfer(N, 20.0, BETA2, fs=FS)
        np.testing.assert_allclose(np.abs(H), 1.0, atol=1e-12)

    def test_attenuation_is_folded_in(self):
        """The step is the whole linear operator, loss included."""
        H = linear_step_transfer(N, 20.0, BETA2, fs=FS, alpha_dB=ALPHA)
        expected = 10 ** (-ALPHA * 20.0 / 20)      # amplitude, not power
        np.testing.assert_allclose(np.abs(H), expected, rtol=1e-12)

    def test_backward_is_the_conjugate_of_forward(self):
        """What makes noiseless back-propagation exact."""
        forward = linear_step_transfer(N, 20.0, BETA2, fs=FS)
        backward = linear_step_transfer(N, 20.0, BETA2, fs=FS, direction=-1)
        np.testing.assert_allclose(backward, np.conj(forward), atol=1e-15)

    def test_steps_compose_by_adding_their_lengths(self):
        """Two half-steps are one step -- the split-step's own premise."""
        half = linear_step_transfer(N, 10.0, BETA2, fs=FS, alpha_dB=ALPHA)
        whole = linear_step_transfer(N, 20.0, BETA2, fs=FS, alpha_dB=ALPHA)
        np.testing.assert_allclose(half * half, whole, rtol=1e-12)

    def test_zero_dispersion_and_zero_length_are_identity(self):
        np.testing.assert_allclose(linear_step_transfer(N, 0.0, BETA2, fs=FS),
                                   np.ones(N), atol=1e-15)


class TestApplyFrequencyResponse(unittest.TestCase):

    def test_a_flat_response_returns_the_signal(self):
        x = np.random.default_rng(0).normal(size=N) + 0j
        np.testing.assert_allclose(apply_frequency_response(x, np.ones(N)), x,
                                   atol=1e-12)

    def test_it_broadcasts_over_the_leading_axes(self):
        """A batch of polarization pairs is filtered without a loop."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 2, N)) + 1j * rng.normal(size=(4, 2, N))
        H = linear_step_transfer(N, 20.0, BETA2, fs=FS)
        together = apply_frequency_response(x, H)
        self.assertEqual(together.shape, x.shape)
        for item in range(4):
            for pol in range(2):
                with self.subTest(item=item, pol=pol):
                    np.testing.assert_allclose(
                        together[item, pol],
                        apply_frequency_response(x[item, pol], H), atol=1e-15)


class TestTheCompositionIsTheOldFunction(unittest.TestCase):

    def test_apply_chromatic_dispersion_is_the_two_halves(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        for z, direction in ((2.0, 1), (80.0, -1)):
            with self.subTest(z=z, direction=direction):
                H = linear_step_transfer(N, z, BETA2, fs=FS, alpha_dB=ALPHA,
                                         direction=direction)
                np.testing.assert_allclose(
                    apply_chromatic_dispersion(x, z, BETA2, alpha_dB=ALPHA,
                                               fs=FS, direction=direction),
                    apply_frequency_response(x, H), rtol=1e-12)


class TestStepTransfers(unittest.TestCase):

    def test_a_linear_schedule_needs_one_row(self):
        """The property the cache rests on: equal steps, one operator."""
        table, index, _ = step_transfers(N, np.full(200, 0.4), beta2=BETA2,
                                         fs=FS)
        self.assertEqual(table.shape, (1, N))
        self.assertEqual(set(index.tolist()), {0})

    def test_a_logarithmic_schedule_needs_one_row_each(self):
        lengths = np.array([0.2, 0.6, 1.8, 5.4])
        table, index, _ = step_transfers(N, lengths, beta2=BETA2, fs=FS)
        self.assertEqual(table.shape, (4, N))
        np.testing.assert_array_equal(index, [0, 1, 2, 3])
        for row, z in enumerate(lengths):
            with self.subTest(z=z):
                np.testing.assert_allclose(
                    table[row], linear_step_transfer(N, z, BETA2, fs=FS),
                    atol=1e-15)

    def test_the_rows_follow_the_order_the_steps_are_applied(self):
        """Unsorted lengths must still map each step to its own row."""
        lengths = np.array([5.0, 1.0, 5.0, 2.0])
        table, index, _ = step_transfers(N, lengths, beta2=BETA2, fs=FS)
        self.assertEqual(table.shape, (3, N))          # three distinct
        for step, z in enumerate(lengths):
            with self.subTest(step=step):
                np.testing.assert_allclose(
                    table[index[step]],
                    linear_step_transfer(N, z, BETA2, fs=FS), atol=1e-15)

    def test_an_unchanged_schedule_is_not_rebuilt(self):
        """What makes a Monte-Carlo pay for this once instead of per pass."""
        previous = step_transfers(N, np.full(8, 0.4), beta2=BETA2, fs=FS)
        again = step_transfers(N, np.full(8, 0.4), beta2=BETA2, fs=FS,
                               previous=previous)
        self.assertIs(again[0], previous[0])

    def test_every_dependency_invalidates_the_cache(self):
        lengths = np.full(8, 0.4)
        previous = step_transfers(N, lengths, beta2=BETA2, fs=FS)
        changes = {
            "n_samples": dict(n_samples=2 * N),
            "fs": dict(fs=2 * FS),
            "beta2": dict(beta2=2 * BETA2),
            "alpha_dB": dict(alpha_dB=ALPHA),
            "direction": dict(direction=-1),
            "lengths": dict(lengths=np.full(8, 0.8)),
        }
        for name, override in changes.items():
            with self.subTest(changed=name):
                params = dict(n_samples=N, lengths=lengths, beta2=BETA2,
                              fs=FS)
                params.update(override)
                rebuilt = step_transfers(params.pop("n_samples"),
                                         params.pop("lengths"),
                                         previous=previous, **params)
                self.assertIsNot(rebuilt[0], previous[0])


if __name__ == "__main__":
    unittest.main()
