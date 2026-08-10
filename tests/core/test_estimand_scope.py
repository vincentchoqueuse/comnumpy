r"""Shared or per path: what an estimator measures on a multi-path signal (D49).

Once a chain carries two polarizations, every estimator has to answer a
question it could ignore while signals were one-dimensional. A carrier
frequency offset is **one** physical quantity -- one laser beating
against one local oscillator -- so it is estimated jointly, over all the
paths at once. An IQ imbalance belongs to a receiver, a DC offset to a
converter, a residual rotation to whichever equalizer output produced
it: one estimate each.

Letting numpy broadcast decides that by accident, and it decided it
wrong in both directions. This file pins the two answers, and the two
defects that made the question worth asking:

* ``BlindIQCompensator`` on a ``(2, N)`` signal used to stack the real
  and imaginary parts of *both* polarizations into one 4-row matrix,
  keep the top-left 2x2 block of the covariance, and return a signal
  far worse than the one it was given -- with no error;
* ``DCCorrector`` never broadcast its own documented ``axis``: the mean
  of a ``(P, N)`` record along ``axis=-1`` came back as ``(P,)`` and the
  subtraction raised a shape error.
"""
import unittest

import numpy as np

from comnumpy.core.compensators import (BlindCFOCompensator,
                                        BlindIQCompensator,
                                        BlindPhaseCompensation, DCCorrector,
                                        DataAidedPhaseCompensator)
from comnumpy.core.processors import BlindPhaseTracker
from comnumpy.core.utils import get_alphabet
from comnumpy.core.validators import validate_single_path
from comnumpy.exceptions import ShapeError

N = 2000


def qpsk(seed=0, size=N):
    alphabet = get_alphabet("QAM", 4)
    rng = np.random.default_rng(seed)
    return alphabet, alphabet[rng.integers(0, 4, size=size)]


class TestSharedEstimand(unittest.TestCase):
    """One laser, one offset: estimated over every path at once."""

    def dual_pol_with_offset(self, omega=0.01):
        _, symbols = qpsk(size=(2, N))
        return symbols, symbols * np.exp(1j * omega * np.arange(N))

    def test_the_offset_is_estimated_jointly_and_applied_to_both(self):
        sent, received = self.dual_pol_with_offset()
        compensator = BlindCFOCompensator()
        corrected = compensator(received)
        self.assertAlmostEqual(float(compensator.w0_), 0.01, places=4)
        self.assertEqual(np.shape(compensator.w0_), ())   # one number
        for path in (0, 1):
            with self.subTest(path=path):
                residual = np.abs(np.angle(
                    np.mean(corrected[path] * np.conj(sent[path]))))
                self.assertLess(float(residual), 1e-2)

    def test_the_joint_estimate_beats_one_path_alone(self):
        """Twice the data: the whole reason to estimate it jointly."""
        errors = {"joint": [], "single": []}
        for seed in range(6):
            _, symbols = qpsk(seed=seed, size=(2, 400))
            noisy = symbols + 0.35 * (np.random.default_rng(100 + seed)
                                      .normal(size=(2, 400)))
            received = noisy * np.exp(1j * 0.01 * np.arange(400))
            joint = BlindCFOCompensator()
            joint(received)
            single = BlindCFOCompensator()
            single(received[0])
            errors["joint"].append(abs(float(joint.w0_) - 0.01))
            errors["single"].append(abs(float(single.w0_) - 0.01))
        self.assertLess(float(np.mean(errors["joint"])),
                        float(np.mean(errors["single"])))


class TestPerPathEstimand(unittest.TestCase):
    """One receiver, one imbalance: an estimate each."""

    def test_the_iq_compensator_no_longer_pools_the_paths(self):
        _, symbols = qpsk()
        # the second path is badly imbalanced, the first is clean
        received = np.stack([symbols,
                             3 * np.real(symbols) + 1j * np.imag(symbols)])
        corrected = BlindIQCompensator()(received)
        for path in (0, 1):
            with self.subTest(path=path):
                ratio = (np.var(np.real(corrected[path]))
                         / np.var(np.imag(corrected[path])))
                self.assertAlmostEqual(float(ratio), 1.0, delta=0.05)

    def test_one_path_alone_gives_the_same_answer_as_inside_a_pair(self):
        _, symbols = qpsk()
        skewed = 3 * np.real(symbols) + 1j * np.imag(symbols)
        alone = BlindIQCompensator()(skewed)
        together = BlindIQCompensator()(np.stack([symbols, skewed]))[1]
        np.testing.assert_allclose(together, alone, rtol=1e-9)

    def test_the_dc_corrector_broadcasts_its_own_axis(self):
        offsets = np.array([[1.0 + 2j], [-3.0 - 1j]])
        _, symbols = qpsk(size=(2, N))
        corrected = DCCorrector(axis=-1)(symbols + offsets)
        np.testing.assert_allclose(np.mean(corrected, axis=-1), 0.0,
                                   atol=1e-12)

    def test_the_phase_compensator_fits_one_angle_per_path(self):
        alphabet, symbols = qpsk(size=(2, N))
        received = symbols * np.array([[np.exp(-1j * 0.3)],
                                       [np.exp(-1j * 0.15)]])
        compensator = BlindPhaseCompensation(alphabet)
        compensator(received)
        np.testing.assert_allclose(np.ravel(compensator.theta_),
                                   [0.3, 0.15], atol=1e-2)

    def test_the_phase_tracker_produces_one_trajectory_per_path(self):
        alphabet, symbols = qpsk(size=(2, 120))
        received = symbols * np.array([[np.exp(-1j * 0.2)],
                                       [np.exp(1j * 0.2)]])
        tracker = BlindPhaseTracker(4, alphabet, phase_steps=32)
        corrected = tracker(received)
        self.assertEqual(corrected.shape, received.shape)
        self.assertEqual(tracker.theta_.shape, received.shape)
        # opposite rotations: a joint tracker would average them to zero
        self.assertLess(float(np.mean(tracker.theta_[0])), -0.1)
        self.assertGreater(float(np.mean(tracker.theta_[1])), 0.1)

    def test_a_single_path_still_returns_plain_scalars(self):
        """Scalar in, scalar out -- the rule the rest of the library uses."""
        alphabet, symbols = qpsk()
        compensator = BlindPhaseCompensation(alphabet)
        compensator(symbols * np.exp(-1j * 0.3))
        self.assertIsInstance(compensator.theta_, float)
        iq = BlindIQCompensator()
        iq(symbols)
        self.assertIsInstance(iq.alpha_, complex)


class TestGuard(unittest.TestCase):
    """What the blocks that were not generalized say instead of crashing."""

    def test_a_data_aided_block_refuses_a_multi_path_signal(self):
        _, reference = qpsk(size=200)
        received = np.stack([reference, reference])
        with self.assertRaises(ShapeError) as ctx:
            DataAidedPhaseCompensator(reference=reference)(received)
        message = str(ctx.exception)
        self.assertIn("one phase against a reference", message)
        self.assertIn("2 paths", message)
        self.assertIn("D49", message)

    def test_the_validator_passes_a_single_path_through(self):
        for shape in ((8,), (1, 8), (3, 1, 8)):
            with self.subTest(shape=shape):
                validate_single_path(np.zeros(shape), "Block", "a delay")

    def test_the_validator_names_what_the_block_measures(self):
        with self.assertRaises(ShapeError) as ctx:
            validate_single_path(np.zeros((4, 8)), "Block", "a delay")
        self.assertIn("Block estimates a delay", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
