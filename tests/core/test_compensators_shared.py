"""The D49 question, answered rather than asked (shared vs per-path)."""
import unittest

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.compensators import (DataAidedComplexGainCompensator,
                                        DataAidedPhaseCompensator)
from comnumpy.exceptions import ShapeError

RNG = np.random.default_rng(20260811)
REFERENCE = (RNG.normal(size=(2, 256)) + 1j * RNG.normal(size=(2, 256)))


class TestTheDecisionIsRequired(unittest.TestCase):
    """Silence is not an answer, and broadcasting would give a wrong one."""

    def test_a_multipath_signal_without_shared_is_refused(self):
        for block in (DataAidedComplexGainCompensator(reference=REFERENCE),
                      DataAidedPhaseCompensator(reference=REFERENCE)):
            with self.subTest(block=type(block).__name__):
                with self.assertRaises(ShapeError) as ctx:
                    block(REFERENCE)
                self.assertIn("shared=", str(ctx.exception))

    def test_a_single_path_signal_never_asks(self):
        """The question only exists once there are paths to confuse."""
        gain = 2 * np.exp(1j * np.pi / 3)
        block = DataAidedComplexGainCompensator(reference=REFERENCE[0])
        corrected = block(gain * REFERENCE[0])
        self.assertLess(np.abs(corrected - REFERENCE[0]).max(), 1e-12)


class TestSharedIsOneQuantity(unittest.TestCase):
    r"""One gain behind both paths, fitted on all of the data.

    This is the dual-polarization case the GN tutorial measures: the mean
    nonlinear phase comes from :math:`|E_x|^2 + |E_y|^2`, so it is common
    to the two polarizations by construction.
    """

    def test_one_gain_recovers_both_paths(self):
        gain = 2 * np.exp(1j * np.pi / 3)
        block = DataAidedComplexGainCompensator(reference=REFERENCE,
                                                shared=True)
        corrected = block(gain * REFERENCE)
        self.assertLess(np.abs(corrected - REFERENCE).max(), 1e-12)
        self.assertIsInstance(block.gain_, complex)

    def test_the_joint_fit_beats_either_path_alone(self):
        """Not merely tidier: it is the better estimator, it sees twice the data."""
        gain = 2 * np.exp(1j * np.pi / 3)
        noisy = gain * REFERENCE + 0.4 * (RNG.normal(size=REFERENCE.shape)
                                          + 1j * RNG.normal(size=REFERENCE.shape))
        joint = DataAidedComplexGainCompensator(reference=REFERENCE,
                                                shared=True).fit(noisy).gain_
        alone = [DataAidedComplexGainCompensator(
            reference=REFERENCE[path]).fit(noisy[path]).gain_ for path in (0, 1)]
        truth = 1 / gain
        self.assertLess(abs(joint - truth),
                        max(abs(one - truth) for one in alone))

    def test_the_phase_compensator_shares_too(self):
        block = DataAidedPhaseCompensator(reference=REFERENCE, shared=True)
        rotated = np.exp(1j * 0.7) * REFERENCE
        self.assertLess(np.abs(block(rotated) - REFERENCE).max(), 1e-10)


class TestPerPathIsOneEach(unittest.TestCase):
    """After a butterfly equalizer each output carries its own residue."""

    def test_each_path_gets_its_own_gain(self):
        gains = np.array([2 * np.exp(1j * 0.3), 5 * np.exp(-1j * 1.1)])
        block = DataAidedComplexGainCompensator(reference=REFERENCE,
                                                shared=False)
        corrected = block(gains[:, None] * REFERENCE)
        self.assertLess(np.abs(corrected - REFERENCE).max(), 1e-12)
        self.assertLess(np.abs(np.asarray(block.gain_) - 1 / gains).max(), 1e-12)

    def test_sharing_a_per_path_gain_does_not_work(self):
        """The two answers are not interchangeable, which is why it asks."""
        gains = np.array([2 * np.exp(1j * 0.3), 5 * np.exp(-1j * 1.1)])
        received = gains[:, None] * REFERENCE
        wrong = DataAidedComplexGainCompensator(reference=REFERENCE,
                                                shared=True)(received)
        self.assertGreater(np.abs(wrong - REFERENCE).max(), 1.0)


class TestItWiresIntoAChain(unittest.TestCase):
    """The point of the keyword: the block can now live in the chain.

    ``wiring`` feeds it the reference the chain generated in the same
    pass, which is the mechanism its own docstring advertises and which a
    dual-polarization signal could not reach before.
    """

    def test_the_reference_arrives_through_wiring(self):
        from comnumpy.core.generators import SymbolGenerator
        from comnumpy.core.mappers import SymbolMapper
        from comnumpy.core.processors import Amplifier
        from comnumpy.core.utils import get_alphabet

        chain = Sequential(
            [SymbolGenerator(4, name="source"),
             SymbolMapper(get_alphabet("QAM", 4), name="tx"),
             Amplifier(2 * np.exp(1j * np.pi / 5), name="channel"),
             DataAidedComplexGainCompensator(reference=np.zeros(1),
                                             shared=True, name="gain")],
            taps=["tx"], wiring={"gain.reference": "tx"})
        out = chain.seed(0)((2, 128))
        self.assertLess(np.abs(out - chain.tapped_["tx"]).max(), 1e-12)


if __name__ == "__main__":
    unittest.main()
