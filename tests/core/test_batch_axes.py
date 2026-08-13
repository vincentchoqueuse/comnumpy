"""The batch contract, block by block.

Leading axes ahead of a block's event axes are batch axes: independent
trials. The contract has three families, and each family has its own
failure mode, so each gets its own lock:

* a **deterministic** block broadcasts its one configuration over the
  batch -- row ``i`` of a batched call must equal the block applied to
  row ``i`` alone;
* a **stochastic** block draws independently per event -- trials must
  not share a realization, and what one event *is* must be declared;
* an **adaptive** block carries independent state per event -- one
  equalizer per pair, never one equalizer smeared over the batch.
"""
import unittest

import numpy as np

from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.utils import Constellation, get_alphabet
from comnumpy.exceptions import ShapeError
from comnumpy.mimo.compensators import BlindDualMIMOCompensator
from comnumpy.ofdm.chains import OFDMReceiver, OFDMTransmitter
from comnumpy.optical.channels import PhaseNoise


class TestDeterministicBlocksBroadcast(unittest.TestCase):
    """Row i of a batched call equals the block applied to row i."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = (rng.standard_normal((3, 64))
                  + 1j * rng.standard_normal((3, 64)))
        self.h = np.array([1.0, 0.4 + 0.2j, 0.1])

    def test_fir_channel_all_modes(self):
        for mode in ("full", "same", "valid"):
            with self.subTest(mode=mode):
                y = FIRChannel(self.h, mode=mode)(self.x)
                row = FIRChannel(self.h, mode=mode)(self.x[1])
                self.assertEqual(y.shape[:-1], (3,))
                np.testing.assert_allclose(y[1], row)

    def test_linear_equalizer_both_methods(self):
        x = FIRChannel(self.h)(self.x)
        for method in ("zf", "mmse"):
            with self.subTest(method=method):
                y = LinearEqualizer(self.h, method=method, sigma2=0.1)(x)
                row = LinearEqualizer(self.h, method=method,
                                      sigma2=0.1)(x[1])
                np.testing.assert_allclose(y[1], row)

    def test_mapper_demapper_and_ofdm_chains(self):
        constellation = Constellation("QAM", 16)
        rng = np.random.default_rng(1)
        indices = rng.integers(0, 16, (3, 64))
        symbols = SymbolMapper(constellation)(indices)
        waveform = OFDMTransmitter(16, 4)(symbols)
        received = OFDMReceiver(16, 4, h=self.h)(waveform)
        detected = SymbolDemapper(constellation)(received)
        for name, batched, row in (
                ("mapper", symbols, SymbolMapper(constellation)(indices[1])),
                ("tx", waveform, OFDMTransmitter(16, 4)(symbols[1])),
                ("rx", received, OFDMReceiver(16, 4, h=self.h)(waveform[1])),
                ("demapper", detected,
                 SymbolDemapper(constellation)(received[1]))):
            with self.subTest(block=name):
                np.testing.assert_allclose(batched[1], row)


class TestStochasticBlocksDrawIndependently(unittest.TestCase):

    def test_awgn_noise_differs_across_the_batch(self):
        x = np.zeros((2, 4096), dtype=complex)
        y = AWGN(sigma2=1.0, seed=0)(x)
        self.assertFalse(np.allclose(y[0], y[1]))

    def test_one_laser_per_pair_independent_across_trials(self):
        """per='pair': the two rows of a pair share the walk, and two
        trials of a batch do not."""
        x = np.ones((3, 2, 512), dtype=complex)
        y = PhaseNoise(1e-3, seed=0)(x)
        phase = np.unwrap(np.angle(y), axis=-1)
        np.testing.assert_allclose(phase[0, 0], phase[0, 1])
        self.assertFalse(np.allclose(phase[0, 0], phase[1, 0]))

    def test_a_bare_pair_still_shares_one_walk(self):
        x = np.ones((2, 512), dtype=complex)
        y = PhaseNoise(1e-3, seed=0)(x)
        np.testing.assert_allclose(np.angle(y[0]), np.angle(y[1]))

    def test_the_ambiguous_shape_is_refused(self):
        """(n, N) with n != 2 is not a pair and not declared batch:
        the block refuses it and names both resolutions."""
        with self.assertRaises(ShapeError):
            PhaseNoise(1e-3, seed=0)(np.ones((3, 64), dtype=complex))

    def test_per_row_and_per_signal_resolve_it(self):
        x = np.ones((3, 512), dtype=complex)
        rows = PhaseNoise(1e-3, per="row", seed=0)(x)
        self.assertFalse(np.allclose(np.angle(rows[0]), np.angle(rows[1])))
        shared = PhaseNoise(1e-3, per="signal", seed=0)(x)
        np.testing.assert_allclose(np.angle(shared[0]), np.angle(shared[1]))


class TestAdaptiveBlocksCarryStatePerEvent(unittest.TestCase):

    def frames(self, n_trials=3, n=400, seed=0):
        alphabet = np.asarray(get_alphabet("PSK", 4))
        rng = np.random.default_rng(seed)
        s = alphabet[rng.integers(0, 4, (n_trials, 2, n))]
        # a fixed rotation mixes the two polarizations of every trial
        theta = 0.4
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        return alphabet, s, np.einsum("ij,kjn->kin", rotation, s)

    def test_each_pair_gets_its_own_equalizer(self):
        """Pair i of a batched pass equals a fresh compensator run on
        pair i alone -- the state is per event, not smeared."""
        alphabet, _, x = self.frames()
        compensator = BlindDualMIMOCompensator(L=2, alphabet=alphabet,
                                               mu=1e-3)
        y = compensator(x)
        assert compensator.H_ is not None
        self.assertEqual(compensator.H_.shape, (3, 2, 2 * (2 * 2 + 1)))
        alone = BlindDualMIMOCompensator(L=2, alphabet=alphabet,
                                         mu=1e-3)(x[1])
        np.testing.assert_allclose(y[1], alone)

    def test_partial_fit_resumes_each_pair_from_its_own_state(self):
        alphabet, _, x = self.frames()
        batched = BlindDualMIMOCompensator(L=2, alphabet=alphabet, mu=1e-3)
        batched(x)
        batched(x)
        alone = BlindDualMIMOCompensator(L=2, alphabet=alphabet, mu=1e-3)
        alone(x[1])
        alone(x[1])
        assert batched.H_ is not None and alone.H_ is not None
        np.testing.assert_allclose(batched.H_[1], alone.H_)

    def test_a_stale_batch_state_is_refused(self):
        alphabet, _, x = self.frames(n_trials=3)
        compensator = BlindDualMIMOCompensator(L=2, alphabet=alphabet,
                                               mu=1e-3)
        compensator(x)
        _, _, other = self.frames(n_trials=4, seed=1)
        with self.assertRaises(ShapeError):
            compensator(other)

    def test_one_polarization_is_refused(self):
        alphabet = np.asarray(get_alphabet("PSK", 4))
        compensator = BlindDualMIMOCompensator(L=2, alphabet=alphabet)
        with self.assertRaises(ShapeError):
            compensator(np.ones(64, dtype=complex))


if __name__ == "__main__":
    unittest.main()
