"""The two blocks the PDM page stands on: the emulator and the tracker.

`PMDEmulator` claims to be unitary, seeded and frequency-selective;
`BlindPhaseSearchCompensator` claims to follow a Wiener phase within
its grid and its window. Each claim is a test, because the tutorial
built on them quotes their numbers as physics.
"""
import unittest

import numpy as np

from comnumpy.core.compensators import BlindPhaseSearchCompensator
from comnumpy.core.utils import get_alphabet
from comnumpy.exceptions import ShapeError
from comnumpy.optical.channels import PMDEmulator


class TestPMDEmulator(unittest.TestCase):

    def field_pair(self, n=256, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((2, n))
                + 1j * rng.standard_normal((2, n)))

    def test_energy_is_conserved(self):
        """The emulator is a product of unitaries: lossless, exactly."""
        x = self.field_pair()
        y = PMDEmulator(5e-12, n_sections=8, fs=64e9, seed=1)(x)
        self.assertAlmostEqual(float(np.sum(np.abs(y) ** 2)),
                               float(np.sum(np.abs(x) ** 2)), places=8)

    def test_same_seed_same_fibre(self):
        x = self.field_pair()
        first = PMDEmulator(5e-12, fs=64e9, seed=3)(x)
        second = PMDEmulator(5e-12, fs=64e9, seed=3)(x)
        np.testing.assert_allclose(first, second)
        third = PMDEmulator(5e-12, fs=64e9, seed=4)(x)
        self.assertFalse(np.allclose(first, third))

    def test_a_rotation_mixes_the_polarizations(self):
        x = np.zeros((2, 64), dtype=complex)
        x[0] = 1.0
        y = PMDEmulator(0.0, n_sections=1, seed=1)(x)
        self.assertGreater(float(np.mean(np.abs(y[1]) ** 2)), 1e-3)

    def test_zero_dgd_is_frequency_flat(self):
        """Without DGD the emulator is one wavelength-flat Jones matrix:
        applying it twice with the same seed equals composing the
        matrices, and every frequency sees the same rotation -- a pure
        tone in must be a pure tone out."""
        n = 128
        tone = np.exp(2j * np.pi * 13 * np.arange(n) / n)
        x = np.stack([tone, np.zeros(n, dtype=complex)])
        y = PMDEmulator(0.0, n_sections=4, seed=2)(x)
        spectrum = np.abs(np.fft.fft(y, axis=-1))
        occupied = np.sum(spectrum > 1e-6 * spectrum.max(), axis=-1)
        self.assertTrue(np.all(occupied <= 1))

    def test_dgd_delays_the_principal_states_against_each_other(self):
        """One section, aligned input: the group delay difference between
        the two rows is the declared tau."""
        fs, tau = 64e9, 20e-12
        n = 4096
        rng = np.random.default_rng(5)
        pulse = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        emulator = PMDEmulator(tau, n_sections=1, fs=fs, seed=7)
        assert emulator.rotations_ is not None
        rotation = emulator.rotations_[0]
        # drive the two principal states one at a time, through the
        # inverse rotation, so each row sees a pure delay of +-tau/2
        for row, sign in ((0, +1), (1, -1)):
            x = np.zeros((2, n), dtype=complex)
            x[row] = pulse
            y = emulator(np.einsum("ij,jn->in", rotation.conj().T, x))
            cross = np.fft.ifft(np.fft.fft(y[row]) *
                                np.conj(np.fft.fft(pulse)))
            lag = int(np.argmax(np.abs(cross)))
            lag = lag - n if lag > n // 2 else lag
            # multiplying the spectrum by exp(+j w tau/2) advances the
            # signal (fft convention e^{-jwn}), hence the minus sign
            self.assertEqual(lag, -round(sign * tau / 2 * fs))

    def test_rms_dgd_matches_the_declared_value(self):
        """The theory the emulator must obey: section DGDs add in
        quadrature, and the ensemble DGD is Maxwellian with RMS equal
        to the declared tau (Poole & Wagner). Measured at one frequency
        over many section draws, from the eigenvalues of the
        group-delay operator j dT/dw T^H."""
        fs, tau, n = 64e9, 10e-12, 512
        dgds = []
        for seed in range(300):
            emulator = PMDEmulator(tau, n_sections=8, fs=fs, seed=seed)
            responses = []
            for k in (1, 2):     # T at two neighbouring frequencies
                tone = np.exp(2j * np.pi * k * np.arange(n) / n)
                columns = []
                for row in range(2):
                    x = np.zeros((2, n), dtype=complex)
                    x[row] = tone
                    y = emulator(x)
                    columns.append(np.fft.fft(y, axis=-1)[:, k] / n)
                responses.append(np.stack(columns, axis=1))
            delta_omega = 2 * np.pi * fs / n
            u = responses[1] @ responses[0].conj().T
            phases = np.angle(np.linalg.eigvals(u))
            dgds.append(abs(phases[0] - phases[1]) / delta_omega)
        rms = float(np.sqrt(np.mean(np.square(dgds))))
        self.assertLess(abs(rms - tau) / tau, 0.10)
        mean = float(np.mean(dgds))
        self.assertLess(abs(mean - np.sqrt(8 / (3 * np.pi)) * tau) / tau,
                        0.10)

    def test_one_polarization_is_refused(self):
        with self.assertRaises(ShapeError):
            PMDEmulator(1e-12, fs=64e9)(np.ones(64, dtype=complex))


class TestBlindPhaseSearch(unittest.TestCase):

    def qpsk_frame(self, n=4096, seed=0):
        alphabet = get_alphabet("PSK", 4)
        rng = np.random.default_rng(seed)
        return alphabet, alphabet[rng.integers(0, 4, n)]

    def test_tracks_a_wiener_walk_within_the_grid(self):
        alphabet, s = self.qpsk_frame()
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.normal(0.0, np.sqrt(1e-4), s.size))
        compensator = BlindPhaseSearchCompensator(alphabet)
        y = compensator(s * np.exp(1j * walk))
        assert compensator.phase_ is not None
        interior = slice(32, -32)          # the window edges are biased
        self.assertLess(
            float(np.max(np.abs(compensator.phase_ - walk)[interior])),
            2 * (np.pi / 2) / 32,
            "the estimate left the walk by more than two grid steps")
        self.assertLess(float(np.max(np.abs(y - s)[interior])), 0.2)

    def test_rows_of_a_pair_get_independent_trajectories(self):
        alphabet, s = self.qpsk_frame(n=1024)
        x = np.stack([s * np.exp(1j * 0.3), s * np.exp(-1j * 0.2)])
        compensator = BlindPhaseSearchCompensator(alphabet, half_window=8)
        compensator(x)
        assert compensator.phase_ is not None
        # the estimate lives modulo pi/2 (quadrant symmetry): compare
        # each trajectory to its truth in that quotient
        for row, truth in ((0, 0.3), (1, -0.2)):
            residual = np.median(compensator.phase_[row]) - truth
            residual = (residual + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self.assertAlmostEqual(float(residual), 0.0, delta=0.03)

    def test_the_quadrant_ambiguity_is_documented_not_resolved(self):
        """A pi/2 offset is invisible to the search, by symmetry."""
        alphabet, s = self.qpsk_frame(n=1024)
        y = BlindPhaseSearchCompensator(alphabet,
                                        half_window=8)(s * 1j)
        self.assertLess(float(np.max(np.abs(y - s * 1j))), 0.06)


if __name__ == "__main__":
    unittest.main()
