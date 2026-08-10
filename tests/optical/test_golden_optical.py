"""Golden tests for the optical module (decision D7).

Fast, assertion-only versions of the validation/ scripts: each test pins
the simulation to an analytical reference, so a normalization or sign
regression in the SSFM fails CI immediately.
"""
import unittest

import numpy as np

from comnumpy.optical.channels import ChromaticDispersion
from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import compute_beta2

FS = 1e12


class TestChromaticDispersion(unittest.TestCase):
    """Gaussian pulse broadening: T1 = T0 * sqrt(1 + (z/L_D)^2)."""

    def test_gaussian_broadening_matches_theory(self):
        N, T0 = 4096, 20e-12
        t = (np.arange(N) - N // 2) / FS
        x = np.exp(-t**2 / (2 * T0**2)).astype(complex)
        beta2 = compute_beta2(1550, 17, 299792458) * 1e-24  # s^2/km
        L_D = T0**2 / abs(beta2)

        z = 2 * L_D
        y = ChromaticDispersion(z, alpha_dB=0, fs=FS)(x)
        intensity = np.abs(y) ** 2
        center = np.sum(t * intensity) / np.sum(intensity)
        rms = np.sqrt(np.sum((t - center) ** 2 * intensity) / np.sum(intensity))
        T1_meas = rms * np.sqrt(2)
        T1_theo = T0 * np.sqrt(1 + (z / L_D) ** 2)
        self.assertAlmostEqual(T1_meas / T1_theo, 1.0, places=10)

    def test_energy_conserved_without_loss(self):
        x = (np.random.default_rng(0).normal(size=1024)
             + 1j * np.random.default_rng(1).normal(size=1024))
        y = ChromaticDispersion(100.0, alpha_dB=0, fs=FS)(x)
        self.assertAlmostEqual(
            float(np.sum(np.abs(y) ** 2) / np.sum(np.abs(x) ** 2)), 1.0, places=10)

    def test_forward_backward_identity(self):
        x = (np.random.default_rng(2).normal(size=1024)
             + 1j * np.random.default_rng(3).normal(size=1024))
        y = ChromaticDispersion(80.0, alpha_dB=0.2, fs=FS)(x)
        z = ChromaticDispersion(80.0, alpha_dB=0.2, fs=FS, direction=-1)(y)
        np.testing.assert_allclose(z, x, atol=1e-12)


class TestKerrSPM(unittest.TestCase):
    """Kerr-only propagation: phi_NL = gamma * P * L(_eff)."""

    P0, GAMMA, L = 5e-3, 1.3, 80.0

    def spm_phase(self, alpha_dB, StPS):
        x = np.sqrt(self.P0) * np.ones(256, dtype=complex)
        link = FiberLink(1, L_span=self.L, StPS=StPS, fs=FS, noise_scaling=0,
                         fiber=FiberSpec(alpha_dB, gamma=self.GAMMA,
                                         cd_coefficient=0))
        y = link(x)
        return float(np.angle(y[0] * np.conj(x[0])))

    def test_lossless_phase_exact(self):
        self.assertAlmostEqual(self.spm_phase(0, 1),
                               self.GAMMA * self.P0 * self.L, places=12)

    def test_lossy_phase_converges_to_effective_length(self):
        alpha = np.log(10) / 10 * 0.2
        L_eff = (1 - np.exp(-alpha * self.L)) / alpha
        phi_theo = self.GAMMA * self.P0 * L_eff
        rel_err = abs(self.spm_phase(0.2, 200) - phi_theo) / phi_theo
        self.assertLess(rel_err, 1e-4)


class TestSoliton(unittest.TestCase):
    """Fundamental soliton is shape-invariant through the full SSFM."""

    def test_fundamental_soliton_preserved(self):
        N, T0, gamma = 2048, 25e-12, 1.3
        t = (np.arange(N) - N // 2) / FS
        beta2 = compute_beta2(1550, 17, 299792458) * 1e-24
        P0 = abs(beta2) / (gamma * T0**2)
        L_D = T0**2 / abs(beta2)
        x = (np.sqrt(P0) / np.cosh(t / T0)).astype(complex)

        L = L_D  # one dispersion length is enough to catch a sign error
        link = FiberLink(1, L_span=L, StPS=int(2 * L), fs=FS, noise_scaling=0,
                         fiber=FiberSpec(0, gamma=gamma, cd_coefficient=17))
        y = link(x)
        nmse = float(np.sum(np.abs(np.abs(y) - np.abs(x)) ** 2)
                     / np.sum(np.abs(x) ** 2))
        self.assertLess(nmse, 1e-9)

        # counter-check: without Kerr the same pulse disperses
        link_lin = FiberLink(1, L_span=L, StPS=1, fs=FS, noise_scaling=0,
                             fiber=FiberSpec(0, gamma=0, cd_coefficient=17))
        y_lin = link_lin(x)
        nmse_lin = float(np.sum(np.abs(np.abs(y_lin) - np.abs(x)) ** 2)
                         / np.sum(np.abs(x) ** 2))
        self.assertGreater(nmse_lin, 1e-3)


class TestDBP(unittest.TestCase):
    """Noiseless DBP with matched steps inverts the fiber exactly."""

    def test_roundtrip_machine_precision(self):
        rng = np.random.default_rng(42)
        N = 1024
        X = rng.normal(size=N) + 1j * rng.normal(size=N)
        Xf = np.fft.fft(X)
        Xf[np.abs(np.fft.fftfreq(N)) > 0.2] = 0
        x = np.fft.ifft(Xf)
        x *= np.sqrt(2e-3) / np.sqrt(np.mean(np.abs(x) ** 2))  # 3 dBm

        params = dict(L_span=80.0, StPS=20, fs=50e9,
                      fiber=FiberSpec(0.2, gamma=1.3, cd_coefficient=17))
        y = FiberLink(2, noise_scaling=0, **params)(x)
        x_hat = DBP(2, **params)(y)
        nmse = float(np.sum(np.abs(x_hat - x) ** 2) / np.sum(np.abs(x) ** 2))
        self.assertLess(nmse, 1e-20)


if __name__ == "__main__":
    unittest.main()
