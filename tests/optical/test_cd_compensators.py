"""Tests for the chromatic dispersion FIR compensators (annex A.5).

The least-squares design is pinned against numerical quadrature rather
than against itself: the Gram matrix ``Q`` and the cross-correlation
vector ``d`` are both integrals, and every closed form used in
``compensators.py`` is checked here with ``scipy.integrate.quad`` on the
real and imaginary parts separately.

Two defects motivated these tests (annex A.5):

* ``q_col`` was allocated to zero and never filled, so ``toeplitz``
  returned an upper triangular matrix instead of the Hermitian Toeplitz
  Gram matrix. The full band hides it (``q(p) = sinc(p) = 0`` makes
  ``Q = I`` whatever the first column holds), a reduced band does not.
* ``d`` was hard-coded to the full-band closed form and ignored
  ``w_vect`` entirely, so a reduced band changed ``Q`` but not ``d``.
"""
import unittest

import numpy as np
import numpy.linalg as LA
from scipy.integrate import quad

from src.comnumpy.optical.compensators import (
    ChromaticDispersionFIRCompensator,
    ChromaticDispersionLSFIRCompensator,
    _cd_cross_correlation,
    _cd_gram_matrix,
)

Z = 1000.0        # km
FS = 20e9         # Hz
N_TAPS = 55       # matches the 2*floor(2*pi*K)+1 span of the truncated design

FULL_BAND = [-np.pi, np.pi]
HALF_BAND = [-np.pi / 2, np.pi / 2]
OFFSET_BAND = [-0.4 * np.pi, 0.9 * np.pi]   # asymmetric: Q is complex, not real
BANDS = [FULL_BAND, HALF_BAND, OFFSET_BAND]


def complex_quad(func, a, b):
    """Integrate a complex-valued integrand by parts (real then imaginary)."""
    real = quad(lambda w: func(w).real, a, b, limit=500, epsabs=1e-13, epsrel=1e-13)[0]
    imag = quad(lambda w: func(w).imag, a, b, limit=500, epsabs=1e-13, epsrel=1e-13)[0]
    return real + 1j * imag


def gram_matrix_by_quadrature(N, Omega_1, Omega_2):
    """Reference ``Q[m, l] = (1/2pi) int exp(-j (l-m) Omega) dOmega``."""
    Q = np.zeros((N, N), dtype=complex)
    for m in range(N):
        for l in range(N):
            p = l - m
            Q[m, l] = complex_quad(
                lambda w, p=p: np.exp(-1j * p * w) / (2 * np.pi), Omega_1, Omega_2
            )
    return Q


def cross_correlation_by_quadrature(K, N, Omega_1, Omega_2):
    """Reference ``d[n] = (1/2pi) int exp(j K Omega^2) exp(j n Omega) dOmega``."""
    bound = N // 2
    return np.array([
        complex_quad(
            lambda w, n=n: np.exp(1j * (K * w**2 + n * w)) / (2 * np.pi),
            Omega_1, Omega_2,
        )
        for n in range(-bound, bound + 1)
    ])


def frequency_response(h, w):
    """DTFT of the causal-centered taps ``h`` at pulsations ``w``."""
    bound = len(h) // 2
    n_vect = np.arange(-bound, bound + 1)
    return np.exp(-1j * np.outer(w, n_vect)) @ h


class TestGramMatrix(unittest.TestCase):
    """The Gram matrix must be Hermitian Toeplitz, on every band."""

    def test_is_hermitian(self):
        # Fails before the fix on any band but the default one: the
        # unfilled q_col made toeplitz() return an upper triangular matrix.
        for band in BANDS:
            with self.subTest(band=band):
                Q = _cd_gram_matrix(N_TAPS, *band)
                self.assertTrue(np.allclose(Q, Q.conj().T))

    def test_matches_numerical_quadrature(self):
        for band in BANDS:
            with self.subTest(band=band):
                Q = _cd_gram_matrix(N_TAPS, *band)
                Q_ref = gram_matrix_by_quadrature(9, *band)
                np.testing.assert_allclose(Q[:9, :9], Q_ref, atol=1e-12)

    def test_full_band_is_identity(self):
        # q(p) = sinc(p) = 0 for p != 0: this is what hid the q_col defect.
        Q = _cd_gram_matrix(N_TAPS, *FULL_BAND)
        np.testing.assert_allclose(Q, np.eye(N_TAPS), atol=1e-12)

    def test_reduced_band_is_rank_deficient(self):
        # Documents why the system is solved and not inverted: about
        # N * (1 - (O2 - O1) / 2pi) eigenvalues are numerically zero.
        Q = _cd_gram_matrix(N_TAPS, *HALF_BAND)
        eigenvalues = LA.eigvalsh(Q)
        self.assertLess(np.sum(eigenvalues > 1e-8), N_TAPS)
        self.assertGreater(np.sum(eigenvalues > 1e-8), N_TAPS // 4)


class TestCrossCorrelation(unittest.TestCase):
    """``d`` must follow the design band, not just the full band."""

    def setUp(self):
        self.K = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS).K

    def test_matches_numerical_quadrature(self):
        # Fails before the fix on every band but the default one: the
        # closed form was hard-coded to the bounds [-pi, pi].
        for band in BANDS:
            with self.subTest(band=band):
                d = _cd_cross_correlation(self.K, N_TAPS, *band)
                d_ref = cross_correlation_by_quadrature(self.K, N_TAPS, *band)
                np.testing.assert_allclose(d, d_ref, atol=1e-12)

    def test_reduced_band_differs_from_full_band(self):
        # Guards against a regression that would silently drop w_vect again.
        d_full = _cd_cross_correlation(self.K, N_TAPS, *FULL_BAND)
        d_half = _cd_cross_correlation(self.K, N_TAPS, *HALF_BAND)
        self.assertGreater(np.abs(d_full - d_half).max(), 1e-2)

    def test_band_additivity(self):
        # The integral is additive over adjacent bands, the closed form must be.
        d_left = _cd_cross_correlation(self.K, N_TAPS, -np.pi, 0.3)
        d_right = _cd_cross_correlation(self.K, N_TAPS, 0.3, np.pi)
        d_full = _cd_cross_correlation(self.K, N_TAPS, *FULL_BAND)
        np.testing.assert_allclose(d_left + d_right, d_full, atol=1e-12)


class TestLSFilter(unittest.TestCase):
    """End-to-end properties of the solved filter."""

    def test_filter_solves_the_quadrature_normal_equations(self):
        for band in BANDS:
            with self.subTest(band=band):
                comp = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=band)
                Q_ref = _cd_gram_matrix(N_TAPS, *band)
                d_ref = cross_correlation_by_quadrature(comp.K, N_TAPS, *band)
                residual = LA.norm(Q_ref @ comp.h - d_ref)
                self.assertLess(residual, 1e-9)

    def test_default_band_tends_to_truncated_design_in_the_interior(self):
        # On the full band Q = I, so h = d. Where the erf arguments
        # saturate (|n| << 2 pi K) the LS design must converge to the
        # Savory truncation; near |n| ~ 2 pi K the two legitimately differ.
        previous = None
        for fs in [20e9, 40e9, 80e9]:
            savory = ChromaticDispersionFIRCompensator(Z, fs=fs)
            N = len(savory.h)
            comp = ChromaticDispersionLSFIRCompensator(Z, N, fs=fs)
            tap_scale = 1 / np.sqrt(4 * np.pi * comp.K)
            interior = slice(N // 4, 3 * N // 4)
            relative = np.abs(comp.h[interior] - savory.h[interior]).max() / tap_scale
            if previous is not None:
                self.assertLess(relative, previous)
            previous = relative
        self.assertLess(previous, 0.05)

    def test_default_band_taps_are_unchanged(self):
        # Regression pin: correcting q_col and d must leave the real use
        # case bit-stable. Values measured before the fix.
        comp = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS)
        self.assertAlmostEqual(float(np.abs(comp.h).max()), 0.1618095891387068, places=12)
        self.assertAlmostEqual(complex(comp.h[27]).real, 0.08492770422154605, places=12)
        self.assertAlmostEqual(complex(comp.h[27]).imag, 0.09149261866112145, places=12)

    def test_reduced_band_beats_full_band_inside_its_band(self):
        # The point of w_vect: spend the taps where the signal is.
        w = np.linspace(*HALF_BAND, 2001)
        target = np.exp(1j * ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS).K * w**2)
        err = {}
        for tag, band in [("full", FULL_BAND), ("half", HALF_BAND)]:
            comp = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=band)
            err[tag] = np.abs(frequency_response(comp.h, w) - target).max()
        self.assertLess(err["half"], 1e-3)
        self.assertLess(err["half"], err["full"] / 10)

    def test_reduced_band_taps_stay_bounded(self):
        # An explicit inverse of the rank-deficient Q returned taps of
        # order 1e9 here; the solve keeps them at the scale of the signal.
        comp = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=HALF_BAND)
        self.assertLess(np.abs(comp.h).max(), 1.0)

    def test_compensates_a_dispersed_pulse(self):
        from src.comnumpy.optical.channels import ChromaticDispersion
        x = np.zeros(512, dtype=complex)
        x[256] = 1.0
        y = ChromaticDispersion(Z, fs=FS)(x)
        comp = ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS)
        delay = (N_TAPS - 1) // 2
        x_hat = comp(y)[delay:delay + 512]
        self.assertGreater(np.abs(x_hat[256]), 0.97)


class TestValidation(unittest.TestCase):

    def test_even_length_rejected(self):
        with self.assertRaises(ValueError):
            ChromaticDispersionLSFIRCompensator(Z, 54, fs=FS)

    def test_reversed_band_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=[np.pi, -np.pi])
        self.assertIn("Omega_1 < Omega_2", str(ctx.exception))

    def test_degenerate_band_rejected(self):
        with self.assertRaises(ValueError):
            ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=[0.5, 0.5])

    def test_wrong_band_length_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            ChromaticDispersionLSFIRCompensator(Z, N_TAPS, fs=FS, w_vect=[-np.pi, 0.0, np.pi])
        self.assertIn("expected exactly 2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
