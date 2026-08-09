"""Golden test for the MIMO detectors (fast version of validation/mimo_zf_ml_ber.py)."""
import unittest

import numpy as np

from comnumpy.mimo.detectors import LinearDetector, MaximumLikelihoodDetector

BPSK = np.array([1.0 + 0.0j, -1.0 + 0.0j])


class TestMIMODetectorGolden(unittest.TestCase):

    def test_zf_matches_rayleigh_closed_form_and_ml_beats_it(self):
        """2x2 iid Rayleigh, BPSK, 8 dB: ZF = diversity-1 closed form; ML < ZF."""
        snr_dB, n_real, n_sym = 8.0, 1500, 40
        rng = np.random.default_rng(42)
        sigma2 = 10 ** (-snr_dB / 10)
        err_zf = err_ml = total = 0
        for _ in range(n_real):
            H = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
            bits = rng.integers(0, 2, (2, n_sym))
            x = BPSK[bits]
            noise = np.sqrt(sigma2 / 2) * (rng.normal(size=x.shape)
                                           + 1j * rng.normal(size=x.shape))
            y = H @ x + noise
            err_zf += np.sum(LinearDetector(BPSK, H=H)(y) != bits)
            err_ml += np.sum(MaximumLikelihoodDetector(BPSK, H=H)(y) != bits)
            total += bits.size

        g = 1 / sigma2
        ber_theo = 0.5 * (1 - np.sqrt(g / (1 + g)))
        ber_zf = err_zf / total
        ber_ml = err_ml / total
        self.assertLess(abs(ber_zf - ber_theo) / ber_theo, 0.15,
                        f"ZF {ber_zf} vs closed form {ber_theo}")
        self.assertLess(ber_ml, ber_zf / 2)


if __name__ == "__main__":
    unittest.main()
