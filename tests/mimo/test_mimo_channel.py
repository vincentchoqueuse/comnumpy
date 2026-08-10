
import unittest
import numpy as np
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser
from comnumpy.mimo.channels import FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import LinearDetector, MaximumLikelihoodDetector


class TestMIMOChannelChain(unittest.TestCase):

    def setUp(self):
        # Basic parameters
        self.N_r, self.N_t = 3, 2
        self.M = 4
        self.sigma2 = 10**-3
        self.N = 10
        self.alphabet = get_alphabet("PSK", self.M)
        self.H = rayleigh_channel(self.N_r, self.N_t)

    def _compute_detector_ser(self, detector):

        chain = Sequential([
            SymbolGenerator(self.M, name="tx"),
            SymbolMapper(self.alphabet),
            FlatMIMOChannel(self.H),
            AWGN(sigma2=self.sigma2),
            detector
            ], taps=["tx"])

        Y = chain((self.N_t, self.N))
        data_tx = chain.tap("tx")
        return compute_ser(data_tx, Y)

    def test_zf_one_shot(self):
        detector = LinearDetector(self.alphabet, H=self.H, method="zf")
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_mmse_one_shot(self):
        detector = LinearDetector(self.alphabet, H=self.H, sigma2=self.sigma2, method="mmse")
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_ml_one_shot(self):
        detector = MaximumLikelihoodDetector(self.alphabet, H=self.H)
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
