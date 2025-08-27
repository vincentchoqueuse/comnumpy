
import unittest
import numpy as np
from src.comnumpy.core import Sequential, Recorder
from src.comnumpy.core.generators import SymbolGenerator
from src.comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from src.comnumpy.core.utils import get_alphabet
from src.comnumpy.core.channels import AWGN, FIRChannel
from src.comnumpy.core.compensators import LinearEqualizer
from src.comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo
from src.comnumpy.mimo.channels import FlatMIMOChannel, AWGN
from src.comnumpy.mimo.utils import rayleigh_channel
from src.comnumpy.mimo.detectors import LinearDetector, MaximumLikelihoodDetector


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
            SymbolGenerator(self.M),
            Recorder(name="recorder_tx"),
            SymbolMapper(self.alphabet),
            FlatMIMOChannel(self.H),
            AWGN(self.sigma2),
            detector
            ])

        Y = chain((self.N_t, self.N))
        data_tx = chain["recorder_tx"].get_data()
        return compute_ser(data_tx, Y)

    def test_zf_one_shot(self):
        detector = LinearDetector(self.alphabet, self.H, method="zf")
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_mmse_one_shot(self):
        detector = LinearDetector(self.alphabet, self.H, sigma2=self.sigma2, method="mmse")
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_ml_one_shot(self):
        detector = MaximumLikelihoodDetector(self.alphabet, self.H)
        ser = self._compute_detector_ser(detector)
        np.testing.assert_allclose(ser, 0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()