
import unittest
import numpy as np
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.metrics import compute_ser


class TestFIRChannelChain(unittest.TestCase):

    def setUp(self):
        # Basic parameters
        self.M = 16
        self.sigma2 = 10**-3
        self.alphabet = get_alphabet("QAM", self.M)
        self.h = np.array([1 + 0.1j, 0.2 - 0.1j, -0.1 + 0.5j])
        self.N = 1000


    def _compute_equalizer_ser(self, equalizer):
        chain = Sequential([
            SymbolGenerator(self.M, name="tx"),
            SymbolMapper(self.alphabet),
            FIRChannel(self.h),
            AWGN(sigma2=self.sigma2),
            equalizer,
            SymbolDemapper(self.alphabet),
            ], taps=["tx"])

        y = chain(self.N)
        data_tx = chain.tap("tx")
        return compute_ser(data_tx, y)

    def test_zf_one_shot(self):
        equalizer = LinearEqualizer(self.h, method="zf")
        ser = self._compute_equalizer_ser(equalizer)
        np.testing.assert_allclose(ser, 0, atol=1e-8)

    def test_mmse_one_shot(self):
        equalizer = LinearEqualizer(self.h, method="mmse")
        ser = self._compute_equalizer_ser(equalizer)
        np.testing.assert_allclose(ser, 0, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
