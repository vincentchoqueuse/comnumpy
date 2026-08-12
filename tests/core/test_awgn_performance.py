import unittest
import numpy as np
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo


class TestAWGNChainPerformance(unittest.TestCase):

    def test_one_shot(self):
        M = 16
        N = 1000000
        modulation = "QAM"
        SNR_dB = 16
        alphabet = get_alphabet(modulation, M)

        # create chain
        chain = Sequential([
            SymbolGenerator(M, name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=SNR_dB),
            SymbolDemapper(alphabet),
            ], taps=["tx"])

        # run chain
        y = chain(N)

        # evaluate metrics
        data_tx = chain.tap("tx")
        ser = compute_ser(data_tx, y)

        snr_per_bit = (10**(SNR_dB/10))/np.log2(M)
        theory = compute_metric_awgn_theo(modulation, M, snr_per_bit)
        ser_theo = theory["ser"]
        np.testing.assert_allclose(ser, ser_theo, atol=1e-3)



if __name__ == '__main__':
    unittest.main()
