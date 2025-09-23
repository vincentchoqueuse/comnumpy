import unittest
import numpy as np
from src.comnumpy.core import Sequential, Recorder
from src.comnumpy.core.generators import SymbolGenerator
from src.comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from src.comnumpy.core.utils import get_alphabet
from src.comnumpy.core.channels import AWGN
from src.comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo


class TestAWGNChainPerformance(unittest.TestCase):

    def test_one_shot(self):
        M = 16
        N = 1000000
        modulation = "QAM"
        SNR_dB = 16
        alphabet = get_alphabet(modulation, M)

        # create chain
        chain = Sequential([
            SymbolGenerator(M),
            Recorder(name="recorder_tx"),
            SymbolMapper(alphabet),
            AWGN(SNR_dB, unit="snr_dB"),
            SymbolDemapper(alphabet),
            ])

        # run chain
        y = chain(N)

        # evaluate metrics
        data_tx = chain["recorder_tx"].get_data()
        ser = compute_ser(data_tx, y)

        snr_per_bit = (10**(SNR_dB/10))/np.log2(M)
        ser_theo = compute_metric_awgn_theo(modulation, M, snr_per_bit, "ser")
        np.testing.assert_allclose(ser, ser_theo, atol=1e-3)



if __name__ == '__main__':
    unittest.main()