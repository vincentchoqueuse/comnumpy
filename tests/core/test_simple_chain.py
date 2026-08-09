
import unittest
import numpy as np
from src.comnumpy.core import Sequential
from src.comnumpy.core.generators import SymbolGenerator
from src.comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from src.comnumpy.core.utils import get_alphabet
from src.comnumpy.core.metrics import compute_ser


class TestSimpleChain(unittest.TestCase):

    def test_one_shot(self):
        M = 16
        N = 1000000
        modulation = "QAM"
        alphabet = get_alphabet(modulation, M)

        # create chain
        chain = Sequential([
            SymbolGenerator(M, name="tx"),
            SymbolMapper(alphabet),
            SymbolDemapper(alphabet),
            ], taps=["tx"])

        # run chain
        y = chain(N)

        # evaluate metrics
        data_tx = chain.tap("tx")
        ser = compute_ser(data_tx, y)
        np.testing.assert_allclose(ser, 0, atol=1e-8)



if __name__ == '__main__':
    unittest.main()
