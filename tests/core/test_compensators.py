
import unittest
import numpy as np
from src.comnumpy.core.compensators import Normalizer


class TestCoreCompensator(unittest.TestCase):

    def test_normalizer(self):

        X = np.array([1, 2, 3, 4])
        
        # test max
        normalizer = Normalizer(method='max', value=2.0)
        Y = normalizer(X)
        Y_ref = np.array([0.5, 1. , 1.5, 2. ])
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

        # test var
        normalizer = Normalizer(method='var', value=2.0)
        Y = normalizer(X)

        scale = np.sqrt(np.var(X))  
        Y_ref = (X / scale) * np.sqrt(2.0)
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
