import unittest
import numpy as np
from src.comnumpy.core.processors import Upsampler

class TestUpsampler(unittest.TestCase):
    def test_upsample_factor_2(self):
        X = np.array([1, 2, 3])
        upsampler = Upsampler(L=2)
        Y = upsampler(X)
        expected_output = np.array([1, 0, 2, 0, 3, 0])
        np.testing.assert_array_equal(Y, expected_output)

    def test_upsample_factor_3_with_phase(self):
        X = np.array([1, 2])
        upsampler = Upsampler(L=3, phase=1)
        Y = upsampler(X)
        expected_output = np.array([0, 1, 0, 0, 2, 0])
        np.testing.assert_array_equal(Y, expected_output)

    def test_upsample_with_scale(self):
        X = np.array([1, 2])
        upsampler = Upsampler(L=2, scale=2.0)
        Y = upsampler(X)
        expected_output = np.array([2, 0, 4, 0])
        np.testing.assert_array_equal(Y, expected_output)

    def test_upsample_with_axis(self):
        X = np.array([[1, 2], [3, 4]])
        upsampler = Upsampler(L=2, axis=-1)
        Y = upsampler(X)
        expected_output = np.array([[1., 0., 2., 0.], [3., 0., 4., 0.]])
        np.testing.assert_array_equal(Y, expected_output)

if __name__ == '__main__':
    unittest.main()
