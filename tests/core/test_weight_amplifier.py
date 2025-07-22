import unittest
import numpy as np
from src.comnumpy.core.processors import WeightAmplifier

class TestWeightAmplifier(unittest.TestCase):

    def test_weights_applied_along_axis_0(self):
        weight_amplifier = WeightAmplifier(weight=np.array([2, 3]), axis=0)
        X = np.array([[1, 2], [3, 4]])
        Y = weight_amplifier(X)
        expected_output = np.array([[2, 4], [9, 12]])
        np.testing.assert_array_equal(Y, expected_output)

    def test_weights_applied_along_axis_1(self):
        weight_amplifier = WeightAmplifier(weight=np.array([2, 3]), axis=1)
        X = np.array([[1, 2], [3, 4]])
        Y = weight_amplifier(X)
        expected_output = np.array([[2, 6], [6, 12]])
        np.testing.assert_array_equal(Y, expected_output)

    def test_weights_applied_along_negative_axis(self):
        weight_amplifier = WeightAmplifier(weight=np.array([2, 3]), axis=-1)
        X = np.array([[1, 2], [3, 4]])
        Y = weight_amplifier(X)
        expected_output = np.array([[2, 6], [6, 12]])
        np.testing.assert_array_equal(Y, expected_output)

if __name__ == '__main__':
    unittest.main()
