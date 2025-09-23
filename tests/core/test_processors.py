
import unittest
import numpy as np
from src.comnumpy.core.processors import Complex2Real, AutoConcatenator


class TestCoreProcessor(unittest.TestCase):

    def test_complex2real(self):
        X = np.array([1+2j, 3+4j, 5+0j])
        processor_real = Complex2Real(part="real")
        Y_real = processor_real(X)
        Y_real_ref = np.array([1., 3., 5.])
        np.testing.assert_allclose(Y_real, Y_real_ref, atol=1e-8)

        processor_imag = Complex2Real(part="imag")
        Y_imag = processor_imag(X)
        Y_imag_ref = np.array([2., 4., 0.])
        np.testing.assert_allclose(Y_imag, Y_imag_ref, atol=1e-8)


    def test_autoconcatenator(self):

        # test 1
        input_copy_mask = np.array([True, False, True])
        output_original_mask = np.array([True, True, True, False, False])
        output_copy_mask = np.array([False, False, False, True, True])
        X = np.array([1, 2, 3])
        concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask)
        Y = concatenator(X)
        Y_ref = np.array([1, 2, 3, 1, 3])
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

        # test 2
        input_copy_mask = np.array([True, False])
        output_original_mask = np.array([True, True, False, False, False])
        output_copy_mask = np.array([False, False, False, True, False])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask)
        Y = concatenator(X)
        Y_ref = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [1, 2, 3], [0, 0, 0]])
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

        # test 3
        input_copy_mask = np.array([False, True, True])
        output_original_mask = np.array([False, True, True, True, False])
        output_copy_mask = np.array([True, False, False, False, True])
        X = np.array([[1, 2, 3], [4, 5, 6]])
        concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask, axis=-1)
        Y = concatenator(X)
        Y_ref = np.array([[2, 1, 2, 3, 3],[5, 4, 5, 6, 6]])
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
