
import unittest
import numpy as np
from comnumpy.exceptions import ShapeError
from comnumpy.core.filters import BWFilter
from comnumpy.core.processors import (
    Amplifier, AutoConcatenator, BlindPhaseTracker, Complex2Real, DataAdder,
    DataExtractor, Downsampler, SampleRemover,
)



class TestCoreProcessor(unittest.TestCase):
    atol = 1e-8

    def test_complex2real(self):
        X = np.array([1+2j, 3+4j, 5+0j])
        processor_real = Complex2Real(part="real")
        Y_real = processor_real(X)
        Y_real_ref = np.array([1., 3., 5.])
        np.testing.assert_allclose(Y_real, Y_real_ref, atol=self.atol)

        processor_imag = Complex2Real(part="imag")
        Y_imag = processor_imag(X)
        Y_imag_ref = np.array([2., 4., 0.])
        np.testing.assert_allclose(Y_imag, Y_imag_ref, atol=self.atol)


    def test_autoconcatenator(self):
        # test 1: 1D input, default axis -1
        concatenator = AutoConcatenator(
            input_copy_mask=np.array([True, False, True]),
            output_original_mask=np.array([True, True, True, False, False]),
            output_copy_mask=np.array([False, False, False, True, True]))
        X = np.array([1, 2, 3])
        Y = concatenator(X)
        Y_ref = np.array([1, 2, 3, 1, 3])
        np.testing.assert_allclose(Y, Y_ref, atol=self.atol)

        # test 2: 2D input, declared axis 0
        concatenator = AutoConcatenator(
            input_copy_mask=np.array([True, False]),
            output_original_mask=np.array([True, True, False, False, False]),
            output_copy_mask=np.array([False, False, False, True, False]),
            axis=0)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = concatenator(X)
        Y_ref = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [1, 2, 3], [0, 0, 0]])
        np.testing.assert_allclose(Y, Y_ref, atol=self.atol)

        # test 3: 2D input, default axis -1 (block content axis)
        concatenator = AutoConcatenator(
            input_copy_mask=np.array([False, True, True]),
            output_original_mask=np.array([False, True, True, True, False]),
            output_copy_mask=np.array([True, False, False, False, True]))
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = concatenator(X)
        Y_ref = np.array([[2, 1, 2, 3, 3], [5, 4, 5, 6, 6]])
        np.testing.assert_allclose(Y, Y_ref, atol=self.atol)

    def test_extractor(self):
        x = np.arange(10)

        extractor1 = DataExtractor(3)
        y = extractor1(x)
        y_ref = np.array([3])
        np.testing.assert_allclose(y, y_ref, atol=self.atol)

        extractor2 = DataExtractor((2, 8))
        y = extractor2(x)
        y_ref = np.array([2, 3, 4, 5, 6, 7])
        np.testing.assert_allclose(y, y_ref, atol=self.atol)

        extractor3 = DataExtractor((1, 9, 2))
        y = extractor3(x)
        y_ref = np.array([1, 3, 5, 7])
        np.testing.assert_allclose(y, y_ref, atol=self.atol)

        X = np.arange(20).reshape(4, 5)
        extractor4 = DataExtractor((1, 3))
        Y = extractor4(X)
        Y_ref = np.array([[ 5,  6,  7,  8,  9], [10, 11, 12, 13, 14]])
        np.testing.assert_allclose(Y, Y_ref, atol=self.atol)


class TestDownsamplerFilter(unittest.TestCase):
    """Downsampler(use_filter=True) used to raise AttributeError: the
    `filter` field was never declared (and slots=True made it fatal)."""

    L = 4

    def _signals(self):
        # two tones on the DFT grid: one in band, one above the cutoff.
        # BWFilter(1/L) cuts at 1/L of Nyquist, i.e. 0.125 cycle/sample here.
        # decimating by 4 folds bin 30 exactly onto bin 2, i.e. on top of the
        # wanted tone -- no downstream filter can separate them afterwards.
        n = np.arange(128)
        clean = np.cos(2 * np.pi * 2 * n / 128)              # f = 0.016
        interferer = 0.5 * np.cos(2 * np.pi * 30 * n / 128)  # f = 0.234
        return clean, clean + interferer

    def test_use_filter_runs(self):
        # regression: this used to raise AttributeError
        clean, x = self._signals()
        y = Downsampler(self.L, use_filter=True)(x)
        self.assertEqual(y.shape, clean[::self.L].shape)

    def test_filter_field_is_declared(self):
        down = Downsampler(self.L)
        self.assertIsInstance(down.filter, BWFilter)
        self.assertAlmostEqual(down.filter.wn, 1 / self.L)

    def test_out_of_band_tone_is_rejected(self):
        clean, x = self._signals()
        y = Downsampler(self.L, use_filter=True)(x)
        # the decimated signal is the decimation of the clean tone
        np.testing.assert_allclose(y, clean[::self.L], atol=1e-10)

    def test_without_filter_the_tone_aliases(self):
        clean, x = self._signals()
        y = Downsampler(self.L)(x)
        # 0.5 * cos(2*pi*30/128 * n) folds straight onto the wanted tone
        self.assertAlmostEqual(float(np.max(np.abs(y - clean[::self.L]))), 0.5, places=10)

    def test_filter_is_applied_before_decimation(self):
        """Anti-aliasing only works upstream: filtering the already
        decimated signal cannot undo the folding."""
        clean, x = self._signals()
        y = Downsampler(self.L, use_filter=True)(x)

        filter_then_decimate = BWFilter(1 / self.L)(x)[::self.L]
        decimate_then_filter = BWFilter(1 / self.L)(x[::self.L])

        np.testing.assert_allclose(y, filter_then_decimate, atol=1e-12)
        # the wrong order leaves the aliased component in place: once folded
        # onto the wanted tone it is indistinguishable from it
        self.assertAlmostEqual(
            float(np.max(np.abs(decimate_then_filter - clean[::self.L]))), 0.5, places=10)


class TestAmplifier(unittest.TestCase):
    """Amplifier is element-wise: y[n] = g x[n], no axis parameter."""

    def test_scalar_gain_scales_every_entry(self):
        X = np.array([[1, 2], [3, 4]])
        Y = Amplifier(gain=3)(X)
        # the removed `axis` branch returned [[1, 6], [3, 12]] for axis=-1
        np.testing.assert_allclose(Y, np.array([[3, 6], [9, 12]]))

    def test_axis_parameter_is_gone(self):
        with self.assertRaises(TypeError):
            Amplifier(gain=3, axis=-1)

    def test_shape_agnostic(self):
        for shape in [(5,), (2, 3), (2, 3, 4)]:
            X = np.arange(int(np.prod(shape))).reshape(shape) + 1.0
            np.testing.assert_allclose(Amplifier(gain=0.5)(X), 0.5 * X)


class TestSerialOnlyBlocks(unittest.TestCase):
    """SampleRemover and DataAdder are 1D only and must say so (D38)."""

    def test_sample_remover_1d_still_works(self):
        np.testing.assert_allclose(SampleRemover(N_start=2, length=3)(np.arange(8)),
                                   np.array([0, 1, 5, 6, 7]))

    def test_sample_remover_rejects_2d(self):
        with self.assertRaises(ShapeError) as ctx:
            SampleRemover(N_start=1, length=1)(np.arange(6).reshape(2, 3))
        message = str(ctx.exception)
        self.assertIn("(2, 3)", message)      # observed
        self.assertIn("1D Serial signal", message)  # expected
        self.assertIn("Parallel2Serial", message)   # action

    def test_data_adder_1d_still_works(self):
        np.testing.assert_allclose(DataAdder(np.array([-1, -1]), N_start=2)(np.arange(5)),
                                   np.array([0, 1, -1, -1, 2, 3, 4]))

    def test_data_adder_rejects_2d_input(self):
        with self.assertRaises(ShapeError) as ctx:
            DataAdder(np.array([-1, -1]))(np.arange(6).reshape(2, 3))
        message = str(ctx.exception)
        self.assertIn("(2, 3)", message)
        self.assertIn("1D Serial signal", message)
        self.assertIn("Parallel2Serial", message)

    def test_data_adder_rejects_2d_symbol(self):
        with self.assertRaises(ShapeError) as ctx:
            DataAdder(np.zeros((2, 2)))(np.arange(5))
        self.assertIn("1D symbol sequence", str(ctx.exception))

    def test_remover_inverts_adder(self):
        x = np.arange(7)
        symbol = np.array([-1, -2, -3])
        y = DataAdder(symbol, N_start=2)(x)
        np.testing.assert_allclose(SampleRemover(N_start=2, length=len(symbol))(y), x)


class TestBlindPhaseTracker(unittest.TestCase):
    """The block must not draw (D25, D42); it exposes theta_ instead (D23)."""

    def _tracker(self):
        alphabet = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        rng = np.random.default_rng(42)
        s = alphabet[rng.integers(0, 4, size=12)]
        return BlindPhaseTracker(L=3, alphabet=alphabet, phase_steps=16), s

    def test_plot_parameter_is_gone(self):
        alphabet = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        with self.assertRaises(TypeError):
            BlindPhaseTracker(L=3, alphabet=alphabet, plot=True)

    def test_forward_does_not_draw(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        tracker, s = self._tracker()
        tracker(s * np.exp(1j * 0.3))
        self.assertEqual(plt.get_fignums(), [])

    def test_theta_is_exposed_and_correct(self):
        tracker, s = self._tracker()
        x = s * np.exp(1j * 0.3)
        y = tracker(x)
        self.assertEqual(tracker.theta_.shape, (len(x),))
        # the estimated phase tracks the applied 0.3 rad within one search step
        step = np.pi / 2 / 16
        np.testing.assert_allclose(tracker.theta_, 0.3, atol=step)
        np.testing.assert_allclose(y, s, atol=1e-2)

    def test_theta_is_none_before_any_call(self):
        tracker, _ = self._tracker()
        self.assertIsNone(tracker.theta_)


if __name__ == '__main__':
    unittest.main()
