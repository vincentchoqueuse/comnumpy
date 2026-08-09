"""Tests for the internal array-backend dispatch (decision D3)."""
import sys
import types
import unittest

import numpy as np
from scipy import fft as scipy_fft

from comnumpy._backend import (fft, fftfreq, fftshift, get_array_module,
                               get_fft_module, ifft, ifftshift)


class TestNumpyPath(unittest.TestCase):
    """numpy arrays must route to scipy.fft, bit-exact."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.normal(size=(3, 16)) + 1j * rng.normal(size=(3, 16))

    def test_get_array_module_is_numpy(self):
        self.assertIs(get_array_module(self.x), np)
        self.assertIs(get_array_module(self.x, 3.0, None), np)

    def test_get_fft_module_is_scipy(self):
        self.assertIs(get_fft_module(self.x), scipy_fft)
        self.assertIs(get_fft_module(), scipy_fft)

    def test_wrappers_match_scipy(self):
        np.testing.assert_array_equal(fft(self.x, axis=-1),
                                      scipy_fft.fft(self.x, axis=-1))
        np.testing.assert_array_equal(ifft(self.x, n=32, norm="ortho"),
                                      scipy_fft.ifft(self.x, n=32, norm="ortho"))
        np.testing.assert_array_equal(fftshift(self.x, axes=-1),
                                      scipy_fft.fftshift(self.x, axes=-1))
        np.testing.assert_array_equal(ifftshift(self.x, axes=-1),
                                      scipy_fft.ifftshift(self.x, axes=-1))
        np.testing.assert_array_equal(fftfreq(16, d=0.5),
                                      scipy_fft.fftfreq(16, d=0.5))
        np.testing.assert_array_equal(fftfreq(16, d=0.5, like=self.x),
                                      scipy_fft.fftfreq(16, d=0.5))


class TestCupyDispatch(unittest.TestCase):
    """A fake cupy in sys.modules proves the dispatch without a GPU."""

    def setUp(self):
        self.fake_cupy = types.ModuleType("cupy")
        self.fake_fft = types.ModuleType("cupyx.scipy.fft")
        fake_cupyx = types.ModuleType("cupyx")
        fake_scipy = types.ModuleType("cupyx.scipy")
        self._saved = {name: sys.modules.get(name)
                       for name in ("cupy", "cupyx", "cupyx.scipy",
                                    "cupyx.scipy.fft")}
        sys.modules["cupy"] = self.fake_cupy
        sys.modules["cupyx"] = fake_cupyx
        sys.modules["cupyx.scipy"] = fake_scipy
        sys.modules["cupyx.scipy.fft"] = self.fake_fft

        class FakeCupyArray:
            pass
        FakeCupyArray.__module__ = "cupy"
        self.gpu_array = FakeCupyArray()

    def tearDown(self):
        for name, mod in self._saved.items():
            if mod is None:
                del sys.modules[name]
            else:
                sys.modules[name] = mod

    def test_cupy_array_selects_cupy_modules(self):
        self.assertIs(get_array_module(self.gpu_array), self.fake_cupy)
        self.assertIs(get_array_module(np.zeros(2), self.gpu_array),
                      self.fake_cupy)
        self.assertIs(get_fft_module(self.gpu_array), self.fake_fft)

    def test_numpy_arrays_never_touch_cupy(self):
        self.assertIs(get_array_module(np.zeros(2)), np)
        self.assertIs(get_fft_module(np.zeros(2)), scipy_fft)


if __name__ == "__main__":
    unittest.main()
