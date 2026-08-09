"""Internal array-backend dispatch (decision D3).

comnumpy has no GPU dependency. This module regroups the FFT calls of
the signal path behind thin dispatchers that route to the library
owning the input array: numpy arrays go to ``scipy.fft`` (exactly as
before), CuPy arrays go to ``cupyx.scipy.fft`` (which reimplements the
scipy API). CuPy is imported only when a CuPy array is actually seen,
so it never becomes a dependency.

Deliberately *not* routed through here: metadata paths that are always
CPU-side numpy (e.g. the ``CarrierAllocation`` mask reordering), and
the visualizers (matplotlib needs host arrays anyway).

This module is internal; its API may change without notice.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Optional

import numpy as np

__all__ = ["get_array_module", "get_fft_module",
           "fft", "ifft", "fftshift", "ifftshift", "fftfreq"]


def get_array_module(*arrays: Any) -> ModuleType:
    """Return numpy, or cupy if any argument is a CuPy array.

    Detection is by the module of the array's type, so cupy is never
    imported for pure-numpy workloads.
    """
    for x in arrays:
        if type(x).__module__.partition(".")[0] == "cupy":
            import cupy  # pyright: ignore[reportMissingImports] -- not a dependency (D1/D3)
            return cupy
    return np


def get_fft_module(*arrays: Any) -> ModuleType:
    """FFT namespace for the library owning ``arrays`` (scipy.fft API)."""
    if get_array_module(*arrays) is np:
        from scipy import fft as fft_module  # local import (D36)
        return fft_module
    import cupyx.scipy.fft as fft_module  # pyright: ignore[reportMissingImports] -- scipy.fft reimplementation
    return fft_module


def fft(x: np.ndarray, n: Optional[int] = None, axis: int = -1,
        norm: Optional[str] = None) -> np.ndarray:
    return get_fft_module(x).fft(x, n=n, axis=axis, norm=norm)


def ifft(x: np.ndarray, n: Optional[int] = None, axis: int = -1,
         norm: Optional[str] = None) -> np.ndarray:
    return get_fft_module(x).ifft(x, n=n, axis=axis, norm=norm)


def fftshift(x: np.ndarray, axes: Optional[int] = None) -> np.ndarray:
    return get_fft_module(x).fftshift(x, axes=axes)


def ifftshift(x: np.ndarray, axes: Optional[int] = None) -> np.ndarray:
    return get_fft_module(x).ifftshift(x, axes=axes)


def fftfreq(n: int, d: float = 1.0, *, like: Any = None) -> np.ndarray:
    """Sample frequencies; pass ``like=x`` to get them on x's device."""
    return get_fft_module(like).fftfreq(n, d=d)
