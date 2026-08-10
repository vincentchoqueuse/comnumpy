"""Plotting functions for frequency-domain OFDM signals.

Like the core visualizers, these are plain functions operating on
arrays extracted with ``Sequential(taps=...)`` -- never in-chain
blocks. Every function takes ``ax=None`` and returns the axis
(decision D25).
"""
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

__all__ = ["plot_subcarrier_amplitude"]


def plot_subcarrier_amplitude(X: np.ndarray, *,
                              reduction: Optional[Literal["mean"]] = "mean",
                              title: str = "Subcarrier amplitude",
                              ax: Optional[Axes] = None) -> Axes:
    """Stem plot of the per-subcarrier amplitude of a Block-layout signal.

    Useful before an IFFT or after an FFT block in an OFDM system.

    Parameters
    ----------
    X : np.ndarray
        Frequency-domain signal in Block layout ``(..., T, F)``.
    reduction : {"mean", None}, keyword-only
        ``"mean"`` (default) averages the amplitude over the OFDM-symbol
        axis ``T``; ``None`` superimposes one stem series per symbol.
    title : str, keyword-only
        Axis title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes
    """
    amplitudes = np.abs(np.asarray(X))
    if amplitudes.ndim < 2:
        raise ValueError(
            f"expected a Block layout (..., T, F), got {amplitudes.shape}")
    N_sc = amplitudes.shape[-1]
    subcarrier_indices = np.arange(-N_sc // 2, N_sc // 2)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    if reduction == "mean":
        ax.stem(subcarrier_indices, amplitudes.mean(axis=-2))
    elif reduction is None:
        for row in amplitudes.reshape(-1, N_sc):
            ax.stem(subcarrier_indices, row)
    else:
        raise ValueError("Invalid reduction option. Choose None or 'mean'.")
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    return ax
