import logging

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from comnumpy.core import Processor
from comnumpy.core.channels import AWGN  # noqa: F401 -- AWGN is element-wise (shape-agnostic); re-exported here for convenience
from .validators import validate_input

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BaseMIMOChannel(Processor):

    r"""Base class for Multiple-Input Multiple-Output (MIMO) channels.

    Signal Model
    ------------
    Subclasses implement the deterministic part of the MIMO observation
    model

    .. math::

        \mathbf{y}[n] = \sum_{l=0}^{L-1} \mathbf{H}[l] \, \mathbf{x}[n-l]
        + \mathbf{b}[n]

    where

    * :math:`\mathbf{H}[l]` is the channel matrix of size
      :math:`N_r \times N_t` of the :math:`l`-th tap,
    * :math:`\mathbf{x}[n]` is the :math:`N_t \times 1` transmitted vector,
    * :math:`\mathbf{b}[n]` is the additive noise, contributed by chaining
      a separate element-wise :class:`~comnumpy.core.channels.AWGN` block
      (this class applies only the noiseless convolution).

    Axes: *declared axis* -- expects ``(..., ant, N)`` with antennas on
    axis -2.

    Parameters
    ----------
    H : np.ndarray, optional
        Channel matrix :math:`\mathbf{H}` of shape ``(N_r, N_t)`` (flat
        channel) or stacked taps ``(L, N_r, N_t)`` (selective channel).
    extend : bool, keyword-only
        If True, a selective channel outputs the full convolution
        (``N + L - 1`` samples); if False, the output is truncated to
        ``N`` samples. Default is True.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"mimo_channel"``.

    References
    ----------
    D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
    Cambridge University Press, 2005, Chapter 7.
    """
    H: Optional[np.array] = None
    extend: bool = field(default=True, kw_only=True)
    name: str = field(default="mimo_channel", kw_only=True)

    def info(self):
        """Describe the channel (taps, conditioning); logged and returned."""
        H = self.H
        if H.ndim == 2:
            H = H[None, :, :]

        L, _, _ = H.shape
        lines = [f"* MIMO Channel ({L} tap(s)):"]
        for index in range(L):
            H_tap = H[index]
            _, S, _ = np.linalg.svd(H_tap)
            lines.append(f"tap {index}:\n{H_tap}")
            lines.append(f"Condition Number={np.linalg.cond(H_tap)}")
            lines.append(f"singular value={S}")
            lines.append(f"norm={np.linalg.norm(H_tap)}")
        description = "\n".join(lines)
        logger.info("%s", description)
        return description

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass(slots=True)
class FlatMIMOChannel(BaseMIMOChannel):
    r"""Flat (frequency-non-selective) MIMO channel.

    Signal Model
    ------------
    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n]

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t`, :math:`\mathbf{x}[n]` the
    :math:`N_t \times 1` transmitted vector and :math:`\mathbf{b}[n]`
    the additive noise, contributed by chaining a separate element-wise
    :class:`~comnumpy.core.channels.AWGN` block (this class applies only
    the noiseless product).

    Axes: *declared axis* -- expects ``(..., ant, N)`` with antennas on
    axis -2, validated against :math:`N_t`.

    Parameters
    ----------
    H : np.ndarray, optional
        Channel matrix :math:`\mathbf{H}` of shape ``(N_r, N_t)``.
    extend : bool, keyword-only
        Unused for a flat channel (inherited). Default is True.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"mimo_channel"``.

    Raises
    ------
    ShapeError
        If the antenna axis of the input does not match :math:`N_t`.

    References
    ----------
    D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
    Cambridge University Press, 2005, Chapter 7.

    Examples
    --------
    >>> H = np.array([[1.0, 0.0], [0.0, 2.0]])
    >>> x = np.array([[1.0, 1.0], [1.0, -1.0]])
    >>> FlatMIMOChannel(H)(x)
    array([[ 1.,  1.],
           [ 2., -2.]])
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        validate_input(X, self.H.shape[1])
        return np.matmul(self.H, X)


@dataclass(slots=True)
class SelectiveMIMOChannel(BaseMIMOChannel):
    r"""Frequency-selective (multi-tap) MIMO channel.

    Signal Model
    ------------
    .. math::

        \mathbf{y}[n] = \sum_{l=0}^{L-1} \mathbf{H}[l] \, \mathbf{x}[n-l]
        + \mathbf{b}[n]

    where :math:`\mathbf{H}[l]` is the channel matrix of size
    :math:`N_r \times N_t` of the :math:`l`-th of :math:`L` taps,
    :math:`\mathbf{x}[n]` the :math:`N_t \times 1` transmitted vector and
    :math:`\mathbf{b}[n]` the additive noise, contributed by chaining a
    separate element-wise :class:`~comnumpy.core.channels.AWGN` block
    (this class applies only the noiseless convolution).

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2, validated against :math:`N_t`.

    Parameters
    ----------
    H : np.ndarray, optional
        Stacked channel matrices :math:`\mathbf{H}[l]` of shape
        ``(L, N_r, N_t)``.
    extend : bool, keyword-only
        If True, the output holds the full convolution (``N + L - 1``
        samples); if False, it is truncated to ``N`` samples. Default is
        True.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"mimo_channel"``.

    Raises
    ------
    ShapeError
        If the antenna axis of the input does not match :math:`N_t`.

    References
    ----------
    D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
    Cambridge University Press, 2005, Chapter 7.

    Examples
    --------
    >>> H = np.array([[[1.0]], [[0.5]]])
    >>> x = np.array([[1.0, 0.0, 0.0]])
    >>> SelectiveMIMOChannel(H)(x)
    array([[1. +0.j, 0.5+0.j, 0. +0.j, 0. +0.j]])
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        validate_input(X, self.H.shape[1])
        L, N_r, N_t = self.H.shape
        N = X.shape[1]

        # create X_matrix
        if self.extend:
            X_matrix = np.zeros((L*N_t, N + L-1), dtype=complex)
            for l_index in range(L):
                X_matrix[l_index*N_t:(l_index+1)*N_t, l_index:l_index+N] = X
        else:
            X_matrix = np.zeros((L*N_t, N), dtype=complex)
            for l_index in range(L):
                X_matrix[l_index*N_t:(l_index+1)*N_t, l_index:] = X[:, :N-l_index]

        # create H matrix
        H_matrix = np.zeros((N_r, L * N_t), dtype=complex)
        for indice in range(L):
            H_matrix[:, indice * N_t:(indice + 1) * N_t] = self.H[indice]

        return np.matmul(H_matrix, X_matrix)


