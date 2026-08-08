import numpy as np
from dataclasses import dataclass
from typing import Optional
from comnumpy.core import Processor
from comnumpy.core.channels import AWGN  # noqa: F401 -- AWGN is element-wise (shape-agnostic); re-exported here for convenience
from .validators import validate_input


@dataclass
class BaseMIMOChannel(Processor):

    r"""
    A base class for modeling Multiple-Input Multiple-Output (MIMO) channels.

    This class provides a framework for simulating MIMO communication channels,
    including methods for setting channel matrices, configuring signal-to-noise
    ratios (SNR), generating noise, and processing input signals.

    Signal Model
    ------------

    .. math ::

        \mathbf{y}[n] = \sum_{l=0}^{L}\mathbf{H}[l]x[n-l] + \mathbf{b}[n]

    where

    * :math:`\mathbf{H}[l]` is a channel matrix of size :math:`N_r \times N_t` corresponding the :math:`l^{th}` channel tap,
    * :math:`\mathbf{x}[n]` is a :math:`N_t` vector containing the transmitted data,
    * :math:`\mathbf{b}[n]\sim \mathcal{N}_c(\mathbf{0},\sigma^2\mathbf{I}_{N_r})` is a :math:`N_r` vector containing the additive white Gaussian noise.

    Attributes
    ----------
    P : float
        Transmit power.
    H : Optional[np.array]
        List of channel matrices for each tap. Each matrix should have equal dimensions.
    extend : bool
        Flag to extend the input signal.
    name : str
        Name of the processor.
    """
    H: Optional[np.array] = None
    extend: bool = True
    name: str = "mimo_channel"

    def info(self):
        H = self.H
        if H.ndim == 2:
            H = H[None, :, :]

        L, N_r, N_t = H.shape

        print(f"* MIMO Channel ({L} tap(s)):")
        for index in range(L):
            H = self.H[index]
            print(f"tap {index}:\n{H}")
            condition_number = np.linalg.cond(H)
            _, S, _ = np.linalg.svd(H)
            norm = np.linalg.norm(H)
            print("Condition Number=", condition_number)
            print(f"singular value={S}")
            print(f"norm={norm}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class FlatMIMOChannel(BaseMIMOChannel):
    r"""
    Flat (frequency-non-selective) MIMO channel.

    Applies a single matrix multiplication :math:`\mathbf{y}[n] = \mathbf{H}\mathbf{x}[n]`
    without frequency selectivity.
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        validate_input(X, self.H.shape[1])
        return np.matmul(self.H, X)


@dataclass
class SelectiveMIMOChannel(BaseMIMOChannel):
    r"""
    Frequency-selective MIMO channel.

    Applies a multi-tap convolution using the channel matrices stored in ``H``.
    The ``extend`` flag controls whether the output signal is extended or truncated.
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


