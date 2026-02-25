import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from comnumpy.core import Processor
from comnumpy.core.utils import compute_sigma2
from .validators import validate_input


@dataclass
class AWGN(Processor):
    r"""
    A class representing an Additive White Gaussian Noise (AWGN) for MIMO channel.

    This class models an AWGN channel, which adds complex Gaussian noise to a signal.
    It is characterized by a noise power specified by sigma squared (sigma2).

    Signal Model
    ------------

    .. math::

       y_u[n] = x_u[n] + b_u[n]

    where:

    * :math:`b[n]\sim \mathcal{N}(0, \sigma^2)` is a Gaussian additive noise.

    For complex signals, a circular Gaussian noise is applied to the signal.
    The value of :math:`\sigma^2` is computed with respect to the method specified as input.

    Attributes
    ----------
    value : float, optional
        The value associated with the given method. Default is 1
    unit : str, optional
        The unit to compute the noise power ("sigma2", "snr", "snr_dB", "snr_dBm"). Default is "sigma2"
    sigma2s : float, optional
        Signal power. default is 1
    seed : int, optional
        The seed for the noise generator.
    sigma2s_method: Literal["fixed", "measured"]
        The method used to obtain the signal power
    name : str
        Name of the channel instance.
    """
    value: float = 1.
    unit: Literal["sigma2", "snr", "snr_dB", "snr_dBm"] = "sigma2"
    sigma2s: float = 1.
    sigma2s_method: Literal["fixed", "measured"] = "fixed"
    seed: int = None
    name: str = 'awgn'

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def get_sigma2s(self, X):
        # extract signal power
        match self.sigma2s_method:
            case "measured": 
                sigma2s = np.sum(np.abs(X)**2) / np.prod(X.shape)
            case "fixed":
                sigma2s = self.sigma2s
            case _:
                raise ValueError(f"Unknown sigma2s_method='{self.sigma2s_method}'. Expected one of: 'fixed', 'measured'.")

        return sigma2s

    def noise_rvs(self, X):
        is_complex = np.iscomplexobj(X)

        # compute signal variance
        sigma2s = self.get_sigma2s(X)
        
        # compute noise variance
        sigma2n = compute_sigma2(self.value, self.unit, sigma2s)

        # apply noise
        shape = X.shape
        if is_complex:
            scale = np.sqrt(sigma2n/2)
            B_r = self.rng.normal(scale=scale, size=shape)
            B_i = self.rng.normal(scale=scale, size=shape)
            B = B_r + 1j * B_i
        else:
            scale = np.sqrt(sigma2n)
            B = self.rng.normal(scale=scale, size=shape)

        # save values
        self.sigma2n = sigma2n
        self._B = B

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.noise_rvs(X)
        Y = X + self._B
        return Y

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
        if H.ndims == 2:
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


