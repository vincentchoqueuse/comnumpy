import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from scipy import signal
from comnumpy.core.generics import Processor
from .utils import compute_sigma2


@dataclass
class AWGN(Processor):
    r"""
    A class representing an Additive White Gaussian Noise (AWGN) channel.

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
        The value associated with the given method. Defaut is 1
    unit : str, optional
        The unit to compute the noise power ("nat", "dB", "dBm"). Default is "var_nat"
    sigma2s : float, optional
        Signal power. default is 1
    seed : int, optional
        The seed for the noise generator.
    estimate_sigma2s : bool
        Whether to estimate the signal power.
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
                sigma2s = np.sum(np.abs(x)**2) / np.prod(X.shape)
            case "fixed":
                sigma2s = self.sigma2s
            case _:
                raise ValueError(f"Unknown sigma2s_method='{self.sigma2s_method}'. Expected one of: 'fixed', 'measured'.")

        return sigma2s

    def noise_rvs(self, x):
        is_complex = np.iscomplexobj(x)

        # compute sigma2s
        sigma2s = self.get_sigma2s(x)
        sigma2n = compute_sigma2(self.value, self.unit, sigma2s)
        shape = x.shape
        if is_complex:
            scale = np.sqrt(sigma2n / 2)
            b_r = self.rng.normal(scale=scale, size=shape)
            b_i = self.rng.normal(scale=scale, size=shape)
            b = b_r + 1j * b_i
        else:
            scale = np.sqrt(sigma2n)
            b = self.rng.normal(scale=scale, size=shape)

        self._b = b
        self.sigma2 = sigma2n

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.noise_rvs(x)
        y = x + self._b
        return y


@dataclass
class FIRChannel(Processor):
    r"""
    Finite Impulse Response (FIR) channel with given impulse response.

    Signal Model
    ------------

    The output signal :math:`y[n]` is computed as the convolution of the input signal :math:`x[n]` with the impulse response :math:`h[l]`:

    .. math::

       y[n] = \sum_{l=0}^{L-1} h[l] x[n-l]

    where:

    - :math:`h[l]` is the impulse response of the channel.
    - :math:`L` is the length of the impulse response.

    Attributes
    ----------
    h : np.ndarray
        The impulse response of the FIR channel. This should be a 1-dimensional numpy array.
    name : str, optional
        The name of the channel instance (default is "fir").
    """
    h: np.array
    mode: Literal["full", "same", "valid"] = "full"
    is_mimo: bool = False
    name: str = "fir"

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = signal.convolve(x, self.h, mode=self.mode)
        return y
