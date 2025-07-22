import numpy as np
from dataclasses import dataclass
from comnumpy.core import Processor


@dataclass
class IQImbalance(Processor):
    r"""
    Apply IQ imbalance impairments.
    
    Signal Model
    ------------

    .. math::
        y[n] = \alpha x[n] + \beta x^*[n]

    where :
    
    * :math:`(\alpha,\beta) \in \mathbb{C}^2` corresponds to the complex weights for the signal and its complex conjugate.

    Attributes
    ----------
    alpha : complex number
        A complex number specifying the :math:`\alpha` parameter
    beta : complex number
        A complex number specifying the :math:`\beta` parameter
        
    """
    alpha: complex
    beta: complex
    name: str = "iq_impairment"

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.alpha*x + self.beta*np.conj(x)
        return y


@dataclass
class CFO(Processor):
    r"""
    Apply Carrier Frequency Offset (CFO).

    Signal Model
    ------------

    .. math::
        y[n] = x[n] e^{j\omega_0 n}

    where :
    
    * :math:`\omega_0` corresponds to the normalized carrier frequency offset (in rad/samples).

    Attributes
    ----------
    cfo : float
        the normalized carrier frequency offset

    """
    cfo: float
    name: str = "cfo_impairment"

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        n_vect = np.arange(N)
        y = x * np.exp(1j*self.cfo*n_vect)
        return y


@dataclass
class Delay(Processor):
    r"""
    Introduces a delay to the input signal.

    Signal Model
    ------------

    .. math::
        y[n] = x[n - \tau]

    where :
    
    * :math:`\tau \in \mathbb{N}^+` is the number of samples to delay the input signal (non-negative integer)

    Attributes
    ----------
    tau : int
        The number of samples to delay the input signal.
    pad_zeros : bool
        If True, pads the delayed signal with zeros to match the input size.
    """
    tau: int
    pad_zeros: bool = True
    name: str = "delay_impairment"

    def __post_init__(self):
        if self.tau < 0:
            raise ValueError("Delay must be a non-negative integer.")

    def forward(self, x: np.ndarray) -> np.ndarray:

        x_delayed = x[self.tau:]
        if self.pad_zeros:
            y = np.zeros(len(x))
            y[:len(x_delayed)] = x_delayed
        else:
            y = x_delayed

        return y
