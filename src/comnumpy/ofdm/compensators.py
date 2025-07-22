import numpy as np
from typing import Literal
from dataclasses import dataclass, field
from scipy.fft import fft, fftshift
from comnumpy.core import Processor, Sequential
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial, WeightAmplifier
from .processors import CarrierAllocator, IFFTProcessor, CyclicPrefixer
from .utils import get_standard_carrier_allocation


@dataclass
class FrequencyDomainEqualizer(WeightAmplifier):
    r"""
    A frequency domain equalizer that applies weights to compensate for channel effects in the frequency domain.

    This class extends the `WeightAmplifier` to operate in the frequency domain, using the Fast Fourier Transform (FFT) to compute weights that equalize the input signal. The equalizer can optionally shift the zero-frequency component to the center of the spectrum.

    Signal Model
    ------------

    Given a channel impulse response :math:`h[l]`, the Frequency Domain Equalizer applies the following weight amplifier

    .. math::

        y[n] = \left(\frac{1}{H[n]}\right) \cdot x[n]

    * :math:`H[n]` corresponds to the inverse of the :math:`n`-th bin of the channel's Discrete Fourier Transform (DFT).

    Attributes
    ----------
    h : np.ndarray, optional
        The impulse response of the channel to be equalized. Default is None.
    shift : bool, optional
        If True, applies a frequency shift to center the zero-frequency component. Default is True.
    axis : int, optional
        The axis along which to compute the FFT and apply the weights. Default is -2.
    name : str
        Name of the frequency domain equalizer instance.

    Example
    -------
    >>> equalizer = FrequencyDomainEqualizer(h=np.array([1, 0.5, 0.2]))
    >>> X = np.random.randn(4, 3, 2)  # Example input tensor
    >>> Y = equalizer(X)
    """
    h : np.ndarray = None
    axis: int = 0
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = "ortho"
    weight: np.ndarray = field(init=False, default=None)
    name : str = "frequency domain equalizer"

    def __post_init__(self):
        if self.h is None:
            raise ValueError("The impulse response 'h' must be provided.")

    def prepare(self, X):
        """
        Compute the amplifier weight from the channel impulse response
        """
        N_sc = X.shape[self.axis]
        Hw = fft(self.h, n=N_sc,  axis=self.axis)
        weight = 1./Hw 
        if self.shift:
            weight = fftshift(weight)
        self.weight = weight

