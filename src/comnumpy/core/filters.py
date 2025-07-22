import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy import signal
from comnumpy.core.generics import Processor


@dataclass
class SRRCFilter(Processor):
    """
    Implement a Square-Root Raised Cosine (SRRC) FIR filter.

    This filter corresponds to a square-root raised cosine FIR filter.
    It has a rolloff factor specified by beta.
    The filter is truncated to span symbols, and each symbol period contains sps samples. The order of the filter, sps*span, must be even. 
    
    The filter energy is normalized to 1.

    Attributes
    ----------
    rho : float
        Roll-off factor for the filter.
    N_h : int
        Length of the filter before oversampling.
    oversampling : int
        Oversampling factor for the filter.
    scale : float
        Amplitude scaling factor for the filter.
    method : Literal["auto", "time", "fft"]
        Method for filter design ("auto", "time", "fft").

    """

    rho: float
    oversampling: int
    N_h: int = 10
    norm: bool = True
    scale: float = 1.0
    method: Literal['lfilter', 'time', 'fft'] = "lfilter"
    is_mimo: bool = True
    axis: int = -1
    name: str = "SRRCFilter"

    def h(self, t=None):

        if t is None:
            N = self.N_h*self.oversampling
            n_vect = np.arange(-N, N+1)
            t = n_vect / self.oversampling

        rho = self.rho
        h = np.zeros(len(t))

        for index, t_temp in enumerate(t):
            if t_temp == 0:
                h_temp = (1 + rho*((4/np.pi)-1))
            else:
                if np.abs(t_temp) == (1/(4*rho)):
                    term1 = (1 + (2/np.pi))
                    term2 = (1 - (2/np.pi))
                    coef = rho / np.sqrt(2)
                    coef2 = np.pi / (4*rho)
                    h_temp = coef*(term1*np.sin(coef2) + term2*np.cos(coef2))
                else:
                    term1 = np.pi * t_temp
                    coef1 = (4 * rho * t_temp)
                    num = np.sin(term1*(1-rho)) + coef1*np.cos(term1*(1+rho))
                    den = term1*(1 - coef1**2)
                    h_temp = num/den

            h[index] = h_temp

        if self.norm:
            h = h / np.sqrt(np.sum(h**2))

        return h

    def H(self, NFFT):
        """Frequency response for fft method"""
        # see hager code on LDBP
        h = self.h()
        filter_delay = self.oversampling*self.N_h
        H_tmp = np.concatenate((h, np.zeros(NFFT-len(h))))
        H_tmp = np.roll(H_tmp, -filter_delay)
        H = np.fft.fft(H_tmp, n=NFFT)
        return H

    def get_delay(self):
        return self.N_h

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.method == "lfilter":
            h = self.h()
            y = signal.lfilter(h, 1, x, axis=-1)

        if self.method == "fft":
            NFFT = len(x)
            fft_x = np.fft.fft(x, NFFT)
            fft_h = self.H(NFFT)
            y = self.scale*np.fft.ifft(fft_x*fft_h, NFFT)

        return y


@dataclass
class BWFilter(Processor):
    r"""
    Implements a frequency domain low-pass brick wall filter.

    This filter attenuates frequencies above a specified critical frequency.
    The critical frequency is normalized from 0 to 1.

    Signal Model
    ------------

    The filter operates in the frequency domain by applying a brick wall (ideal) low-pass filter to the input signal.
    The filter's transfer function :math:`H` is defined as:

    .. math::

       H[k] =
       \begin{cases}
       1 & \text{if } |w[k]| \leq w_n \\
       0 & \text{if } |w[k]| > w_n
       \end{cases}

    where:
    
    -  :math:`w[k]` is the angular frequency of the k-th component.
    -  :math:`w_n` is the critical angular frequency.

    The filtering process involves:

    1. Computing the Fast Fourier Transform (FFT) of the input signal.
    2. Applying the brick wall filter in the frequency domain.
    3. Performing the inverse FFT to obtain the filtered time-domain signal.


    Attributes
    ----------
    wn : float
        The critical frequency or frequencies. Values should be normalized from 0 to 1,
        where 1 is the Nyquist frequency. This determines the cutoff point for the filter.
    """
    wn: float
    is_mimo: bool = False

    def forward(self, x: np.ndarray) -> np.ndarray:

        if x.ndim > 1:
            raise NotImplementedError("BW Filter: only 1D signals are supported.")

        NFFT = len(x)
        w = np.fft.fftfreq(NFFT, d=1)
        H = (abs(w) <= self.wn).astype(float)
        fft_x = np.fft.fft(x, NFFT)
        y = np.fft.ifft(H*fft_x, NFFT)
        return y
