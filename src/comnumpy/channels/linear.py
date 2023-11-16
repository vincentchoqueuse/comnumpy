from numpy.random import randn
from scipy import signal
import numpy as np
from .core import Channel

class FIR_Channel(Channel):

    r"""
    Finite Impulse Response (FIR) channel with D taps.
    The output of a FIR channel layer is given by
    .. math::
        y_l[n] = \sum_{d=0}^{D-1} h_d y_{l-1}[n-d]
    """

    def __init__(self, h, name="fir"):
        self.h = h
        self.name = name

    def freqresp(self, NFFT=512, apply_fftshift=True):
        # Compute the frequency response (DTFT) of the filter
        _, H = signal.freqz(self.h, worN=NFFT, whole=True)
        w = np.fft.fftfreq(len(H))
        if apply_fftshift:
            w = np.fft.fftshift(w)
            H = np.fft.fftshift(H)
        return w, H

    def forward(self, x):
        y = signal.convolve(x, self.h)
        return y