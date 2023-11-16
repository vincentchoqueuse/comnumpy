import numpy as np
from scipy import signal
from .core import Processor


class Upsampler(Processor):
    """
    Implement a upsampler with integer upsampling factor.
    """

    def __init__(self, oversampling, scale=1, name="upsampler"):
        self.oversampling = oversampling
        self.scale = scale
        self.name = name

    def forward(self,x):
        N = len(x)
        y = np.zeros(self.oversampling*N, dtype=complex)
        y[::self.oversampling] = x
        return self.scale*y


class Downsampler(Processor):
    """Downsampler
    Implement a downsampler with integer downsampling factor.

    It does not implement any anti-aliasing filter
    """

    def __init__(self, oversampling, pre_delay=0, post_delay=0, scale=1, name="downsampler"):
        self.oversampling = oversampling
        self.scale = scale
        self.name = name
        self.pre_delay = pre_delay
        self.post_delay = post_delay

    def forward(self,x):
        pre_delay = self.pre_delay
        post_delay = self.post_delay
        N = len(x)
        y = x[pre_delay::self.oversampling]
        y = y[post_delay:]
        return self.scale*y


class SRRC_filter(Processor):
    """ 
    Implement SRR Filter with the coefficients, b, that correspond to a square-root raised cosine FIR filter with rolloff factor specified by beta. The filter is truncated to span symbols, and each symbol period contains sps samples. The order of the filter, sps*span, must be even. The filter energy is 1.
    
    :param rho: roll off factor
    :param N_h: length of filter before oversampling
    :param oversampling: oversampling factor
    :param scale: amplitude scaling factor
    :param method: method ("auto", "time", "fft")
    
    """
    # https://en.wikipedia.org/wiki/Root-raised-cosine_filter

    def __init__(self, rho, oversampling, N_h=10, norm=True, scale=1, method="auto", name="srrc_filter"):
        self.rho = rho  # roll-off factor
        self.N_h = N_h
        self.oversampling = oversampling  # oversampling factor
        self.norm = norm
        self.method = method  #‘auto’, ‘direct’, ‘fft’
        self.scale=scale
        self.name = name

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
                    den = term1*(1 - coef1**2 )
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
        
    def forward(self,x):
        
        if self.method == "auto":
            h = self.h()
            y = signal.convolve(x, h, method=self.method)

        if self.method == "fft": 
            NFFT = len(x)
            fft_x = np.fft.fft(x, NFFT)
            fft_h = self.H(NFFT)
            y = self.scale*np.fft.ifft(fft_x*fft_h, NFFT)

        """
        if self.method == "fft_old":
            NFFT = len(x)
            t = np.fft.fftfreq(NFFT, d=self.oversampling/NFFT)
            h = self.h(t=t)
            fft_x = np.fft.fft(x, NFFT)
            fft_h = np.fft.fft(h, NFFT)
            y = self.scale*np.fft.ifft(fft_x*fft_h, NFFT)    
        """
        return y


class BW_filter(Processor):
    """
    Implements a frequency domain low-pass brick wall filter 
    
    :param wn: The critical frequency or frequencies. Wn units are normalized from 0 to 1, where 1 is the Nyquist frequency 
    """
    def __init__(self, wn):
        self.wn = wn  # oversampling factor

    def forward(self,x):
        NFFT = len(x)
        w = np.fft.fftfreq(NFFT,d=1)
        H = (abs(w) <= self.wn).astype(float)
        fft_x = np.fft.fft(x, NFFT)
        y = np.fft.ifft(H*fft_x, NFFT)
        return y

