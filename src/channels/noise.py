from numpy.random import randn
import numpy as np
from scipy.stats import norm
from .core import Channel

def awgn_psk(order, snr_per_bit, type):

    gamma_b = snr_per_bit
    k = int(np.log2(order))

    if order == 2:    
        # see book Proakis "Digital communication", p 271
        argument = np.sqrt(2*gamma_b)
        value = norm.sf(argument)

    if order == 4:
        # see book Proakis "Digital communication", p 272
        argument = np.sqrt(2*gamma_b)
        term = norm.sf(argument)
        value = 2*term*(1-0.5*term)

    if order > 4:
        M = order
        argument = np.sqrt(2*k*gamma_b)*np.sin(np.pi/M)
        value = 2*norm.sf(argument)

    if type == "bin":
        value = value/k

    return value


def awgn_qam(order, snr_per_bit, type):

    gamma_b = snr_per_bit

    # see book Proakis "Digital communication", p 280
    M = order 
    k = np.log2(order)
    argument = np.sqrt(3*k*gamma_b/(M-1))
    P_sqrt_M = 2*(1-1/np.sqrt(M))*norm.sf(argument)

    value = 1-(1-P_sqrt_M)**2

    if type == "bin":
        value = value/k
    
    return value


def awgn_theo(modulation, order, snr_per_bit, type):
    if modulation == "PSK":
        value = awgn_psk(order, snr_per_bit, type)

    if modulation == "QAM":
        value = awgn_qam(order, snr_per_bit, type)

    return value


class AWGN(Channel):

    def __init__(self, sigma2=0, name="awgn"):
        self.sigma2 = sigma2
        self._b = None
        self.name = name

    def rvs(self,N):
        b_r = randn(N)
        b_i = randn(N)
        coef = np.sqrt(self.sigma2/2)
        self._b = coef * ( b_r + 1j * b_i)

    def forward(self,x):
        N = len(x)
        self.rvs(N)
        y = x + self._b
        return y


class Phase_Noise(Channel):

    def __init__(self, sigma2, name="phase_noise"):
        self.sigma2 = sigma2
        self._b = None
        self.name = name

    def rvs(self, N):
        sigma2 = self.sigma2
        scale = np.sqrt(sigma2)
        noise = norm.rvs(loc=0, scale=scale, size=N)
        b = np.cumsum(noise)
        self._b = b

    def forward(self, x):
        N = len(x)
        self.rvs(N)
        y = x * np.exp(1j*self._b)
        return y