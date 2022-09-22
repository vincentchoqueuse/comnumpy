from .core import Processor
import numpy as np


class OM_Algorithm(Processor):

    def __init__(self, os, name="om"):
        self.os = os
        self.name = name

    def estimate_tau(self, x):
        n_vect = np.arange(len(x))
        x2 = np.abs(x)**2
        coef = -(1/(2*np.pi))
        tau = coef*np.angle(np.sum(x2*np.exp(-2j*np.pi*n_vect / self.os)))
        return tau

    def compensation_tau(self, x, tau):
        fftx = np.fft.fft(x)
        freq = np.fft.fftfreq(len(fftx))
        ffty = fftx*np.exp(2j*np.pi*tau*freq*self.os)
        y = np.fft.ifft(ffty)
        return y

    def forward(self, x):
        tau = self.estimate_tau(x)
        print(tau)
        y = 2*self.compensation_tau(x, tau)
        return y



#class Gardner_Algorithm(Processor):

    #https://publications.lib.chalmers.se/records/fulltext/161391.pdf
