from .core import Processor
import numpy as np
from scipy.signal import lfilter, filtfilt, firwin

class Tx_Modem(Processor):

    def __init__(self,f0, Fs, name="tx_modem"):
        self.Fs = Fs 
        self.f0 = f0
        self.name = name

    def forward(self,data):

        N = len(data)
        t = np.arange(N)/self.Fs
        
        data_r = np.real(data)*np.cos(2*np.pi*self.f0*t)
        data_i = -np.imag(data)*np.sin(2*np.pi*self.f0*t)
        data_out = data_r+data_i

        return data_out


class Rx_Modem(Processor):

    def __init__(self,f0, Fs, filter=True, cutoff=None, name="rx_modem"):
        self.Fs = Fs 
        self.f0 = f0
        self.filter = filter
        if cutoff == None:
            cutoff = (f0/Fs)
        self.cutoff = cutoff
        self.lowpass_order = 51   
        self.name = name

    def get_h(self):
        return firwin(self.lowpass_order, self.cutoff)

    def get_delay(self):
        return self.lowpass_order / (2*self.Fs)

    def forward(self,data):
        #Quadrature demodulation
        N = len(data)
        t = np.arange(N)/self.Fs
        data_r = 2*data*np.cos(2*np.pi*self.f0*t)
        data_i = -2*data*np.sin(2*np.pi*self.f0*t)

        if self.filter:
            h = self.get_h()
            data_r = filtfilt(h, 1, data_r)
            data_i = filtfilt(h, 1, data_i)

        data_out = data_r + 1j*data_i
        return data_out
