import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .core import Analyser
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

def set_figure(num):
    if num:
        plt.figure(num)
    else:
        plt.figure()


class Time_Scope(Analyser):

    """
    This class implements a basic Time Scope
    """

    def __init__(self, Fs=1, num=None, name="time_scope"):
        self.num = num
        self.Fs = Fs
        self.name = name

    def analyse(self, x):
        set_figure(self.num)
        t = np.arange(len(x))/self.Fs
        if np.iscomplexobj(x):
            plt.plot(t, np.real(x))
            plt.plot(t, np.imag(x))
        else:
            plt.plot(t, x)

        plt.xlabel("time [s]")
        plt.title(self.name)


class Spectrum_Scope(Analyser):

    """
    This class implements a basic Spectrum Scope
    """

    def __init__(self, F_s=1, norm=True, dB= True, xlim=None, ylim=None, num=None, apply_fftshift=False, label="spectrum_scope", name="spectrum_scope"):
        self.num = num
        self.F_s = F_s
        self.norm = norm
        self.apply_fftshift = apply_fftshift
        self.xlim = xlim 
        self.ylim = ylim
        self.dB = dB
        self.label = label
        self.name = name  

    def forward(self, x):

        set_figure(self.num)
        fft_x = np.fft.fft(x)
        freq = np.fft.fftfreq(len(fft_x), d = 1/self.F_s)
        modulus = np.abs(fft_x)**2

        if self.apply_fftshift:
            fft_x = np.fft.fftshift(fft_x)
            freq = np.fft.fftshift(freq)

        if self.norm:
            max_modulus = np.max(modulus)
            modulus = (1/max_modulus)*modulus

        if self.dB:
            plt.plot(freq, 10*np.log10(modulus))
            plt.ylabel("PSD [dB]")
        else:
            plt.plot(freq, modulus)
            plt.ylabel("PSD")

        if self.xlim:
            plt.xlim(self.xlim)
        if self.ylim:
            plt.ylim(self.ylim)
    
        plt.xlabel("freq [Hz]")
        plt.title(self.label)

        return x


class IQ_Scope(Analyser):

    """
    This class implements a basic IQ Scope
    """

    def __init__(self, num=None, axis=None, nlim = None, name="iq_scope"):
        self.num = num
        self.nlim = nlim
        self.axis = axis
        self.name = name

    def get_signal(self, x):
        if self.nlim:
            nlim = self.nlim
            y = x[nlim[0]:nlim[1]]
        else:
            y = x
        return y

    def forward(self, x):
        
        y = self.get_signal(x)
        set_figure(self.num)
        plt.plot(np.real(y), np.imag(y), ".", label=self.name)
        plt.xlabel("real part")
        plt.ylabel("imag part")
        plt.title(self.name)

        if self.axis:
            plt.axis(self.axis)

        return x


class KDE_Scope(Analyser):

    """
    This class implements a basic KDE scope
    """

    # Plot bivariate distributions using kernel density estimation.

    def __init__(self, bw_adjust=1, thresh=0.05, num=None, name="scope"):
        self.name = name
        self.bw_adjust = bw_adjust
        self.thresh = thresh
        self.num = num

    def analyse(self, x):
        plt.figure(self.num)
        sns.kdeplot(x=np.real(x), y=np.imag(x), bw_adjust=self.bw_adjust, thresh=self.thresh, fill=True)