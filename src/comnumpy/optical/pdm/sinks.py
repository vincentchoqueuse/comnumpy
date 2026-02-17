from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.io.wavfile import write
from comnumpy.core.processors import Downsampler
from comnumpy.optical.pdm.trail_remover import TrailRemover
from comnumpy.core.generics import Processor

mpl.rcParams['agg.path.chunksize'] = 10000

def set_figure(num):
    if num:
        plt.figure(num)
    else:
        plt.figure()


@dataclass
class Time_Scope(Processor):
    r"""
    A basic Time Scope for visualizing signals in the time domain.

    This class provides a simple interface to plot the real and imaginary parts of a 
    complex signal, or just the signal itself if it's real, against time.

    Attributes
    ----------
    fs : float
        Sampling frequency used to calculate the time axis.
    num : int or None
        The figure number to be used for the plot. If None, a new figure is created.
    complex_type : str
        Type of complex representation: "abs", "real", or "pow". Default is "abs".
    title : str
        Title of the plot.
    name : str
        Name of the Time Scope instance.
    marker : str
        Marker style for the plot. Default is "-".
    """

    fs: float = 1
    num: Optional[int] = None
    complex_type: str = "abs"
    title: str = "time scope"
    name: str = "time_scope"
    marker: str = "-"

    def forward(self, x: np.ndarray) -> np.ndarray:
        set_figure(self.num)
        t = np.arange(len(x)) / self.fs
        if np.iscomplexobj(x):
            if self.complex_type == "real":
                plt.plot(t, np.real(x), self.marker)
            elif self.complex_type == "abs":
                plt.plot(t, np.abs(x), self.marker)
            elif self.complex_type == "pow":
                plt.plot(t, np.abs(x)**2, self.marker)
        else:
            plt.plot(t, x, self.marker)

        plt.xlabel("time [s]")
        plt.title(self.name)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


@dataclass
class Spectrum_Scope(Processor):
    r"""
    A basic Spectrum Scope for visualizing the power spectral density (PSD) of a signal.

    This class provides functionality to compute the Fast Fourier Transform (FFT) of a signal
    and plot its spectrum, optionally using decibels for the amplitude and applying a shift
    for zero frequency at the center.

    Attributes
    ----------
    fs : float
        Sampling frequency of the input signal.
    norm : bool
        If True, normalizes the spectrum to its maximum value.
    dB : bool
        If True, converts the spectrum to decibel scale.
    xlim : tuple or None
        Limits for the x-axis (frequency).
    ylim : tuple or None
        Limits for the y-axis (amplitude).
    num : int or None
        The figure number for plotting.
    shift : bool
        If True, shifts the zero frequency to the center of the spectrum.
    title : str
        Title of the plot.
    name : str
        Name of the Spectrum Scope instance.
    """

    fs: float = 1
    norm: bool = True
    dB: bool = True
    xlim: Optional[Tuple] = None
    ylim: Optional[Tuple] = None
    num: Optional[int] = None
    shift: bool = False
    title: str = "spectrum_scope"
    name: str = "spectrum_scope"

    def forward(self, x: np.ndarray) -> np.ndarray:
        set_figure(self.num)
        fft_x = np.fft.fft(x)
        freq = np.fft.fftfreq(len(fft_x), d=1/self.fs)

        if self.shift:
            fft_x = np.fft.fftshift(fft_x)

        modulus = np.abs(fft_x)**2
        modulus = np.fft.fftshift(modulus)
        freq = np.fft.fftshift(freq)

        if self.norm:
            max_modulus = np.max(modulus)
            modulus = (1/max_modulus) * modulus

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
        plt.title(self.title)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


@dataclass
class IQ_Scope(Processor):
    r"""
    A basic IQ Scope for visualizing the In-phase (I) and Quadrature (Q) components 
    of a complex signal.

    This class provides a simple interface to plot the I and Q components of a signal 
    on a 2D scatter plot.

    Attributes
    ----------
    num : int or None
        The figure number for plotting.
    axis : tuple or None
        Limits for the axis of the plot (xmin, xmax, ymin, ymax).
    nlim : tuple or None
        Limits for the sample numbers to be plotted (n_start, n_end).
    title : str
        Title of the plot.
    name : str
        Name of the IQ Scope instance.
    """

    num: Optional[int] = None
    axis: Optional[Tuple] = None
    nlim: Optional[Tuple[int, int]] = None
    title: str = "iq scope"
    name: str = "iq_scope"

    def get_signal(self, x: np.ndarray) -> np.ndarray:
        if self.nlim:
            y = x[self.nlim[0]:self.nlim[1]]
        else:
            y = x
        return y

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.get_signal(x)
        set_figure(self.num)
        plt.plot(np.real(y), np.imag(y), ".", label=self.name)
        plt.xlabel("real part")
        plt.ylabel("imag part")
        plt.title(self.title)

        if self.axis:
            plt.axis(self.axis)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


@dataclass
class IQ_Scope_PostProcessing(IQ_Scope):
    r"""
    IQ Scope with post-processing: downsampling and trail removal.

    Extends IQ_Scope with automatic downsampling and trail removal before visualization.

    Attributes
    ----------
    oversampling : int
        Oversampling factor for downsampling.
    srrc_taps : int
        Number of SRRC filter taps.
    """

    oversampling: int = 2
    srrc_taps: int = 30
    downsampler: Downsampler = field(init=False, repr=False)
    trail_remover: TrailRemover = field(init=False, repr=False)

    def __post_init__(self):
        self.downsampler = Downsampler(self.oversampling, phase=self.srrc_taps * self.oversampling)
        self.trail_remover = TrailRemover(trail=self.srrc_taps * self.oversampling)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_downsampled = self.downsampler(x)
        x_processed = self.trail_remover(x_downsampled)

        y = self.get_signal(x_processed)
        set_figure(self.num)
        plt.plot(np.real(y), np.imag(y), ".", label=self.name)
        plt.xlabel("real part")
        plt.ylabel("imag part")
        plt.title(self.title)

        if self.axis:
            plt.axis(self.axis)
        return x


@dataclass
class KDE_Scope(Processor):
    r"""
    A basic Kernel Density Estimation (KDE) Scope for visualizing bivariate distributions.

    This class provides functionality to plot the kernel density estimation of a 
    complex signal's In-phase (I) and Quadrature (Q) components.

    Attributes
    ----------
    bw_adjust : float
        Bandwidth adjustment for kernel density estimation.
    thresh : float
        Threshold for the density estimate.
    num : int or None
        The figure number for plotting.
    name : str
        Name of the KDE Scope instance.
    """

    bw_adjust: float = 1
    thresh: float = 0.05
    num: Optional[int] = None
    name: str = "scope"

    def forward(self, x: np.ndarray) -> np.ndarray:
        plt.figure(self.num)
        sns.kdeplot(x=np.real(x), y=np.imag(x), bw_adjust=self.bw_adjust, thresh=self.thresh, fill=True)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


@dataclass
class Audio_Recorder(Processor):
    r"""
    A class for recording audio data to a file.

    This class allows for writing an audio signal to a file in a 16-bit integer format.

    Attributes
    ----------
    filename : str
        The name of the file where the audio data will be saved.
    rate : int
        The sampling rate of the audio data in Hz.
    """

    filename: str
    rate: int = 44100

    def forward(self, x: np.ndarray) -> np.ndarray:
        write(self.filename, self.rate, x.astype(np.int16))
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)