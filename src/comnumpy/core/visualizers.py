import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from comnumpy.core.generics import Processor
from scipy.signal import welch


mpl.rcParams['agg.path.chunksize'] = 10000


def create_subplots(num, fig_indices):
    num_plots = len(fig_indices)
    if num_plots == 1:
        width = 6
    else:
        width = 3 * num_plots

    fig, axes = plt.subplots(num_plots, 1, num=num, figsize=(8, width))

    if num_plots == 1:
        axes = [axes]

    return fig, axes


def get_sliced_data(X, idx, axis=-1):
    if X.ndim == 1:
        # For SISO, return the entire signal if idx is not specified, otherwise return the specific index
        x_sliced = X
    else:
        slices = [slice(None)] * X.ndim
        slices[axis] = idx
        x_sliced = X[tuple(slices)]
    return x_sliced


def plot_chain_profiling(chain, input, title='Processor Timings', N_test=100, orientation="horizontal"):
    r"""
    Plot the profiling results of a chain of processors using a box plot to visualize the distribution of execution times.

    This function runs a specified chain of processors multiple times, collects the profiling results, and visualizes them
    using a box plot to show the distribution of execution times for each method in the chain.

    Parameters
    ----------
    chain : object
        An object representing the chain of processors to be profiled. It must have a method `profile_execution_time`
        that takes `input` as an argument and returns a dictionary of execution times for each method in the chain.

    input : any
        The input data to be passed to the `profile_execution_time` method of the chain. The type and structure of
        this input depend on the specific implementation of the chain.

    title : str, optional
        The title of the box plot. Default is 'Box Plot of Method Timings'.

    N_test : int, optional
        The number of times to run the chain and collect profiling results. Default is 100. 
        
        
    .. note::
        Increasing the number :code:`N_test` provides a more accurate representation of the execution time distribution but takes longer to compute.

    Returns
    -------
    None
        This function does not return any value. It displays a box plot of the profiling results.

    """

    # run chain
    results = []

    for num_test in range(N_test):
        output = chain.profile_execution_time(input)
        results.append(output)

    # Extract keys from the first dictionary to use as labels
    keys = results[0].keys()

    # Convert the list of dictionaries to a NumPy array
    data_array = np.array([[result[key] for key in keys] for result in results])

    # Create a box plot
    plt.figure(figsize=(12, 6))
    plt.boxplot(data_array, labels=keys, orientation=orientation)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.grid(True)


@dataclass
class TimeScope(Processor):
    """
    A basic Time Scope for visualizing signals in the time domain.

    This class provides a simple interface to plot the real and imaginary parts of a complex signal, or just the signal itself if it's real, against time.

    Attributes
    ----------
    fs : float
        Sampling frequency used to calculate the time axis. Default is 1.0.
    num : Optional[int]
        The figure number to be used for the plot. If None, a new figure is created.
    complex_type : str
        Determines which part of the complex signal to plot. Options are "real", "abs", and "pow".
        Default is "abs".
    title : str
        Title of the plot. Default is "Time Scope".
    name : str
        Name of the time scope instance. Default is "time_scope".
    marker : str
        Marker style for the plot. Default is "-".
    axis : int
        The axis along which to plot the signal for MIMO. Default is -1.
    fig_indices : tuple
        Indices of the figures to plot for MIMO signals.

    """
    num: Optional[int]
    fs: float = 1.0
    plot_type: Literal["full", "real", "abs", "pow"] = "real"
    fig_indices: Tuple[int, ...] = (0,)
    slices: Tuple[slice, ...] = (slice(None),)
    marker: str = "-"
    is_mimo: bool = True
    axis: int = 0
    title: str = "Time Scope"
    name: str = "time_scope"

    def forward(self, X: np.ndarray) -> np.ndarray:
        fig, axes = create_subplots(self.num, self.fig_indices)

        for ax, idx in zip(axes, self.fig_indices):
            x_sliced = get_sliced_data(X, idx, axis=self.axis)
            t = np.arange(len(x_sliced)) / self.fs  # Assuming the specified axis is time

            match self.plot_type:
                case "real":
                    x_sliced_2_plot = np.real(x_sliced)
                case "abs":
                    x_sliced_2_plot = np.abs(x_sliced)
                case "pow":
                    x_sliced_2_plot = np.abs(x_sliced)**2
                case "full":
                    x_sliced_2_plot = x_sliced

            ax.plot(t, x_sliced_2_plot, self.marker)
            ax.set_xlabel("time [s]")
            ax.set_title(f"{self.title} - Stream {idx} ({self.plot_type})")

        plt.tight_layout()
        return X


@dataclass
class SpectrumScope(Processor):
    """
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
        Limits for the x-axis (frequency). None for no limits.
    ylim : tuple or None
        Limits for the y-axis (amplitude). None for no limits.
    num : int or None
        The figure number for plotting. If None, a new figure is created.
    shift : bool
        If True, shifts the zero frequency to the center of the spectrum.
    title : str
        Title of the plot. Default is "Spectrum Scope".
    name : str
        Name of the Spectrum Scope instance.
    axis : int
        The axis along which to compute the FFT for MIMO. Default is -1.
    fig_indices : tuple
        Indices of the figures to plot for MIMO signals.
    slices : tuple
        Slices to apply to the input signal for selecting which parts to plot.

    """
    fs: float = 1.0
    norm: bool = True
    dB: bool = True
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    num: Optional[int] = None
    shift: bool = False
    fig_indices: Tuple[int, ...] = (0,)
    slices: Tuple[slice, ...] = (slice(None),)
    is_mimo: bool = False
    axis: int = 0
    title: str = "Spectrum Scope"
    name: str = "SpectrumScope"

    def forward(self, X: np.ndarray) -> np.ndarray:
        fig, axes = create_subplots(self.num, self.fig_indices)

        for ax, idx in zip(axes, self.fig_indices):

            x_sliced = get_sliced_data(X, idx, axis=self.axis)
            fft_x = np.fft.fft(x_sliced, axis=-1)
            freq = np.fft.fftfreq(x_sliced.shape[-1], d=1/self.fs)

            if self.shift:
                fft_x = np.fft.fftshift(fft_x, axes=-1)
                freq = np.fft.fftshift(freq)

            modulus = np.abs(fft_x)**2
            if self.norm:
                max_modulus = np.max(modulus)
                modulus = modulus / max_modulus

            if self.dB:
                ax.plot(freq, 10*np.log10(modulus))
                ax.set_ylabel("PSD [dB]")
            else:
                ax.plot(freq, modulus)
                ax.set_ylabel("PSD")

            if self.xlim:
                ax.set_xlim(self.xlim)
            if self.ylim:
                ax.set_ylim(self.ylim)

            ax.set_xlabel("freq [Hz]")
            ax.set_title(f"{self.title} - Stream {idx}")

        plt.tight_layout()
        return X


@dataclass
class IQScope(Processor):
    """
    A basic IQ Scope for visualizing the In-phase (I) and Quadrature (Q) components
    of a complex signal.

    This class provides a simple interface to plot the I and Q components of a signal
    on a 2D scatter plot. It allows for specification of the range of samples to plot
    and custom axis limits.

    Attributes
    ----------
    num : Optional[int]
        The figure number for plotting. If None, a new figure is created.
    title : str
        Title of the plot. Default is "IQ Scope".
    name : str
        Name of the IQ Scope instance. Default is "IQScope".
    axis : int
        The axis along which to plot the IQ components for MIMO. Default is -1.
    fig_indices : tuple
        Indices of the figures to plot for MIMO signals.
    slices : tuple
        Slices to apply to the input signal for selecting which parts to plot.

    """
    num: Optional[int] = None
    title: str = "IQ Scope"  
    fig_indices: Tuple[int, ...] = (0,)
    slices: Tuple[slice, ...] = (slice(None),)
    is_mimo: bool = True
    axis: int = 0
    name: str = "IQScope"

    def forward(self, X: np.ndarray) -> np.ndarray:
        fig, axes = create_subplots(self.num, self.fig_indices)

        for ax, idx in zip(axes, self.fig_indices):
            x_sliced = get_sliced_data(X, idx, axis=self.axis)
            ax.plot(np.real(x_sliced), np.imag(x_sliced), ".", label=f"Stream {idx}")
            ax.set_xlabel("real part")
            ax.set_ylabel("imag part")
            ax.set_title(f"{self.title} - Stream {idx}")
            ax.axis("equal")
            ax.legend()

        return X


@dataclass
class KDEScope(Processor):
    """
    A basic Kernel Density Estimation (KDE) Scope for visualizing bivariate distributions.

    This class provides functionality to plot the kernel density estimation of a complex signal's In-phase (I) and Quadrature (Q) components.
    It uses Seaborn's kdeplot for visualization.

    Attributes
    ----------
    bw_adjust : float
        Bandwidth adjustment for kernel density estimation. Larger values make the
        estimation smoother. Default is 1.0.
    thresh : float
        Threshold for the density estimate. Only regions with density above this
        threshold will be plotted. Default is 0.05.
    num : Optional[int]
        The figure number for plotting. If None, a new figure is created.
    name : str
        Title of the KDE plot. Default is "KDE Scope".

    """
    bw_adjust: float = 1.0
    thresh: float = 0.05
    num: Optional[int] = None
    is_mimo: bool = False
    name: str = "KDE Scope"

    def forward(self, x: np.ndarray) -> np.ndarray:
        fig, axes = create_subplots(self.num, self.fig_indices)

        sns.kdeplot(x=np.real(x), y=np.imag(x), bw_adjust=self.bw_adjust, thresh=self.thresh, fill=True)
        return x


@dataclass
class Scope(Processor):
    """
    A generic Scope class that instantiates the appropriate visualizer based on the type attribute.

    Attributes
    ----------
    scope_type : str
        The type of scope to instantiate. Options are "time", "spectrum", "iq", and "kde".
    kwargs : dict
        Additional keyword arguments to pass to the specific scope class.
    """
    scope_type: Literal["time", "spectrum", "iq", "kde"]
    kwargs: dict

    def __new__(cls, scope_type: str, **kwargs) -> Processor:
        scope_classes = {
            "time": TimeScope,
            "spectrum": SpectrumScope,
            "iq": IQScope,
            "kde": KDEScope,
        }

        if scope_type not in scope_classes:
            raise ValueError(f"Unknown scope type: {scope_type}")

        return scope_classes[scope_type](**kwargs)


@dataclass
class WelchScope(Processor):
    """
    Scope for visualizing the power spectral density (PSD) of a signal using Welch’s method.

    This class provides functionality to compute and plot the PSD of a signal
    optionally using decibels for the amplitude and applying a shift for zero frequency at the center.

    Attributes
    ----------
    F_s : float
        Sampling frequency of the input signal.
    nperseg : int
        The number of samples by window
    norm : bool
        If True, normalizes the spectrum to its maximum value.
    dB : bool
        If True, converts the spectrum to decibel scale.
    xlim : tuple or None
        Limits for the x-axis (frequency). None for no limits.
    ylim : tuple or None
        Limits for the y-axis (amplitude). None for no limits.
    num : int or None
        The figure number for plotting. If None, a new figure is created.
    shift : bool
        If True, shifts the zero frequency to the center of the spectrum.
    label : str
        Label for the plot.
    name : str
        Name of the Spectrum Scope instance.

    """

    def __init__(self, fs=1, nperseg=None, norm=True, dB= True, xlim=None, ylim=None, num=None, title="PSD_scope", name="spectrum_scope"):
        self.num = num
        self.fs = fs
        self.norm = norm
        self.nperseg = nperseg
        self.xlim = xlim 
        self.ylim = ylim
        self.dB = dB
        self.title = title
        self.name = name

    def forward(self, x):
        plt.figure(self.num)
        freq, modulus = welch(x, self.fs, nperseg=self.nperseg, noverlap=0, return_onesided=False, scaling='spectrum')
        freq = np.fft.fftshift(freq)
        modulus = np.fft.fftshift(modulus)

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
        plt.title(self.title)
        return x
