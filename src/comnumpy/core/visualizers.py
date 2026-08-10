"""Plotting functions for time-domain signals.

Visualization is deliberately *not* part of a chain: chains describe
the communication system only, signals are extracted with
``Sequential(taps=...)`` and handed to the plain functions below. Every
function takes ``ax=None``, draws on the given axis (creating one only
when needed) and returns it (decision D25); none of them calls
``plt.show()``.

For a 2D input, each row is treated as one stream and overlaid on the
same axis with a legend; slice beforehand for anything fancier.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections.abc import Mapping
from typing import Literal, Optional, Tuple
from scipy.signal import welch
from scipy.stats import gaussian_kde

__all__ = [
    "plot_time", "plot_spectrum", "plot_welch", "plot_iq", "plot_kde",
    "plot_error_rate", "plot_chain_profiling",
]


def _as_streams(x: np.ndarray) -> np.ndarray:
    """View the input as (n_streams, N); 1D becomes a single stream."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x
    raise ValueError(
        f"expected a 1D signal or a (n_streams, N) array, got {x.shape} "
        "-- slice the extra axes before plotting.")


def plot_time(x: np.ndarray, *, fs: float = 1.0,
              plot_type: Literal["real", "abs", "pow"] = "real",
              marker: str = "-", title: str = "Time domain",
              ax: Optional[Axes] = None) -> Axes:
    """Plot a signal against time.

    Parameters
    ----------
    x : np.ndarray
        Signal, 1D or ``(n_streams, N)`` (streams are overlaid).
    fs : float, keyword-only
        Sampling frequency used for the time axis. Default 1.0.
    plot_type : {"real", "abs", "pow"}, keyword-only
        Quantity to plot. Default ``"real"``.
    marker : str, keyword-only
        Matplotlib format string. Default ``"-"``.
    title : str, keyword-only
        Axis title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes
    """
    streams = _as_streams(x)
    reduce = {"real": np.real, "abs": np.abs,
              "pow": lambda v: np.abs(v) ** 2}[plot_type]
    if ax is None:
        _, ax = plt.subplots()
    t = np.arange(streams.shape[-1]) / fs
    for idx, stream in enumerate(streams):
        ax.plot(t, reduce(stream), marker, label=f"stream {idx}")
    if streams.shape[0] > 1:
        ax.legend()
    ax.set_xlabel("time [s]")
    ax.set_ylabel(plot_type)
    ax.set_title(title)
    return ax


def plot_spectrum(x: np.ndarray, *, fs: float = 1.0, norm: bool = True,
                  dB: bool = True, shift: bool = False,
                  xlim: Optional[Tuple[float, float]] = None,
                  ylim: Optional[Tuple[float, float]] = None,
                  title: str = "Spectrum",
                  ax: Optional[Axes] = None) -> Axes:
    """Plot the periodogram (squared FFT modulus) of a signal.

    Parameters
    ----------
    x : np.ndarray
        Signal, 1D or ``(n_streams, N)`` (streams are overlaid).
    fs : float, keyword-only
        Sampling frequency. Default 1.0.
    norm : bool, keyword-only
        Normalize to the maximum. Default True.
    dB : bool, keyword-only
        Decibel scale. Default True.
    shift : bool, keyword-only
        Put the zero frequency at the center. Default False.
    xlim, ylim : tuple, optional, keyword-only
        Axis limits.
    title : str, keyword-only
        Axis title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes
    """
    streams = _as_streams(x)
    if ax is None:
        _, ax = plt.subplots()
    freq = np.fft.fftfreq(streams.shape[-1], d=1 / fs)
    if shift:
        freq = np.fft.fftshift(freq)
    for idx, stream in enumerate(streams):
        fft_x = np.fft.fft(stream)
        if shift:
            fft_x = np.fft.fftshift(fft_x)
        modulus = np.abs(fft_x) ** 2
        if norm:
            modulus = modulus / np.max(modulus)
        ax.plot(freq, 10 * np.log10(modulus) if dB else modulus,
                label=f"stream {idx}")
    if streams.shape[0] > 1:
        ax.legend()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("freq [Hz]")
    ax.set_ylabel("PSD [dB]" if dB else "PSD")
    ax.set_title(title)
    return ax


def plot_welch(x: np.ndarray, *, fs: float = 1.0, nperseg: Optional[int] = None,
               norm: bool = True, dB: bool = True,
               xlim: Optional[Tuple[float, float]] = None,
               ylim: Optional[Tuple[float, float]] = None,
               title: str = "Welch PSD",
               ax: Optional[Axes] = None) -> Axes:
    """Plot the power spectral density using Welch's method.

    Parameters
    ----------
    x : np.ndarray
        Signal, 1D.
    fs : float, keyword-only
        Sampling frequency. Default 1.0.
    nperseg : int, optional, keyword-only
        Samples per Welch segment (``scipy.signal.welch``).
    norm, dB, xlim, ylim, title, ax
        Same conventions as :func:`plot_spectrum`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    freq, modulus = welch(x, fs, nperseg=nperseg, noverlap=0,
                          return_onesided=False, scaling="spectrum")
    freq = np.fft.fftshift(freq)
    modulus = np.fft.fftshift(modulus)
    if norm:
        modulus = modulus / np.max(modulus)
    ax.plot(freq, 10 * np.log10(modulus) if dB else modulus)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("freq [Hz]")
    ax.set_ylabel("PSD [dB]" if dB else "PSD")
    ax.set_title(title)
    return ax


def plot_iq(x: np.ndarray, *, marker: str = ".", title: str = "Constellation",
            reference: Optional[np.ndarray] = None, label: Optional[str] = None,
            ax: Optional[Axes] = None) -> Axes:
    """Scatter the in-phase and quadrature components of a signal.

    Parameters
    ----------
    x : np.ndarray
        Complex signal, 1D or ``(n_streams, N)`` (streams are overlaid).
    marker : str, keyword-only
        Matplotlib format string. Default ``"."``.
    title : str, keyword-only
        Axis title.
    reference : np.ndarray, optional, keyword-only
        Ideal constellation to overlay as black crosses -- normally the
        alphabet the signal was drawn from. A received cloud means little
        without the points it is supposed to be near: the overlay is what
        turns the picture into a measurement of how far it drifted, which
        is why it is here rather than left to each caller.
    label : str, optional, keyword-only
        Legend entry for the scatter. Without it a multi-stream signal
        labels its streams ``stream 0``, ``stream 1``, ... and a
        single-stream one is not labelled at all; pass a label to
        overlay two signals on one axis (before and after a
        compensator, say) and tell them apart.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> alphabet = get_alphabet("QAM", 16)
    >>> rng = np.random.default_rng(0)
    >>> received = alphabet[rng.integers(0, 16, 200)] + 0.1 * rng.standard_normal(200)
    >>> ax = plot_iq(received, reference=alphabet, title="16-QAM")
    >>> len(ax.lines)             # the cloud, and the ideal points
    2
    """
    streams = _as_streams(x)
    if ax is None:
        _, ax = plt.subplots()
    for idx, stream in enumerate(streams):
        if label is not None:
            entry = label if streams.shape[0] == 1 else f"{label} {idx}"
        else:
            entry = f"stream {idx}"
        ax.plot(np.real(stream), np.imag(stream), marker, label=entry)
    if reference is not None:
        ideal = np.ravel(reference)
        ax.plot(np.real(ideal), np.imag(ideal), "kx", markersize=9,
                label="reference")
    if streams.shape[0] > 1 or label is not None or reference is not None:
        ax.legend()
    ax.set_xlabel("real part")
    ax.set_ylabel("imag part")
    ax.set_title(title)
    ax.axis("equal")
    return ax


def plot_kde(x: np.ndarray, *, bw_adjust: float = 1.0, thresh: float = 0.05,
             levels: int = 10, title: str = "Density",
             ax: Optional[Axes] = None) -> Axes:
    """Contour plot of the I/Q kernel density estimate of a signal.

    Parameters
    ----------
    x : np.ndarray
        Complex signal (flattened before estimation).
    bw_adjust : float, keyword-only
        Bandwidth multiplier of ``scipy.stats.gaussian_kde``. Default 1.0.
    thresh : float, keyword-only
        Fraction of the peak density below which nothing is drawn.
        Default 0.05.
    levels : int, keyword-only
        Number of contour levels. Default 10.
    title : str, keyword-only
        Axis title.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    data = np.vstack([np.real(x).ravel(), np.imag(x).ravel()])
    kde = gaussian_kde(data)
    # scipy stubs type `factor` loosely; it is a scalar bandwidth
    kde.set_bandwidth(np.asarray(kde.factor).item() * bw_adjust)
    xi, yi = np.mgrid[data[0].min():data[0].max():100j,
                      data[1].min():data[1].max():100j]
    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
    ax.contourf(xi, yi, np.ma.masked_less(zi, thresh * zi.max()),
                levels=levels)
    ax.set_xlabel("real part")
    ax.set_ylabel("imag part")
    ax.set_title(title)
    return ax


def plot_error_rate(x: np.ndarray,
                    measured: Mapping[str, np.ndarray] | np.ndarray, *,
                    theory: Optional[Mapping[str, np.ndarray]] = None,
                    x_theory: Optional[np.ndarray] = None,
                    xlabel: str = "SNR [dB]",
                    ylabel: str = "error rate",
                    title: str = "", ax: Optional[Axes] = None) -> Axes:
    """Plot Monte-Carlo error rates, and the curves they are read against.

    The figure every sweep ends with: measured points as hollow markers,
    the closed form they are supposed to reach as a line **in the same
    colour**, a logarithmic ordinate and a grid on both decades. Fourteen
    scripts of this repository drew it by hand before this existed.

    Zeros are dropped rather than plotted. A sweep point where no error
    was seen means the estimate ran out of samples, not that the error
    rate is zero, and a logarithmic axis has no place to put it.

    Parameters
    ----------
    x : np.ndarray
        Abscissa of the measurements, usually an SNR in dB.
    measured : mapping of str to np.ndarray, or np.ndarray
        Simulated error rates. A bare array is one unnamed curve.
    theory : mapping of str to np.ndarray, optional, keyword-only
        Reference curves. A key also present in ``measured`` is drawn in
        that curve's colour, so a pair reads as one statement; any other
        key gets its own colour.
    x_theory : np.ndarray, optional, keyword-only
        Abscissa of the reference curves, when they are evaluated on a
        finer grid than the measurements. Defaults to ``x``.
    xlabel, ylabel, title : str, optional, keyword-only
        Axis labels and title.
    ax : matplotlib.axes.Axes, optional, keyword-only
        Axis to draw on; created when None (decision D25).

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> snr = np.arange(0, 12, 2)
    >>> ax = plot_error_rate(snr, {"QPSK": 10.0 ** (-snr / 8)},
    ...                      theory={"QPSK": 10.0 ** (-snr / 8 - 0.05)})
    >>> len(ax.lines)                      # one marker set, one curve
    2
    """
    if ax is None:
        _, ax = plt.subplots()
    curves = ({"": np.asarray(measured)}
              if not isinstance(measured, Mapping) else dict(measured))
    reference = dict(theory or {})
    abscissa = np.asarray(x, dtype=float)
    fine = abscissa if x_theory is None else np.asarray(x_theory, dtype=float)

    colors = {}
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        seen = values > 0
        # A measurement that has a reference curve is drawn as markers
        # alone, so that the pair reads as one statement rather than as
        # two curves; one that has none is joined, because a scatter of
        # points is not a curve the eye can follow.
        style = "o" if name in reference else "o-"
        line, = ax.semilogy(abscissa[seen], values[seen], style,
                            fillstyle="none",
                            label=name or "simulation")
        colors[name] = line.get_color()
    for name, values in reference.items():
        values = np.asarray(values, dtype=float)
        seen = values > 0
        label = f"{name}, theory" if name else "theory"
        ax.semilogy(fine[seen], values[seen], "-", color=colors.get(name),
                    label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both")
    if len(curves) + len(reference) > 1 or any(curves):
        ax.legend()
    return ax


def plot_chain_profiling(chain: object, input: object, *,
                         ax: Optional[Axes] = None,
                         title: str = "Processor timings",
                         N_test: int = 100,
                         orientation: Literal["horizontal", "vertical"] = "horizontal") -> Axes:
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

    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If None, a new figure and axis are created.

    title : str, optional
        The title of the box plot. Default is 'Box Plot of Method Timings'.

    N_test : int, optional
        The number of times to run the chain and collect profiling results. Default is 100.

    orientation : {"horizontal", "vertical"}, optional
        Orientation of the box plot. Default is "horizontal".

    .. note::
        Increasing the number :code:`N_test` provides a more accurate representation of the execution time distribution but takes longer to compute.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the box plot of the profiling results (decision D25).

    """
    profile = getattr(chain, "profile_execution_time", None)
    if profile is None:
        raise TypeError(
            f"expected a chain exposing profile_execution_time(), got "
            f"{type(chain).__name__} -- pass a Sequential instance.")

    # run chain
    results = [profile(input) for _ in range(N_test)]

    # Extract keys from the first dictionary to use as labels
    keys = results[0].keys()

    # Convert the list of dictionaries to a NumPy array
    data_array = np.array([[result[key] for key in keys] for result in results])

    # Create a box plot
    if ax is None:
        # constrained layout: block names are long and would be clipped
        _, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.boxplot(data_array, tick_labels=list(keys), orientation=orientation)
    ax.set_title(title)
    # a chain spans several decades -- an FFT and a demapper are not the
    # same order of magnitude -- so a linear axis shows one block and
    # flattens all the others against zero
    if orientation == "horizontal":
        ax.set_xscale("log")
        ax.set_xlabel("Time (s)")
    else:
        ax.set_yscale("log")
        ax.set_ylabel("Time (s)")
    ax.grid(True, which="both")
    return ax
