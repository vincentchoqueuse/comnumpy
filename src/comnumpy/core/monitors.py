import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from comnumpy.core.generics import Processor
from .processors import DataExtractor


@dataclass(slots=True)
class Recorder(Processor):
    r"""Pass-through recorder that stores the signal for later retrieval.

    The recorded array is available afterwards via :meth:`get_data`,
    e.g. for plotting, criterion computation or pilot extraction.

    Signal Model
    ------------
    The input signal is stored and passed through unchanged:

    .. math::

        y[n] = x[n]

    Axes: *element-wise* -- identity pass-through, shape-agnostic.

    Parameters
    ----------
    extractor : DataExtractor, optional
        Data extractor associated with the recorder. Default is a
        ``DataExtractor`` with ``selector=None``.
    name : str, optional, keyword-only
        Name of the recorder instance. Default is ``"recorder"``.

    Examples
    --------
    >>> recorder = Recorder()
    >>> y = recorder(np.array([1.0, 2.0]))
    >>> print(recorder.get_data())
    [1. 2.]
    """
    extractor: DataExtractor = field(default_factory=lambda: DataExtractor(selector=None))
    name: str = field(default="recorder", kw_only=True)
    # internal state (declared for slots, D40a)
    data: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        self.data = None

    def get_data(self):
        return self.data

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.data = X
        return X


@dataclass(slots=True)
class Logger(Processor):
    r"""Pass-through logger that prints the signal content.

    Signal Model
    ------------
    The input signal is printed and passed through unchanged:

    .. math::

        y[n] = x[n]

    Axes: *element-wise* -- identity pass-through, shape-agnostic.

    Parameters
    ----------
    num : int, optional
        A number associated with the logger, shown in the printed line.
    name : str, optional, keyword-only
        Name of the logger instance. Default is ``"logger"``.

    Examples
    --------
    >>> logger = Logger(num=1)
    >>> y = logger(np.array([1, 2]))
    Data logger (1, logger): [1 2]
    """
    num: Optional[int] = None
    name: str = field(default="logger", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        print(f"Data logger ({self.num}, {self.name}): {X}")
        return X


@dataclass(slots=True)
class Debugger(Processor):
    r"""Pass-through debugger that prints properties of the incoming signal.

    Signal Model
    ------------
    The input signal is passed through unchanged while its shape and
    basic statistics (max of real/imaginary parts, mean, variance) are
    printed:

    .. math::

        y[n] = x[n]

    Axes: *element-wise* -- identity pass-through, shape-agnostic
    (statistics are computed over all elements).

    This class takes no constructor parameters.

    Examples
    --------
    >>> Debugger()
    Debugger(debug=False, name='debugger')
    """
    name: str = field(default="debugger", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        print(f"Data logger : {self.name}")
        print(f"shape: {X.shape}")
        print(f"max (real part): {np.max(np.real(X))}")
        print(f"max (imag part): {np.max(np.imag(X))}")
        print(f"mean: {np.mean(X)}")
        print(f"var: {np.var(X)}")
        return X


@dataclass(slots=True)
class PowerReporter(Processor):
    r"""Pass-through reporter that prints the average power of the signal.

    Signal Model
    ------------
    The input signal is passed through unchanged while its empirical
    average power is printed:

    .. math::

        y[n] = x[n], \qquad
        \widehat{P} = \frac{1}{N} \sum_{n=0}^{N-1} \left|x[n]\right|^2

    where :math:`N` is the total number of samples.

    Axes: *element-wise* -- identity pass-through, shape-agnostic (the
    power is averaged over all elements).

    Parameters
    ----------
    num : int, optional
        A number associated with the power reporter.
    verbose : bool, optional, keyword-only
        If True (default), prints the measured power :math:`\widehat{P}`.
    name : str, optional, keyword-only
        Name of the power reporter instance. Default is ``"power"``.

    Examples
    --------
    >>> reporter = PowerReporter()
    >>> y = reporter(np.array([1.0+0.0j, 0.0+1.0j]))
    Power reporter (power): 1.0
    """
    num: Optional[int] = None
    verbose: bool = field(default=True, kw_only=True)
    name: str = field(default="power", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.verbose:
            P = np.mean(np.abs(X)**2)
            print(f"Power reporter ({self.name}): {P}")
        return X


class TimeSignalMonitor(Processor):
    r"""Pass-through monitor printing amplitude and power statistics of the signal.

    Signal Model
    ------------
    The input signal is passed through unchanged while statistics of its
    modulus :math:`|x[n]|` (min, max, mean, standard deviation, RMS,
    energy, average power and optionally PAPR) are printed as a table:

    .. math::

        y[n] = x[n]

    Axes: *element-wise* -- identity pass-through, shape-agnostic
    (statistics are computed over all elements).

    Parameters
    ----------
    compute_PAPR : bool, default=False
        If True, computes and displays the Peak-to-Average Power Ratio
        (PAPR).
    PAPR_unit : str, default="dB"
        Unit for displaying PAPR. Options are ``"dB"`` or ``"linear"``.
    title : str, default="Signal Information"
        Title displayed at the top of the output summary table.
    name : str, default="signal_info_printer"
        Internal identifier for the processor.

    Examples
    --------
    >>> monitor = TimeSignalMonitor(title="Stats")
    >>> y = monitor.forward(np.array([1.0, 1.0]))
    <BLANKLINE>
    Stats
    -----
    Min            : 1.0000
    Max            : 1.0000
    Mean           : 1.0000
    Std Dev        : 0.0000
    RMS            : 1.0000
    Energy         : 2.0000
    Avg Power      : 1.0000
    -----
    """
    def __init__(self, compute_PAPR=False, PAPR_unit="dB", title="Signal Information", name="signal_info_printer"):
        super().__init__()  # initialize the Processor base fields (debug, Y)
        self.compute_PAPR = compute_PAPR
        self.PAPR_unit = PAPR_unit
        self.title = title
        self.name = name
        self.stats = {}

    def _compute_stats(self, x):
        abs_x = np.abs(x)
        self.stats["Min"] = np.min(abs_x)
        self.stats["Max"] = np.max(abs_x)
        self.stats["Mean"] = np.mean(abs_x)
        self.stats["Std Dev"] = np.std(abs_x)
        self.stats["RMS"] = np.sqrt(np.mean(abs_x**2))
        self.stats["Energy"] = np.sum(abs_x**2)
        self.stats["Avg Power"] = np.mean(abs_x**2)
        if self.compute_PAPR:
            # local import to avoid a circular dependency (ofdm imports core)
            from comnumpy.ofdm.metrics import compute_PAPR
            self.stats[f"PAPR ({self.PAPR_unit})"] = compute_PAPR(x, unit=self.PAPR_unit)

    def _print_stats(self):
        print(f"\n{self.title}")
        print("-" * len(self.title))
        for key, value in self.stats.items():
            print(f"{key:<15}: {value:.4f}")
        print("-" * len(self.title))

    def forward(self, x):
        self._compute_stats(x)
        self._print_stats()
        return x
