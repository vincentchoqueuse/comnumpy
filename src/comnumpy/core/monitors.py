import numpy as np
from dataclasses import dataclass, field
from comnumpy.core.generics import Processor
from .processors import DataExtractor


@dataclass
class Recorder(Processor):
    """
    Implements a basic Recorder for signal processing.

    This class acts as a recorder in a signal processing chain. It stores the input signal and allows for later retrieval. 

    .. HINT::
        Putting a recorder on a chain allows to store the signal for later purpose (such as plotting, criterion computation, pilot extraction, ...).

    Signal Model
    ------------
    
    The input data is not modified and is directly sent to the output.

    .. math::
        \mathbf{Y} = \mathbf{X}

    Attributes
    ----------
    extractor: DataExtractor (optional)
        : data extraction 
    name : str (optional)
        Name of the recorder instance. Default is 'recorder'.

    Methods
    -------
    get_data():
        Retrieves the recorded data.

    """
    is_mimo: bool = True
    extractor: DataExtractor = field(default_factory=lambda: DataExtractor(selector=None))
    name: str = "recorder"

    def __post_init__(self):
        self.data = None

    def get_data(self):
        return self.data

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.data = X
        return X


@dataclass
class Logger(Processor):
    """
    This class implements a basic Logger that lets the signal pass through.

    Attributes
    ----------
    num : int, optional
        A number associated with the logger.
    name : str
        Name of the logger instance.
    """
    num: int = None
    name: str = "logger"

    def forward(self, X: np.ndarray) -> np.ndarray:
        print(f"Data logger ({self.num}, {self.name}): {X}")
        return X


@dataclass
class Debugger(Processor):
    """
    This class prints several information about the incomming signal for debugging purposes

    Attributes
    ----------
    name : str
        Name of the logger instance.
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        print(f"Data logger : {self.name}")
        print(f"shape: {X.shape}")
        print(f"max (real part): {np.max(np.real(X))}")
        print(f"max (imag part): {np.max(np.imag(X))}")
        print(f"mean: {np.mean(X)}")
        print(f"var: {np.var(X)}")
        return X


@dataclass
class PowerReporter(Processor):
    """
    This class implements a basic Power Reporter that lets the signal pass through.

    Attributes
    ----------
    num : int, optional
        A number associated with the power reporter.
    verbose : bool
        Whether to print detailed information.
    name : str
        Name of the power reporter instance.
    """
    num: int = None
    verbose: bool = True
    name: str = "power"

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.verbose:
            P = np.mean(np.abs(X)**2)
            print(f"Power reporter ({self.name}): {P}")
        return X


class TimeSignalMonitor(Processor):
    """
    A signal analysis processor for computing and displaying basic amplitude and power statistics.

    This class takes a real or complex signal and computes key characteristics such as:
    minimum, maximum, mean, standard deviation, RMS, energy, average power, and optionally PAPR.
    The results are printed in a clean and formatted table in the terminal.

    Parameters
    ----------
    compute_PAPR : bool, default=False
        If True, computes and displays the Peak-to-Average Power Ratio (PAPR).

    PAPR_unit : str, default="dB"
        Unit for displaying PAPR. Options are "dB" or "linear".

    title : str, default="Signal Information"
        Title displayed at the top of the output summary table.

    name : str, default="signal_info_printer"
        Internal identifier for the processor.
    """
    def __init__(self, compute_PAPR=False, PAPR_unit="dB", title="Signal Information", name="signal_info_printer"):
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
