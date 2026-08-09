"""comnumpy: a Python library for communication system prototyping and simulation.

The root package exposes what the canonical example needs (decision D36);
everything else lives in its subpackage (``comnumpy.core``, ``comnumpy.ofdm``,
``comnumpy.mimo``, ``comnumpy.optical``).
"""

__version__ = "1.0.0.dev0"

from .exceptions import ComnumpyError, ShapeError, NotFittedError
from .serialization import to_json, from_json
from .sweep import sweep
from .core import (
    Processor, Sequential,
    SymbolGenerator, SymbolMapper, SymbolDemapper,
    AWGN, compute_ser, get_alphabet, ebn0_to_snr_dB, esn0_to_snr_dB,
)

__all__ = [
    "ComnumpyError", "ShapeError", "NotFittedError",
    "Processor", "Sequential",
    "SymbolGenerator", "SymbolMapper", "SymbolDemapper",
    "AWGN", "compute_ser", "get_alphabet", "ebn0_to_snr_dB", "esn0_to_snr_dB",
    "to_json", "from_json", "sweep",
]
