from importlib import import_module

from .generics import Processor, Sequential
from .monitors import Recorder
from .generators import SymbolGenerator, GaussianGenerator
from .mappers import SymbolMapper, SymbolDemapper
from .channels import AWGN, FIRChannel
from .filters import SRRCFilter, BWFilter
from .processors import Upsampler, Downsampler, Serial2Parallel, Parallel2Serial, Amplifier, DataExtractor
from .metrics import compute_ser, compute_ber, compute_evm, compute_metric_awgn_theo, compute_ccdf
from .utils import get_alphabet, hard_projector, ebn0_to_snr_dB, esn0_to_snr_dB
from .frames import FieldRole, FrameField, FrameStructure, Framer, Deframer
from .sequences import zadoff_chu, schmidl_cox_preamble, barker, golay_pair, m_sequence

__all__ = [
    "Processor", "Sequential", "Recorder",
    "SymbolGenerator", "GaussianGenerator",
    "SymbolMapper", "SymbolDemapper",
    "AWGN", "FIRChannel",
    "SRRCFilter", "BWFilter",
    "Upsampler", "Downsampler", "Serial2Parallel", "Parallel2Serial",
    "Amplifier", "DataExtractor",
    "compute_ser", "compute_ber", "compute_evm", "compute_metric_awgn_theo", "compute_ccdf",
    "get_alphabet", "hard_projector", "ebn0_to_snr_dB", "esn0_to_snr_dB",
    "FieldRole", "FrameField", "FrameStructure", "Framer", "Deframer",
    "zadoff_chu", "schmidl_cox_preamble", "barker", "golay_pair", "m_sequence",
    # lazily loaded (they pull matplotlib, see D36):
    "Scope", "plot_chain_profiling",
]

# PEP 562 lazy loading: importing comnumpy must not import matplotlib (D36).
_LAZY = {"Scope": ".visualizers", "plot_chain_profiling": ".visualizers"}


def __getattr__(name):
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
