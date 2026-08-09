from importlib import import_module

from .generics import Processor, Sequential
from .generators import SymbolGenerator, GaussianGenerator
from .mappers import SymbolMapper, SymbolDemapper
from .channels import AWGN, FIRChannel, TappedDelayLineChannel
from .capacity import (awgn_capacity, constellation_capacity, bicm_capacity,
                       rayleigh_ergodic_capacity, mimo_ergodic_capacity,
                       outage_capacity, waterfilling)
from .fading import (DopplerSpectrum, PowerDelayProfile, rayleigh_process,
                     get_delay_profile, register_delay_profile,
                     available_delay_profiles)
from .filters import SRRCFilter, BWFilter
from .processors import Upsampler, Downsampler, Serial2Parallel, Parallel2Serial, Amplifier, DataExtractor
from .metrics import compute_ser, compute_ber, compute_evm, compute_metric_awgn_theo, compute_ccdf, signal_report
from .utils import get_alphabet, hard_projector, ebn0_to_snr_dB, esn0_to_snr_dB
from .frames import FieldRole, FrameField, FrameStructure, Framer, Deframer
from .sequences import zadoff_chu, schmidl_cox_preamble, barker, golay_pair, m_sequence

__all__ = [
    "Processor", "Sequential",
    "SymbolGenerator", "GaussianGenerator",
    "SymbolMapper", "SymbolDemapper",
    "AWGN", "FIRChannel", "TappedDelayLineChannel",
    "awgn_capacity", "constellation_capacity", "bicm_capacity",
    "rayleigh_ergodic_capacity", "mimo_ergodic_capacity",
    "outage_capacity", "waterfilling",
    "DopplerSpectrum", "PowerDelayProfile", "rayleigh_process",
    "get_delay_profile", "register_delay_profile", "available_delay_profiles",
    "SRRCFilter", "BWFilter",
    "Upsampler", "Downsampler", "Serial2Parallel", "Parallel2Serial",
    "Amplifier", "DataExtractor",
    "compute_ser", "compute_ber", "compute_evm", "compute_metric_awgn_theo", "compute_ccdf",
    "signal_report",
    "get_alphabet", "hard_projector", "ebn0_to_snr_dB", "esn0_to_snr_dB",
    "FieldRole", "FrameField", "FrameStructure", "Framer", "Deframer",
    "zadoff_chu", "schmidl_cox_preamble", "barker", "golay_pair", "m_sequence",
    # lazily loaded (they pull matplotlib, see D36):
    "plot_time", "plot_spectrum", "plot_welch", "plot_iq", "plot_kde",
    "plot_chain_profiling",
]

# PEP 562 lazy loading: importing comnumpy must not import matplotlib (D36).
_LAZY = {name: ".visualizers"
         for name in ("plot_time", "plot_spectrum", "plot_welch",
                      "plot_iq", "plot_kde", "plot_chain_profiling")}


def __getattr__(name):
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
