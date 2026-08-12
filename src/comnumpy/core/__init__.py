from importlib import import_module
from typing import TYPE_CHECKING, Any

from .generics import Processor, Sequential
from .generators import SymbolGenerator, GaussianGenerator
from .mappers import SymbolMapper, SymbolDemapper
from .channels import AWGN, FIRChannel, TappedDelayLineChannel
from .capacity import (awgn_capacity, constellation_capacity, bicm_capacity,
                       rayleigh_ergodic_capacity, mimo_ergodic_capacity,
                       outage_capacity, waterfilling)
from .information import compute_gmi, compute_llr, compute_mi, compute_ngmi
from .shaping import (maxwell_boltzmann, distribution_entropy,
                      composition_from_distribution, shaping_gain_dB,
                      ConstantCompositionMatcher, SphereShaper,
                      DistributionMatcher, DistributionDematcher)
from .fading import (DopplerSpectrum, PowerDelayProfile, rayleigh_process,
                     get_delay_profile, register_delay_profile,
                     available_delay_profiles)
from .filters import SRRCFilter, BWFilter
from .processors import Upsampler, Downsampler, Serial2Parallel, Parallel2Serial, Amplifier, DataExtractor
from .metrics import compute_ser, compute_ber, compute_evm, compute_metric_awgn_theo, compute_ccdf, signal_report
from .utils import (Constellation, get_alphabet, hard_projector,
                    ebn0_to_snr_dB, esn0_to_snr_dB)
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
    "compute_llr", "compute_mi", "compute_gmi", "compute_ngmi",
    "maxwell_boltzmann", "distribution_entropy",
    "composition_from_distribution", "shaping_gain_dB",
    "ConstantCompositionMatcher", "SphereShaper",
    "DistributionMatcher", "DistributionDematcher",
    "DopplerSpectrum", "PowerDelayProfile", "rayleigh_process",
    "get_delay_profile", "register_delay_profile", "available_delay_profiles",
    "SRRCFilter", "BWFilter",
    "Upsampler", "Downsampler", "Serial2Parallel", "Parallel2Serial",
    "Amplifier", "DataExtractor",
    "compute_ser", "compute_ber", "compute_evm", "compute_metric_awgn_theo", "compute_ccdf",
    "signal_report",
    "Constellation", "get_alphabet", "hard_projector", "ebn0_to_snr_dB",
    "esn0_to_snr_dB",
    "FieldRole", "FrameField", "FrameStructure", "Framer", "Deframer",
    "zadoff_chu", "schmidl_cox_preamble", "barker", "golay_pair", "m_sequence",
    # lazily loaded (they pull matplotlib, see D36):
    "plot_time", "plot_spectrum", "plot_welch", "plot_iq", "plot_kde",
    "plot_error_rate", "plot_chain_profiling",
]

if TYPE_CHECKING:
    # the names below are resolved at runtime by __getattr__; importing
    # them here would pull matplotlib, so the type checker gets them and
    # the interpreter does not
    from .visualizers import (plot_chain_profiling, plot_error_rate,
                              plot_iq, plot_kde, plot_spectrum, plot_time,
                              plot_welch)

# PEP 562 lazy loading: importing comnumpy must not import matplotlib (D36).
_LAZY = {name: ".visualizers"
         for name in ("plot_time", "plot_spectrum", "plot_welch",
                      "plot_iq", "plot_kde", "plot_error_rate",
                      "plot_chain_profiling")}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
