from dataclasses import dataclass, field
import numpy as np
from typing import Union, Optional
from comnumpy.core import Processor, Sequential
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.ofdm.processors import (
    CarrierAllocator, CarrierExtractor, IFFTProcessor, FFTProcessor,
    CyclicPrefixer, CyclicPrefixRemover
)
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer


@dataclass(slots=True)
class OFDMTransmitter(Processor):
    r"""
    OFDM transmitter processing chain.

    This class encapsulates the sequence of operations to generate
    an OFDM time-domain transmit signal from input data symbols.

    Processing steps:
    - Serial to parallel conversion
    - Allocation of data and pilot symbols to OFDM subcarriers
    - IFFT to move from frequency to time domain
    - Addition of cyclic prefix to mitigate inter-symbol interference
    - Parallel to serial conversion to output the time-domain signal

    Parameters
    ----------
    N_carrier_data : int
        Number of carriers (subcarriers) for data
    carrier_type : np.ndarray or list
        Array indicating the carrier allocation types (e.g., data, pilot, null).
    N_cp : int
        Length of the cyclic prefix.
    pilots : list or np.ndarray
        Pilot symbols to be inserted on pilot subcarriers.


    Attributes
    ----------
    chain : Processor
        Sequential processor implementing the transmitter chain.
    """
    N_carrier_data: int
    N_cp: int
    carrier_type: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    pilots: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    chain: Processor = field(init=False, repr=False)

    def __post_init__(self):

        if self.carrier_type is None:
            self.carrier_type = np.ones(self.N_carrier_data, dtype=int)

        self.chain = Sequential([
            Serial2Parallel(self.N_carrier_data),
            CarrierAllocator(carrier_type=self.carrier_type, pilots=self.pilots, name="carrier_allocator_tx"),
            IFFTProcessor(),
            CyclicPrefixer(self.N_cp),
            Parallel2Serial()
        ])

    def forward(self, X):
        return self.chain(X)


@dataclass(slots=True)
class OFDMReceiver(Processor):
    r"""
    OFDM receiver processing chain.

    This class encapsulates the sequence of operations to recover
    data symbols from a received OFDM time-domain signal.

    Processing steps:
    - Serial to parallel conversion
    - Cyclic prefix removal
    - FFT to transform to frequency domain
    - Frequency-domain equalization using channel response
    - Extraction of data and pilot subcarriers
    - Parallel to serial conversion to output recovered data symbols

    Parameters
    ----------
    N_carrier_data : int
        Number of carriers (subcarriers) for data
    carrier_type : np.ndarray or list
        Array indicating the carrier allocation types (e.g., data, pilot, null).
    N_cp : int
        Length of the cyclic prefix.
    h : np.ndarray or list
        Channel impulse or frequency response used for equalization.

    Attributes
    ----------
    chain : Processor
        Sequential processor implementing the receiver chain.
    """
    N_carrier_data: int
    N_cp: int
    h: Union[np.ndarray, list] = field(default_factory=lambda: np.array([1.0]), kw_only=True)
    carrier_type: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    chain: Processor = field(init=False, repr=False)

    def __post_init__(self):
        if self.carrier_type is None:
            self.carrier_type = np.ones(self.N_carrier_data, dtype=int)

        N_carriers = len(self.carrier_type)
        self.chain = Sequential([
            Serial2Parallel(N_carriers + self.N_cp),
            CyclicPrefixRemover(self.N_cp),
            FFTProcessor(),
            FrequencyDomainEqualizer(h=self.h),
            CarrierExtractor(self.carrier_type),
            Parallel2Serial()
        ])

    def forward(self, X):
        return self.chain(X)
