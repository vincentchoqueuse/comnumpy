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
    r"""OFDM transmitter chain: allocation, IFFT, cyclic prefix, serialization.

    Signal Model
    ------------
    The serial data symbols are split into blocks, allocated to the data
    subcarriers (pilot and null subcarriers are filled according to the
    allocation mask), and each allocated block :math:`X_t[k]` of length
    :math:`N` is brought to the time domain with the orthonormal IDFT:

    .. math::

        x_t[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X_t[k] \,
        e^{i 2 \pi k n / N}, \qquad n = 0, \dots, N-1

    A cyclic prefix of length :math:`N_{cp}` (the last :math:`N_{cp}`
    samples of the block) is prepended to each block, and the extended
    blocks are serialized into the time-domain signal :math:`y[m]`.

    Axes: *declared axis* -- consumes the serial axis -1 (length
    :math:`T \, N_{data}`), produces a serial time-domain signal of
    length :math:`T (N + N_{cp})` on axis -1.

    Parameters
    ----------
    N_carrier_data : int
        Number of data subcarriers :math:`N_{data}` per OFDM symbol.
    N_cp : int
        Length :math:`N_{cp}` of the cyclic prefix.
    carrier_type : np.ndarray or list, optional, keyword-only
        Subcarrier allocation mask (e.g. data, pilot, null), passed to
        :class:`~comnumpy.ofdm.processors.CarrierAllocator`. Default is
        an all-data mask of length ``N_carrier_data``.
    pilots : np.ndarray or list, optional, keyword-only
        Pilot values, one per pilot subcarrier of the allocation mask.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"ofdm_transmitter"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> transmitter = OFDMTransmitter(4, 2)
    >>> x = np.arange(8) + 0j  # 2 OFDM symbols of 4 data subcarriers
    >>> y = transmitter(x)
    >>> print(y.shape)  # 2 blocks of length 4 + 2
    (12,)
    """
    N_carrier_data: int
    N_cp: int
    carrier_type: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    pilots: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    name: str = field(default="ofdm_transmitter", kw_only=True)
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
    r"""OFDM receiver chain: prefix removal, FFT, equalization, extraction.

    Signal Model
    ------------
    The serial received signal is split into blocks of length
    :math:`N + N_{cp}`, the cyclic prefix is discarded, and each block
    :math:`r_t[n]` is brought to the frequency domain with the
    orthonormal DFT before one-tap zero-forcing equalization:

    .. math::

        R_t[k] = \frac{1}{\sqrt{N}} \sum_{n=0}^{N-1} r_t[n] \,
        e^{-i 2 \pi k n / N}, \qquad
        Y_t[k] = \frac{R_t[k]}{H[k]}

    where :math:`H[k]` is the :math:`N`-point DFT of the channel impulse
    response :math:`h[l]`. The data subcarriers are then extracted
    according to the allocation mask and serialized.

    Axes: *declared axis* -- consumes the serial axis -1 (length
    :math:`T (N + N_{cp})`), produces the serial data symbols of length
    :math:`T \, N_{data}` on axis -1.

    Parameters
    ----------
    N_carrier_data : int
        Number of data subcarriers :math:`N_{data}` per OFDM symbol.
    N_cp : int
        Length :math:`N_{cp}` of the cyclic prefix.
    h : np.ndarray or list, keyword-only
        Channel impulse response :math:`h[l]` used by the frequency
        domain equalizer. Default is the ideal channel ``[1.0]``.
    carrier_type : np.ndarray or list, optional, keyword-only
        Subcarrier allocation mask (e.g. data, pilot, null), passed to
        :class:`~comnumpy.ofdm.processors.CarrierExtractor`. Default is
        an all-data mask of length ``N_carrier_data``.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"ofdm_receiver"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> x = np.arange(8) + 0j
    >>> y = OFDMTransmitter(4, 2)(x)
    >>> x_est = OFDMReceiver(4, 2)(y)
    >>> print(np.round(x_est.real, 6))
    [0. 1. 2. 3. 4. 5. 6. 7.]
    """
    N_carrier_data: int
    N_cp: int
    h: Union[np.ndarray, list] = field(default_factory=lambda: np.array([1.0]), kw_only=True)
    carrier_type: Optional[Union[np.ndarray, list]] = field(default=None, kw_only=True)
    name: str = field(default="ofdm_receiver", kw_only=True)
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
