import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
from comnumpy._backend import fft, ifft, fftshift, ifftshift  # cupy-compatible (D3)
from comnumpy.core import Processor
from comnumpy.core.processors import AutoConcatenator
from comnumpy.exceptions import ShapeError
from .allocation import CarrierAllocation
from .utils import plot_carrier_allocation



@dataclass(slots=True)
class CyclicPrefixer(AutoConcatenator):
    r"""
    Processor for adding a cyclic prefix to combat multipath interference.

    Signal Model
    ------------
    The cyclic prefix is a copy of the last :math:`N_{cp}` samples of each
    block, prepended to the block to combat multipath interference and
    inter-symbol interference:

    .. math::
        y[n] = \left\{\begin{array}{cl}
        x[N - N_{cp} + n] & \text{for } 0 \le n < N_{cp},\\
        x[n - N_{cp}] & \text{for } N_{cp} \le n < N + N_{cp},
        \end{array}\right.

    where:

    * :math:`x[n]` is the input block of length :math:`N`,
    * :math:`y[n]` is the output block of length :math:`N + N_{cp}`,
    * :math:`N_{cp}` is the length of the cyclic prefix.

    Axes: *axis -1* -- the prefix is prepended along the block content
    axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Parameters
    ----------
    N_cp : int
        Length :math:`N_{cp}` of the cyclic prefix to be added. Must be a
        non-negative integer. Default is 10.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"cp adder"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> X = np.arange(10)
    >>> prefixer = CyclicPrefixer(N_cp=3)
    >>> print(prefixer(X))
    [7 8 9 0 1 2 3 4 5 6 7 8 9]
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> prefixer = CyclicPrefixer(N_cp=2)
    >>> print(prefixer(X))
    [[2 3 1 2 3]
     [5 6 4 5 6]]
    """
    N_cp: int = 10
    name: str = field(default="cp adder", kw_only=True)

    def __post_init__(self):
        if not (isinstance(self.N_cp, int) and self.N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")

    def prepare(self, X: np.ndarray):
        """Initialize the mask based on the signal shape"""
        input_length = X.shape[-1]
        output_mask_length = input_length + self.N_cp

        input_copy_mask = np.zeros(input_length)
        input_copy_mask[-self.N_cp:] = 1

        output_original_mask = np.zeros(output_mask_length)
        output_original_mask[self.N_cp:] = 1

        output_copy_mask = np.zeros(output_mask_length)
        output_copy_mask[:self.N_cp] = 1

        self.input_copy_mask = input_copy_mask.astype(bool)
        self.output_original_mask = output_original_mask.astype(bool)
        self.output_copy_mask = output_copy_mask.astype(bool)


@dataclass(slots=True)
class CyclicPrefixRemover(Processor):
    r"""
    Processor for removing a cyclic prefix from the input data.

    Signal Model
    ------------
    The removal of the cyclic prefix discards the first :math:`N_{cp}`
    samples of each block:

    .. math::
        y[n] = x[n + N_{cp}], \qquad 0 \le n < N,

    where:

    * :math:`x[n]` is the input block of length :math:`N + N_{cp}` that contains the cyclic prefix,
    * :math:`y[n]` is the output block of length :math:`N` after removing the cyclic prefix,
    * :math:`N_{cp}` is the length of the cyclic prefix.

    Axes: *axis -1* -- the prefix is removed along the block content
    axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Parameters
    ----------
    N_cp : int
        Length :math:`N_{cp}` of the cyclic prefix to be removed. Must be a
        non-negative integer.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"cp remover"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> X = np.arange(13)
    >>> remover = CyclicPrefixRemover(N_cp=3)
    >>> print(remover(X))
    [ 3  4  5  6  7  8  9 10 11 12]
    """
    N_cp: int
    name: str = field(default="cp remover", kw_only=True)

    def __post_init__(self):
        if not (isinstance(self.N_cp, int) and self.N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Remove the cyclic prefix along the block content axis
        return X[..., self.N_cp:]


@dataclass(slots=True)
class HermitianPrefixer(AutoConcatenator):
    r"""
    Processor for enforcing Hermitian symmetry on frequency-domain blocks, so that the IDFT output is real-valued.

    Signal Model
    ------------
    Each input block of length :math:`N` is extended to a block of length
    :math:`2(N+1)` whose spectrum is Hermitian symmetric: the DC and Nyquist
    subcarriers are set to zero and the conjugate mirror of the data is
    appended. When ``shift`` is False, the output is given by

    .. math::

        y[k] = \left\{\begin{array}{cl}
        0 &\text{for }k=0 \text{ and } k=N+1,\\
        x[k-1] &\text{for }k=1, \cdots, N,\\
        x^*[2N+1-k] &\text{for }k=N+2, \cdots, 2N+1,
        \end{array}\right.

    where:

    * :math:`x[k]` is the input block of length :math:`N`,
    * :math:`y[k]` is the Hermitian-symmetric output block of length :math:`2(N+1)`.

    When ``shift`` is True, the two halves are swapped (conjugate copy
    first), matching an fftshift-ed subcarrier ordering.

    Axes: *axis -1* -- the Hermitian extension is built along the block
    content axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Parameters
    ----------
    shift : bool, optional, keyword-only
        Whether to swap the original and conjugate halves. Default is False.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"hermitian prefixer"``.

    References
    ----------
    J. Armstrong, "OFDM for Optical Communications", Journal of Lightwave
    Technology, vol. 27, no. 3, pp. 189-204, 2009.

    Examples
    --------
    >>> X = np.arange(1, 4) + 1j*np.arange(1, 4)
    >>> prefixer = HermitianPrefixer()
    >>> print(prefixer(X))
    [0.+0.j 1.+1.j 2.+2.j 3.+3.j 0.+0.j 3.-3.j 2.-2.j 1.-1.j]
    >>> x = np.arange(1, 7) + 1j*np.arange(1, 7)
    >>> X = np.reshape(x, (2, 3))
    >>> prefixer = HermitianPrefixer(shift=True)
    >>> print(prefixer(X))
    [[0.+0.j 3.-3.j 2.-2.j 1.-1.j 0.+0.j 1.+1.j 2.+2.j 3.+3.j]
     [0.+0.j 6.-6.j 5.-5.j 4.-4.j 0.+0.j 4.+4.j 5.+5.j 6.+6.j]]
    """
    shift: bool = field(default=False, kw_only=True)
    name: str = field(default="hermitian prefixer", kw_only=True)

    def __post_init__(self):
        self.input_copy_mask = None
        self.output_original_mask = None
        self.output_copy_mask = None

    def prepare(self, X: np.ndarray):
        input_length = X.shape[-1]
        output_mask_length = 2*(input_length + 1)  # add 0 for the DC and nyquist componetns.

        input_copy_mask = np.ones(input_length)

        output_original_mask = np.zeros(output_mask_length)
        output_original_mask[1:input_length+1] = 1

        output_copy_mask = np.zeros(output_mask_length)
        output_copy_mask[-input_length:] = 1

        # construct copy mask for original data
        output_original_mask = np.zeros(output_mask_length)
        if self.shift:
            output_original_mask[-input_length:] = 1
        else:
            output_original_mask[1:input_length+1] = 1

        # construct copy mask for duplicated data
        output_copy_mask = np.zeros(output_mask_length)
        if self.shift:
            output_copy_mask[1:input_length+1] = 1
        else:
            output_copy_mask[-input_length:] = 1

        self.input_copy_mask = input_copy_mask.astype(bool)
        self.output_original_mask = output_original_mask.astype(bool)
        self.output_copy_mask = output_copy_mask.astype(bool)

    def process_copy(self, X: np.ndarray):
        return np.conjugate(np.flip(X, axis=-1))



@dataclass(slots=True)
class FFTProcessor(Processor):
    r"""
    Processor for performing Fast Fourier Transform (FFT) on the input data.

    Signal Model
    ------------
    The FFT computes the Discrete Fourier Transform (DFT), which maps a
    block of time-domain samples to its frequency-domain representation.
    With the default orthonormal normalization (``norm="ortho"``), the DFT
    of a block of length :math:`N` is given by:

    .. math::
        y[k] = \frac{1}{\sqrt{N}}\sum_{n=0}^{N-1} x[n] \cdot e^{-i 2 \pi k n / N}

    where:

    * :math:`x[n]` is the input block of time-domain samples,
    * :math:`y[k]` is the output value at subcarrier :math:`k`,
    * :math:`N` is the block length.

    Axes: *axis -1* -- the FFT changes the meaning of axis -1
    (time -> frequency), never its position (see CONVENTIONS.md).

    Parameters
    ----------
    shift : bool, optional
        If True, applies the FFT shift which swaps the low and high frequency components.
        Default is False.
    norm : {"ortho", "backward", "forward"}, optional, keyword-only
        Normalization mode for FFT. "ortho" means orthonormal FFT is computed.
        Default is "ortho".
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"fft"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 11.2.

    Examples
    --------
    >>> fft_processor = FFTProcessor()
    >>> x = np.ones(4)
    >>> print(np.round(np.abs(fft_processor(x)), 3))
    [2. 0. 0. 0.]
    """
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = field(default="ortho", kw_only=True)
    name: str = field(default="fft", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = fft(X, norm=self.norm, axis=-1)
        if self.shift:
            Y = fftshift(Y, axes=-1)
        return Y


@dataclass(slots=True)
class IFFTProcessor(Processor):
    r"""
    Processor for performing Inverse Fast Fourier Transform (IFFT) on the input data.

    Signal Model
    ------------
    The IFFT computes the Inverse Discrete Fourier Transform (IDFT), which
    maps a block of frequency-domain samples back to the time domain. With
    the default orthonormal normalization (``norm="ortho"``), the IDFT of a
    block of length :math:`N` is given by:

    .. math::
        y[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} x[k] \cdot e^{i 2 \pi k n / N}

    where:

    * :math:`x[k]` is the input value at subcarrier :math:`k`,
    * :math:`y[n]` is the output block of time-domain samples,
    * :math:`N` is the block length.

    Axes: *axis -1* -- the IFFT changes the meaning of axis -1
    (frequency -> time), never its position (see CONVENTIONS.md).

    Parameters
    ----------
    shift : bool, optional
        If True, applies the IFFT shift which swaps the low and high frequency components.
        Default is False.
    norm : {"ortho", "backward", "forward"}, optional, keyword-only
        Normalization mode for IFFT. "ortho" means orthonormal IFFT is computed.
        Default is "ortho".
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"ifft"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 11.2.

    Examples
    --------
    >>> ifft_processor = IFFTProcessor()
    >>> X = np.array([2.0, 0.0, 0.0, 0.0])
    >>> print(np.round(np.abs(ifft_processor(X)), 3))
    [1. 1. 1. 1.]
    """
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = field(default="ortho", kw_only=True)
    name: str = field(default="ifft", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.shift:
            X = ifftshift(X, axes=-1)
        Y = ifft(X, norm=self.norm, axis=-1)
        return Y


@dataclass(slots=True)
class CarrierAllocator(Processor):
    r"""
    Processor for allocating data to specific subcarriers.

    Signal Model
    ------------
    The Carrier Allocator assigns data to specific subcarriers based on a
    predefined subcarrier type array. It supports the insertion of pilot
    values on the pilot subcarriers and zeros on the null subcarriers:

    .. math::
        y[k] = \left\{\begin{array}{cl}
        x[m_k] & \text{if } s[k] = 1 \text{ (data)}\\
        p[j_k] & \text{if } s[k] = 2 \text{ (pilot)}\\
        0 & \text{if } s[k] = 0 \text{ (null)}\\
        \end{array}\right.

    where:

    * :math:`x[m]` is the input data vector, with :math:`m_k` indexing the data subcarriers in increasing subcarrier index :math:`k`,
    * :math:`p[j]` is the pilot value vector, with :math:`j_k` indexing the pilot subcarriers in increasing :math:`k`,
    * :math:`s[k]` is the subcarrier type array, where each element specifies the type of the :math:`k`-th subcarrier,
    * :math:`y[k]` is the output data vector over the full set of subcarriers.

    The indices :math:`m_k` and :math:`j_k` are determined by the positions in the `carrier_type` array where the values are 1 and 2, respectively.

    Axes: *declared axis* -- expects the Block layout ``(..., T, F)``
    (or ``(..., N_data)`` for a period-1 allocation) and validates it in
    ``prepare()`` (decision D18). The mask says *where*, the ``pilots``
    argument says *what*: transmitter and receiver share the same
    :class:`~comnumpy.ofdm.allocation.CarrierAllocation` object, which
    removes the "diverging masks" class of bugs.

    Parameters
    ----------
    carrier_type : CarrierAllocation or np.ndarray
        The allocation, providing the subcarrier type array :math:`s[k]`.
        A :class:`CarrierAllocation` is described in physical order and
        converted once with ``to_fft_order()`` (decision D16); a raw 1D
        array is taken as an FFT-order mask.
    pilots : np.ndarray or scalar, optional, keyword-only
        Pilot values :math:`p[j]`, one per PILOT subcarrier of an OFDM
        symbol (or a scalar broadcast to all of them). Default is an
        empty array.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"carrier allocator"``.

    Raises
    ------
    ShapeError
        If the last axis of the input does not hold ``N_data`` data
        subcarriers, or if a periodic (scattered) allocation is fed a 1D
        signal without the OFDM-symbol axis.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> carrier_type = np.array([1, 2, 0, 1, 2, 1])
    >>> pilots = np.array([-1, -1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, pilots=pilots)
    >>> X = np.array([1, 2, 3])
    >>> print(allocator(X))
    [ 1 -1  0  2 -1  3]

    >>> carrier_type = np.array([1, 0, 0, 1, 1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type)
    >>> X = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> print(allocator(X))
    [[1 0 0 4 7]
     [2 0 0 5 8]
     [3 0 0 6 9]]

    >>> from comnumpy.ofdm.allocation import get_allocation
    >>> allocator = CarrierAllocator(get_allocation("802.11a"), pilots=1.0)
    >>> Y = allocator(np.ones((10, 48)))
    >>> print(Y.shape)
    (10, 64)
    """
    carrier_type: object
    pilots: Optional[np.ndarray] = field(default=None, kw_only=True)
    name: str = field(default="carrier allocator", kw_only=True)
    # internal state (declared for slots, D40a)
    mask: np.ndarray = field(init=False, repr=False)
    N: int = field(init=False, repr=False)
    N_data: int = field(init=False, repr=False)
    period: int = field(init=False, repr=False)

    def __post_init__(self):
        self.initialize_masks()

    def initialize_masks(self):
        if isinstance(self.carrier_type, CarrierAllocation):
            # physical order -> FFT order, once and explicitly (D16)
            self.mask = self.carrier_type.to_fft_order()
        else:
            self.mask = np.atleast_2d(np.asarray(self.carrier_type))

        self.period = self.mask.shape[0]
        self.N = self.mask.shape[1]
        n_data = np.sum(self.mask == 1, axis=1)
        if not np.all(n_data == n_data[0]):
            raise ValueError(
                f"the number of data subcarriers must be constant over the "
                f"period, got {n_data.tolist()}")
        self.N_data = int(n_data[0])

        if self.pilots is None:
            self.pilots = np.array([])

    def _pilot_values(self, n_pilots):
        pilots = np.asarray(self.pilots)
        if pilots.ndim == 0:
            return np.broadcast_to(pilots, (n_pilots,))
        if len(pilots) != n_pilots:
            raise ValueError(
                f"incompatible number of pilots ({n_pilots} needed for this "
                f"OFDM symbol, {len(pilots)} provided)")
        return pilots

    def set_carrier_type(self, carrier_type):
        self.carrier_type = carrier_type
        self.initialize_masks()

    def prepare(self, X: np.ndarray):
        if X.shape[-1] != self.N_data:
            raise ShapeError(
                f"CarrierAllocator expects (..., T, {self.N_data}) "
                f"(N_data={self.N_data} data subcarriers per OFDM symbol), "
                f"got {X.shape} -- fix Serial2Parallel(N_sub={self.N_data}) "
                f"or the allocation.")
        if self.period > 1 and X.ndim < 2:
            raise ShapeError(
                f"CarrierAllocator with a period-{self.period} allocation "
                f"expects a Block layout (..., T, {self.N_data}), got 1D "
                f"{X.shape} -- add the OFDM-symbol axis with Serial2Parallel.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        new_shape = X.shape[:-1] + (self.N,)
        Y = np.zeros(new_shape, dtype=X.dtype)

        if self.period == 1:
            row = self.mask[0]
            Y[..., row == 1] = X
            n_pilots = int(np.sum(row == 2))
            if n_pilots > 0:
                Y[..., row == 2] = self._pilot_values(n_pilots)
        else:
            # scattered pattern: mask row l applies to OFDM symbol t = l mod period
            T = X.shape[-2]
            for t in range(T):
                row = self.mask[t % self.period]
                Y[..., t, row == 1] = X[..., t, :]
                n_pilots = int(np.sum(row == 2))
                if n_pilots > 0:
                    Y[..., t, row == 2] = self._pilot_values(n_pilots)
        return Y

    def plot(self, ax=None, shift=False):
        """
        Plot the carrier allocation; returns the axis (decision D25).
        """
        if isinstance(self.carrier_type, CarrierAllocation):
            return self.carrier_type.plot(ax=ax)
        return plot_carrier_allocation(np.ravel(self.mask), ax=ax, shift=shift, title="Carrier Allocation")


@dataclass(slots=True)
class CarrierExtractor(Processor):
    r"""
    Processor for extracting data from specific subcarriers.

    Signal Model
    ------------
    The Carrier Extractor extracts the data subcarriers based on a
    predefined subcarrier type array, undoing the mapping of the
    :class:`CarrierAllocator`:

    .. math::
        y[m] = x[k_m]

    where:

    * :math:`x[k]` is the input data over the full set of subcarriers,
    * :math:`k_m` is the index of the :math:`m`-th subcarrier of type 1 (data subcarrier) in the input vector,
    * :math:`y[m]` is the output data vector.

    The indices :math:`k_m` are determined by the positions in the `carrier_type` array where the value is 1.
    The content of the pilot subcarriers (type 2) can be recorded with ``pilot_recorder``.

    Axes: *declared axis* -- expects the Block layout ``(..., T, N_fft)``
    (or ``(..., N_fft)`` for a period-1 allocation) and validates it in
    ``prepare()`` (decision D18). Share the same
    :class:`~comnumpy.ofdm.allocation.CarrierAllocation` object with the
    transmitter's :class:`CarrierAllocator`.

    Parameters
    ----------
    carrier_type : CarrierAllocation or np.ndarray
        The allocation, providing the subcarrier type array used to find
        the data indices :math:`k_m`. A :class:`CarrierAllocation` is
        described in physical order and converted once with
        ``to_fft_order()`` (decision D16); a raw 1D array is taken as an
        FFT-order mask.
    pilot_recorder : callable, optional, keyword-only
        Function to record the content associated to pilot values if required. Default is None.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"carrier extractor"``.

    Raises
    ------
    ShapeError
        If the last axis of the input does not hold ``N_fft`` subcarriers,
        or if a periodic (scattered) allocation is fed a 1D signal without
        the OFDM-symbol axis.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> from comnumpy.core import Recorder
    >>> carrier_type = np.array([1, 2, 0, 1, 2, 1])
    >>> pilot_recorder = Recorder()
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, pilots=np.array([-1, -2]))
    >>> extractor = CarrierExtractor(carrier_type=carrier_type, pilot_recorder=pilot_recorder)
    >>> Z = allocator(np.array([1, 2, 3]))
    >>> print(extractor(Z))
    [1 2 3]
    >>> print(pilot_recorder.get_data())
    [-1 -2]

    >>> carrier_type = np.array([1, 0, 0, 1, 1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type)
    >>> extractor = CarrierExtractor(carrier_type=carrier_type)
    >>> X = np.array([[1, 4, 7], [2, 5, 8]])
    >>> Z = allocator(X)
    >>> print(Z)
    [[1 0 0 4 7]
     [2 0 0 5 8]]
    >>> print(extractor(Z))
    [[1 4 7]
     [2 5 8]]
    """
    carrier_type: object
    pilot_recorder: Optional[Callable] = field(default=None, kw_only=True)
    name: str = field(default="carrier extractor", kw_only=True)
    # internal state (declared for slots, D40a)
    mask: np.ndarray = field(init=False, repr=False)
    period: int = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.carrier_type, CarrierAllocation):
            self.mask = self.carrier_type.to_fft_order()
        else:
            self.mask = np.atleast_2d(np.asarray(self.carrier_type))
        self.period = self.mask.shape[0]

    def prepare(self, X: np.ndarray):
        N_fft = self.mask.shape[1]
        if X.shape[-1] != N_fft:
            raise ShapeError(
                f"CarrierExtractor expects (..., T, {N_fft}) "
                f"(N_fft={N_fft} subcarriers), got {X.shape} -- use the "
                f"same allocation as the transmitter's CarrierAllocator.")
        if self.period > 1 and X.ndim < 2:
            raise ShapeError(
                f"CarrierExtractor with a period-{self.period} allocation "
                f"expects a Block layout (..., T, {N_fft}), got 1D {X.shape}.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.period == 1:
            row = self.mask[0]
            X_data = X[..., row == 1]
            X_pilots = X[..., row == 2]
        else:
            T = X.shape[-2]
            data_parts, pilot_parts = [], []
            for t_index in range(T):
                row = self.mask[t_index % self.period]
                data_parts.append(X[..., t_index, row == 1])
                pilot_parts.append(X[..., t_index, row == 2])
            X_data = np.stack(data_parts, axis=-2)
            # pilot counts may differ per symbol: keep the raw list then
            X_pilots = pilot_parts

        # Save pilots if needed
        if self.pilot_recorder:
            self.pilot_recorder(X_pilots)

        return X_data

    def plot(self, ax=None, shift=False):
        """
        Plot the carrier allocation; returns the axis (decision D25).
        """
        if isinstance(self.carrier_type, CarrierAllocation):
            return self.carrier_type.plot(ax=ax)
        return plot_carrier_allocation(np.ravel(self.mask), ax=ax, shift=shift, title="Carrier Allocation")

