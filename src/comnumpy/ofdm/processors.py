import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
from scipy.fft import fft, ifft, fftshift, ifftshift
from comnumpy.core import Processor
from comnumpy.core.processors import AutoConcatenator
from .utils import plot_carrier_allocation



@dataclass(slots=True)
class CyclicPrefixer(AutoConcatenator):
    r"""
    Processor for adding a cyclic prefix to combat multi-path interference.

    Signal Model
    ------------
    The cyclic prefix is a portion of the signal that is prepended to the original data to combat
    multipath interference and inter-symbol interference in communication systems. The addition of
    the cyclic prefix involves copying the last `N_cp` samples of the original data and placing
    them at the beginning along the specified axis.

    Mathematically, if `X` is the original input signal, the operation can be described as:

    .. math::
        \mathbf{y}[n] = \begin{bmatrix}
        \mathbf{0}_{N_{cp},N-N_{cp}} & \mathbf{I}_{N_{cp},N_{cp}} \\
        \mathbf{I}_{N-N_{cp},N-N_{cp}} & \mathbf{0}_{N-N_{cp},N_{cp}}\\
        \mathbf{0}_{N_{cp},N-N_{cp}} & \mathbf{I}_{N_{cp},N_{cp}}\\
        \end{bmatrix}
        \mathbf{x}[n]

    where:

    * :math:`N_{cp}` is the length of the cyclic prefix,
    * :math:`\mathbf{x}[n]` is the input signal of size :math:`N`,
    * :math:`\mathbf{y}[n]` is the output signal of size :math:`N+N_{cp}` after adding the cyclic prefix.

    Axes: *axis -1* -- the prefix is prepended along the block content
    axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Attributes
    ----------
    N_cp : int
        Length of the cyclic prefix to be added. Must be a non-negative integer.

    Example 1
    ---------

    >>> X = np.arange(10)
    >>> prefixer = CyclicPrefixer(N_cp=3)
    >>> print(prefixer(X))
    [7 8 9 0 1 2 3 4 5 6 7 8 9]

    Example 2
    ---------

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
    The cyclic prefix is a portion of the signal that is prepended to the original data to combat
    multipath interference and inter-symbol interference in communication systems. The removal of
    the cyclic prefix involves discarding the first `N_cp` samples along the specified axis.

    Mathematically, if `X` is the input signal with a cyclic prefix, the operation can be described as:

    .. math::
        \mathbf{y}[n] =\begin{bmatrix}
        \mathbf{0}_{N,N_{cp}} & \mathbf{I}_{N,N}
        \end{bmatrix}
        \mathbf{x}[n]

    where:

    * :math:`N_{cp}` is the length of the cyclic prefix,
    * :math:`\mathbf{x}[n]` is the input signal of size :math:`N+N_{cp}` that contains the cyclic prefix,
    * :math:`\mathbf{y}[n]` is the output signal of size :math:`N` after removing the cyclic prefix.

    Axes: *axis -1* -- the prefix is removed along the block content
    axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Attributes
    ----------
    N_cp : int
        Length of the cyclic prefix to be removed. Must be a non-negative integer.
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
    Processor for preparing data to enforce Hermitian symmetry, useful in signal processing applications.

    Signal Model
    ------------
    The HermitianPrefixer generates masks that can be used to enforce Hermitian symmetry on a given signal.

    Mathematically, the masks are designed to handle the input signal `X` and prepare it for Hermitian operations.
    The process involves creating masks that identify the portions of the signal to be copied and transformed. When shift is false, the output
    is given by

    .. math ::

        y[n] = \left\{\begin{array}{cl}
        0 &\text{if }n=0, N+2,\\
        x[n] &\text{for }n=1, \cdots, N+1,\\
        x^*[n-N+2)] &\text{for }n=N+2, \cdots, 2N+1.
        \end{array}\right.

    Axes: *axis -1* -- the Hermitian extension is built along the block
    content axis (Block layout ``(..., T, F)``, see CONVENTIONS.md).

    Attributes
    ----------
    shift : bool, keyword-only
        Whether to apply a shift to the masks. Default is False.
    name : str, keyword-only
        The name of the prefixer. Default is "hermitian prefixer".

    Example 1
    ---------

    >>> X = np.arange(1, 4) + 1j*np.arange(1, 4)
    >>> prefixer = HermitianPrefixer()
    >>> print(prefixer(X))
    [0.+0.j 1.+1.j 2.+2.j 3.+3.j 0.+0.j 3.-3.j 2.-2.j 1.-1.j]

    Example 2
    ---------

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
    The Fast Fourier Transform (FFT) is an algorithm to compute the Discrete Fourier Transform (DFT)
    and its inverse efficiently. The DFT transforms a sequence of values in the time domain into
    a sequence of values in the frequency domain.

    Mathematically, the DFT of a sequence :math:`x[n]` of length :math:`N` is given by:

    .. math::
        y[k] = \frac{1}{\sqrt{N}}\sum_{l=0}^{N-1} x[l] \cdot e^{-i 2 \pi k l / N}


    The FFT operation can be represented in matrix form as:

    .. math::
        \mathbf{y}[n] = \mathbf{W} \mathbf{x}[n]

    * :math:`\mathbf{W}` is the DFT matrix of size \( N \times N \),
    * :math:`\mathbf{x}[n]` is the input vector of time-domain samples,
    * :math:`\mathbf{y}[n]` is the output vector of frequency-domain samples.

    Axes: *axis -1* -- the FFT changes the meaning of axis -1
    (time -> frequency), never its position (see CONVENTIONS.md).

    Attributes
    ----------
    shift : bool, optional
        If True, applies the FFT shift which swaps the low and high frequency components.
        Default is False.
    norm : {"ortho", "backward", "forward"}, optional
        Normalization mode for FFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
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
    The Inverse Fast Fourier Transform (IFFT) is an algorithm to compute the Inverse Discrete Fourier Transform (IDFT)
    efficiently. The IDFT transforms a sequence of values in the frequency domain back into the time domain.

    Mathematically, the IDFT of a sequence :math:`x[k]` of length :math:`N` is given by:

    .. math::
        y[l] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} x[k] \cdot e^{i 2 \pi k n / N}

    The IFFT operation can be represented in matrix form as:

    .. math::
        \mathbf{y}[n] = \mathbf{W}^{H} \mathbf{x}[n]

    * :math:`\mathbf{W}^{H}` is the inverse DFT matrix of size :math:`N \times N`,
    * :math:`\mathbf{x}[n]` is the input vector of frequency-domain samples,
    * :math:`\mathbf{y}[n]` is the output vector of time-domain samples.

    Axes: *axis -1* -- the IFFT changes the meaning of axis -1
    (frequency -> time), never its position (see CONVENTIONS.md).

    Attributes
    ----------
    shift : bool, optional
        If True, applies the IFFT shift which swaps the low and high frequency components.
        Default is False.
    norm : {"ortho", "backward", "forward"}, optional
        Normalization mode for IFFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
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
    The Carrier Allocator assigns data to specific subcarriers based on a predefined subcarrier type array.
    It supports the insertion of pilot values and ensures Hermitian symmetry for certain subcarriers.

    Mathematically, the allocation can be described as:

    .. math::
        y[n] = \left\{\begin{array}{cl}
        x[m] & \text{if } s[n] = 1 \\
        p[k] & \text{if } s[n] = 2 \\
        0 & \text{if } s[n] = 0 \\
        \end{array}\right.

    where:

    * :math:`x[m]` is the input data vector, with :math:`m` indexing the data subcarriers,
    * :math:`p[k]` is the pilot value vector, with :math:`k` indexing the pilot subcarriers,
    * :math:`s[n]` is the subcarrier type array, where each element specifies the type of the :math:`n`-th subcarrier,
    * :math:`y[n]` is the output data vector, with allocated subcarriers.

    The indices :math:`m` and :math:`k` are determined by the positions in the `carrier_type` array where the values are 1 and 2, respectively.

    Attributes
    ----------
    carrier_type : np.ndarray
        Array specifying the type of each subcarrier.
    pilots : np.ndarray, optional
        Array of pilot values to be inserted into the subcarriers. Default is an empty array.
    axis : int, optional
        Axis along which to allocate subcarriers. Default is -1, the
        block content axis of the Block layout ``(..., T, F)``.

    Example 1
    ---------
    >>> carrier_type = np.array([1, 2, 0, 1, 2, 1])
    >>> pilots = np.array([-1, -1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, pilots=pilots)
    >>> X = np.array([1, 2, 3])
    >>> print(allocator(X))
    [ 1 -1  0  2 -1  3]

    Example 2
    ---------
    >>> carrier_type = np.array([1, 0, 0, 1, 1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type)
    >>> X = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> print(allocator(X))
    [[1 0 0 4 7]
     [2 0 0 5 8]
     [3 0 0 6 9]]
    """
    carrier_type: np.ndarray
    pilots: Optional[np.ndarray] = field(default=None, kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="carrier allocator", kw_only=True)
    # internal state (declared for slots, D40a)
    N: int = field(init=False, repr=False)
    N_data: int = field(init=False, repr=False)
    N_pilots: int = field(init=False, repr=False)
    index_data: np.ndarray = field(init=False, repr=False)
    index_pilots: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.initialize_masks()

    def initialize_masks(self):
        self.N = len(self.carrier_type)
        self.N_data = np.sum(self.carrier_type == 1)
        self.N_pilots = np.sum(self.carrier_type == 2)

        # Check carrier dimension
        if self.pilots is None:
            self.pilots = np.array([])

        if self.N_pilots != len(self.pilots):
            raise ValueError(f"Incompatible number of pilots ({self.N_pilots} needed, {len(self.pilots)} provided)")

        # Initialize vector
        self.index_data = (self.carrier_type == 1)
        self.index_pilots = (self.carrier_type == 2)

    def set_carrier_type(self, carrier_type):
        self.carrier_type = carrier_type
        self.initialize_masks()

    def forward(self, X: np.ndarray) -> np.ndarray:
        # validate size
        N_data_validation = X.shape[self.axis]
        if self.N_data != N_data_validation:
            raise ValueError(f"Incompatible number of subcarriers ({N_data_validation} provided in input data, {self.N_data} expected in carrier type)")

        # Initialize the output array
        new_shape = list(X.shape)
        new_shape[self.axis] = self.N
        Y = np.zeros(new_shape, dtype=X.dtype)

        # Create a slicing object to index along the specified axis
        slices = [slice(None)] * len(X.shape)

        # Assign data
        slices[self.axis] = self.index_data
        Y[tuple(slices)] = X

        # Assign pilots (reshaped to broadcast along the allocation axis,
        # so that 1D and N-D inputs are both supported)
        if self.N_pilots > 0:
            slices[self.axis] = self.index_pilots
            pilot_shape = [1] * Y.ndim
            pilot_shape[self.axis % Y.ndim] = self.N_pilots
            Y[tuple(slices)] = np.asarray(self.pilots).reshape(pilot_shape)

        return Y

    def plot(self, shift=False):
        """
        Plot the carrier allocation
        """
        plot_carrier_allocation(self.carrier_type, shift=shift, title="Carrier Allocation")


@dataclass(slots=True)
class CarrierExtractor(Processor):
    r"""
    Processor for extracting data from specific subcarriers.

    Signal Model
    ------------
    The Carrier Extractor extracts data from specific subcarriers based on a predefined subcarrier type array.
    It supports the extraction of pilot values and ensures Hermitian symmetry for certain subcarriers.

    Mathematically, the extraction can be described as:

    .. math::
        y[n] = x[k_n]

    where:

    * :math:`x[n]` is the input data,
    * :math:`k_m` is the index of the :math:`m`-th subcarrier of type 1 (data subcarrier) in the input vector,
    * :math:`y[n]` is the output data.

    The indices :math:`k_m` are determined by the positions in the `carrier_type` array where the value is 1.

    Attributes
    ----------
    carrier_type : np.ndarray
        Array specifying the type of each subcarrier.
    pilot_recorder : callable, optional
        Function to record the content associated to pilot values if required. Default is None.
    axis : int, optional
        Axis along which to extract subcarriers. Default is -1, the
        block content axis of the Block layout ``(..., T, F)``.

    Example 1
    ---------
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

    Example 2
    ---------
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
    carrier_type: np.ndarray
    pilot_recorder: Optional[Callable] = field(default=None, kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="carrier extractor", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Create a slicing object to index along the specified axis
        slices = [slice(None)] * X.ndim

        # Extract data
        slices[self.axis] = self.carrier_type == 1
        X_data = X[tuple(slices)]

        # Extract pilots
        slices[self.axis] = self.carrier_type == 2
        X_pilots = X[tuple(slices)]

        # Save pilot if needed
        if self.pilot_recorder:
            self.pilot_recorder(X_pilots)

        return X_data

    def plot(self, shift=False):
        """
        Plot the carrier allocation
        """
        plot_carrier_allocation(self.carrier_type, shift=shift, title="Carrier Allocation")

