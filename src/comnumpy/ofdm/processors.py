import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Callable
from scipy.fft import fft, ifft, fftshift, ifftshift
from comnumpy.core import Sequential, Processor
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial, AutoConcatenator
from comnumpy.core.validators import validate_real
from .utils import plot_carrier_allocation



@dataclass
class OFDMTransmitter(Processor):
    """
    OFDM Transmitter for converting digital data into an OFDM signal.

    Signal Model
    ------------
    The OFDM Transmitter processes input data through a series of steps:

    1. Serial to Parallel Conversion: Converts the serial data stream into parallel sub-streams.
    2. Carrier Allocation: Allocates data to specific subcarriers.
    3. Inverse Fast Fourier Transform (IFFT): Transforms the frequency-domain data into time-domain signals.
    4. Cyclic Prefixing: Adds a cyclic prefix to combat multipath interference.
    5. Parallel to Serial Conversion: Converts the parallel streams back into a serial data stream.

    Attributes
    ----------
    nb_carrier_data : int
        Number of data carriers used in OFDM.
    carrier_type : np.ndarray
        Array specifying the carrier type.
    nb_cp : int
        Number of samples in the Cyclic Prefix (CP).
    chain : Sequential
        Processing chain encapsulating the sequence of OFDM transmission steps.
    """
    nb_carrier_data: int
    carrier_type: np.ndarray
    nb_cp: int
    chain: Sequential = None

    def __post_init__(self):
        self.chain = Sequential([
            Serial2Parallel(self.nb_carrier_data),
            CarrierAllocator(self.carrier_type),
            IFFTProcessor(),
            CyclicPrefixer(self.nb_cp),
            Parallel2Serial()
        ])

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.chain(x)


@dataclass
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

    Attributes
    ----------
    N_cp : int
        Length of the cyclic prefix to be added. Must be a non-negative integer.
    axis : int, optional
        Axis along which to add the cyclic prefix. Default is the first dimension


    Example 1
    ---------

    >>> X = np.arange(10)
    >>> prefixer = CyclicPrefixer(N_cp=3)
    >>> Y = prefixer(X)
    [7 8 9 0 1 2 3 4 5 6 7 8 9]

    Example 2
    ---------

    >>> X = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> prefixer = CyclicPrefixer(N_cp=2)
    >>> Y = prefixer(X)
    [[2 5 8]
    [3 6 9]
    [1 4 7]
    [2 5 8]
    [3 6 9]]      
    """
    N_cp: int = 10
    axis: int = 0
    name: str = "cp adder"

    def __post_init__(self):
        if not (isinstance(self.N_cp, int) and self.N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")

    def prepare(self, X: np.ndarray):
        """Initialize the mask based on the signal shape"""
        input_length = X.shape[self.axis]
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


@dataclass
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

    Attributes
    ----------
    N_cp : int
        Length of the cyclic prefix to be removed. Must be a non-negative integer.
    axis : int, optional
        Axis along which to remove the cyclic prefix. Default is the first dimension
    """
    N_cp: int
    axis: int = 0
    name: str = "cp remover"

    def __post_init__(self):
        if not (isinstance(self.N_cp, int) and self.N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Define slices for removing the cyclic prefix
        slices = [slice(None)] * X.ndim
        data_slice = slices.copy()
        data_slice[self.axis] = slice(self.N_cp, None)

        # Remove the cyclic prefix
        return X[tuple(data_slice)]


@dataclass
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

    Attributes
    ----------
    axis : int
        The axis along which to apply the Hermitian operation. Default is 0.
    shift : bool
        Whether to apply a shift to the masks. Default is False.
    name : str
        The name of the prefixer. Default is "hermitian prefixer".

    Example 1
    ---------

    >>> X = np.arange(1, 4) + 1j*np.arange(1, 4)
    >>> prefixer = HermitianPrefixer()
    >>> Y = prefixer(X)
    [0.+0.j 1.+1.j 2.+2.j 3.+3.j 0.+0.j 3.-3.j 2.-2.j 1.-1.j]

    Example 2
    ---------

    >>> x = np.arange(1, 7) + 1j*np.arange(1, 7)
    >>> X = np.reshape(x, (3, 2), order="F")
    >>> prefixer = HermitianPrefixer(shift=True)
    >>> Y = prefixer(X)
    [[0.+0.j 0.+0.j]
    [3.-3.j 6.-6.j]
    [2.-2.j 5.-5.j]
    [1.-1.j 4.-4.j]
    [0.+0.j 0.+0.j]
    [1.+1.j 4.+4.j]
    [2.+2.j 5.+5.j]
    [3.+3.j 6.+6.j]]
    """
    axis: int = 0
    shift: bool = False
    name: str = "hermitian prefixer"

    def __post_init__(self):
        self.input_copy_mask = None
        self.output_original_mask = None
        self.output_copy_mask = None

    def prepare(self, X: np.ndarray):
        input_length = X.shape[self.axis]
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
        return np.conjugate(np.flip(X, axis=self.axis))
    


@dataclass
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

    Attributes
    ----------
    axis : int, optional
        Axis along which to perform the FFT. Default is the first axis
    shift : bool, optional
        If True, applies the FFT shift which swaps the low and high frequency components.
        Default is False.
    norm : {"ortho", "backward", "forward"}, optional
        Normalization mode for FFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
    """
    axis: int = 0
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = "ortho"
    name: str = "fft"

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = fft(X, norm=self.norm, axis=self.axis)
        if self.shift:
            Y = fftshift(Y, axes=self.axis)
        return Y


@dataclass
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

    Attributes
    ----------
    axis : int, optional
        Axis along which to perform the FFT. Default is the first axis
    shift : bool, optional
        If True, applies the IFFT shift which swaps the low and high frequency components.
        Default is True.
    norm : {"ortho", "backward", "forward"}, optional
        Normalization mode for IFFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
    """
    axis: int = 0
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = "ortho"
    name: str = "ifft"

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.shift:
            X = ifftshift(X, axes=self.axis)
        Y = ifft(X, norm=self.norm, axis=self.axis)
        return Y


@dataclass
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
        Axis along which to allocate subcarriers. Default is the first dimension.

    Example 1
    ---------
    >>> carrier_type = np.array([1, 2, 0, 1, 2, 1])
    >>> pilots = np.array([-1, -1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, pilots=pilots)
    >>> X = np.array([1, 2, 3])
    [1 2 3]
    >>> Y = allocator(X)
    [ 1 -1  0  2 -1  3]

    Example 2
    ---------
    >>> carrier_type = np.array([1, 0, 0, 1, 1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, axis=-1)
    >>> X = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    [[1 4 7]
    [2 5 8]
    [3 6 9]]
    >>> Y = allocator(X)
    [[1 0 0 4 7]
    [2 0 0 5 8]
    [3 0 0 6 9]]
    """
    carrier_type: np.ndarray
    pilots: Optional[np.ndarray] = None
    axis: int = 0
    name: str = "carrier allocator"

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

        # Assign pilots
        if self.N_pilots > 0:
            slices[self.axis] = self.index_pilots
            Y[tuple(slices)] = self.pilots[:, np.newaxis]

        return Y

    def plot(self, shift=False):
        """
        Plot the carrier allocation
        """
        plot_carrier_allocation(self.carrier_type, shift=shift, title="Carrier Allocation")


@dataclass
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
        Axis along which to extract subcarriers. Default is the first dimension

    Example 1
    ---------
    >>> carrier_type = np.array([1, 2, 0, 1, 2, 1])
    >>> pilots = np.array([-1, -2])
    >>> pilot_recorder = Recorder()
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, pilots=pilots)
    >>> extractor = CarrierExtractor(carrier_type=carrier_type, pilot_recorder=pilot_recorder)
    >>> X = np.array([1, 2, 3])
    >>> Z = allocator(X)
    [ 1 -1  0  2 -2  3]
    >>> Y = extractor(Z)
    [1 2 3]
    >>> pilot_recorded = pilot_recorder.get_data()
    [-1 -2]

    Example 2
    ---------
    >>> carrier_type = np.array([1, 0, 0, 1, 1])
    >>> allocator = CarrierAllocator(carrier_type=carrier_type, axis=-1)
    >>> extractor = CarrierExtractor(carrier_type=carrier_type, axis=-1)
    >>> X = np.array([[1, 4, 7], [2, 5, 8]])
    >>> Z = allocator(X)
    [[1 0 0 4 7]
    [2 0 0 5 8]]
    >>> Y = extractor(Z)
    [[1 4 7]
    [2 5 8]]

    """
    carrier_type: np.ndarray
    pilot_recorder: Optional[Callable] = None
    axis: int = 0
    name: str = "carrier extractor"

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

