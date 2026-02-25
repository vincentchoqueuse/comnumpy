import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Literal, Union, Tuple, List
from scipy import signal
from comnumpy.core.generics import Processor
from comnumpy.core.filters import BWFilter


@dataclass
class Upsampler(Processor):
    r"""
    Upsampler class for increasing the sampling rate of a signal along a specified axis.

    This class increases the sample rate of the incoming signal by inserting :math:`L – 1` zeros between samples along the specified axis.

    Signal Model
    ------------

    .. math::
        y[n] = \begin{cases}
        \alpha x[(n-\tau)/ L] & \text{if } (n-\tau)\% L = 0 \\\\
        0 & \text{otherwise}
        \end{cases}

    Attributes
    ----------
    L : int
        The upsampling factor.
    phase : int, optional
        The number of samples :math:`\tau \in \mathbb{N}` by which to offset the upsampled sequence (default: 0).
    scale : float, optional
        The scaling factor :math:`\alpha`  applied to the upsampled signal (default: 1.0).
    axis : int, optional
        The axis along which to perform the upsampling operation (default: -1).
    use_filter: bool, optional
        Apply a lowpass filter at the output (default: false)
    name : str, optional
        The name of the processor (default: "upsampler").

    Example 1
    ---------
    >>> import numpy as np
    >>> X = np.array([1, 2, 3])
    >>> upsampler = Upsampler(L=2)
    >>> Y = upsampler(X)
    >>> print(Y)
    [1. 0. 2. 0. 3. 0.]

    Example 2
    ---------
    >>> X = np.array([1, 2])
    >>> upsampler = Upsampler(L=3, phase=1)
    >>> Y = upsampler(X)
    >>> print(Y)
    [0. 1. 0. 0. 2. 0.]

    Example 3
    ---------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(X[0, :])
    [1 2]
    >>> upsampler = Upsampler(L=2, axis=-1)
    >>> Y = upsampler(X)
    >>> print(Y)
    [[1. 0. 2. 0.]
    [3. 0. 4. 0.]]
    """
    L: int
    phase: int = 0
    scale: float = 1.0
    is_mimo: bool = True
    axis: int = -1
    use_filter: bool = False
    name: str = "upsampler"

    def __post_init__(self):
        self.filter = BWFilter(1/self.L)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Determine the shape of the output array
        output_shape = list(X.shape)
        output_shape[self.axis] = self.L * X.shape[self.axis]
        output_shape = tuple(output_shape)

        # Initialize the output array with zeros
        Y = np.zeros(output_shape, dtype=X.dtype)

        # Create a slice object for the specified axis
        slices = [slice(None)] * X.ndim
        slices[self.axis] = slice(self.phase, None, self.L)

        # Perform the operation along the specified axis
        Y[tuple(slices)] = X

        if self.use_filter:
            Y = self.filter(Y)

        return self.scale * Y


@dataclass
class Downsampler(Processor):
    r"""
    Downsampler class for decreasing the sampling rate of a signal along a specified axis.

    This class decreases the sample rate of the input signal by keeping the first sample and then every Lth sample after the first along the specified axis.

    Signal Model
    ------------
    The decimation process can be described mathematically as follows:

    .. math::
        y[n] = x[n \cdot L]

    where :math:`L` is the downsampling factor, :math:`x` is the input signal, and :math:`y` is the output signal.

    Attributes
    ----------
    L : int
        The downsampling factor, which determines how many samples are skipped between each retained sample.
    phase : int, optional
        The number of samples by which to offset the downsampled sequence (default: 0).
    scale : float, optional
        The scaling factor applied to the downsampled signal (default: 1.0).
    axis : int, optional
        The axis along which to perform the downsampling operation (default: -1).
    use_filter: bool, optional
        Apply a lowpass filter at the output (default: false)
    name : str, optional
        The name of the processor (default: "downsampler").

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([1, 2, 3, 4, 5, 6])
    >>> downsampler = Downsampler(L=2)
    >>> Y = downsampler(X)
    >>> print(Y)
    [1. 3. 5.]
    """
    L: int
    phase: int = 0
    scale: float = 1.0
    is_mimo: bool = True
    axis: int = -1
    use_filter: bool = False
    name: str = "downsampler"

    def forward(self, X: np.ndarray) -> np.ndarray:

        if self.use_filter:
            X = self.filter(X)

        # Create a slice object for the specified axis
        slices = [slice(None)] * X.ndim
        slices[self.axis] = slice(self.phase, None, self.L)

        # Apply the slice to the input array
        Y = X[tuple(slices)]
        return self.scale * Y


@dataclass
class Serial2Parallel(Processor):
    r"""
    A class for converting a serial data stream into parallel data streams.

    This class reshapes a multi-dimensional array (serial stream) into another multi-dimensional array (parallel data streams)
    using a specified number of subcarriers and an ordering scheme. If necessary, the input data is either padded with zeros
    to fit the specified structure or is truncated.

    Attributes
    ----------
    N_sub : int
        The number of subcarriers, which defines the size of the last dimension in the reshaped array. Must be a positive integer.
    order : str, optional
        The order in which to reshape the array. 'F' means to reshape in column-major (Fortran-style) order,
        and 'C' means to reshape in row-major (C-style) order. Default is 'F'.
    method : Literal["zero-padding", "truncate"], optional
        The method to handle data that does not fit perfectly into the reshaped structure.
        Options are 'zero-padding' to pad with zeros or 'truncate' to remove excess data. Default is 'zero-padding'.
    name : str, optional
        Name of the instance. Default is "S2P".

    Example 1
    ---------

    >>> processor = Serial2Parallel(3)
    >>> X = np.arange(5)
    >>> print(X)
    [0 1 2 3 4]
    >>> print(X.shape)
    (5,)
    >>> Y = processor(X)
    >>> print(Y.shape)
    (3, 2)
    >>> print(Y[:, 0])
    [0 1 2]
    >>> print(Y[:, 1])
    [3 4 0]

    Example 2
    ---------

    >>> processor = Serial2Parallel(3)
    >>> X = np.arange(10).reshape(2, 5)
    >>> print(X)
    [[0 1 2 3 4]
    [5 6 7 8 9]]
    >>> print(X.shape)
    (2, 5)
    >>> print(X[0, :])
    [0 1 2 3 4]
    >>> Y = processor(X)
    >>> print(Y.shape)
    (2, 3, 2)
    >>> print(Y[0, :, 0])
    [0 1 2]
    >>> print(Y[0, :, 1])
    [3 4 0]

    """
    N_sub: int
    order: str = "F"
    method: Literal["zero-padding", "truncate"] = "zero-padding"
    name: str = "S2P"

    def __post_init__(self):
        if not (self.N_sub > 0):
            raise ValueError("N_sub must be a positive number.")

    def set_N_sub(self, N_sub):
        self.N_sub = N_sub

    def forward(self, X: np.ndarray) -> np.ndarray:
        N_sub = self.N_sub
        N = X.shape[-1]  # Number of elements in the last dimension
        M = N // N_sub

        if N % N_sub != 0:
            if self.method == "zero-padding":
                M += 1
                new_shape = X.shape[:-1] + (N_sub*M,)
                X_processed = np.zeros(new_shape, dtype=X.dtype)
                X_processed[..., :N] = X
            elif self.method == "truncate":
                X_processed = X[..., :M * N_sub]
        else:
            X_processed = X

        # Reshape along the last axis
        new_shape = X_processed.shape[:-1] + (N_sub, M)
        Y = X_processed.reshape(new_shape, order=self.order)
        return Y


@dataclass
class Parallel2Serial(Processor):
    """
    A class for converting parallel data streams into a serial data stream.

    This class reshapes a multi-dimensional array (parallel data streams) into another multi-dimensional array
    where the last dimension is flattened into a serial data stream, using a specified ordering scheme.

    Attributes
    ----------
    order : str, optional
        The order in which to reshape the array. 'F' means to reshape in column-major (Fortran-style) order,
        and 'C' means to reshape in row-major (C-style) order. Default is 'F'.
    name : str, optional
        Name of the instance. Default is "P2S".

    Example 1
    ---------
    >>> processor = Parallel2Serial()
    >>> X = np.array([[0, 3], [1, 4], [2, 0]])
    >>> print(X.shape)
    (3, 2)
    >>> Y = processor(X)
    >>> print(Y.shape)
    (6,)
    >>> print(Y)
    [0 1 2 3 4 0]

    Example 2
    ---------
    >>> processor = Parallel2Serial()
    >>> X = np.array([[[0, 3],[1, 4], [2, 0]], [[5, 8], [6, 9], [7, 0]]])
    >>> print(X.shape)
    (2, 3, 2)
    >>> Y =processor(X)
    >>> print(Y.shape)
    (2, 6)
    >>> print(Y[0, :])
    [0 1 2 3 4 0]
    """
    order: str = "F"
    name: str = "P2S"

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Reshape the array to flatten the last dimension
        new_shape = X.shape[:-2] + (X.shape[-2] * X.shape[-1],)
        x = X.reshape(new_shape, order=self.order)
        return x

@dataclass
class Amplifier(Processor):
    """
    A class for amplifying or attenuating a signal along a specified axis.

    This class multiplies an input signal by a specified gain factor, effectively amplifying or attenuating the signal based on the gain value. The gain can be applied to all elements or selectively along a specified axis.

    Attributes
    ----------
    gain : float
        The amplification factor by which the signal will be multiplied.
    axis : int or None, optional
        The axis along which to apply the gain. If None, the gain is applied to the entire array. Default is None.
    name : str
        Name of the signal amplifier instance.

    Example 1
    ---------
    >>> amplifier = Amplifier(gain=2)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(X)
    [[1 2]
    [3 4]]
    >>> Y = amplifier(X)
    >>> print(Y)
    [[2 4]
    [6 8]]

    Example 2
    ---------
    >>> amplifier = Amplifier(gain=3, axis=-1)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(X)
    [[1 2]
    [3 4]]
    >>> Y = amplifier(X)
    >>> print(Y)
    [[ 1.  6.]
    [ 3. 12.]]
    """
    gain: float = 1.0
    axis: int | None = None
    name: str = "signal_amplifier"

    def forward(self, X: np.ndarray) -> np.ndarray:
        gain = self.gain
        if self.axis is None:
            # Apply gain to the entire array
            Y = gain * X
        else:
            # Apply gain along the specified axis
            gain_shape = [1] * X.ndim
            gain_shape[self.axis] = gain
            Y = gain_shape * X
        return Y


@dataclass
class WeightAmplifier(Processor):
    """
    Applies weights to a MIMO (Multiple Input Multiple Output) parallel signal along a specified axis.

    This class multiplies each parallel stream of the input signal by a corresponding weight along a specified axis. The weights are applied selectively to the elements along the specified axis.

    Signal Model
    ------------

    .. math::

       y_l[n] = w_l x_l[n]

    where the coefficient :math:`w_l` specifies the weight.

    Attributes
    ----------
    weight : np.ndarray
        Weights to be applied to each parallel stream of the input signal (1D array).
    axis : int, optional
        The axis along which to apply the weights. Default is -1 (the last axis).
    name : str
        Name of the weight amplifier instance.

    Example 1
    ---------
    >>> weight_amplifier = WeightAmplifier(weight=np.array([2, 3]), axis=0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(X)
    [[1 2]
     [3 4]]
    >>> Y = weight_amplifier.forward(X)
    >>> print(Y)
    [[2 4]
     [9 12]]

    Example 2
    ---------
    >>> weight_amplifier = WeightAmplifier(weight=np.array([2, 3]), axis=-1)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(X)
    [[1 2]
     [3 4]]
    >>> Y = weight_amplifier(X)
    >>> print(Y)
    [[2 6]
     [6 12]]
    """
    weight: Optional[np.ndarray] = None
    axis: int = -1
    name: str = "parallel_signal_weight"

    def __post_init__(self):
        if self.weight.ndim != 1:
            raise ValueError(f"The weight vector should be a 1D array (current shape: {self.weight.shape})")

    def validate_input(self, X: np.ndarray):
        if len(self.weight) != X.shape[self.axis]:
            raise ValueError(f"Dimension of the weight vector and input signal along axis {self.axis} does not match")

    def get_weight(self, input_shape=None):
        return self.weight

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.validate_input(X)
        # Apply weights along the specified axis
        self.weight = self.get_weight(X.shape)
        weight_shape = [1] * X.ndim
        weight_shape[self.axis] = len(self.weight)
        Y = self.weight.reshape(weight_shape) * X
        return Y


@dataclass
class Complex2Real(Processor):
    r"""
    A processor class to extract the real or imaginary part of a complex array.

    Attributes
    ----------
    part : Literal["real", "imag"]
        Specifies which part of the complex number to extract.
        Can be either "real" or "imag". Default is "real".

    validate_input : bool
        If True, validates that the input array is purely real or imaginary
        based on the specified part. Default is False.

    Example 1
    ---------

    >>> processor = Complex2Real(part="real")
    >>> X = np.array([1+2j, 3+4j, 5+0j])
    >>> processor(X)
    array([1., 3., 5.])

    Example 2
    ---------
    >>> processor_imag = Complex2Real(part="imag")
    >>> X = np.array([1+2j, 3+4j, 5+0j])
    >>> processor_imag(X)
    array([2., 4., 0.])

    Example 3
    ---------
    >>> processor_real2 = Complex2Real(part="imag", validate_input=True)
    >>> X = np.array([1+2j, 3+4j, 5+0j])
    >>> processor(X)  # Raises ValueError
    ValueError: The input data is not real since the imaginary part is non-zero.
    """

    part: Literal["real", "imag"] = "real"
    validate_input : bool = False

    def forward(self, X):
        match self.part:
            case "real":
                if self.validate_input and (not np.isclose(np.ravel(X).imag, 0, atol=10**-7).all()):
                    raise ValueError("the input data is not real since the imag part is non zero ")
                Y = X.real
            case "imag":
                if self.validate_input and (not np.isclose(np.ravel(X).real, 0, atol=10**-7).all()):
                    raise ValueError("the input data is not imaginary since the real part is non zero ")
                Y = X.imag

        return Y


@dataclass
class AutoConcatenator(Processor):
    r"""
    A class to automatically concatenate data along a specified axis using masks.

    This class facilitates the extraction and concatenation of data from an input array based on
    predefined masks. It ensures that the output array is constructed correctly by validating
    the shapes and contents of the masks.

    Attributes
    ----------
    input_copy_mask : Optional[np.ndarray]
        A boolean mask used to extract a portion of the input data to be copied.

    output_original_mask : Optional[np.ndarray]
        A boolean mask indicating where the original data should be placed in the output array.

    output_copy_mask : Optional[np.ndarray]
        A boolean mask indicating where the copied data should be placed in the output array.

    axis : int
        The axis along which to perform the concatenation. Default is 0.

    name : str
        The name of the processor. Default is "auto concatenator".

    Example 1
    ---------

    >>> input_copy_mask = np.array([True, False, True])
    >>> output_original_mask = np.array([True, True, True, False, False])
    >>> output_copy_mask = np.array([False, False, False, True, True])
    >>> X = np.array([1, 2, 3])
    [1 2 3]
    >>> concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask)
    >>> Y = concatenator(X)
    [1 2 3 1 3]

    Example 2
    ---------

    >>> input_copy_mask = np.array([True, False])
    >>> output_original_mask = np.array([True, True, False, False, False])
    >>> output_copy_mask = np.array([False, False, False, True, False])
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    [[1 2 3]
    [4 5 6]]
    >>> concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask)
    >>> Y = concatenator(X)
    [[1 2 3]
    [4 5 6]
    [0 0 0]
    [1 2 3]
    [0 0 0]]

    Example 3
    ---------

    >>> input_copy_mask = np.array([False, True, True])
    >>> output_original_mask = np.array([False, True, True, True, False])
    >>> output_copy_mask = np.array([True, False, False, False, True])
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    [[1 2 3]
    [4 5 6]]
    >>> concatenator = AutoConcatenator(input_copy_mask, output_original_mask, output_copy_mask, axis=-1)
    >>> result = concatenator(X)
    [[2 1 2 3 3]
    [5 4 5 6 6]]

    """
    input_copy_mask: Optional[np.ndarray] = None
    output_original_mask: Optional[np.ndarray] = None
    output_copy_mask: Optional[np.ndarray] = None
    axis: int = 0
    name: str = "auto concatenator"

    def __post_init__(self):

        # Check if the sizes of output_original_mask and output_copy_mask are the same
        if self.output_original_mask.shape != self.output_copy_mask.shape:
            raise ValueError("output_original_mask and output_copy_mask must have the same shape.")

        # Check the number of True values in the masks
        if self.input_copy_mask is not None:
            num_true_input_copy = np.sum(self.input_copy_mask)
            num_true_output_copy = np.sum(self.output_copy_mask)
            num_true_output_original = np.sum(self.output_original_mask)

            if num_true_input_copy != num_true_output_copy:
                raise ValueError("The number of True values in input_copy_mask must be equal to the number of True values in output_copy_mask.")
                
        # check if there is no overlap between allocated value
        if np.any(np.logical_and(self.output_original_mask, self.output_copy_mask)):
            raise ValueError("The two output masks overlap.")
                

    def extract_copy(self, X: np.ndarray):
        """
        Extract a copy  from the input signal on the specified axis
        """
        shape = list(X.shape)
        shape[self.axis] = np.sum(self.input_copy_mask)
        slices = [slice(None)] * len(X.shape)
        slices[self.axis] = self.input_copy_mask
        X_copy = X[tuple(slices)]
        return X_copy

    def process_copy(self, X: np.ndarray):
        return X

    def forward(self, X: np.ndarray) -> np.ndarray:

        if X.shape[self.axis] != len(self.input_copy_mask):
            raise ValueError(f"input signal for the dimension {self.axis} and input_copy_mask must have the same shape.")    
            
        X_copy = self.extract_copy(X)
        X_copy_processed = self.process_copy(X_copy)

        # Prepare the output array
        shape = list(X.shape)
        shape[self.axis] = len(self.output_original_mask)
        Y = np.zeros(shape, dtype=X.dtype)

        # Create slicing object
        slices = [slice(None)] * len(X.shape)

        # Assign original data
        slices[self.axis] = self.output_original_mask
        Y[tuple(slices)] = X

        # Assign copied data
        slices[self.axis] = self.output_copy_mask
        Y[tuple(slices)] = X_copy_processed
        return Y



@dataclass
class SampleRemover(Processor):
    """
    Deletes samples from a signal.

    This class removes a specified number of samples starting from a given index in the signal.

    Attributes
    ----------
    N_start : int
        Index of the first sample to delete.
    length : int
        Number of samples to delete.
    name : str
        Name of the symbol remover instance. Default is "SymbolRemover".
    """
    N_start: int = 0
    length: int = 0
    name: str = "symbol remover"

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(len(x) - self.length, dtype=x.dtype)
        y[:self.N_start] = x[:self.N_start]
        y[self.N_start:] = x[self.N_start + self.length:]
        return y


@dataclass
class DelayRemover(Processor):
    """
    Removes an initial delay from a signal.

    This class removes a specified number of initial samples (delay) from the signal.

    Attributes
    ----------
    delay : int
        Number of initial samples to remove.
    name : str
        Name of the delay remover instance. Default is "DelayRemover".
    """
    delay: int
    is_mimo: bool = True
    axis: int = -1
    name: str = "delay remover"

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Create a slice object for the specified axis
        slices = [slice(None)] * x.ndim
        slices[self.axis] = slice(self.delay, None)

        # Apply the slice to the input array
        y = x[tuple(slices)]
        return y


@dataclass
class DataAdder(Processor):
    """
    Inserts symbol samples into a signal.

    This class inserts a specified symbol into a signal at a given index.

    Attributes
    ----------
    symbol : np.ndarray
        Symbol to be inserted into the signal.
    N_start : int
        Index at which to insert the symbol.
    name : str
        Name of the data adder instance. Default is "DataAdder".
    """
    symbol: np.ndarray
    N_start: int = 0
    name: str = "Data Adder"

    def validate_input(self, x: np.ndarray):
        if self.N_start < 0 or self.N_start > len(x):
            raise ValueError("N_start is out of bounds for the input signal.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_input(x)
        y = np.concatenate((x[:self.N_start], self.symbol, x[self.N_start:]))
        return y


@dataclass
class DataExtractor(Processor):
    """
    Extract a segment from a signal using NumPy-style indexing.

    Parameters
    ----------
    selector : int, slice, tuple, list, ndarray or None
        - int -> a single index
        - tuple(start, stop[, step]) -> converted to slice
        - list/ndarray -> explicit indices
        - None -> passthrough (no extraction)
    name : str
        Instance name

    Examples
    --------
    >>> x = np.arange(10)

    # single index
    >>> extractor1 = DataExtractor(3)
    >>> extractor1(x)
    array([3])

    # slice with tuple
    >>> extractor2 = DataExtractor((2, 8))
    >>> extractor2(x)
    array([2, 3, 4, 5, 6, 7])

    # slice with step
    >>> extractor3 = DataExtractor((1, 9, 2))
    >>> extractor3(x)
    array([1, 3, 5, 7])

    # multidimensional example
    >>> x2d = np.arange(20).reshape(4, 5)
    >>> extractor4 = DataExtractor((1, 3))
    >>> extractor4(x2d)
    array([[ 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    """
    selector: Optional[Union[int, slice, tuple, list, np.ndarray]] = None
    name: str = "Data Extractor"

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return x

        # transformer un tuple en slice si c'est un tuple simple
        if isinstance(self.selector, tuple):
            self.selector = slice(*self.selector)

        return x[self.selector]


@dataclass
class Resampler(Processor):
    """
    A class for resampling a signal.

    This class changes the sampling rate of a signal by a rational factor.
    It performs upsampling by the specified 'up' factor, followed by downsampling by the specified 'down' factor, effectively changing the sampling rate by a factor of up/down.

    Attributes
    ----------
    up : int
        The upsampling factor. Must be a positive integer.
    down : int
        The downsampling factor. Must be a positive integer.
    name : str
        Name of the resampler instance. Default is "Resampler".

    """
    up: int
    down: int
    name: str = "Resampler"

    def __post_init__(self):

        if self.up <= 0 or self.down <= 0:
            raise ValueError("Both 'up' and 'down' factors must be positive integers.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = signal.resample_poly(x, self.up, self.down)
        return y


@dataclass
class Clipper(Processor):
    r"""
    Clipper class for clipping signal values to a specified threshold.

    Signal Model
    ------------

    .. math::

       y[n] = \frac{x[n]}{|x[n]|} \cdot \min{(|x[n]|, \tau)}

    Attributes
    ----------
    threshold : float
        The threshold value :math:`\tau` for clipping.
    name : str
        Name of the clipper instance.

    """
    threshold: float
    name: str = "Clipper"

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, -self.threshold, self.threshold)
        return y



class BlindPhaseTracker(Processor):
    """
    A class implementing a blind phase tracking algorithm using a grid search approach.

    This algorithm estimates and compensates for unknown phase rotations in a received signal 
    by minimizing the local Error Vector Magnitude (EVM) around each sample. It assumes 
    the signal belongs to a known modulation alphabet (e.g., QAM or PSK).

    Parameters
    ----------
    L : int
        The number of neighboring symbols on each side used to compute the local EVM cost.
    alphabet : np.ndarray
        The set of complex symbols representing the modulation constellation.
    phase_steps : int, optional
        The number of discrete phase candidates evaluated within the range [-π/4, π/4). 
        Default is 10.

    Methods
    -------
    hard_projector(z)
        Projects input samples to the nearest symbols from the constellation.
    evm_cost(x, n, phi)
        Computes the local EVM cost at index `n` for a candidate phase shift `phi`.
    forward(x)
        Applies blind phase correction to the input signal `x` and plots the estimated 
        phase evolution over time.
    """

    L: int
    alphabet: np.ndarray
    phase_steps: int = 10
    phases: np.ndarray = field(init=False)

    
    def __post_init__(self):
        self.phases = np.linspace(-np.pi/4, np.pi/4, self.phase_steps, endpoint=False)

    def hard_projector(self, z):
        z = np.atleast_2d(np.ravel(z))
        distances = np.abs(z.T - self.alphabet)**2
        indices = np.argmin(distances, axis=1)
        symbols = self.alphabet[indices]
        return indices, symbols

    def evm_cost(self, x, n, phi):
        # Compute local EVM cost around index n
        total_error = 0
        count = 0
        for m in range(-self.L, self.L + 1):
            idx = n + m
            if 0 <= idx < len(x):
                rotated = x[idx] * np.exp(-1j * phi)
                _, closest = self.hard_projector(rotated)
                total_error += np.abs(rotated - closest)**2
                count += 1
        return total_error / count if count > 0 else np.inf

    def forward(self, x):
        y_corrected = np.zeros_like(x, dtype=complex)
        optimal_phases = []

        for n in range(len(x)):
            costs = [self.evm_cost(x, n, phi) for phi in self.phases]
            best_phi_idx = np.argmin(costs)
            best_phi = self.phases[best_phi_idx]
            optimal_phases.append(best_phi)
            y_corrected[n] = x[n] * np.exp(-1j * best_phi)

        # Plot estimated phase evolution
        if self.plot:
            plt.figure(figsize=(10, 4))
            plt.plot(optimal_phases, label="Estimated Phase (rad)")
            plt.title("Estimated Phase per Sample")
            plt.xlabel("Sample Index")
            plt.ylabel("Phase (radians)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return y_corrected