import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Union
from comnumpy.core.generics import Processor
from comnumpy.core.filters import BWFilter
from comnumpy.exceptions import ShapeError


def _required_mask(mask: Optional[np.ndarray], block: str,
                   name: str) -> np.ndarray:
    """A mask a subclass builds in ``prepare()``, or a message.

    These blocks are legitimately incomplete until the chain calls
    ``prepare()``: returning the value rather than only asserting it is
    what lets the rest of the method treat it as an array.
    """
    if mask is None:
        raise ValueError(
            f"{block} was used before {name} was built. Got None, "
            f"expected a boolean or index mask -- either pass it at "
            f"construction, or run the block inside a Sequential, which "
            f"calls prepare() before the first sample.")
    return mask

__all__ = [
    "Upsampler", "Downsampler", "Serial2Parallel", "Parallel2Serial",
    "Amplifier", "WeightAmplifier", "Complex2Real", "AutoConcatenator",
    "SampleRemover", "DelayRemover", "DataAdder", "DataExtractor", "Resampler",
    "Clipper", "BlindPhaseTracker",
]


@dataclass(slots=True)
class Upsampler(Processor):
    r"""
    Increases the sampling rate of a signal by inserting zeros between samples.

    Signal Model
    ------------
    Upsampling by an integer factor :math:`L` inserts :math:`L-1` zeros
    between consecutive input samples, so that the output rate is
    :math:`L` times the input rate:

    .. math::

        y[n] = \left\{\begin{array}{cl}
        \alpha \, x\!\left[\dfrac{n-\tau}{L}\right] & \text{if } (n-\tau) \bmod L = 0,\\[6pt]
        0 & \text{otherwise,}
        \end{array}\right.

    where:

    * :math:`x[n]` is the input sequence of length :math:`N`,
    * :math:`y[n]` is the output sequence of length :math:`L N`,
    * :math:`L` is the upsampling factor,
    * :math:`\tau` is the phase offset of the retained samples,
    * :math:`\alpha` is an output scaling factor.

    Zero insertion creates :math:`L-1` spectral images of the input
    spectrum. When ``use_filter`` is set, a lowpass filter of normalized
    cutoff :math:`1/L` is applied to the zero-stuffed sequence, which
    turns the operation into an interpolation.

    Axes: *declared axis* -- the zero stuffing runs along ``axis``
    (default -1, the sample axis of the Serial layout ``(..., N)``) and
    the remaining axes are left untouched.

    Parameters
    ----------
    L : int
        Upsampling factor :math:`L`. Must be a positive integer.
    phase : int, optional, keyword-only
        Phase offset :math:`\tau \in \mathbb{N}` of the retained samples.
        Default is 0.
    scale : float, optional, keyword-only
        Output scaling factor :math:`\alpha`. Default is 1.0.
    axis : int, optional, keyword-only
        Axis along which the upsampling is performed. Default is -1.
    use_filter : bool, optional, keyword-only
        If True, apply the anti-imaging lowpass filter of normalized
        cutoff :math:`1/L` at the output. Default is False.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"upsampler"``.

    References
    ----------
    A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*,
    3rd ed., Pearson, 2010, Chapter 4 (changing the sampling rate by
    discrete-time processing).

    Examples
    --------
    >>> print(Upsampler(L=2)(np.array([1, 2, 3])))
    [1. 0. 2. 0. 3. 0.]
    >>> print(Upsampler(L=3, phase=1)(np.array([1, 2])))
    [0. 1. 0. 0. 2. 0.]
    >>> print(Upsampler(L=2)(np.array([[1, 2], [3, 4]])))
    [[1. 0. 2. 0.]
     [3. 0. 4. 0.]]
    """
    L: int
    phase: int = field(default=0, kw_only=True)
    scale: float = field(default=1.0, kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    use_filter: bool = field(default=False, kw_only=True)
    name: str = field(default="upsampler", kw_only=True)
    # internal state (declared for slots, D40a)
    filter: Optional[BWFilter] = field(init=False, repr=False, default_factory=lambda: None)

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

        if self.use_filter and self.filter is not None:
            Y = self.filter(Y)

        return self.scale * Y


@dataclass(slots=True)
class Downsampler(Processor):
    r"""
    Decreases the sampling rate of a signal by keeping one sample out of every L.

    Signal Model
    ------------
    Downsampling (decimation) by an integer factor :math:`L` keeps one
    sample out of every :math:`L`, starting at the phase :math:`\tau`:

    .. math::

        y[n] = \alpha \, x[n L + \tau]

    where:

    * :math:`x[n]` is the input sequence of length :math:`N`,
    * :math:`y[n]` is the output sequence of length :math:`\lceil (N-\tau)/L \rceil`,
    * :math:`L` is the downsampling factor,
    * :math:`\tau` is the phase of the retained samples,
    * :math:`\alpha` is an output scaling factor.

    Decimation folds the input spectrum :math:`L` times; the input must
    therefore be bandlimited to :math:`1/L` beforehand, otherwise aliasing
    occurs. Setting ``use_filter`` applies that anti-aliasing lowpass
    filter of normalized cutoff :math:`1/L` **before** the decimation --
    the mirror image of :class:`Upsampler`, where the anti-imaging filter
    of the same cutoff comes **after** the zero stuffing.

    Axes: *declared axis* -- the decimation runs along ``axis`` (default
    -1, the sample axis of the Serial layout ``(..., N)``) and the
    remaining axes are left untouched.

    Parameters
    ----------
    L : int
        Downsampling factor :math:`L`. Must be a positive integer.
    phase : int, optional, keyword-only
        Phase :math:`\tau \in \mathbb{N}` of the retained samples.
        Default is 0.
    scale : float, optional, keyword-only
        Output scaling factor :math:`\alpha`. Default is 1.0.
    axis : int, optional, keyword-only
        Axis along which the downsampling is performed. Default is -1.
    use_filter : bool, optional, keyword-only
        If True, apply the anti-aliasing lowpass filter of normalized
        cutoff :math:`1/L` before the decimation. Default is False.
        The filter is a :class:`comnumpy.core.filters.BWFilter`, which
        only accepts a 1D input.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"downsampler"``.

    References
    ----------
    A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*,
    3rd ed., Pearson, 2010, Chapter 4 (changing the sampling rate by
    discrete-time processing).

    Examples
    --------
    >>> print(Downsampler(L=2)(np.arange(6)))
    [0. 2. 4.]
    >>> print(Downsampler(L=2, phase=1)(np.arange(6)))
    [1. 3. 5.]
    >>> print(Downsampler(L=2)(np.arange(12).reshape(2, 6)))
    [[ 0.  2.  4.]
     [ 6.  8. 10.]]

    Anti-aliasing: an in-band tone corrupted by an out-of-band one is
    decimated exactly with ``use_filter=True``, and aliased without it.
    The cutoff is :math:`1/L = 1/4` of the Nyquist frequency, i.e. 0.125
    cycle per sample.

    >>> n = np.arange(128)
    >>> clean = np.cos(2 * np.pi * 2 * n / 128)             # f = 0.016, kept
    >>> x = clean + 0.5 * np.cos(2 * np.pi * 30 * n / 128)  # f = 0.234, rejected
    >>> y = Downsampler(L=4, use_filter=True)(x)
    >>> print(round(float(np.max(np.abs(y - clean[::4]))), 6))
    0.0
    >>> print(round(float(np.max(np.abs(Downsampler(L=4)(x) - clean[::4]))), 3))
    0.5
    """
    L: int
    phase: int = field(default=0, kw_only=True)
    scale: float = field(default=1.0, kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    use_filter: bool = field(default=False, kw_only=True)
    name: str = field(default="downsampler", kw_only=True)
    # internal state (declared for slots, D40a)
    filter: Optional[BWFilter] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        self.filter = BWFilter(1/self.L)

    def forward(self, X: np.ndarray) -> np.ndarray:

        # anti-aliasing filtering comes BEFORE the decimation
        if self.use_filter and self.filter is not None:
            X = self.filter(X)

        # Create a slice object for the specified axis
        slices = [slice(None)] * X.ndim
        slices[self.axis] = slice(self.phase, None, self.L)

        # Apply the slice to the input array
        Y = X[tuple(slices)]
        return self.scale * Y


@dataclass(slots=True)
class Serial2Parallel(Processor):
    r"""
    Converts a serial data stream into parallel blocks (Serial to Block layout).

    Signal Model
    ------------
    The serial axis of length :math:`N` is cut into :math:`T` consecutive
    blocks of :math:`F` samples. The sample of index :math:`n` of the
    serial stream becomes the sample of index :math:`f` of the block
    :math:`t`:

    .. math::

        y[t, f] = x\!\left[t F + f\right], \qquad
        0 \le t < T, \quad 0 \le f < F, \quad T = \left\lceil \frac{N}{F} \right\rceil

    Equivalently, :math:`t = \lfloor n / F \rfloor` and
    :math:`f = n \bmod F`: the **serial index runs fastest on the last
    axis**. This is exactly a C-order (row-major) ``reshape``, which is
    the normative Serial :math:`(..., N)` to Block :math:`(..., T, F)`
    conversion of CONVENTIONS.md (decision D2). Any implementation
    requiring a transposition -- for instance a Fortran-order reshape
    producing :math:`(F, T)` -- violates that convention.

    When :math:`N` is not a multiple of :math:`F`, the serial axis is
    either zero-padded up to :math:`T F` samples or truncated down to
    :math:`\lfloor N/F \rfloor F` samples, depending on ``method``.

    Axes: *declared axis* -- consumes the serial axis -1 of the Serial
    layout ``(..., N)`` and produces the Block layout ``(..., T, F)``;
    the leading axes are carried through unchanged.

    Parameters
    ----------
    N_sub : int
        Block length :math:`F` (the number of subcarriers in an OFDM
        chain), which becomes the size of the last axis of the output.
        Must be a positive integer.
    method : {"zero-padding", "truncate"}, optional, keyword-only
        How to handle a serial length :math:`N` that is not a multiple of
        :math:`F`: ``"zero-padding"`` appends zeros up to :math:`T F`
        samples, ``"truncate"`` drops the trailing incomplete block.
        Default is ``"zero-padding"``.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"S2P"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2 (serial-to-parallel conversion).

    Examples
    --------
    >>> Y = Serial2Parallel(3)(np.arange(5))
    >>> print(Y.shape)
    (2, 3)
    >>> print(Y)
    [[0 1 2]
     [3 4 0]]
    >>> print(Serial2Parallel(3, method="truncate")(np.arange(5)))
    [[0 1 2]]
    >>> Y = Serial2Parallel(3)(np.arange(10).reshape(2, 5))
    >>> print(Y.shape)
    (2, 2, 3)
    """
    N_sub: int
    method: Literal["zero-padding", "truncate"] = field(default="zero-padding", kw_only=True)
    name: str = field(default="S2P", kw_only=True)

    def __post_init__(self):
        if not (self.N_sub > 0):
            raise ValueError("N_sub must be a positive number.")

    def set_N_sub(self, N_sub: int) -> None:
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
                raise ValueError(
                    f"{type(self).__name__}: unknown method "
                    f"{self.method!r}, expected 'zero-padding' or "
                    f"'truncate' -- the input length {N} is not a "
                    f"multiple of N_sub={N_sub}, so one of the two has "
                    f"to say what to do with the remainder.")
        else:
            X_processed = X

        # Pure C-order reshape into the Block layout (..., T, F)
        new_shape = X_processed.shape[:-1] + (M, N_sub)
        Y = X_processed.reshape(new_shape)
        return Y


@dataclass(slots=True)
class Parallel2Serial(Processor):
    r"""
    Converts parallel blocks back into a serial data stream (Block to Serial layout).

    Signal Model
    ------------
    Inverse of :class:`Serial2Parallel`: the :math:`T` blocks of
    :math:`F` samples are concatenated back into a single serial axis of
    length :math:`T F`,

    .. math::

        y[t F + f] = x[t, f], \qquad
        0 \le t < T, \quad 0 \le f < F

    that is, the block content index :math:`f` runs fastest. This is a
    pure C-order (row-major) ``reshape`` of the last two axes, the
    normative Block :math:`(..., T, F)` to Serial :math:`(..., T F)`
    conversion of CONVENTIONS.md (decision D2). No transposition is
    involved, so ``Parallel2Serial()(Serial2Parallel(F)(x))`` returns
    ``x`` whenever :math:`F` divides its length.

    Axes: *declared axis* -- consumes the Block layout ``(..., T, F)``
    and produces the Serial layout ``(..., T*F)``; the leading axes are
    carried through unchanged.

    Parameters
    ----------
    name : str, optional
        Name of the processor instance. Default is ``"P2S"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2 (parallel-to-serial conversion).

    Examples
    --------
    >>> print(Parallel2Serial()(np.array([[0, 1, 2], [3, 4, 5]])))
    [0 1 2 3 4 5]
    >>> Y = Parallel2Serial()(np.arange(12).reshape(2, 2, 3))
    >>> print(Y.shape)
    (2, 6)
    >>> print(Y[0])
    [0 1 2 3 4 5]
    >>> x = np.arange(6)
    >>> print(np.array_equal(Parallel2Serial()(Serial2Parallel(3)(x)), x))
    True
    """
    name: str = "P2S"

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Pure C-order reshape flattening the last two axes
        new_shape = X.shape[:-2] + (X.shape[-2] * X.shape[-1],)
        x = X.reshape(new_shape)
        return x


@dataclass(slots=True)
class Amplifier(Processor):
    r"""
    Scales a signal by a constant gain (amplification or attenuation).

    Signal Model
    ------------
    The amplifier is a memoryless linear device applying a constant gain
    :math:`g` to every sample:

    .. math::

        y[n] = g \, x[n]

    where :math:`g > 1` amplifies, :math:`g < 1` attenuates, and the
    output power is :math:`g^2` times the input power.

    The gain is a single scalar applied to the whole array. To weight the
    parallel streams of a signal individually, i.e. one gain per entry of
    a declared axis, use :class:`WeightAmplifier`.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    gain : float
        Gain :math:`g` applied to every sample. Default is 1.0.
    name : str, optional, keyword-only
        Name of the amplifier instance. Default is
        ``"signal_amplifier"``.

    References
    ----------
    S. C. Cripps, *RF Power Amplifiers for Wireless Communications*,
    2nd ed., Artech House, 2006 (memoryless amplitude gain).

    Examples
    --------
    >>> print(Amplifier(gain=2)(np.array([[1, 2], [3, 4]])))
    [[2 4]
     [6 8]]
    >>> print(Amplifier(gain=0.5)(np.array([2.0, 4.0])))
    [1. 2.]
    """
    gain: float = 1.0
    name: str = field(default="signal_amplifier", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.gain * X


@dataclass(slots=True)
class WeightAmplifier(Processor):
    r"""
    Applies one complex weight per parallel stream along a declared axis.

    Signal Model
    ------------
    Each stream :math:`l` of a parallel (for instance multi-antenna)
    signal is scaled by its own weight :math:`w_l`:

    .. math::

        y_l[n] = w_l \, x_l[n], \qquad 0 \le l < L

    where:

    * :math:`x_l[n]` is the :math:`l`-th input stream,
    * :math:`w_l` is the weight applied to that stream,
    * :math:`L` is the number of streams, i.e. the size of the input
      along ``axis``.

    This is the diagonal (per-stream) case of a linear precoder or of a
    one-tap equalizer; :class:`comnumpy.ofdm.compensators.FrequencyDomainEqualizer`
    derives from it with :math:`w_k = 1/H[k]`.

    Axes: *declared axis* -- the weights are applied along ``axis``
    (default -1); ``len(weight)`` must equal ``X.shape[axis]``, which is
    checked at every call.

    Parameters
    ----------
    weight : np.ndarray
        1D array of weights :math:`w_l`, of length :math:`L`.
    axis : int, optional, keyword-only
        Axis carrying the parallel streams. Default is -1.
    name : str, optional, keyword-only
        Name of the processor instance. Default is
        ``"parallel_signal_weight"``.

    Raises
    ------
    ValueError
        If ``weight`` is not a 1D array, or if its length does not match
        the size of the input along ``axis``.

    References
    ----------
    A. Goldsmith, *Wireless Communications*, Cambridge University Press,
    2005, Chapter 10 (multiple-antenna systems, transmit and receive
    weighting).

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(WeightAmplifier(weight=np.array([2, 3]), axis=-1)(X))
    [[ 2  6]
     [ 6 12]]
    >>> print(WeightAmplifier(weight=np.array([2, 3]), axis=0)(X))
    [[ 2  4]
     [ 9 12]]
    """
    weight: np.ndarray
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="parallel_signal_weight", kw_only=True)

    def __post_init__(self):
        self.weight = np.asarray(self.weight)
        if self.weight.ndim != 1:
            raise ValueError(f"The weight vector should be a 1D array (current shape: {self.weight.shape})")

    def validate_input(self, X: np.ndarray) -> None:
        if len(self.weight) != X.shape[self.axis]:
            raise ValueError(f"Dimension of the weight vector and input signal along axis {self.axis} does not match")

    def get_weight(self, input_shape: Optional[tuple[int, ...]] = None
                   ) -> np.ndarray:
        return self.weight

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.validate_input(X)
        # Apply weights along the specified axis
        self.weight = self.get_weight(X.shape)
        weight_shape = [1] * X.ndim
        weight_shape[self.axis] = len(self.weight)
        Y = self.weight.reshape(weight_shape) * X
        return Y


@dataclass(slots=True)
class Complex2Real(Processor):
    r"""
    Extracts the real or the imaginary part of a complex signal.

    Signal Model
    ------------
    Writing the input in Cartesian form
    :math:`x[n] = x_I[n] + i \, x_Q[n]`, the processor returns one of the
    two quadrature components:

    .. math::

        y[n] = \left\{\begin{array}{cl}
        x_I[n] = \Re\{x[n]\} & \text{if } \texttt{part} = \text{"real"},\\
        x_Q[n] = \Im\{x[n]\} & \text{if } \texttt{part} = \text{"imag"}.
        \end{array}\right.

    The output is real-valued. With ``validate_input=True`` the
    discarded component is checked to be numerically zero (tolerance
    :math:`10^{-7}`), which turns the block into a guard: it then only
    changes the dtype of a signal that is already known to be real (or
    purely imaginary), and refuses to silently drop energy.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    part : {"real", "imag"}, optional
        Quadrature component to extract, :math:`x_I[n]` or
        :math:`x_Q[n]`. Default is ``"real"``.
    validate_input : bool, optional, keyword-only
        If True, raise instead of discarding a non-zero component.
        Default is False.

    Raises
    ------
    ValueError
        If ``validate_input`` is True and the discarded component is not
        numerically zero.

    Examples
    --------
    >>> X = np.array([1+2j, 3+4j, 5+0j])
    >>> Complex2Real(part="real")(X)
    array([1., 3., 5.])
    >>> Complex2Real(part="imag")(X)
    array([2., 4., 0.])
    >>> Complex2Real(part="imag", validate_input=True)(X)
    Traceback (most recent call last):
        ...
    ValueError: the input data is not imaginary since the real part is non zero
    """

    part: Literal["real", "imag"] = "real"
    validate_input: bool = field(default=False, kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        match self.part:
            case "real":
                if self.validate_input and (not np.isclose(np.imag(np.ravel(X)), 0, atol=10**-7).all()):
                    raise ValueError("the input data is not real since the imag part is non zero ")
                Y = np.real(X)
            case "imag":
                if self.validate_input and (not np.isclose(np.real(np.ravel(X)), 0, atol=10**-7).all()):
                    raise ValueError("the input data is not imaginary since the real part is non zero ")
                Y = np.imag(X)

        return Y


@dataclass(slots=True)
class AutoConcatenator(Processor):
    r"""
    Copies part of a signal and reassembles original and copy through boolean masks.

    Signal Model
    ------------
    Let :math:`m^{\text{in}}`, :math:`m^{\text{orig}}` and
    :math:`m^{\text{copy}}` be the three boolean masks, of lengths
    :math:`N` (input) and :math:`M` (output). Writing
    :math:`\mathcal{I} = \{ i : m^{\text{in}}_i \}`,
    :math:`\mathcal{O} = \{ i : m^{\text{orig}}_i \}` and
    :math:`\mathcal{C} = \{ i : m^{\text{copy}}_i \}` for the sorted
    positions selected by each mask, the output is

    .. math::

        y\!\left[\mathcal{O}_j\right] = x[j], \quad 0 \le j < N,
        \qquad
        y\!\left[\mathcal{C}_j\right] = g\!\left(x\!\left[\mathcal{I}_j\right]\right),
        \quad 0 \le j < |\mathcal{I}|,

    and :math:`y[i] = 0` for every position covered by neither mask. The
    map :math:`g(\cdot)` is the hook ``process_copy`` (identity by
    default), which subclasses override to transform the copy.

    Consistency requires :math:`|\mathcal{O}| = N`,
    :math:`|\mathcal{I}| = |\mathcal{C}|`, and
    :math:`\mathcal{O} \cap \mathcal{C} = \varnothing`; all three are
    checked. Prepending a copy of the tail of each block is exactly the
    cyclic prefix, which is why
    :class:`comnumpy.ofdm.processors.CyclicPrefixer` derives from this
    class and only builds the three masks.

    Axes: *declared axis* -- the masks index ``axis`` (default -1);
    ``len(input_copy_mask)`` must equal ``X.shape[axis]``, and the output
    has size ``len(output_original_mask)`` along that axis.

    Parameters
    ----------
    input_copy_mask : np.ndarray of bool, optional, keyword-only
        Mask :math:`m^{\text{in}}` selecting, in the input, the entries
        to be copied. Length ``X.shape[axis]``.
    output_original_mask : np.ndarray of bool, optional, keyword-only
        Mask :math:`m^{\text{orig}}` giving, in the output, the positions
        of the original data. Must select exactly ``X.shape[axis]``
        entries.
    output_copy_mask : np.ndarray of bool, optional, keyword-only
        Mask :math:`m^{\text{copy}}` giving, in the output, the positions
        of the copied data. Same length as ``output_original_mask``, and
        must select as many entries as ``input_copy_mask``.
    axis : int, optional, keyword-only
        Axis along which the concatenation is performed. Default is -1.
    name : str, optional, keyword-only
        Name of the processor instance. Default is
        ``"auto concatenator"``.

    Raises
    ------
    ValueError
        If the two output masks have different shapes, select a different
        number of entries than expected, or overlap; or if the input size
        along ``axis`` does not match ``input_copy_mask``.

    Examples
    --------
    >>> concatenator = AutoConcatenator(
    ...     input_copy_mask=np.array([True, False, True]),
    ...     output_original_mask=np.array([True, True, True, False, False]),
    ...     output_copy_mask=np.array([False, False, False, True, True]))
    >>> print(concatenator(np.array([1, 2, 3])))
    [1 2 3 1 3]
    >>> print(concatenator(np.array([[1, 2, 3], [4, 5, 6]])))
    [[1 2 3 1 3]
     [4 5 6 4 6]]
    """
    input_copy_mask: Optional[np.ndarray] = field(default=None, kw_only=True)
    output_original_mask: Optional[np.ndarray] = field(default=None, kw_only=True)
    output_copy_mask: Optional[np.ndarray] = field(default=None, kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="auto concatenator", kw_only=True)

    def __post_init__(self):
        # Subclasses build their masks in prepare(); nothing to validate yet
        if self.output_original_mask is None and self.output_copy_mask is None:
            return

        # Check if the sizes of output_original_mask and output_copy_mask are the same
        if self.output_copy_mask is None or self.output_original_mask is None:
            raise ValueError(
                "AutoConcatenator needs both output masks or neither: "
                "got one of them set and the other None.")
        if self.output_original_mask.shape != self.output_copy_mask.shape:
            raise ValueError("output_original_mask and output_copy_mask must have the same shape.")

        # Check the number of True values in the masks
        if self.input_copy_mask is not None:
            num_true_input_copy = np.sum(self.input_copy_mask)
            num_true_output_copy = np.sum(self.output_copy_mask)

            if num_true_input_copy != num_true_output_copy:
                raise ValueError("The number of True values in input_copy_mask must be equal to the number of True values in output_copy_mask.")

        # check if there is no overlap between allocated value
        if np.any(np.logical_and(self.output_original_mask, self.output_copy_mask)):
            raise ValueError("The two output masks overlap.")


    def extract_copy(self, X: np.ndarray) -> np.ndarray:
        """
        Extract a copy  from the input signal on the specified axis
        """
        slices: list[slice | np.ndarray] = [slice(None)] * len(X.shape)
        slices[self.axis] = _required_mask(self.input_copy_mask,
                                           type(self).__name__,
                                           "input_copy_mask")
        X_copy = X[tuple(slices)]
        return X_copy

    def process_copy(self, X: np.ndarray) -> np.ndarray:
        return X

    def forward(self, X: np.ndarray) -> np.ndarray:
        block = type(self).__name__
        input_copy_mask = _required_mask(self.input_copy_mask, block,
                                         "input_copy_mask")
        output_original_mask = _required_mask(self.output_original_mask,
                                              block, "output_original_mask")
        output_copy_mask = _required_mask(self.output_copy_mask, block,
                                          "output_copy_mask")

        if X.shape[self.axis] != len(input_copy_mask):
            raise ValueError(f"input signal for the dimension {self.axis} and input_copy_mask must have the same shape.")

        X_copy = self.extract_copy(X)
        X_copy_processed = self.process_copy(X_copy)

        # Prepare the output array
        shape = list(X.shape)
        shape[self.axis] = len(output_original_mask)
        Y = np.zeros(shape, dtype=X.dtype)

        # Create slicing object
        slices: list[slice | np.ndarray] = [slice(None)] * len(X.shape)

        # Assign original data
        slices[self.axis] = output_original_mask
        Y[tuple(slices)] = X

        # Assign copied data
        slices[self.axis] = output_copy_mask
        Y[tuple(slices)] = X_copy_processed
        return Y



@dataclass(slots=True)
class SampleRemover(Processor):
    r"""
    Deletes a contiguous run of samples from a serial signal.

    Signal Model
    ------------
    A window of :math:`D` samples starting at index :math:`n_0` is cut
    out of the input, and the two remaining segments are spliced
    together:

    .. math::

        y[n] = \left\{\begin{array}{cl}
        x[n] & \text{for } 0 \le n < n_0,\\
        x[n + D] & \text{for } n_0 \le n < N - D,
        \end{array}\right.

    where :math:`x[n]` has length :math:`N`, :math:`n_0` is the index of
    the first deleted sample, :math:`D` the number of deleted samples,
    and :math:`y[n]` has length :math:`N - D`. It is the exact inverse of
    :class:`DataAdder` when the same :math:`n_0` is used.

    Axes: *declared axis* -- operates on a 1D serial signal ``(N,)``; the
    splice uses flat indexing and does not broadcast.

    Parameters
    ----------
    N_start : int
        Index :math:`n_0` of the first sample to delete. Default is 0.
    length : int, optional, keyword-only
        Number :math:`D` of samples to delete. Default is 0.
    name : str, optional, keyword-only
        Name of the processor instance. Default is
        ``"symbol remover"``.

    Raises
    ------
    comnumpy.ShapeError
        If the input is not a 1D Serial signal ``(N,)``.

    Examples
    --------
    >>> print(SampleRemover(N_start=2, length=3)(np.arange(8)))
    [0 1 5 6 7]
    >>> print(SampleRemover(N_start=0, length=2)(np.arange(5)))
    [2 3 4]
    >>> SampleRemover(N_start=1, length=1)(np.arange(6).reshape(2, 3))
    Traceback (most recent call last):
        ...
    comnumpy.exceptions.ShapeError: SampleRemover expects a 1D Serial signal (N,), got 2D (2, 3) -- ...
    """
    N_start: int = 0
    length: int = field(default=0, kw_only=True)
    name: str = field(default="symbol remover", kw_only=True)

    def prepare(self, x: np.ndarray) -> None:
        if x.ndim != 1:
            raise ShapeError(
                f"SampleRemover expects a 1D Serial signal (N,), got "
                f"{x.ndim}D {x.shape} -- flatten the stream with "
                f"Parallel2Serial, or apply the block to each stream "
                f"separately.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(len(x) - self.length, dtype=x.dtype)
        y[:self.N_start] = x[:self.N_start]
        y[self.N_start:] = x[self.N_start + self.length:]
        return y


@dataclass(slots=True)
class DelayRemover(Processor):
    r"""
    Removes the first samples of a signal to compensate a known delay.

    Signal Model
    ------------
    A pure advance by the known integer delay :math:`d`, i.e. a shift
    that discards the transient introduced upstream (filter group delay,
    channel delay):

    .. math::

        y[n] = x[n + d], \qquad 0 \le n < N - d

    where :math:`x[n]` has length :math:`N`, :math:`d` is the delay to
    remove, and :math:`y[n]` has length :math:`N - d`.

    Axes: *declared axis* -- the samples are dropped along ``axis``
    (default -1, the sample axis of the Serial layout ``(..., N)``); the
    remaining axes are untouched.

    Parameters
    ----------
    delay : int
        Number :math:`d` of leading samples to remove.
    axis : int, optional, keyword-only
        Axis along which the delay is removed. Default is -1.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"delay remover"``.

    Examples
    --------
    >>> print(DelayRemover(3)(np.arange(8)))
    [3 4 5 6 7]
    >>> print(DelayRemover(1)(np.array([[1, 2, 3], [4, 5, 6]])))
    [[2 3]
     [5 6]]
    """
    delay: int
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="delay remover", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Create a slice object for the specified axis
        slices = [slice(None)] * x.ndim
        slices[self.axis] = slice(self.delay, None)

        # Apply the slice to the input array
        y = x[tuple(slices)]
        return y


@dataclass(slots=True)
class DataAdder(Processor):
    r"""
    Inserts a known sequence into a serial signal at a given index.

    Signal Model
    ------------
    A sequence :math:`s[m]` of length :math:`D` (a preamble, a pilot
    block, a guard sequence) is spliced into the input at index
    :math:`n_0`:

    .. math::

        y[n] = \left\{\begin{array}{cl}
        x[n] & \text{for } 0 \le n < n_0,\\
        s[n - n_0] & \text{for } n_0 \le n < n_0 + D,\\
        x[n - D] & \text{for } n_0 + D \le n < N + D,
        \end{array}\right.

    where :math:`x[n]` has length :math:`N` and :math:`y[n]` has length
    :math:`N + D`. Applying :class:`SampleRemover` with the same
    :math:`n_0` and :math:`D` restores :math:`x[n]`.

    Axes: *declared axis* -- operates on a 1D serial signal ``(N,)``; the
    splice is a concatenation along the first axis.

    Parameters
    ----------
    symbol : np.ndarray
        Sequence :math:`s[m]` to insert, of length :math:`D`.
    N_start : int, optional, keyword-only
        Insertion index :math:`n_0`, in ``[0, N]``. Default is 0.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"Data Adder"``.

    Raises
    ------
    comnumpy.ShapeError
        If the input, or ``symbol``, is not a 1D Serial signal ``(N,)``.
    ValueError
        If ``N_start`` is outside ``[0, len(x)]``.

    Examples
    --------
    >>> print(DataAdder(np.array([-1, -1]), N_start=2)(np.arange(5)))
    [ 0  1 -1 -1  2  3  4]
    >>> print(DataAdder(np.array([-1, -1]))(np.arange(3)))
    [-1 -1  0  1  2]
    >>> DataAdder(np.array([-1, -1]))(np.arange(6).reshape(2, 3))
    Traceback (most recent call last):
        ...
    comnumpy.exceptions.ShapeError: DataAdder expects a 1D Serial signal (N,), got 2D (2, 3) -- ...
    """
    symbol: np.ndarray
    N_start: int = field(default=0, kw_only=True)
    name: str = field(default="Data Adder", kw_only=True)

    def prepare(self, x: np.ndarray) -> None:
        if x.ndim != 1:
            raise ShapeError(
                f"DataAdder expects a 1D Serial signal (N,), got "
                f"{x.ndim}D {x.shape} -- flatten the stream with "
                f"Parallel2Serial, or apply the block to each stream "
                f"separately.")
        symbol = np.asarray(self.symbol)
        if symbol.ndim != 1:
            raise ShapeError(
                f"DataAdder expects a 1D symbol sequence (D,), got "
                f"{symbol.ndim}D {symbol.shape} -- ravel the inserted "
                f"sequence before passing it as symbol=.")

    def validate_input(self, x: np.ndarray):
        if self.N_start < 0 or self.N_start > len(x):
            raise ValueError("N_start is out of bounds for the input signal.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_input(x)
        y = np.concatenate((x[:self.N_start], self.symbol, x[self.N_start:]))
        return y


@dataclass(slots=True)
class DataExtractor(Processor):
    r"""
    Extracts a segment of a signal using NumPy-style indexing.

    Signal Model
    ------------
    The block keeps the samples whose index belongs to the selection
    :math:`\mathcal{S}` described by ``selector``:

    .. math::

        y[m] = x\!\left[\mathcal{S}_m\right], \qquad
        0 \le m < |\mathcal{S}|

    With ``selector = (n_0, n_1, \Delta)`` the selection is the
    arithmetic progression
    :math:`\mathcal{S} = \{ n_0, n_0 + \Delta, \dots \} \cap [n_0, n_1)`,
    so :math:`n_0` is the start index, :math:`n_1` the (excluded) stop
    index and :math:`\Delta` the step. An explicit list or array gives
    :math:`\mathcal{S}` directly, an integer selects a single sample, and
    ``None`` makes the block the identity :math:`y[n] = x[n]`.

    Axes: *declared axis* -- the selector indexes the first axis, which
    is the sample axis of a 1D Serial signal ``(N,)``; on a
    multidimensional input it selects entries of axis 0 and the trailing
    axes are carried through.

    Parameters
    ----------
    selector : int, slice, tuple, list, np.ndarray or None, optional
        Description of the selection :math:`\mathcal{S}`: an ``int`` for
        a single index, a ``tuple`` ``(start, stop[, step])`` converted to
        a slice, a ``slice``, a list or array of explicit indices, or
        ``None`` for a pass-through. Default is ``None``.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"Data Extractor"``.

    Examples
    --------
    >>> x = np.arange(10)
    >>> print(DataExtractor(3)(x))
    3
    >>> DataExtractor((2, 8))(x)
    array([2, 3, 4, 5, 6, 7])
    >>> DataExtractor((1, 9, 2))(x)
    array([1, 3, 5, 7])
    >>> DataExtractor((1, 3))(np.arange(20).reshape(4, 5))
    array([[ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    """
    selector: Optional[Union[int, slice, tuple[int, ...], list[int],
                             np.ndarray]] = None
    name: str = field(default="Data Extractor", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return x

        # transformer un tuple en slice si c'est un tuple simple
        if isinstance(self.selector, tuple):
            self.selector = slice(*self.selector)

        return x[self.selector]


@dataclass(slots=True)
class Resampler(Processor):
    r"""
    Changes the sampling rate of a signal by a rational factor.

    Signal Model
    ------------
    The rate is changed by the rational factor :math:`P/Q`: the signal is
    upsampled by :math:`P`, lowpass filtered, then decimated by
    :math:`Q`. Writing :math:`h[k]` for the anti-imaging / anti-aliasing
    filter,

    .. math::

        y[n] = \sum_{k} h[k] \; v\!\left[n Q - k\right], \qquad
        v[m] = \left\{\begin{array}{cl}
        x[m / P] & \text{if } m \bmod P = 0,\\
        0 & \text{otherwise,}
        \end{array}\right.

    so that an input of :math:`N` samples yields
    :math:`\lceil N P / Q \rceil` output samples and, for a signal
    bandlimited below :math:`\min(1/P, 1/Q)`, :math:`y[n] \approx
    x_c\!\left(n Q / P\right)` where :math:`x_c` is the underlying
    continuous-time waveform. The three stages are evaluated jointly by
    a polyphase implementation (``scipy.signal.resample_poly``), which
    never computes the discarded samples.

    Axes: *declared axis* -- the resampling runs along the last axis, the
    sample axis of the Serial layout ``(..., N)``.

    Parameters
    ----------
    up : int
        Interpolation factor :math:`P`. Must be a positive integer.
    down : int
        Decimation factor :math:`Q`. Must be a positive integer.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"Resampler"``.

    Raises
    ------
    ValueError
        If ``up`` or ``down`` is not a positive integer.

    References
    ----------
    A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*,
    3rd ed., Pearson, 2010, Chapter 4 (changing the sampling rate by a
    non-integer factor, polyphase implementation).

    Examples
    --------
    >>> x = np.cos(2 * np.pi * 0.02 * np.arange(48))
    >>> y = Resampler(up=3, down=2)(x)
    >>> print(y.shape)
    (72,)
    >>> m = np.arange(20, 52)
    >>> ideal = np.cos(2 * np.pi * 0.02 * (2 / 3) * m)
    >>> print(round(float(np.max(np.abs(y[20:52] - ideal))), 4))
    0.0006
    """
    up: int
    down: int
    name: str = field(default="Resampler", kw_only=True)

    def __post_init__(self):

        if self.up <= 0 or self.down <= 0:
            raise ValueError("Both 'up' and 'down' factors must be positive integers.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        from scipy import signal  # local import (D36)
        y = signal.resample_poly(x, self.up, self.down)
        return y


@dataclass(slots=True)
class Clipper(Processor):
    r"""
    Saturates a real signal to a symmetric amplitude interval.

    Signal Model
    ------------
    The clipper is a memoryless nonlinearity: it is transparent below the
    threshold :math:`\tau` and saturates above it,

    .. math::

        y[n] = \max\left(-\tau, \min\left(x[n], \tau\right)\right)
        = \left\{\begin{array}{cl}
        x[n] & \text{if } |x[n]| \le \tau,\\
        \tau \, \mathrm{sign}(x[n]) & \text{if } |x[n]| > \tau,
        \end{array}\right.

    where :math:`\tau > 0` is the saturation threshold. Clipping reduces
    the peak-to-average power ratio at the cost of in-band distortion and
    spectral regrowth.

    .. NOTE::
        The interval :math:`[-\tau, \tau]` is a *real* interval: this
        block is meant for real-valued signals. For the polar clipping
        :math:`\tau e^{i \angle x[n]}` of a complex baseband envelope,
        use :class:`comnumpy.ofdm.predistorders.HardClipper`.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    threshold : float
        Saturation threshold :math:`\tau`.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"Clipper"``.

    References
    ----------
    S. C. Cripps, *RF Power Amplifiers for Wireless Communications*,
    2nd ed., Artech House, 2006 (memoryless saturating nonlinearity).

    Examples
    --------
    >>> print(Clipper(threshold=1.5)(np.array([-3.0, -1.0, 0.5, 2.0])))
    [-1.5 -1.   0.5  1.5]
    >>> print(Clipper(threshold=1.0)(np.array([[0.2, 5.0], [-5.0, -0.2]])))
    [[ 0.2  1. ]
     [-1.  -0.2]]
    """
    threshold: float
    name: str = field(default="Clipper", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, -self.threshold, self.threshold)
        return y



@dataclass(slots=True)
class BlindPhaseTracker(Processor):
    r"""
    Blind carrier phase recovery by exhaustive phase search (BPS).

    Signal Model
    ------------
    The input carries an unknown, slowly varying carrier phase
    :math:`\theta[n]`:

    .. math::

        x[n] = s[n] \, e^{i \theta[n]} + b[n]

    where :math:`s[n]` belongs to the known alphabet :math:`\mathcal{A}`.
    For each sample, the :math:`B` test phases
    :math:`\varphi_p = \frac{\pi}{4}\left(\frac{2p}{B} - 1\right)`,
    :math:`p = 0, \dots, B-1`, are applied, and the one minimizing the
    local squared error over a sliding window of :math:`2L + 1` samples
    is retained:

    .. math::

        \hat{\theta}[n] = \arg\min_{\varphi_p}
        \frac{1}{2L+1} \sum_{m=-L}^{L}
        \left| x[n+m] e^{-i \varphi_p}
        - \mathcal{D}\!\left(x[n+m] e^{-i \varphi_p}\right) \right|^2

    where :math:`\mathcal{D}(z) = \arg\min_{a \in \mathcal{A}} |z - a|^2`
    is the hard decision on the alphabet. The compensated output is

    .. math::

        y[n] = x[n] \, e^{-i \hat{\theta}[n]}

    The search range is limited to :math:`[-\pi/4, \pi/4)`, the phase
    ambiguity of a square QAM constellation; residual :math:`\pi/2`
    cycle slips must be resolved elsewhere (differential encoding or
    pilots). The window half-length :math:`L` sets the usual trade-off:
    a long window averages the noise, a short one follows a fast phase.

    The estimand is **per path** (D49): a ``(..., P, N)`` signal gets one
    phase trajectory per path, and ``theta_`` has the shape of the
    input. After a butterfly equalizer each output carries its own
    residual rotation, so tracking the paths jointly would fit a single
    phase to two different ones.

    Axes: *declared axis* -- operates on a 1D serial signal ``(N,)``; the
    search loops over the sample axis.

    Parameters
    ----------
    L : int
        Half-length :math:`L` of the sliding window; the local cost is
        averaged over :math:`2L + 1` samples.
    alphabet : np.ndarray
        Constellation :math:`\mathcal{A}` used by the hard decision
        :math:`\mathcal{D}(\cdot)`.
    phase_steps : int, optional
        Number :math:`B` of test phases uniformly spanning
        :math:`[-\pi/4, \pi/4)`. Default is 10.

    Attributes
    ----------
    phases : np.ndarray
        The :math:`B` test phases :math:`\varphi_p`, precomputed from
        ``phase_steps`` alone (parameter-derived, hence no trailing
        underscore).
    theta_ : np.ndarray
        The estimated phase trajectory :math:`\hat{\theta}[n]` of the last
        call, of length :math:`N` (data-derived, hence the trailing
        underscore, D23). A block never draws (D25, D42): plot it from the
        caller, for instance with ``matplotlib.pyplot.plot(tracker.theta_)``.

    References
    ----------
    T. Pfau, S. Hoffmann, R. Noe, "Hardware-Efficient Coherent Digital
    Receiver Concept With Feedforward Carrier Recovery for M-QAM
    Constellations", Journal of Lightwave Technology, vol. 27, no. 8,
    pp. 989-999, 2009, doi: 10.1109/JLT.2008.2010511.

    Examples
    --------
    >>> alphabet = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    >>> rng = np.random.default_rng(42)
    >>> s = alphabet[rng.integers(0, 4, size=12)]
    >>> x = s * np.exp(1j * 0.3)
    >>> tracker = BlindPhaseTracker(L=3, alphabet=alphabet, phase_steps=16)
    >>> y = tracker(x)
    >>> print(round(float(np.max(np.abs(y - s))), 3))
    0.005
    >>> print(tracker.theta_.shape, round(float(np.mean(tracker.theta_)), 3))
    (12,) 0.295
    """

    L: int
    alphabet: np.ndarray
    phase_steps: int = 10
    phases: np.ndarray = field(init=False)
    # estimated quantity (D23): phase trajectory of the last run
    theta_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)


    def __post_init__(self):
        self.phases = np.linspace(-np.pi/4, np.pi/4, self.phase_steps, endpoint=False)

    def hard_projector(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.atleast_2d(np.ravel(z))
        distances = np.abs(z.T - self.alphabet)**2
        indices = np.argmin(distances, axis=1)
        symbols = self.alphabet[indices]
        return indices, symbols

    def evm_cost(self, x: np.ndarray, n: int, phi: float) -> float:
        # Compute local EVM cost around index n
        total_error = 0.0
        count = 0
        for m in range(-self.L, self.L + 1):
            idx = n + m
            if 0 <= idx < len(x):
                rotated = x[idx] * np.exp(-1j * phi)
                _, closest = self.hard_projector(rotated)
                # hard_projector answers with a one-element array, since
                # it accepts a record as well as a single sample
                total_error += float(np.sum(np.abs(rotated - closest)**2))
                count += 1
        return float(total_error / count) if count > 0 else float(np.inf)

    def _track(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Track one path: the search itself, on a 1-D record."""
        y_corrected = np.zeros_like(x, dtype=complex)
        optimal_phases = []

        for n in range(len(x)):
            costs = [self.evm_cost(x, n, phi) for phi in self.phases]
            best_phi_idx = np.argmin(costs)
            best_phi = self.phases[best_phi_idx]
            optimal_phases.append(best_phi)
            y_corrected[n] = x[n] * np.exp(-1j * best_phi)

        return y_corrected, np.asarray(optimal_phases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        signal = np.asarray(x)
        if signal.ndim == 1:
            y_corrected, phases = self._track(signal)
        else:
            # one trajectory per path (D49): after a butterfly equalizer
            # each output carries its own residual rotation, so tracking
            # them jointly would fit one phase to two different ones
            tracked = [self._track(path)
                       for path in signal.reshape(-1, signal.shape[-1])]
            y_corrected = np.stack([y for y, _ in tracked]).reshape(signal.shape)
            phases = np.stack([p for _, p in tracked]).reshape(signal.shape)

        # estimated quantity (D23), exposed for the caller to plot (D25, D42)
        self.theta_ = phases
        return y_corrected
