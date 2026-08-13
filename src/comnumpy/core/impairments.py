import numpy as np
from dataclasses import dataclass, field
from comnumpy.core import Processor

__all__ = ["IQImbalance", "CFO", "Delay"]


@dataclass(slots=True)
class IQImbalance(Processor):
    r"""Apply an IQ imbalance impairment.

    Signal Model
    ------------
    The output is a widely-linear combination of the input signal and its
    complex conjugate:

    .. math::

        y[n] = \alpha x[n] + \beta x^*[n]

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    alpha : complex
        Complex weight :math:`\alpha` applied to the direct signal
        :math:`x[n]`.
    beta : complex
        Complex weight :math:`\beta` applied to the conjugate signal
        :math:`x^*[n]`.
    name : str, optional, keyword-only
        Name of the impairment instance. Default is ``"iq_impairment"``.

    References
    ----------
    I. Fatadin, S. J. Savory, D. Ives, "Compensation of Quadrature
    Imbalance in an Optical QPSK Coherent Receiver," IEEE Photonics
    Technology Letters 20(20), 2008.

    Examples
    --------
    >>> iq = IQImbalance(alpha=1.0, beta=0.1)
    >>> print(iq(np.array([1.0+1.0j, 1.0-1.0j])))
    [1.1+0.9j 1.1-0.9j]
    """
    alpha: complex
    beta: complex
    name: str = field(default="iq_impairment", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.alpha*x + self.beta*np.conj(x)
        return y


@dataclass(slots=True)
class CFO(Processor):
    r"""Apply a Carrier Frequency Offset (CFO).

    Signal Model
    ------------
    The input signal is multiplied by a complex exponential phase ramp:

    .. math::

        y[n] = x[n] \, e^{j \omega_0 n}

    where :math:`\omega_0` is the normalized carrier frequency offset in
    rad/sample.

    Axes: *axis -1* -- the time index :math:`n` runs along the last
    axis; leading axes are batch, every row sees the same offset.

    Parameters
    ----------
    cfo : float
        Normalized carrier frequency offset :math:`\omega_0` in
        rad/sample.
    name : str, optional, keyword-only
        Name of the impairment instance. Default is ``"cfo_impairment"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 5.

    Examples
    --------
    >>> cfo = CFO(cfo=np.pi/2)
    >>> y = cfo(np.ones(4, dtype=complex))
    >>> print(np.round(y, 2) + 0.0)
    [ 1.+0.j  0.+1.j -1.+0.j  0.-1.j]
    """
    cfo: float
    name: str = field(default="cfo_impairment", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_vect = np.arange(x.shape[-1])
        y = x * np.exp(1j*self.cfo*n_vect)
        return y


@dataclass(slots=True)
class Delay(Processor):
    r"""Discard the first :math:`\tau` samples of the input signal (delay removal).

    Signal Model
    ------------
    The block drops the first :math:`\tau` samples, i.e. it advances the
    signal by :math:`\tau` samples:

    .. math::

        y[n] = x[n + \tau], \qquad 0 \le n < N - \tau

    When ``pad_zeros`` is True, the output is zero-padded at the end to
    keep the input length :math:`N`; otherwise the output has length
    :math:`N - \tau`.

    Axes: *axis -1* -- samples are dropped along the last axis;
    leading axes are batch.

    Parameters
    ----------
    tau : int
        Number of samples :math:`\tau` to discard (non-negative integer).
    pad_zeros : bool, optional, keyword-only
        If True (default), pads the output with trailing zeros to match
        the input size :math:`N`.
    name : str, optional, keyword-only
        Name of the impairment instance. Default is ``"delay_impairment"``.

    Raises
    ------
    ValueError
        If ``tau`` is negative (validated at construction).

    References
    ----------
    H. Meyr, M. Moeneclaey, S. Fechtel, *Digital Communication
    Receivers*, Wiley, 1998, Chapter 4 (timing offsets).

    Examples
    --------
    >>> delay = Delay(2)
    >>> print(delay(np.array([1.0, 2.0, 3.0, 4.0])))
    [3. 4. 0. 0.]
    """
    tau: int
    pad_zeros: bool = field(default=True, kw_only=True)
    name: str = field(default="delay_impairment", kw_only=True)

    def __post_init__(self):
        if self.tau < 0:
            raise ValueError("Delay must be a non-negative integer.")

    def forward(self, x: np.ndarray) -> np.ndarray:

        x_delayed = x[..., self.tau:]
        if self.pad_zeros:
            y = np.zeros(x.shape, dtype=x.dtype)
            y[..., :x_delayed.shape[-1]] = x_delayed
        else:
            y = x_delayed

        return y
