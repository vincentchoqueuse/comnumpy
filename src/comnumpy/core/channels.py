import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core.generics import Processor


@dataclass(slots=True)
class AWGN(Processor):
    r"""Additive white Gaussian noise channel.

    Signal Model
    ------------
    .. math::

        y[n] = x[n] + b[n], \qquad
        b[n] \sim \mathcal{CN}\left(0, \sigma^2\right)

    When parameterized by ``snr_dB``, the variance is derived from the
    measured input power :math:`P_x = \mathbb{E}\left[|x[n]|^2\right]`:

    .. math::

        \sigma^2 = P_x \, 10^{-\mathrm{SNR_{dB}}/10}

    For real-valued inputs, a real Gaussian noise of variance
    :math:`\sigma^2` is applied instead of a circular complex one.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    snr_dB : float, keyword-only
        Signal-to-noise ratio :math:`\mathrm{SNR_{dB}}` in decibels,
        relative to the measured input power. Mutually exclusive with
        ``sigma2``.
    sigma2 : float, keyword-only
        Absolute noise variance :math:`\sigma^2`. Mutually exclusive
        with ``snr_dB``.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the channel instance. Default is ``"awgn"``.

    Attributes
    ----------
    sigma2_ : float
        Variance actually applied. Estimated from the input power when
        parameterized by ``snr_dB`` (data-dependent, hence the trailing
        underscore); equal to ``sigma2`` otherwise.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.2.

    Examples
    --------
    >>> x = np.zeros(10_000, dtype=complex)
    >>> y = AWGN(sigma2=0.01, seed=42)(x)
    >>> print(round(float(np.var(y)), 3))
    0.01
    """
    snr_dB: Optional[float] = field(default=None, kw_only=True)
    sigma2: Optional[float] = field(default=None, kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="awgn", kw_only=True)
    # internal state (created in __post_init__ / forward; declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)
    sigma2_: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    _b: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self) -> None:
        if (self.snr_dB is None) == (self.sigma2 is None):
            raise ValueError(
                "AWGN: specify exactly one of snr_dB= (relative to "
                "measured signal power) or sigma2= (absolute noise "
                "variance); got both or neither."
            )
        self.rng = np.random.default_rng(self.seed)

    def noise_rvs(self, x: np.ndarray) -> None:
        if self.sigma2 is not None:
            sigma2n = self.sigma2
        else:
            # internal invariant: __post_init__ enforced exactly one of
            # snr_dB / sigma2, so snr_dB is set on this branch
            assert self.snr_dB is not None
            P_x = float(np.mean(np.abs(x) ** 2))
            sigma2n = P_x * 10 ** (-self.snr_dB / 10)

        shape = x.shape
        if np.iscomplexobj(x):
            scale = np.sqrt(sigma2n / 2)
            b_r = self.rng.normal(scale=scale, size=shape)
            b_i = self.rng.normal(scale=scale, size=shape)
            b = b_r + 1j * b_i
        else:
            scale = np.sqrt(sigma2n)
            b = self.rng.normal(scale=scale, size=shape)

        self._b = b
        self.sigma2_ = sigma2n

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.noise_rvs(x)
        y = x + self._b
        return y


@dataclass(slots=True)
class FIRChannel(Processor):
    r"""Finite impulse response (FIR) channel with a given impulse response.

    Signal Model
    ------------
    The output :math:`y[n]` is the convolution of the input :math:`x[n]`
    with the impulse response :math:`h[l]` of length :math:`L`:

    .. math::

       y[n] = \sum_{l=0}^{L-1} h[l] \, x[n-l]

    Axes: *declared axis* -- operates on a 1D serial signal ``(N,)``
    (``scipy.signal.convolve`` semantics; output length depends on
    ``mode``).

    Parameters
    ----------
    h : np.ndarray
        The impulse response :math:`h[l]` of the channel, a 1D array of
        length :math:`L`.
    mode : Literal["full", "same", "valid"], optional
        Convolution mode passed to ``scipy.signal.convolve``. Default is
        ``"full"`` (output length :math:`N + L - 1`).
    name : str, optional
        The name of the channel instance. Default is ``"fir"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 9.3 (channels with ISI).

    Examples
    --------
    >>> channel = FIRChannel(np.array([1.0, 0.5]))
    >>> print(channel(np.array([1.0, 0.0, 2.0])))
    [1.  0.5 2.  1. ]
    """
    h: np.ndarray
    mode: Literal["full", "same", "valid"] = "full"
    name: str = "fir"

    def forward(self, x: np.ndarray) -> np.ndarray:
        from scipy import signal  # local import (D36)
        y = signal.convolve(x, self.h, mode=self.mode)
        return y
