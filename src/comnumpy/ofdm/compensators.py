import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass, field
from comnumpy._backend import fft, fftshift  # cupy-compatible (D3)
from comnumpy.core.processors import WeightAmplifier


@dataclass(slots=True)
class FrequencyDomainEqualizer(WeightAmplifier):
    r"""
    One-tap zero-forcing equalizer that compensates the channel per subcarrier in the frequency domain.

    This class extends the `WeightAmplifier`: the weights are computed once
    from the channel impulse response, then each subcarrier is divided by
    the corresponding bin of the channel frequency response.

    Signal Model
    ------------
    Given a channel impulse response :math:`h[l]`, the Frequency Domain
    Equalizer applies the following weight amplifier:

    .. math::

        y[k] = \frac{x[k]}{H[k]}, \qquad
        H[k] = \sum_{l=0}^{L-1} h[l] \, e^{-i 2 \pi k l / N}

    where:

    * :math:`x[k]` is the input value at subcarrier :math:`k`,
    * :math:`h[l]` is the channel impulse response of length :math:`L`,
    * :math:`H[k]` is the :math:`N`-point DFT of :math:`h[l]`, with :math:`N` the number of subcarriers (taken from the input shape),
    * :math:`y[k]` is the equalized output at subcarrier :math:`k`.

    Axes: *declared axis* -- the weights :math:`1/H[k]` are computed and
    applied along ``axis`` (default -1, the block content axis of the
    Block layout ``(..., T, F)``).

    Parameters
    ----------
    h : np.ndarray
        The impulse response :math:`h[l]` of the channel to be equalized.
        Required (a ValueError is raised when it is missing).
    axis : int, optional, keyword-only
        The axis along which to compute the DFT and apply the weights.
        Default is -1.
    shift : bool, optional, keyword-only
        If True, applies an fftshift to the weights, for inputs whose
        zero-frequency subcarrier is centered. Default is False.
    norm : {"ortho", "backward", "forward"}, optional, keyword-only
        FFT normalization mode. Default is "ortho".
    name : str, optional, keyword-only
        Name of the equalizer instance. Default is
        ``"frequency domain equalizer"``.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 2.

    Examples
    --------
    >>> h = np.array([1.0, 0.5])
    >>> equalizer = FrequencyDomainEqualizer(h=h)
    >>> X = np.fft.fft(h, 4)  # the channel frequency response itself
    >>> print(np.round(equalizer(X), 3))
    [1.+0.j 1.+0.j 1.+0.j 1.+0.j]
    """
    h: Optional[np.ndarray] = None
    axis: int = field(default=-1, kw_only=True)
    shift: bool = field(default=False, kw_only=True)
    norm: Literal["ortho", "backward", "forward"] = field(default="ortho", kw_only=True)
    name: str = field(default="frequency domain equalizer", kw_only=True)
    # internal state (declared for slots, D40a)
    weight: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        if self.h is None:
            raise ValueError("The impulse response 'h' must be provided.")

    def prepare(self, X):
        """
        Compute the amplifier weight from the channel impulse response
        """
        N_sc = X.shape[self.axis]
        Hw = fft(self.h, n=N_sc,  axis=self.axis)
        weight = 1./Hw
        if self.shift:
            weight = fftshift(weight)
        self.weight = weight

