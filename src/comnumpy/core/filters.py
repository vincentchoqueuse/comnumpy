import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

__all__ = ["SRRCFilter", "BWFilter"]


@dataclass(slots=True)
class SRRCFilter(Processor):
    r"""Square-root raised cosine (SRRC) FIR pulse-shaping filter.

    Signal Model
    ------------
    The output is the discrete convolution of the input with the truncated
    SRRC impulse response :math:`h[m]`:

    .. math::

        y[n] = \sum_{m=0}^{2 N_h L} h[m] \, x[n - m]

    The taps are the continuous-time SRRC response sampled at
    :math:`t = m / L` symbol periods, where :math:`L` is the oversampling
    factor. With a symbol period normalized to :math:`T = 1` and a
    roll-off factor :math:`\rho \in [0, 1]`:

    .. math::

        h(t) = \frac{\sin\left(\pi t (1 - \rho)\right)
                     + 4 \rho t \cos\left(\pi t (1 + \rho)\right)}
                    {\pi t \left(1 - (4 \rho t)^2\right)}

    completed at its two removable singularities by

    .. math::

        h(0) = 1 + \rho \left(\frac{4}{\pi} - 1\right), \qquad
        h\left(\pm \frac{1}{4 \rho}\right) = \frac{\rho}{\sqrt{2}}
        \left[ \left(1 + \frac{2}{\pi}\right) \sin \frac{\pi}{4 \rho}
             + \left(1 - \frac{2}{\pi}\right) \cos \frac{\pi}{4 \rho}
        \right]

    The response is truncated to :math:`t \in [-N_h, N_h]`, i.e.
    :math:`2 N_h L + 1` taps, and scaled to unit energy
    :math:`\sum_m h^2[m] = 1` when ``norm`` is True.

    **Why the square root, and why on both sides.** Nyquist's first
    criterion states that a pulse :math:`g(t)` produces no intersymbol
    interference when it vanishes at every non-zero multiple of the
    symbol period, :math:`g(mT) = 0` for :math:`m \neq 0`. The raised
    cosine pulse satisfies it. Here the criterion must hold on the
    *cascade* transmit filter / matched receive filter, not on either one
    alone: splitting the raised cosine into two identical square-root
    factors,

    .. math::

        G_{\mathrm{RC}}(f) = H_{\mathrm{SRRC}}(f) \, H_{\mathrm{SRRC}}^*(f)
                           = \left| H_{\mathrm{SRRC}}(f) \right|^2

    gives a receive filter matched to the transmit pulse -- which
    maximizes the sampled signal-to-noise ratio in additive white
    Gaussian noise -- while their convolution is the raised cosine, hence
    ISI-free at the symbol instants. The same ``SRRCFilter`` instance is
    therefore placed at the transmitter and at the receiver.

    Axes: *axis -1* -- both methods filter along the last axis and
    broadcast over the leading ones, so a polarization pair
    ``(..., 2, N)`` is shaped one polarization at a time.

    Parameters
    ----------
    rho : float
        Roll-off factor :math:`\rho \in [0, 1]`. ``rho = 0`` gives the
        sinc pulse (minimum bandwidth :math:`1/2T`), ``rho = 1`` doubles
        the excess bandwidth.
    oversampling : int
        Oversampling factor :math:`L` in samples per symbol; the taps are
        the response sampled at :math:`t = m / L`.
    N_h : int, optional, keyword-only
        Half-length :math:`N_h` of the filter in symbol periods, so the
        FIR has :math:`2 N_h L + 1` taps. Default is 10.
    norm : bool, optional, keyword-only
        If True, normalize the taps to unit energy
        :math:`\sum_m h^2[m] = 1`. Default is True.
    scale : float, optional, keyword-only
        Amplitude scaling factor applied to the output. Default is
        1.0, applied by both methods.
    method : {"lfilter", "fft"}, optional, keyword-only
        Filtering method: ``"lfilter"`` for the causal FIR convolution
        (``scipy.signal.lfilter``), ``"fft"`` for the circular
        frequency-domain product with the delay-compensated response
        returned by :meth:`H`. Default is ``"lfilter"``.
    axis : int, optional, keyword-only
        Filtering axis. Default is -1. The ``"lfilter"`` method honours
        it; the ``"fft"`` method always filters along the last axis.
    name : str, optional, keyword-only
        Name of the filter instance. Default is ``"SRRCFilter"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 9 (signal design for band-limited
    channels: Nyquist criterion, raised cosine and its square root).

    Examples
    --------
    >>> srrc = SRRCFilter(0.25, 4)  # roll-off 0.25, 4 samples per symbol
    >>> h = srrc.h()
    >>> h.shape, round(float(np.sum(h ** 2)), 6)
    ((81,), 1.0)
    >>> g = np.convolve(h, h)  # transmit pulse * matched filter = raised cosine
    >>> print(np.round(np.abs(g[len(h) - 1 + 4::4][:5]), 3))  # t = T, 2T, ...
    [0. 0. 0. 0. 0.]
    """

    rho: float
    oversampling: int
    N_h: int = field(default=10, kw_only=True)
    norm: bool = field(default=True, kw_only=True)
    scale: float = field(default=1.0, kw_only=True)
    method: Literal['lfilter', 'fft'] = field(default="lfilter", kw_only=True)
    axis: int = field(default=-1, kw_only=True)
    name: str = field(default="SRRCFilter", kw_only=True)

    def __post_init__(self) -> None:
        if self.method not in ("lfilter", "fft"):
            raise ValueError(
                f"SRRCFilter: unknown method {self.method!r}; expected "
                f"'lfilter' (causal FIR) or 'fft' (circular, 1D) -- fix the "
                f"method= argument.")

    def h(self, t: Optional[np.ndarray] = None) -> np.ndarray:

        if t is None:
            N = self.N_h*self.oversampling
            n_vect = np.arange(-N, N+1)
            t = n_vect / self.oversampling

        rho = self.rho
        h = np.zeros(len(t))

        for index, t_temp in enumerate(t):
            if t_temp == 0:
                h_temp = (1 + rho*((4/np.pi)-1))
            else:
                # rho = 0 is the sinc pulse: the singularity at 1/(4*rho)
                # moves to infinity and the general branch reduces to it
                if rho != 0 and np.abs(t_temp) == (1/(4*rho)):
                    term1 = (1 + (2/np.pi))
                    term2 = (1 - (2/np.pi))
                    coef = rho / np.sqrt(2)
                    coef2 = np.pi / (4*rho)
                    h_temp = coef*(term1*np.sin(coef2) + term2*np.cos(coef2))
                else:
                    term1 = np.pi * t_temp
                    coef1 = (4 * rho * t_temp)
                    num = np.sin(term1*(1-rho)) + coef1*np.cos(term1*(1+rho))
                    den = term1*(1 - coef1**2)
                    h_temp = num/den

            h[index] = h_temp

        if self.norm:
            h = h / np.sqrt(np.sum(h**2))

        return h

    def H(self, NFFT: int) -> np.ndarray:
        """Frequency response for fft method"""
        from comnumpy._backend import fft  # local import (D36), cupy-compatible (D3)
        # see hager code on LDBP
        h = self.h()
        if NFFT < len(h):
            raise ShapeError(
                f"SRRCFilter(method='fft') needs at least as many samples as "
                f"filter taps ({len(h)} = 2*N_h*oversampling + 1), got {NFFT} "
                f"-- use method='lfilter' or a longer signal")
        filter_delay = self.oversampling*self.N_h
        H_tmp = np.concatenate((h, np.zeros(NFFT-len(h))))
        H_tmp = np.roll(H_tmp, -filter_delay)
        H = fft(H_tmp, n=NFFT)
        return H

    def get_delay(self) -> int:
        return self.N_h

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.method == "lfilter":
            h = self.h()
            from scipy import signal  # local import (D36)
            # asarray: lfilter is typed as returning a tuple when `zi` is
            # given, which it is not here
            y = self.scale * np.asarray(signal.lfilter(h, 1, x, axis=self.axis))
        else:  # "fft" -- validated in __post_init__
            from comnumpy._backend import fft, ifft  # local import (D36), cupy-compatible (D3)
            # x.shape[-1], not len(x): a polarization pair (..., 2, N)
            # would otherwise transform two samples instead of N and
            # fail inside H() with an unrelated message. The response is
            # (NFFT,) and broadcasts over the leading axes.
            NFFT = x.shape[-1]
            fft_x = fft(x, NFFT, axis=-1)
            fft_h = self.H(NFFT)
            y = self.scale*ifft(fft_x*fft_h, NFFT, axis=-1)

        return y


@dataclass(slots=True)
class BWFilter(Processor):
    r"""Ideal (brick-wall) low-pass filter applied in the frequency domain.

    .. note::

       Despite the class name, this block does **not** implement a
       Butterworth response: the mask below is rectangular, i.e. an ideal
       low-pass with infinite roll-off, not the
       :math:`|H(f)|^2 = 1 / (1 + (f/f_c)^{2N})` magnitude of a
       Butterworth filter of order :math:`N`. The docstring describes
       what the code does.

    Signal Model
    ------------
    The input block of :math:`N` samples is transformed, masked and
    transformed back:

    .. math::

        y[n] = \frac{1}{N} \sum_{k=0}^{N-1} H[k] \, X[k] \,
               e^{j 2 \pi k n / N}, \qquad
        X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2 \pi k n / N}

    with the rectangular mask

    .. math::

        H[k] =
        \begin{cases}
        1 & \text{if } |f[k]| \leq w_n / 2 \\
        0 & \text{if } |f[k]| > w_n / 2
        \end{cases}

    where :math:`f[k]` is the frequency of bin :math:`k` in cycles per
    sample (``fftfreq(N, d=1)``, so :math:`|f[k]| \leq 1/2`) and the
    cutoff :math:`w_n` is **normalized to the Nyquist frequency**, the
    convention of ``scipy.signal``: :math:`w_n = 1` passes the whole
    band, :math:`w_n = 1/L` keeps the band of a signal oversampled by
    :math:`L`. Because the mask is applied to a length-:math:`N` DFT, the
    operation is a **circular** convolution with a periodic sinc of
    length :math:`N`, not a linear one.

    Axes: *axis -1* -- the FFT mask acts on the last axis; leading
    axes are batch.

    Parameters
    ----------
    wn : float
        Critical (cutoff) frequency :math:`w_n`, normalized so that
        :math:`w_n = 1` is the Nyquist frequency (``scipy.signal``
        convention), hence :math:`0 < w_n \leq 1`. The retained band in
        cycles per sample is :math:`|f| \leq w_n / 2`.

    Raises
    ------
    ValueError
        If ``wn`` is outside :math:`(0, 1]`.

    References
    ----------
    A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*,
    3rd ed., Prentice Hall, 2010, Chapter 2 (ideal frequency-selective
    filters) and Chapter 8 (filtering by DFT multiplication and its
    circular nature).

    Examples
    --------
    >>> n = np.arange(64)
    >>> x = np.cos(2*np.pi*4*n/64) + np.cos(2*np.pi*20*n/64)  # f = 1/16 and 5/16
    >>> y = BWFilter(0.2)(x)  # cutoff 0.2 x Nyquist = 0.1 cycles/sample
    >>> print(np.round(y[:4].real, 6))
    [1.       0.92388  0.707107 0.382683]
    """
    wn: float

    def __post_init__(self) -> None:
        if not 0 < self.wn <= 1:
            raise ValueError(
                f"BWFilter: expected a cutoff in (0, 1] normalized to the "
                f"Nyquist frequency (scipy.signal convention), got wn="
                f"{self.wn} -- pass wn=1 to keep the whole band, or "
                f"wn=1/L for a signal oversampled by L.")

    def forward(self, x: np.ndarray) -> np.ndarray:

        from comnumpy._backend import fft, ifft, fftfreq  # local import (D36), cupy-compatible (D3)

        # the FFT and the mask act on axis -1; leading axes are batch
        NFFT = x.shape[-1]
        w = fftfreq(NFFT, d=1, like=x)
        # wn is normalized to Nyquist (= 1/2 cycle/sample), hence the /2
        H = (abs(w) <= self.wn / 2).astype(float)
        fft_x = fft(x, NFFT)
        y = ifft(H*fft_x, NFFT)
        if np.isrealobj(x):
            # the mask is symmetric in +/-f, so a real signal stays real:
            # keep its dtype instead of leaking the ifft's complex128
            y = np.real(y)
        return y
