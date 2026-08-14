import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional
from comnumpy.core.generics import Processor
from comnumpy.core.fading import PowerDelayProfile
from comnumpy.exceptions import ShapeError

if TYPE_CHECKING:                      # matplotlib is never imported at
    from matplotlib.axes import Axes   # module load time (decision D36)

__all__ = ["AWGN", "FIRChannel", "TappedDelayLineChannel"]


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

    Axes: *element-wise* -- applied pointwise, shape-agnostic. One
    caveat under a batch (D51): with ``snr_dB=`` the signal power is
    measured over the **whole** input, batch rows included, so rows of
    unequal power share one variance rather than each receiving the
    requested SNR; ``sigma2=`` is batch-neutral.

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

        if sigma2n < 0:
            # a swept sigma2 gone negative would otherwise turn into
            # sqrt(<0) and an all-NaN pass with only a RuntimeWarning
            raise ValueError(
                f"AWGN: the noise variance is negative (sigma2={sigma2n}); "
                f"a variance is non-negative -- check the sweep values.")

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

    Axes: *axis -1* -- the convolution runs along the last axis
    (``scipy.signal.convolve`` semantics; output length depends on
    ``mode``); leading axes are batch, every row through the same taps.

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
        # scipy wants kernel and signal of equal rank: size-1 leading
        # axes keep those dimensions untouched in every mode
        h = np.reshape(self.h, (1,) * (np.ndim(x) - 1) + (-1,))
        y = signal.convolve(x, h, mode=self.mode)
        return y

    def info(self) -> dict[str, Any]:
        """What this channel is, as a dictionary.

        The figures an engineer reads off a tap vector before deciding
        anything: how long it is, how much of the energy sits in the
        first path, and how far the frequency response swings between
        its best and its worst point -- the number that says whether one
        equalizer tap will do.

        Returns
        -------
        dict of str to object

        Examples
        --------
        >>> info = FIRChannel(np.array([1.0, 0.0, 0.5])).info()
        >>> info["n_taps"], round(info["peak_to_notch_dB"], 2)
        (3, 9.54)
        """
        taps = np.asarray(self.h).ravel()
        energy = float(np.sum(np.abs(taps) ** 2))
        response = np.abs(np.fft.fft(taps, 512)) ** 2
        return {
            "kind": "FIR",
            "n_taps": int(taps.size),
            "energy": energy,
            "first_tap_fraction": float(np.abs(taps[0]) ** 2 / energy),
            "peak_to_notch_dB": float(
                10 * np.log10(response.max() / max(response.min(), 1e-300))),
        }

    def plot(self, domain: Literal["impulse", "frequency"] = "impulse", *,
             scale: Literal["linear", "dB"] = "linear",
             fs: Optional[float] = None, ax: "Optional[Axes]" = None):
        """Draw the impulse response, or the frequency response of it.

        See :func:`~comnumpy.core.visualizers.plot_channel_response`.
        """
        from comnumpy.core.visualizers import plot_channel_response  # D36
        return plot_channel_response(np.asarray(self.h), domain=domain,
                                     scale=scale, fs=fs, ax=ax)


@dataclass(slots=True)
class TappedDelayLineChannel(Processor):
    r"""Time-varying multipath fading channel (decision D43).

    Signal Model
    ------------
    The received signal is the sum of the :math:`L` resolvable paths of a
    :class:`~comnumpy.core.fading.PowerDelayProfile`, each weighted by its
    own fading process:

    .. math::

        y[n] = \sum_{l=0}^{L-1} a_l[n] \, x[n - d_l],
        \qquad \mathbb{E}\left[|a_l[n]|^2\right] = \gamma_l

    where :math:`d_l = \mathrm{round}(\tau_l f_s)` is the path delay in
    samples and :math:`\gamma_l` its normalized power. Each :math:`a_l[n]`
    is an independent Rayleigh process with the profile's Doppler
    spectrum (:func:`~comnumpy.core.fading.rayleigh_process`), so the
    channel varies within a single call: this is a time-selective as well
    as frequency-selective model, not block fading. Setting
    ``f_doppler=0`` recovers block fading, one draw per call.

    When the profile declares a Rice factor :math:`K`, the first path
    carries a specular component:

    .. math::

        a_0[n] = \sqrt{\gamma_0}\left(\sqrt{\frac{K}{K+1}} \,
        e^{j \phi} + \sqrt{\frac{1}{K+1}} \, g[n]\right)

    where :math:`g[n]` is the unit-power Rayleigh process of the first
    path, so that :math:`\mathbb{E}[|a_0[n]|^2] = \gamma_0` still
    holds.

    The output keeps the input length: the delayed copies are zero-padded
    at the start, i.e. the channel is causal and the transient is kept.

    Axes: *axis -1* -- expects a 1D serial signal ``(N,)``.

    Parameters
    ----------
    profile : PowerDelayProfile
        Delay profile, from the catalog or hand-built.
    fs : float, keyword-only
        Sampling frequency :math:`f_s` in Hz; sets the mapping from the
        profile's delays in ns onto the sample grid.
    f_doppler : float, optional, keyword-only
        Maximum Doppler :math:`f_D` in Hz. Default 0.0 (block fading).
    seed : int, optional, keyword-only
        Local RNG seed; overridden by chain seeding (``Sequential.seed``).
    name : str, optional, keyword-only
        Name of the channel instance. Default ``"fading"``.

    Attributes
    ----------
    h_ : np.ndarray
        Realized time-varying impulse response of the last run, shape
        ``(L, N)`` -- data-dependent, hence the trailing underscore (D23).
        Row ``l`` is :math:`a_l[n]`; the matching delays are ``delays_``.
    delays_ : np.ndarray
        Path delays in samples used for the last run.

    Raises
    ------
    ShapeError
        If the input is not 1D, or is shorter than the profile's longest
        path at this sampling rate.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 13.1; W. C. Jakes, *Microwave Mobile
    Communications*, Wiley, 1974, Chapter 1.

    Examples
    --------
    >>> from comnumpy.core.fading import get_delay_profile
    >>> channel = TappedDelayLineChannel(get_delay_profile("EVA"),
    ...                                  fs=15.36e6, seed=1)
    >>> y = channel(np.ones(1000, dtype=complex))
    >>> y.shape, channel.delays_.tolist()[:4]
    ((1000,), [0, 2, 5, 6])
    >>> channel.h_.shape          # one fading process per resolvable path
    (8, 1000)
    """
    profile: PowerDelayProfile
    fs: float = field(kw_only=True)
    f_doppler: float = field(default=0.0, kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="fading", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)
    h_: Optional[np.ndarray] = field(init=False, repr=False,
                                     default_factory=lambda: None)
    delays_: Optional[np.ndarray] = field(init=False, repr=False,
                                          default_factory=lambda: None)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def impulse_response(self, n_samples: Optional[int] = None) -> np.ndarray:
        r"""Draw one realization and return it as a tap vector.

        Signal Model
        ------------
        Sounding a channel with an impulse is how its response is
        measured, and it is what a script needs whenever a *known*
        channel is wanted: the equalizers of
        :class:`~comnumpy.core.compensators.LinearEqualizer` and
        :class:`~comnumpy.ofdm.compensators.FrequencyDomainEqualizer`
        take a tap vector, not a fading process. This does exactly that
        -- feeds :math:`\delta[n]` through :meth:`forward` and returns
        the first :math:`L` samples, with :math:`L` one more than the
        longest path in samples.

        The result is dense: paths that fall between the resolvable
        delays of this sampling rate appear as zeros, so the vector can
        be handed straight to a convolution.

        **This draws a new realization** and advances the generator, the
        way calling the block does. Seed it (or the chain) to get the
        same channel twice.

        Axes: *returns 1D* ``(L,)``.

        Parameters
        ----------
        n_samples : int, optional
            Length of the sounding record. Defaults to eight times the
            longest path, which is enough for the profile to fit and
            for a Doppler process to be sampled meaningfully. Only the
            first :math:`L` samples are returned either way.

        Returns
        -------
        np.ndarray
            Complex tap vector of length ``delays_[-1] + 1``.

        Examples
        --------
        >>> from comnumpy.core.fading import get_delay_profile
        >>> channel = TappedDelayLineChannel(get_delay_profile("EPA"),
        ...                                  fs=7.68e6, seed=18)
        >>> h = channel.impulse_response()
        >>> h.shape                  # 410 ns of spread at 7.68 MHz
        (4,)
        >>> bool(np.all(h[channel.delays_] == channel.h_[:, 0]))
        True
        """
        delays, _ = self.profile.to_taps(self.fs)
        length = int(delays[-1]) + 1
        if n_samples is None:
            n_samples = max(8 * length, 64)
        impulse = np.zeros(n_samples, dtype=complex)
        impulse[0] = 1.0
        return self(impulse)[:length]

    def info(self) -> dict[str, Any]:
        """What this channel is, as a dictionary.

        Everything the profile fixes, plus what this sampling rate makes
        of it: a delay spread in nanoseconds becomes a number of taps,
        and a coherence bandwidth is what decides whether a subcarrier
        sees a flat channel.

        Returns
        -------
        dict of str to object

        Examples
        --------
        >>> from comnumpy.core.fading import get_delay_profile
        >>> info = TappedDelayLineChannel(get_delay_profile("EPA"),
        ...                               fs=7.68e6).info()
        >>> info["standard"], info["n_paths"], info["n_taps"]
        ('EPA', 7, 4)
        >>> round(info["rms_delay_spread_ns"]), round(info["fs_Hz"] / 1e6, 2)
        (43, 7.68)
        """
        delays, powers = self.profile.to_taps(self.fs)
        return {
            "kind": "tapped delay line",
            "standard": self.profile.standard,
            "n_paths": self.profile.n_taps,
            "delays_ns": self.profile.delays_ns.tolist(),
            "powers_dB": self.profile.powers_dB.tolist(),
            "rms_delay_spread_ns": self.profile.rms_delay_spread_ns,
            "coherence_bandwidth_Hz": self.profile.coherence_bandwidth_hz,
            "rice_k_dB": getattr(self.profile, "rice_k_dB", None),
            "fs_Hz": float(self.fs),
            "f_doppler_Hz": float(self.f_doppler),
            "n_taps": int(delays[-1]) + 1,
            "resolvable_delays_samples": delays.tolist(),
            "resolvable_powers": powers.tolist(),
        }

    def plot(self, domain: Literal["impulse", "frequency"] = "impulse", *,
             scale: Literal["linear", "dB"] = "linear",
             ax: "Optional[Axes]" = None):
        """Draw one realization, in delay or in frequency.

        A fading channel has no single impulse response, so this draws
        the **last one realized** if the block has been run, and a fresh
        one otherwise -- which advances the generator, exactly as
        :meth:`impulse_response` does.

        See :func:`~comnumpy.core.visualizers.plot_channel_response`.
        """
        from comnumpy.core.visualizers import plot_channel_response  # D36
        if self.h_ is None or self.delays_ is None:
            taps = self.impulse_response()
        else:
            taps = np.zeros(int(self.delays_[-1]) + 1, dtype=complex)
            taps[self.delays_] = self.h_[:, 0]
        return plot_channel_response(taps, domain=domain, scale=scale,
                                     fs=self.fs, ax=ax)

    def prepare(self, x: np.ndarray) -> None:
        from comnumpy.core.fading import validate_taps_fit  # local (D36)
        if x.ndim != 1:
            raise ShapeError(
                f"TappedDelayLineChannel expects a 1D Serial signal (N,), "
                f"got {x.ndim}D {x.shape} -- flatten the stream with "
                f"Parallel2Serial, or apply the channel per stream.")
        validate_taps_fit(self.profile, self.fs, x.shape[-1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        from comnumpy.core.fading import rayleigh_process  # local (D36)

        delays, powers = self.profile.to_taps(self.fs)
        n = x.shape[-1]
        taps = np.empty((delays.size, n), dtype=complex)
        for index, power in enumerate(powers):
            gain = rayleigh_process(n, self.fs, self.f_doppler,
                                    spectrum=self.profile.doppler,
                                    rng=self.rng)
            taps[index] = np.sqrt(power) * gain

        rice_k_dB = getattr(self.profile, "rice_k_dB", None)
        if rice_k_dB is not None:
            # first path carries a specular component of Rice factor K
            k = 10 ** (rice_k_dB / 10)
            phase = np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            diffuse = taps[0] / np.sqrt(k + 1)
            taps[0] = np.sqrt(powers[0] * k / (k + 1)) * phase + diffuse

        y = np.zeros(n, dtype=complex)
        for delay, gain in zip(delays, taps, strict=True):
            if delay:
                y[delay:] += gain[delay:] * x[:n - delay]
            else:
                y += gain * x

        self.h_ = taps
        self.delays_ = delays
        return y
