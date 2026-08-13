import logging

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional
from comnumpy._backend import fftfreq  # cupy-compatible (D3)
from comnumpy.core import Processor
from .devices import ErbiumDopedFiberAmplifier
from .fiber import FiberSpec
from .raman import RamanSolution
from .constants import PLANCK_CONSTANT
from .utils import (get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain,
                    compute_erbium_doped_fiber_N_ase, apply_kerr_nonlinearity,
                    is_polarization_pair, manakov_kerr, watt_to_dbm,
                    step_transfers, apply_frequency_response)

__all__ = ["FiberLink"]

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class FiberLink(Processor):
    r"""Multi-span fiber link simulated with the split-step Fourier method (SSFM).

    Signal Model
    ------------
    Each span integrates the nonlinear Schroedinger equation (NLSE) for
    the complex field envelope :math:`A(z, t)`:

    .. math::

        \frac{\partial A(z, t)}{\partial z} =
        -\frac{\alpha}{2} A(z, t)
        - j \frac{\beta_2}{2} \frac{\partial^2 A(z, t)}{\partial t^2}
        + j \gamma |A(z, t)|^2 A(z, t)

    where :math:`\alpha = \frac{\ln 10}{10} \alpha_{dB}` is the
    attenuation (Np/km), :math:`\beta_2 = -10^3 D \lambda^2 / (2 \pi c)`
    the group velocity dispersion (ps^2/km) and :math:`\gamma` the Kerr
    coefficient (rad/W/km). The NLSE is integrated numerically with
    ``StPS`` split steps per span (symmetric scheme:
    :math:`D(\delta z/2)\,N(\delta z)\,D(\delta z/2)`; asymmetric scheme:
    :math:`N(\delta z)\,D(\delta z)`). After each span, an EDFA
    compensates the loss with power gain :math:`G` and adds circular ASE
    noise of variance :math:`\sigma^2_{\mathrm{ase}}`:

    .. math::

        G = 10^{\alpha_{dB} L_{\mathrm{span}} / 10}, \qquad
        \sigma^2_{\mathrm{ase}} = \kappa \, f_s \, (G - 1) \, h \nu \, n_{sp}, \qquad
        n_{sp} = \frac{\mathrm{NF} / 2}{1 - 1/G}

    with :math:`\mathrm{NF} = 10^{\mathrm{NF_{dB}} / 10}` the amplifier
    noise figure and :math:`\kappa` a noise scaling factor.

    **Two polarizations (Manakov).** A field shaped ``(..., 2, N)`` --
    the antenna/channel axis of D2 -- is propagated with the Manakov
    equation instead:

    .. math::

        \frac{\partial E_{x,y}}{\partial z} =
        -\frac{\alpha}{2} E_{x,y}
        - j \frac{\beta_2}{2} \frac{\partial^2 E_{x,y}}{\partial t^2}
        + j \frac{8\gamma}{9}
          \left(|E_x|^2 + |E_y|^2\right) E_{x,y}

    The two polarizations share the *total* intensity, and the
    :math:`8/9` comes from averaging the nonlinearity over the fast
    random birefringence of a real fibre. It belongs to the **model**,
    not to the fibre: ``fiber.gamma`` stays the fibre's Kerr
    coefficient in both modes, and which equation is integrated is read
    off the shape of the field, exactly as which pumps are on is read
    off their powers (D41). A ``(..., 1, N)`` field is a single
    polarization and keeps the scalar NLSE with :math:`\gamma`.

    Axes: *declared axis* -- a full-field signal ``(N,)`` or ``(..., P, N)``
    with ``P`` in ``{1, 2}`` polarizations. The leading axes are a
    **batch**: ``(B, P, N)`` propagates ``B`` realizations in one call,
    which is 1.8x faster than the same ``B`` calls at 12288 samples and
    3.1x at 1024 -- the split-step loop, the per-span amplifier and the
    parameter precomputation are paid once instead of ``B`` times. A
    batch of single-polarization fields is ``(B, 1, N)``; ``(B, N)``
    reads ``B`` as polarizations and is refused.

    Parameters
    ----------
    N_spans : int
        Number of spans in the fiber link.
    L_span : float, keyword-only
        Span length :math:`L_{\mathrm{span}}` in km. Default is 80.
    StPS : int, keyword-only
        Number of split steps :math:`\delta z` per span. Default is 1.
    fs : float, keyword-only
        Sampling frequency :math:`f_s` in Hz. Default is 1.
    NF_dB : float, keyword-only
        EDFA noise figure :math:`\mathrm{NF_{dB}}` in dB. Default is 4.
    noise_scaling : float, keyword-only
        ASE noise scaling factor :math:`\kappa` (0 disables the ASE
        noise). Default is 1.
    step_type : {"linear", "logarithmic"}, keyword-only
        Step-size distribution within a span. Default is ``"linear"``.
    step_method : {"symmetric", "asymetric"}, keyword-only
        Split-step scheme. Default is ``"symmetric"``.
    use_only_linear : bool, keyword-only
        If True, only the linear effects (CD and attenuation) are
        simulated, in a single exact step per span. Default is False.
    c : float, keyword-only
        Speed of light :math:`c` in m/s. Default is ``SPEED_OF_LIGHT``.
    h : float, keyword-only
        Planck constant :math:`h` in J.s. Default is ``PLANCK_CONSTANT``.
    gamma : float, keyword-only
        Kerr coefficient :math:`\gamma` in rad/W/km. Default is
        ``KERR_COEFFICIENT``.
    lamb : float, keyword-only
        Wavelength :math:`\lambda` in nm. Default is ``WAVELENGTH``.
    alpha_dB : float, keyword-only
        Fiber loss :math:`\alpha_{dB}` in dB/km. Default is ``FIBER_LOSS``.
    cd_coefficient : float, keyword-only
        Dispersion coefficient :math:`D` in ps/nm/km. Default is
        ``CD_COEFFICIENT``.
    nu : float, keyword-only
        Optical carrier frequency :math:`\nu` in Hz. Default is
        ``OPTICAL_CARRIER_FREQUENCY``.
    step_log_factor : float, keyword-only
        Adjustment factor of the logarithmic step-size distribution.
        Default is 0.4.
    name : str, optional, keyword-only
        Name of the link instance. Default is ``"fiber link"``.
    callbacks : dict of str to callable, optional, keyword-only
        Hooks called during propagation; the key ``"post_span"`` is
        called after each span as ``callback(y, num_span=...)``.
    raman : RamanSolution, optional, keyword-only
        Distributed Raman gain profile, from
        :func:`~comnumpy.optical.raman.solve_raman` over **this span
        length** (decision D45). Its on-off gain
        :math:`G_\mathrm{on\text{-}off}(z)` is applied step by step
        inside the split-step loop, so the Kerr term sees the power the
        fibre actually carries -- which is the whole reason a profile is
        used rather than a lumped gain. The EDFA is reduced by the same
        amount, so a span stays transparent whether or not it is pumped,
        and the ASE the solver integrated is added once per span, scaled
        from its reference bandwidth to ``fs``.
    seed : int, optional, keyword-only
        Seed of the local generator, which feeds both the Raman ASE and
        the per-span EDFA seed. Overridden by chain seeding
        (``Sequential.seed``, D6).

    Attributes
    ----------
    raman_step_gain_ : np.ndarray
        Amplitude gain applied at each split-step, read off the Raman
        profile at the step boundaries. Derived from the inputs, hence
        the trailing underscore (D23).
    raman_sigma2_ : float
        Variance of the Raman ASE added at the end of each span, in the
        simulated bandwidth :math:`f_s`.

    Raises
    ------
    ShapeError
        If the input is neither a full-field 1D signal ``(N,)`` nor a
        polarization pair ``(..., P, N)`` with ``P`` in ``{1, 2}``
        (validated in ``prepare()``): a pointwise Kerr step on a
        multi-channel array would silently produce SPM only (no XPM, no
        FWM).

    References
    ----------
    * G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
      2013, Sections 3.2 (CD) and 4.1 (SPM).
    * J. Shao, X. Liang and S. Kumar, "Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems,"
      IEEE Photonics Journal, vol. 6, no. 4, pp. 1-15, Aug. 2014, Art no. 7200515, doi: 10.1109/JPHOT.2014.2340993.
    * O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber
      communications systems," Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003.
    * R.-J. Essiambre, G. Kramer, P. J. Winzer, G. J. Foschini and B. Goebel,
      "Capacity limits of optical fiber networks," Journal of Lightwave
      Technology, vol. 28, no. 4, pp. 662-701, 2010.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = 1e-3 * (rng.normal(size=128) + 1j * rng.normal(size=128))
    >>> link = FiberLink(2, L_span=80.0, StPS=4, fs=10e9, noise_scaling=0)
    >>> y = link(x)
    >>> print(y.shape, y.dtype)
    (128,) complex128
    """
    N_spans: int = 1
    L_span: float = field(default=80, kw_only=True)
    StPS: int = field(default=1, kw_only=True)
    fs: float = field(default=1, kw_only=True)
    fiber: FiberSpec = field(default_factory=FiberSpec, kw_only=True)
    NF_dB: float = field(default=4, kw_only=True)
    noise_scaling: float = field(default=1, kw_only=True)
    step_type: Literal["linear", "logarithmic"] = field(default="linear", kw_only=True)
    step_method: Literal["symmetric", "asymetric"] = field(default="symmetric", kw_only=True)
    use_only_linear: bool = field(default=False, kw_only=True)
    step_log_factor: float = field(default=0.4, kw_only=True)
    name: str = field(default="fiber link", kw_only=True)
    # Callable[..., None]: the 'post_span' hook is called as
    # callback(y, num_span=...), which Callable[[np.ndarray], None]
    # declared away. Not Optional: default_factory=dict never yields None.
    callbacks: Dict[str, Callable[..., None]] = field(default_factory=dict, kw_only=True)
    raman: Optional[RamanSolution] = field(default=None, kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    # internal state (declared for slots, D40a)
    step_size: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    beta2: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    edfa_gain: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    edfa_N_ase: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    raman_step_gain_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    raman_tilt_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    manakov_: bool = field(init=False, repr=False, default=False)
    raman_sigma2_: float = field(init=False, repr=False, default=0.0)
    transfer_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    transfer_index_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    transfer_key_: Optional[tuple] = field(init=False, repr=False, default=None)
    rng_: Optional[np.random.Generator] = field(init=False, repr=False, default=None)

    def prepare(self, x: np.ndarray) -> None:
        self.manakov_ = is_polarization_pair(x, type(self).__name__)
        match self.step_type:
            case "linear":
                self.step_size = get_linear_step_size(self.L_span, self.StPS)
            case "logarithmic":
                self.step_size = get_logarithmic_step_size(self.L_span, self.StPS, alpha_dB=self.fiber.alpha_dB, step_log_factor=self.step_log_factor)
            case _:
                raise NotImplementedError(f"Step type {self.step_type} is not implemented")

        self.beta2 = self.fiber.beta2
        self.rng_ = np.random.default_rng(self.seed)

        # Distributed Raman gain (D45): the profile is sampled at the step
        # boundaries, so each step carries the on-off gain accumulated
        # over exactly that step -- which is what makes *where* the gain
        # happens visible to the Kerr term.
        residual_dB = self.fiber.alpha_dB * self.L_span
        if self.raman is not None:
            self.raman_step_gain_ = self._raman_step_gain()
            # A multi-signal solve carries one gain per channel, so the
            # step gain is no longer a number: it becomes a transfer
            # function, interpolated from the channels onto the FFT grid.
            self.raman_tilt_ = (self._raman_tilt(x.shape[-1])
                                if self.raman.n_signals > 1 else None)
            if self.step_type == "logarithmic":
                logger.warning(
                    "logarithmic steps are sized for the exponential decay "
                    "of an unpumped span, and distributed Raman flattens "
                    "that profile: the longest steps then land where the "
                    "power is highest. Measured on an SPM phase, the "
                    "logarithmic grid is 3x *worse* than the linear one "
                    "once Raman is on (8.2e-4 rad against 2.8e-4 at "
                    "StPS=20). Use step_type='linear' with Raman.")
            # One EDFA cannot undo a tilt: it is flat, so it makes up the
            # *mean* on-off gain and the channels come out spread around
            # transparency by the tilt. That is the physical situation a
            # gain-flattening filter exists to fix, not a modelling
            # shortcut, and `raman.tilt_dB` says how wide the spread is.
            mean_gain_dB = float(np.mean(np.atleast_1d(self.raman.on_off_gain_dB)))
            residual_dB -= mean_gain_dB
            # the ASE the solver integrated over its own reference
            # bandwidth, spread over the simulated one. Averaged over the
            # channels and added flat across the band: the gain is shaped
            # in frequency, this noise is not.
            self.raman_sigma2_ = self.fs * self._ase_densities()[2]
            if residual_dB < 0:
                logger.warning(
                    "Raman over-compensates the span: %.2f dB of on-off gain "
                    "against %.2f dB of loss, so the EDFA is set to "
                    "attenuate by %.2f dB to keep the span transparent.",
                    mean_gain_dB, self.fiber.alpha_dB * self.L_span,
                    -residual_dB)

        # the EDFA makes up whatever the Raman gain did not, so a span
        # stays transparent whether or not it is Raman-pumped
        equivalent_alpha_dB, edfa_density, _ = self._ase_densities()
        self.edfa_gain = compute_erbium_doped_fiber_amplifier_gain(equivalent_alpha_dB, self.L_span)
        # a density becomes a power in the simulated bandwidth; `budget`
        # multiplies the same density by the bandwidth a receiver keeps
        self.edfa_N_ase = self.fs * edfa_density
        self._build_transfer(x.shape[-1])

    def _build_transfer(self, n_samples: int) -> None:
        """Cache one linear-step transfer function per distinct length."""
        assert self.step_size is not None and self.beta2 is not None
        if self.use_only_linear:
            lengths = np.array([self.L_span], dtype=float)
        elif self.step_method == "symmetric":
            lengths = np.asarray(self.step_size, dtype=float) / 2
        else:
            lengths = np.asarray(self.step_size, dtype=float)
        self.transfer_, self.transfer_index_, self.transfer_key_ = step_transfers(
            n_samples, lengths, beta2=self.beta2, fs=self.fs,
            alpha_dB=self.fiber.alpha_dB, direction=1,
            previous=(self.transfer_, self.transfer_index_, self.transfer_key_))

    def _ase_densities(self) -> tuple[float, float, float]:
        """One span's noise, before any bandwidth is chosen.

        Returns the equivalent attenuation the EDFA has to make up, and
        the one-sided ASE spectral densities of the EDFA and of the
        distributed Raman amplifier, in W/Hz **per polarization**, both
        already scaled by ``noise_scaling``.

        It exists so that :meth:`prepare` and :meth:`budget` cannot
        disagree: the first turns these densities into a variance over
        ``fs``, the second into a power over the bandwidth a receiver
        collects, and the physics is written once.
        """
        residual_dB = self.fiber.alpha_dB * self.L_span
        raman_density = 0.0
        if self.raman is not None:
            residual_dB -= float(np.mean(np.atleast_1d(self.raman.on_off_gain_dB)))
            if self.raman.bandwidth_Hz > 0:
                ase_W = float(np.mean(np.atleast_2d(self.raman.ase_W)[:, -1]))
                raman_density = (self.noise_scaling * ase_W
                                 / self.raman.bandwidth_Hz)
        equivalent_alpha_dB = residual_dB / self.L_span
        edfa_density = self.noise_scaling * compute_erbium_doped_fiber_N_ase(
            equivalent_alpha_dB, self.L_span, self.NF_dB,
            h=PLANCK_CONSTANT, nu=self.fiber.carrier_frequency_Hz)
        return equivalent_alpha_dB, edfa_density, raman_density

    def budget(self, bandwidth_Hz: float, *,
               polarizations: int = 1) -> Dict[str, Any]:
        r"""The amplifier noise this link accumulates, in closed form.

        Signal Model
        ------------
        Each of the :math:`N_s` spans adds one amplifier's worth of
        spontaneous emission, and the noise a receiver sees is that
        density integrated over the bandwidth it keeps -- the symbol
        rate, for a matched filter:

        .. math::

            P_{\mathrm{ASE}} = P \, N_s \, N_{\mathrm{ASE}} \, B,
            \qquad
            N_{\mathrm{ASE}} = (G - 1) h \nu \, n_{sp},
            \qquad
            n_{sp} = \frac{\mathrm{NF}/2}{1 - 1/G}

        with :math:`P` the number of polarizations. That factor is the
        whole reason this method exists. :math:`N_{\mathrm{ASE}}` is a
        density **per polarization**, and a dual-polarization link
        carries two independent copies of it; quoting a channel power
        against a single-polarization budget is a 3 dB error that looks
        exactly like a modelling disagreement. Writing the product by
        hand at each call site is how the two conventions drift apart.

        The result is the link's *own* prediction: it owes nothing to a
        simulation, so it is the reference a measured effective SNR is
        checked against. ``noise_scaling`` is applied, so a link with
        its noise switched off budgets zero.

        Parameters
        ----------
        bandwidth_Hz : float
            Bandwidth the receiver collects, in Hz. For a matched filter
            this is the symbol rate, not the simulated ``fs``.
        polarizations : int, optional, keyword-only
            1 (default) for a single-polarization link, 2 for the
            dual-polarization link a coherent system uses -- the same
            convention as
            :func:`~comnumpy.optical.gn_model.gn_model_nli_power`.

        Returns
        -------
        dict
            ``ase_power_W`` and ``ase_power_dBm``, the accumulated noise;
            ``ase_density_W_per_Hz``, one span's density per
            polarization; and the inputs it was built from
            (``n_spans``, ``NF_dB``, ``span_gain_dB``, ``bandwidth_Hz``,
            ``polarizations``).

        Raises
        ------
        ValueError
            If the bandwidth is not positive, or if ``polarizations`` is
            neither 1 nor 2: a fibre carries one or two.

        Examples
        --------
        Ten 100 km spans with a 5 dB noise figure, one polarization,
        read in a 32 GBd matched filter:

        >>> link = FiberLink(10, L_span=100.0, NF_dB=5.0, fs=192e9)
        >>> budget = link.budget(32e9)
        >>> print(f"{budget['ase_power_dBm']:.2f} dBm")
        -21.88 dBm

        The second polarization carries its own noise, and nothing else
        changes:

        >>> print(f"{link.budget(32e9, polarizations=2)['ase_power_dBm']:.2f} dBm")
        -18.87 dBm
        """
        if bandwidth_Hz <= 0:
            raise ValueError(
                f"a noise power needs a bandwidth to be quoted in, got "
                f"bandwidth_Hz={bandwidth_Hz}. For a matched filter this "
                f"is the symbol rate, not the simulated fs.")
        if polarizations not in (1, 2):
            raise ValueError(
                f"a fibre carries one or two polarizations, got "
                f"{polarizations}.")
        equivalent_alpha_dB, edfa_density, raman_density = self._ase_densities()
        density = edfa_density + raman_density
        power = polarizations * self.N_spans * density * bandwidth_Hz
        return {
            "ase_power_W": power,
            "ase_power_dBm": float(watt_to_dbm(power)) if power > 0 else -np.inf,
            "ase_density_W_per_Hz": density,
            "span_gain_dB": equivalent_alpha_dB * self.L_span,
            "n_spans": self.N_spans,
            "NF_dB": self.NF_dB,
            "bandwidth_Hz": bandwidth_Hz,
            "polarizations": polarizations,
        }

    def _raman_step_gain(self) -> np.ndarray:
        """Amplitude gain of each SSFM step, from the Raman profile."""
        assert self.raman is not None and self.step_size is not None
        span_km = float(self.raman.z_km[-1])
        if abs(span_km - self.L_span) > 1e-6 * max(self.L_span, 1.0):
            raise ValueError(
                f"expected a Raman solution over the span length "
                f"{self.L_span} km, got one over {span_km} km -- solve it "
                f"with length_km={self.L_span}, the gain profile has to "
                f"describe this fibre")
        # Sampled at the *half*-step boundaries, not the step boundaries.
        # The gain belongs to the linear operator, exactly like the loss
        # that the linear step already applies over dz/2 in
        # each half; applying a whole step's gain at one point breaks the
        # symmetry of the symmetric split-step and drops it from second
        # order to first. Measured on the SPM phase of a CW field: the
        # error fell as 1/StPS with a single application, and falls as
        # 1/StPS^2 with this one, which is the order of the scheme
        # without Raman.
        ends = np.cumsum(self.step_size)
        starts = np.concatenate(([0.0], ends[:-1]))
        boundaries = np.empty(2 * len(ends) + 1)
        boundaries[0] = 0.0
        boundaries[1::2] = (starts + ends) / 2
        boundaries[2::2] = ends
        profile = np.atleast_2d(self.raman.gain_profile_dB)
        gain_dB = np.stack([np.interp(boundaries, self.raman.z_km, row)
                            for row in profile])
        steps = 10 ** (np.diff(gain_dB, axis=1) / 20)   # (n_signals, 2*StPS)
        return steps[0] if steps.shape[0] == 1 else steps

    def _raman_tilt(self, n_samples: int) -> np.ndarray:
        """Half-step gains as transfer functions over the simulated band.

        A multi-signal solve gives one gain per channel; the field the
        link propagates is the multiplex of those channels (D44), so the
        gain becomes frequency-dependent. Each channel's gain is placed
        at its own optical frequency and interpolated onto the FFT grid,
        the band edges holding the outermost channels' value. Outside
        the solved comb there is nothing to interpolate from, so the
        edge is held rather than extrapolated -- extrapolating a Raman
        tilt off the end of the comb invents gain.
        """
        assert self.raman is not None and self.raman_step_gain_ is not None
        frequency = np.atleast_1d(
            np.asarray(self.raman.frequency_signal_Hz, dtype=float))
        centre = self.fiber.carrier_frequency_Hz
        bins = centre + fftfreq(n_samples, d=1 / self.fs)
        if bins.max() < frequency.min() or bins.min() > frequency.max():
            raise ValueError(
                f"the simulated band [{bins.min() / 1e12:.3f}, "
                f"{bins.max() / 1e12:.3f}] THz does not overlap the Raman "
                f"channels [{frequency.min() / 1e12:.3f}, "
                f"{frequency.max() / 1e12:.3f}] THz -- the fibre's "
                f"wavelength_nm sets the carrier the band is centred on, so "
                f"give FiberSpec the comb centre the solve used")
        order = np.argsort(frequency)
        gains_dB = 20 * np.log10(self.raman_step_gain_)     # (n_signals, 2*StPS)
        tilt = np.stack([np.interp(bins, frequency[order], gains_dB[order, k])
                         for k in range(gains_dB.shape[1])])
        return 10 ** (tilt / 20)

    def _apply_kerr(self, y: np.ndarray, dz: float) -> np.ndarray:
        """Kerr step: scalar NLSE, or Manakov when the field is a pair."""
        gamma, intensity = manakov_kerr(y, self.fiber.gamma, self.manakov_)
        return apply_kerr_nonlinearity(y, dz, gamma, direction=1,
                                       intensity=intensity)

    def _apply_raman(self, y: np.ndarray, index: int) -> np.ndarray:
        """Half-step Raman gain: a number, or a filter when it is tilted."""
        if self.raman_step_gain_ is None:
            return y
        if self.raman_tilt_ is None:
            return self.raman_step_gain_[index] * y
        return apply_frequency_response(y, self.raman_tilt_[index])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # perform SSFM
        y = x
        # set by prepare(), which the Processor base always runs first
        assert (self.beta2 is not None and self.step_size is not None
                and self.edfa_gain is not None and self.edfa_N_ase is not None)
        for num_span in range(self.N_spans):
            # perform for each span
            if self.use_only_linear:
                y = apply_frequency_response(y, self.transfer_[0])
                if self.raman_step_gain_ is not None:
                    # no step loop here, so the whole profile applies at
                    # once -- the span must stay transparent in this mode
                    # too, and the EDFA has already been reduced for it
                    if self.raman_tilt_ is None:
                        y = float(np.prod(self.raman_step_gain_)) * y
                    else:
                        y = apply_frequency_response(
                            y, np.prod(self.raman_tilt_, axis=0))
            else:
                for num_step in range(self.StPS):
                    dz = self.step_size[num_step]
                    # the two half-step Raman gains bracket the Kerr term,
                    # like the two half-step dispersion operators
                    H = self.transfer_[self.transfer_index_[num_step]]
                    if self.step_method == "symmetric":
                        y = self._apply_raman(y, 2*num_step)
                        y = apply_frequency_response(y, H)
                        y = self._apply_kerr(y, dz)
                        y = self._apply_raman(y, 2*num_step+1)
                        y = apply_frequency_response(y, H)

                    if self.step_method == "asymetric":
                        y = self._apply_raman(self._apply_raman(y, 2*num_step), 2*num_step+1)
                        y = self._apply_kerr(y, dz)
                        y = apply_frequency_response(y, H)

            # the ASE the distributed amplifier generated over this span,
            # already amplified by the gain downstream of where it was born
            # (the solver integrates it to z = L, so it is added once)
            if self.raman_sigma2_ > 0:
                assert self.rng_ is not None
                scale = np.sqrt(self.raman_sigma2_ / 2)
                y = y + (self.rng_.normal(scale=scale, size=y.shape)
                         + 1j * self.rng_.normal(scale=scale, size=y.shape))

            # perform amplification for fiber loss compensation. The EDFA
            # draws its seed from the link's generator: built unseeded it
            # made the whole link irreproducible, and seeding the Raman
            # noise alone would only have made it *look* reproducible.
            assert self.rng_ is not None
            edfa = ErbiumDopedFiberAmplifier(
                self.edfa_gain, self.edfa_N_ase,
                seed=int(self.rng_.integers(2 ** 31)))
            y = edfa(y)

            # callback after span if needed
            if 'post_span' in self.callbacks:
                self.callbacks['post_span'](y, num_span=num_span)

        return y

