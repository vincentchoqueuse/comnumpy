import logging

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Dict
from comnumpy.core import Processor
from comnumpy.exceptions import ShapeError
from .devices import ErbiumDopedFiberAmplifier
from .raman import RamanSolution
from .constants import SPEED_OF_LIGHT, PLANCK_CONSTANT, WAVELENGTH, KERR_COEFFICIENT, FIBER_LOSS, CD_COEFFICIENT, OPTICAL_CARRIER_FREQUENCY
from .utils import (compute_beta2, get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain,
                    compute_erbium_doped_fiber_N_ase, apply_chromatic_dispersion, apply_kerr_nonlinearity)

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

    Axes: *declared axis* -- requires a full-field 1D signal (N,).

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
        If the input is not a full-field 1D signal (N,) (validated in
        ``prepare()``): a pointwise Kerr step on a multi-channel array
        would silently produce SPM only (no XPM, no FWM).

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
    NF_dB: float = field(default=4, kw_only=True)
    noise_scaling: float = field(default=1, kw_only=True)
    step_type: Literal["linear", "logarithmic"] = field(default="linear", kw_only=True)
    step_method: Literal["symmetric", "asymetric"] = field(default="symmetric", kw_only=True)
    use_only_linear: bool = field(default=False, kw_only=True)
    c: float = field(default=SPEED_OF_LIGHT, kw_only=True)              # in meters per second
    h: float = field(default=PLANCK_CONSTANT, kw_only=True)             # in Joule seconds
    gamma: float = field(default=KERR_COEFFICIENT, kw_only=True)        # in rad/W/km
    lamb: float = field(default=WAVELENGTH, kw_only=True)               # nm
    alpha_dB: float = field(default=FIBER_LOSS, kw_only=True)           # in dB/km
    cd_coefficient: float = field(default=CD_COEFFICIENT, kw_only=True)  # in ps/nm/km
    nu: float = field(default=OPTICAL_CARRIER_FREQUENCY, kw_only=True)  # optical carrier frequency
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
    raman_sigma2_: float = field(init=False, repr=False, default=0.0)
    rng_: Optional[np.random.Generator] = field(init=False, repr=False, default=None)

    def prepare(self, x: np.ndarray) -> None:
        if x.ndim != 1:
            raise ShapeError(
                f"nonlinear propagation requires a full-field signal (N,); "
                f"got {x.shape}. A pointwise Kerr step on a multi-channel "
                f"array would silently produce SPM only (no XPM, no FWM). "
                f"Multiplex the channels first with "
                f"comnumpy.optical.WDMMultiplexer (decision D44), or use a "
                f"coupled-NLSE model (not implemented)."
            )
        match self.step_type:
            case "linear":
                self.step_size = get_linear_step_size(self.L_span, self.StPS)
            case "logarithmic":
                self.step_size = get_logarithmic_step_size(self.L_span, self.StPS, alpha_dB=self.alpha_dB, step_log_factor=self.step_log_factor)
            case _:
                raise NotImplementedError(f"Step type {self.step_type} is not implemented")

        self.beta2 = compute_beta2(self.lamb, self.cd_coefficient, self.c)
        self.rng_ = np.random.default_rng(self.seed)

        # Distributed Raman gain (D45): the profile is sampled at the step
        # boundaries, so each step carries the on-off gain accumulated
        # over exactly that step -- which is what makes *where* the gain
        # happens visible to the Kerr term.
        residual_dB = self.alpha_dB * self.L_span
        if self.raman is not None:
            self.raman_step_gain_ = self._raman_step_gain()
            if self.step_type == "logarithmic":
                logger.warning(
                    "logarithmic steps are sized for the exponential decay "
                    "of an unpumped span, and distributed Raman flattens "
                    "that profile: the longest steps then land where the "
                    "power is highest. Measured on an SPM phase, the "
                    "logarithmic grid is 3x *worse* than the linear one "
                    "once Raman is on (8.2e-4 rad against 2.8e-4 at "
                    "StPS=20). Use step_type='linear' with Raman.")
            residual_dB -= self.raman.on_off_gain_dB
            # the ASE the solver integrated over its own reference
            # bandwidth, spread over the simulated one
            if self.raman.bandwidth_Hz > 0:
                self.raman_sigma2_ = (self.noise_scaling * float(self.raman.ase_W[-1])
                                      * self.fs / self.raman.bandwidth_Hz)
            if residual_dB < 0:
                logger.warning(
                    "Raman over-compensates the span: %.2f dB of on-off gain "
                    "against %.2f dB of loss, so the EDFA is set to "
                    "attenuate by %.2f dB to keep the span transparent.",
                    self.raman.on_off_gain_dB, self.alpha_dB * self.L_span,
                    -residual_dB)

        # the EDFA makes up whatever the Raman gain did not, so a span
        # stays transparent whether or not it is Raman-pumped
        equivalent_alpha_dB = residual_dB / self.L_span
        self.edfa_gain = compute_erbium_doped_fiber_amplifier_gain(equivalent_alpha_dB, self.L_span)
        self.edfa_N_ase = self.noise_scaling * self.fs * compute_erbium_doped_fiber_N_ase(equivalent_alpha_dB, self.L_span, self.NF_dB, h=self.h, nu=self.nu)

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
        # that apply_chromatic_dispersion already applies over dz/2 in
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
        gain_dB = np.interp(boundaries, self.raman.z_km,
                            self.raman.gain_profile_dB)
        return 10 ** (np.diff(gain_dB) / 20)      # 2*StPS half-step gains

    def forward(self, x: np.ndarray) -> np.ndarray:
        # perform SSFM
        y = x
        # set by prepare(), which the Processor base always runs first
        assert (self.beta2 is not None and self.step_size is not None
                and self.edfa_gain is not None and self.edfa_N_ase is not None)
        for num_span in range(self.N_spans):
            # perform for each span
            if self.use_only_linear:
                y = apply_chromatic_dispersion(y, self.L_span, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)
                if self.raman_step_gain_ is not None:
                    # no step loop here, so the whole profile applies at
                    # once -- the span must stay transparent in this mode
                    # too, and the EDFA has already been reduced for it
                    y = float(np.prod(self.raman_step_gain_)) * y
            else:
                for num_step in range(self.StPS):
                    dz = self.step_size[num_step]
                    # the two half-step Raman gains bracket the Kerr term,
                    # like the two half-step dispersion operators
                    first, second = (self.raman_step_gain_[2*num_step:2*num_step+2]
                                     if self.raman_step_gain_ is not None
                                     else (1.0, 1.0))

                    if self.step_method == "symmetric":
                        y = apply_chromatic_dispersion(first*y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)
                        y = apply_kerr_nonlinearity(y, dz, self.gamma, direction=1)
                        y = apply_chromatic_dispersion(second*y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)

                    if self.step_method == "asymetric":
                        y = apply_kerr_nonlinearity(first*second*y, dz, self.gamma, direction=1)
                        y = apply_chromatic_dispersion(y, dz, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)

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

