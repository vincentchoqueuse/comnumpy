import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Dict
from comnumpy.core import Processor
from comnumpy.exceptions import ShapeError
from .devices import ErbiumDopedFiberAmplifier
from .constants import SPEED_OF_LIGHT, PLANCK_CONSTANT, WAVELENGTH, KERR_COEFFICIENT, FIBER_LOSS, CD_COEFFICIENT, OPTICAL_CARRIER_FREQUENCY
from .utils import (compute_beta2, get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain,
                    compute_erbium_doped_fiber_N_ase, apply_chromatic_dispersion, apply_kerr_nonlinearity)

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
    # internal state (declared for slots, D40a)
    step_size: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    beta2: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    edfa_gain: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    edfa_N_ase: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)

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
        self.edfa_gain = compute_erbium_doped_fiber_amplifier_gain(self.alpha_dB, self.L_span)
        self.edfa_N_ase = self.noise_scaling * self.fs * compute_erbium_doped_fiber_N_ase(self.alpha_dB, self.L_span, self.NF_dB, h=self.h, nu=self.nu)

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
            else:
                for num_step in range(self.StPS):
                    dz = self.step_size[num_step]

                    if self.step_method == "symmetric":
                        y = apply_chromatic_dispersion(y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)
                        y = apply_kerr_nonlinearity(y, dz, self.gamma, direction=1)
                        y = apply_chromatic_dispersion(y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)

                    if self.step_method == "asymetric":
                        y = apply_kerr_nonlinearity(y, dz, self.gamma, direction=1)
                        y = apply_chromatic_dispersion(y, dz, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=1)

            # perform amplification for fiber loss compensation
            edfa = ErbiumDopedFiberAmplifier(self.edfa_gain, self.edfa_N_ase)
            y = edfa(y)

            # callback after span if needed
            if 'post_span' in self.callbacks:
                self.callbacks['post_span'](y, num_span=num_span)

        return y

