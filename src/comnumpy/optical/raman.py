r"""Distributed Raman amplification: gain spectra and the pump equations.

Two objects and one solver, split the way the physics is:

* :class:`RamanGainSpectrum` -- the *shape* of the Raman gain against
  the pump-signal frequency shift, normalized to 1 at its peak, with
  its provenance. It is a property of the glass, so it is catalogued
  the way D15/D20/D43 catalogue the frequency and delay axes: frozen
  value object, registry, and a self-check at construction against the
  figures its source publishes.
* the peak coefficient :math:`g_R / A_\mathrm{eff}` is **not** in that
  object. It is a property of the *fibre* -- SMF, DCF and NZDSF differ
  by a large factor through the effective area and the germanium
  doping -- so it is an argument of the solver, next to the losses.
* :func:`solve_raman` -- the coupled power equations, integrated
  forwards when only the co-propagating pump is on and solved as a
  two-point boundary value problem otherwise.

There is no ``Processor`` here, on purpose. Raman lives in the *power*
domain and the split-step propagation of
:class:`~comnumpy.optical.links.FiberLink` lives in the *field* domain:
a block that multiplied the field by a lumped gain at the end of a span
would describe a discrete amplifier, which is the one thing distributed
amplification is not. What this module produces is the gain profile
:math:`G(z)`, meant to be applied inside the linear step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RamanGainSpectrum", "RamanSolution", "solve_raman",
           "get_gain_spectrum", "register_gain_spectrum",
           "available_gain_spectra"]

PLANCK = 6.62607015e-34          # J s
BOLTZMANN = 1.380649e-23         # J/K
SPEED_OF_LIGHT = 2.99792458e8    # m/s


@dataclass(frozen=True)
class RamanGainSpectrum:
    r"""Normalized Raman gain shape against the pump-signal frequency shift.

    Signal Model
    ------------
    Stimulated Raman scattering transfers power from a pump at
    :math:`\nu_p` to a signal at :math:`\nu_s < \nu_p`, with an
    efficiency that depends only on the Stokes shift

    .. math::

        \Delta \nu = \nu_p - \nu_s

    This object carries the **shape** :math:`\tilde{g}(\Delta \nu)`,
    normalized so that :math:`\max_{\Delta \nu} \tilde{g} = 1`; the
    fibre's peak coefficient :math:`g_R / A_\mathrm{eff}` multiplies it
    (see :func:`solve_raman`). Exactly one of the two closed forms below
    is selected, the D41 way -- one keyword per parameterization, no
    discriminator argument.

    **Damped-oscillator model** (``lorentzian``). The Raman response of
    silica is modelled by a single damped oscillator of period
    :math:`\tau_1` and lifetime :math:`\tau_2`, whose Fourier transform
    is

    .. math::

        \tilde{H}_R(\omega) = \frac{\tau_1^{-2} + \tau_2^{-2}}
             {\tau_1^{-2} - \left(\omega + i \tau_2^{-1}\right)^2},
        \qquad \omega = 2 \pi \Delta \nu

    and the gain is :math:`\operatorname{Im} \tilde{H}_R`, normalized
    here to its peak. :math:`\tilde{H}_R(0) = 1` fixes the
    normalization of the response itself.

    **Triangular model** (``triangular``). Over the first few terahertz
    the gain is linear in the shift, which is what the closed-form
    inter-channel Raman tilt of a WDM comb relies on:

    .. math::

        \tilde{g}(\Delta \nu) =
        \begin{cases}
        \Delta \nu / \Delta \nu_\mathrm{peak}
            & 0 \leq \Delta \nu \leq \Delta \nu_\mathrm{peak}\\
        0 & \text{otherwise}
        \end{cases}

    Normalizing to the peak leaves a single free number,
    :math:`\Delta \nu_\mathrm{peak}`: the slope is then
    :math:`1/\Delta \nu_\mathrm{peak}` in normalized units and the
    absolute slope comes from the fibre's peak coefficient.

    Parameters
    ----------
    lorentzian : tuple of two floats, optional, keyword-only
        :math:`(\tau_1, \tau_2)` in femtoseconds. Mutually exclusive
        with ``triangular``.
    triangular : float, optional, keyword-only
        Peak Stokes shift :math:`\Delta \nu_\mathrm{peak}` in THz.
        Mutually exclusive with ``lorentzian``.
    standard : str, keyword-only
        Name of the model, e.g. ``"Blow-Wood"``.
    reference : str, keyword-only
        Publication the parameters were copied from (same provenance
        rule as D15/D20/D43).

    Raises
    ------
    ValueError
        If neither or both parameterizations are given, or if a time
        constant or the peak shift is not positive.

    References
    ----------
    K. J. Blow, D. Wood, "Theoretical description of transient
    stimulated Raman scattering in optical fibers", IEEE J. Quantum
    Electron., vol. 25, no. 12, pp. 2665-2673, 1989.

    G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
    2013, Sections 2.3 and 8.1.

    Examples
    --------
    >>> spectrum = get_gain_spectrum("blow-wood")
    >>> round(spectrum.peak_shift_THz, 2)
    13.08
    >>> round(float(spectrum.shape(13.08e12)), 6)
    1.0
    """

    lorentzian: Optional[tuple[float, float]] = field(default=None, kw_only=True)
    triangular: Optional[float] = field(default=None, kw_only=True)
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        given = [name for name, value in (("lorentzian", self.lorentzian),
                                          ("triangular", self.triangular))
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                f"expected exactly one parameterization, got "
                f"{given or 'none'} -- pass lorentzian=(tau1_fs, tau2_fs) "
                f"or triangular=peak_shift_THz")
        if self.lorentzian is not None:
            tau1, tau2 = self.lorentzian
            if tau1 <= 0 or tau2 <= 0:
                raise ValueError(
                    f"expected positive time constants in fs, got "
                    f"({tau1}, {tau2})")
            object.__setattr__(self, "lorentzian", (float(tau1), float(tau2)))
        else:
            if self.triangular is None or self.triangular <= 0:
                raise ValueError(
                    f"expected a positive peak shift in THz, got "
                    f"{self.triangular}")
            object.__setattr__(self, "triangular", float(self.triangular))

    # -- the shape --------------------------------------------------------
    def _raw(self, shift_Hz: np.ndarray) -> np.ndarray:
        """Unnormalized gain, for the shift in Hz (may be negative)."""
        shift = np.asarray(shift_Hz, dtype=float)
        if self.lorentzian is not None:
            tau1, tau2 = self.lorentzian[0] * 1e-15, self.lorentzian[1] * 1e-15
            omega = 2 * np.pi * shift
            response = ((tau1 ** -2 + tau2 ** -2)
                        / (tau1 ** -2 - (omega + 1j / tau2) ** 2))
            return np.imag(response)
        peak = self.triangular * 1e12          # type: ignore[operator]
        return np.where((shift >= 0) & (shift <= peak), shift / peak, 0.0)

    def shape(self, shift_Hz: np.ndarray) -> np.ndarray:
        r"""Normalized gain :math:`\tilde{g}(\Delta \nu)`, peak value 1.

        Parameters
        ----------
        shift_Hz : array_like
            Stokes shift :math:`\Delta \nu = \nu_p - \nu_s` in Hz. A
            negative shift means the "signal" is above the pump, where
            the model gives no gain.

        Returns
        -------
        np.ndarray
            Gain shape, between 0 and 1.

        Examples
        --------
        >>> spectrum = get_gain_spectrum("triangular")
        >>> [round(float(spectrum.shape(f)), 3) for f in (0.0, 6e12, -6e12)]
        [0.0, 0.455, 0.0]
        """
        peak = float(np.max(self._raw(self._probe_grid())))
        return np.maximum(self._raw(shift_Hz) / peak, 0.0)

    @staticmethod
    def _probe_grid() -> np.ndarray:
        """Shift grid used to locate the peak and the width, in Hz.

        0 to 40 THz at 0.1 GHz: the silica gain is negligible past
        40 THz, and the step is what fixes the two decimals the
        self-check compares.
        """
        return np.linspace(0.0, 40e12, 400_001)

    @property
    def peak_shift_THz(self) -> float:
        r"""Stokes shift :math:`\Delta \nu_\mathrm{peak}` of the maximum."""
        grid = self._probe_grid()
        return float(grid[int(np.argmax(self._raw(grid)))] / 1e12)

    @property
    def fwhm_THz(self) -> float:
        """Full width at half maximum of the gain shape, in THz."""
        grid = self._probe_grid()
        gain = self._raw(grid)
        above = grid[gain >= gain.max() / 2]
        return float((above[-1] - above[0]) / 1e12)

    def __repr__(self) -> str:
        model = ("lorentzian tau=(%.4g, %.4g) fs" % self.lorentzian
                 if self.lorentzian is not None
                 else "triangular peak=%.4g THz" % self.triangular)
        return (f"RamanGainSpectrum({self.standard}, {model})\n"
                f"  peak at {self.peak_shift_THz:.2f} THz, "
                f"FWHM {self.fwhm_THz:.2f} THz"
                + (f"\n  {self.reference}" if self.reference else ""))


# -- catalog --------------------------------------------------------------
_SPECTRUM_REGISTRY: dict[str, Callable[..., RamanGainSpectrum]] = {}


def register_gain_spectrum(name: str):
    """Register a catalog entry; users can add their own fits.

    Parameters
    ----------
    name : str
        Name the factory answers to in :func:`get_gain_spectrum`.

    Examples
    --------
    >>> @register_gain_spectrum("LAB")
    ... def _lab():
    ...     return RamanGainSpectrum(triangular=14.0, standard="LAB",
    ...                              reference="bench")
    >>> round(get_gain_spectrum("LAB").peak_shift_THz, 1)
    14.0
    """
    def decorator(func: Callable[..., RamanGainSpectrum]):
        _SPECTRUM_REGISTRY[name] = func
        return func
    return decorator


def available_gain_spectra() -> list[str]:
    """Names accepted by :func:`get_gain_spectrum`.

    Returns
    -------
    list of str
        Registered names, sorted.

    Examples
    --------
    >>> [name for name in available_gain_spectra() if "-" in name]
    ['blow-wood']
    """
    return sorted(_SPECTRUM_REGISTRY)


def get_gain_spectrum(standard: str, **kwargs: Any) -> RamanGainSpectrum:
    """Return a catalog gain spectrum by name.

    Parameters
    ----------
    standard : str
        Entry name; see :func:`available_gain_spectra`.
    **kwargs
        Forwarded to the entry factory.

    Raises
    ------
    KeyError
        If the name is not in the catalog.

    Examples
    --------
    >>> get_gain_spectrum("blow-wood").standard
    'Blow-Wood single damped oscillator'
    """
    if standard not in _SPECTRUM_REGISTRY:
        raise KeyError(
            f"unknown Raman gain spectrum {standard!r}; available: "
            f"{available_gain_spectra()} -- register your own with "
            f"@register_gain_spectrum")
    return _SPECTRUM_REGISTRY[standard](**kwargs)


def _check_expect(spectrum: RamanGainSpectrum,
                  expect: Optional[dict[str, float]]) -> RamanGainSpectrum:
    """Verify a catalog entry against figures published with it (D20)."""
    if not expect:
        return spectrum
    checks = {"peak_shift_THz": spectrum.peak_shift_THz,
              "fwhm_THz": spectrum.fwhm_THz}
    for key, wanted in expect.items():
        if key not in checks:
            raise KeyError(f"unknown self-check {key!r}; known: {sorted(checks)}")
        got = checks[key]
        if abs(got - wanted) > max(0.02, 2e-2 * abs(wanted)):
            raise ValueError(
                f"{spectrum.standard}: expected {key} = {wanted} from "
                f"{spectrum.reference or 'the source'}, this model gives "
                f"{got:.3f} -- one of the two is wrong.")
    return spectrum


@register_gain_spectrum("blow-wood")
def _blow_wood() -> RamanGainSpectrum:
    r"""Single damped oscillator, :math:`\tau_1 = 12.2` fs, :math:`\tau_2 = 32` fs.

    Reproduces the peak of the silica Raman gain to within 1 % --
    13.08 THz against the 13.2 THz that is published -- which is what
    the self-check pins.

    It does **not** reproduce the width: this model gives a FWHM of
    9.55 THz where the measured silica spectrum is 5 to 6 THz wide,
    roughly 70 % too broad. That is the known limitation of a single
    oscillator, and it is stated rather than hidden: the model is right
    for the time-domain Raman term of the generalized NLSE and for the
    soliton self-frequency shift, and wrong for designing a multi-pump
    amplifier, where the spectral shape *is* the gain flatness. A
    multi-Lorentzian fit is what that needs; none is shipped, because
    none has been transcribed from its source.
    """
    return _check_expect(
        RamanGainSpectrum(lorentzian=(12.2, 32.0),
                          standard="Blow-Wood single damped oscillator",
                          reference="Blow & Wood, IEEE JQE 25(12), 1989; "
                                    "Agrawal, Nonlinear Fiber Optics, 5th ed., "
                                    "section 2.3"),
        {"peak_shift_THz": 13.2})


@register_gain_spectrum("triangular")
def _triangular(peak_shift_THz: float = 13.2) -> RamanGainSpectrum:
    r"""Linear gain up to the peak, zero beyond.

    The model behind the closed-form inter-channel Raman tilt of a WDM
    comb: over a band a few THz wide the gain is proportional to the
    frequency separation, so the tilt across the comb is affine in the
    channel index. Only valid for :math:`\Delta \nu` well below the
    peak -- it is a slope, not a spectrum, and it says nothing at all
    above :math:`\Delta \nu_\mathrm{peak}`.
    """
    return RamanGainSpectrum(triangular=peak_shift_THz,
                             standard="triangular tilt model",
                             reference="Semrau, Killey & Bayvel, "
                                       "J. Lightwave Technol. 36(14), 2018")


# -- the solver -----------------------------------------------------------
@dataclass(frozen=True)
class RamanSolution:
    r"""Power profiles along the span, and what is read off them.

    Signal Model
    ------------
    All four profiles are functions of the distance :math:`z` from the
    span input. The figures of merit are

    .. math::

        G_\mathrm{on\text{-}off} =
            \frac{P_s(L)}{P_s(L)\big|_{P_p = 0}}
          = \frac{P_s(L)}{P_s(0) e^{-\alpha_s L}},
        \qquad
        G_\mathrm{net} = \frac{P_s(L)}{P_s(0)}

    -- the on-off gain is what the pump buys, the net gain is what
    comes out of the span -- and the effective noise figure

    .. math::

        F_\mathrm{eff} = \frac{1}{G_\mathrm{net}}
            \left(1 + \frac{P_\mathrm{ASE}(L)}{h \nu_s B}\right)

    which is *effective* because it refers the noise of the whole
    distributed span to a hypothetical discrete amplifier; it may be
    below 0 dB, and that is the reason distributed amplification is
    used at all.

    Attributes
    ----------
    z_km : np.ndarray
        Distance grid, in km, from 0 to the span length.
    signal_W : np.ndarray
        Signal power :math:`P_s(z)` in W.
    pump_forward_W, pump_backward_W : np.ndarray
        Pump powers :math:`P_p^{+}(z)` and :math:`P_p^{-}(z)` in W.
        Both are given on the same :math:`z` grid; the backward pump is
        launched at :math:`z = L`.
    ase_W : np.ndarray
        Amplified spontaneous emission :math:`P_\mathrm{ASE}(z)` in W,
        in the reference bandwidth of the solve.
    bandwidth_Hz : float
        Reference bandwidth :math:`B` the ASE was integrated over.
    frequency_signal_Hz : float
        Signal frequency :math:`\nu_s`, needed to turn the ASE into a
        noise figure.
    loss_only_W : np.ndarray
        The signal profile the same fibre would give with the pumps
        off, :math:`P_s(0) e^{-\alpha_s z}`. Carrying it makes the
        on-off gain a ratio of two stored curves rather than a
        recomputation.

    Examples
    --------
    >>> solution = solve_raman(length_km=80.0, gain_peak_W_km=0.4,
    ...                        pump_backward_W=0.2, signal_W=1e-3)
    >>> round(solution.on_off_gain_dB, 2)
    5.97
    """

    z_km: np.ndarray
    signal_W: np.ndarray
    pump_forward_W: np.ndarray
    pump_backward_W: np.ndarray
    ase_W: np.ndarray
    loss_only_W: np.ndarray
    bandwidth_Hz: float
    frequency_signal_Hz: float

    @property
    def on_off_gain_dB(self) -> float:
        """Gain the pump buys, in dB: output with pump over output without."""
        return float(10 * np.log10(self.signal_W[-1] / self.loss_only_W[-1]))

    @property
    def net_gain_dB(self) -> float:
        """Span output over span input, in dB. Negative for a lossy span."""
        return float(10 * np.log10(self.signal_W[-1] / self.signal_W[0]))

    @property
    def gain_profile_dB(self) -> np.ndarray:
        r"""On-off gain accumulated up to each :math:`z`, in dB.

        This is the curve the split-step loop of
        :class:`~comnumpy.optical.links.FiberLink` would consume.
        """
        return 10 * np.log10(self.signal_W / self.loss_only_W)

    @property
    def noise_figure_dB(self) -> float:
        """Effective noise figure of the span, in dB; may be negative."""
        photon = PLANCK * self.frequency_signal_Hz * self.bandwidth_Hz
        net_gain = self.signal_W[-1] / self.signal_W[0]
        return float(10 * np.log10((1 + self.ase_W[-1] / photon) / net_gain))

    @property
    def pump_depletion(self) -> float:
        """Fraction of the launched pump power the signal took away.

        Zero in the undepleted regime the analytic gain formula
        assumes; a large value is the warning that it no longer holds.
        """
        launched = self.pump_forward_W[0] + self.pump_backward_W[-1]
        remaining = self.pump_forward_W[-1] + self.pump_backward_W[0]
        no_raman = (self.pump_forward_W[0] * self._pump_transmission
                    + self.pump_backward_W[-1] * self._pump_transmission)
        if launched <= 0:
            return 0.0
        return float(max(no_raman - remaining, 0.0) / launched)

    _pump_transmission: float = field(default=1.0, repr=False)


def _photon_occupancy(shift_Hz: float, temperature_K: float) -> float:
    r"""Phonon occupancy :math:`1 + \eta` of the Raman ASE.

    .. math::

        \eta(\Delta \nu, T) =
            \left[\exp\!\left(\frac{h \Delta \nu}{k_B T}\right) - 1\right]^{-1}

    At room temperature and a 13 THz shift this is a few per cent, so
    the spontaneous emission is close to -- but not equal to -- the
    zero-temperature value.
    """
    if shift_Hz <= 0 or temperature_K <= 0:
        return 1.0
    return 1.0 + 1.0 / np.expm1(PLANCK * shift_Hz / (BOLTZMANN * temperature_K))


def solve_raman(*, length_km: float, gain_peak_W_km: float,
                signal_W: float = 1e-3,
                pump_forward_W: float = 0.0,
                pump_backward_W: float = 0.0,
                alpha_signal_dB_km: float = 0.2,
                alpha_pump_dB_km: float = 0.25,
                wavelength_signal_nm: float = 1550.0,
                wavelength_pump_nm: float = 1455.0,
                spectrum: Optional[RamanGainSpectrum] = None,
                bandwidth_Hz: float = 12.5e9,
                temperature_K: float = 300.0,
                n_nodes: int = 401,
                tol: float = 1e-8) -> RamanSolution:
    r"""Solve the coupled Raman power equations along one span.

    Signal Model
    ------------
    The signal propagates towards :math:`+z`, the co-propagating pump
    with it and the counter-propagating pump against it. Writing every
    profile in the :math:`+z` coordinate, the powers obey

    .. math::

        \frac{\mathrm{d} P_s}{\mathrm{d}z} &=
            \left[g\left(P_p^{+} + P_p^{-}\right) - \alpha_s\right] P_s \\
        \frac{\mathrm{d} P_p^{+}}{\mathrm{d}z} &=
            -\left[\frac{\nu_p}{\nu_s} g P_s + \alpha_p\right] P_p^{+} \\
        \frac{\mathrm{d} P_p^{-}}{\mathrm{d}z} &=
            +\left[\frac{\nu_p}{\nu_s} g P_s + \alpha_p\right] P_p^{-}

    with :math:`g = (g_R / A_\mathrm{eff}) \,
    \tilde{g}(\nu_p - \nu_s)` the gain coefficient at the working
    shift. The factor :math:`\nu_p / \nu_s` is photon-number
    bookkeeping: one pump photon makes one signal photon, and the
    energy difference goes into a phonon. Spontaneous emission adds

    .. math::

        \frac{\mathrm{d} P_\mathrm{ASE}}{\mathrm{d}z} =
            \left[g P_p^{\mathrm{tot}} - \alpha_s\right] P_\mathrm{ASE}
            + 2 h \nu_s B \, g P_p^{\mathrm{tot}}
              \left[1 + \eta(\Delta \nu, T)\right]

    the last bracket being the phonon occupancy, which is why a Raman
    amplifier is quieter when it is cold.

    **Boundary conditions.** The signal and the co-propagating pump are
    known at the span input, the counter-propagating pump at its
    output:

    .. math::

        P_s(0) = P_\mathrm{in}, \quad
        P_p^{+}(0) = P_f, \quad
        P_p^{-}(L) = P_b, \quad
        P_\mathrm{ASE}(0) = 0

    With :math:`P_b = 0` everything is known at :math:`z = 0` and the
    system is an initial value problem, integrated with
    ``scipy.integrate.solve_ivp``. Otherwise it is a genuine two-point
    boundary value problem and ``scipy.integrate.solve_bvp`` is used,
    started from the undepleted-pump profiles -- a poor initial guess
    is the usual reason that solver fails to converge.

    **Undepleted limit.** When the signal takes a negligible part of
    the pump, the pump decays exponentially and the on-off gain has the
    closed form

    .. math::

        G_\mathrm{on\text{-}off} = \exp\left(g P_p L_\mathrm{eff}\right),
        \qquad
        L_\mathrm{eff} = \frac{1 - e^{-\alpha_p L}}{\alpha_p}

    which is what ``validation/optical_raman.py`` checks the solver
    against, along with photon-number conservation in the lossless
    limit.

    Axes: *not a Processor* -- this is a power-domain solver, and it
    returns profiles rather than transforming a signal. See the module
    docstring.

    Parameters
    ----------
    length_km : float, keyword-only
        Span length :math:`L`, in km.
    gain_peak_W_km : float, keyword-only
        Peak Raman gain coefficient :math:`g_R / A_\mathrm{eff}` of the
        fibre, in :math:`\mathrm{W}^{-1}\mathrm{km}^{-1}`. A property
        of the fibre, not of the glass spectrum.
    signal_W : float, optional, keyword-only
        Launched signal power :math:`P_\mathrm{in}`, in W.
    pump_forward_W : float, optional, keyword-only
        Co-propagating pump :math:`P_f` launched at :math:`z = 0`, in W.
    pump_backward_W : float, optional, keyword-only
        Counter-propagating pump :math:`P_b` launched at :math:`z = L`,
        in W. There is no ``direction`` argument: which pumps are on
        *is* the configuration (D41).
    alpha_signal_dB_km, alpha_pump_dB_km : float, optional, keyword-only
        Fibre attenuation :math:`\alpha_s` and :math:`\alpha_p` at the
        signal and pump wavelengths, in dB/km.
    wavelength_signal_nm, wavelength_pump_nm : float, optional, keyword-only
        Wavelengths setting :math:`\nu_s`, :math:`\nu_p` and hence the
        Stokes shift :math:`\Delta \nu`.
    spectrum : RamanGainSpectrum, optional, keyword-only
        Gain shape used to scale ``gain_peak_W_km`` at the working
        shift. ``None`` (default) uses the peak coefficient as given,
        i.e. assumes the pump sits at the gain peak.
    bandwidth_Hz : float, optional, keyword-only
        Reference bandwidth :math:`B` for the ASE. Default 12.5 GHz,
        the 0.1 nm convention.
    temperature_K : float, optional, keyword-only
        Fibre temperature :math:`T`, which sets the phonon occupancy.
    n_nodes : int, optional, keyword-only
        Number of mesh points of the initial :math:`z` grid.
    tol : float, optional, keyword-only
        Tolerance passed to the solver.

    Returns
    -------
    RamanSolution
        Power profiles and the figures of merit read off them.

    Raises
    ------
    ValueError
        If no pump is on, if a power or a length is negative, or if the
        boundary value problem fails to converge -- an unconverged mesh
        is never returned, because it looks exactly like a plausible
        result.

    References
    ----------
    G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
    2013, Chapter 8.

    M. N. Islam, "Raman amplifiers for telecommunications", IEEE J.
    Sel. Topics Quantum Electron., vol. 8, no. 3, pp. 548-559, 2002.

    Examples
    --------
    >>> solution = solve_raman(length_km=80.0, gain_peak_W_km=0.4,
    ...                        pump_backward_W=0.2)
    >>> round(solution.on_off_gain_dB, 2), round(solution.net_gain_dB, 2)
    (5.97, -10.03)
    """
    from scipy.integrate import solve_bvp, solve_ivp      # local import (D36)

    if length_km <= 0:
        raise ValueError(f"expected a positive span length, got {length_km} km")
    if signal_W <= 0:
        raise ValueError(f"expected a positive signal power, got {signal_W} W")
    if pump_forward_W < 0 or pump_backward_W < 0:
        raise ValueError(
            f"expected non-negative pump powers, got forward "
            f"{pump_forward_W} W and backward {pump_backward_W} W")
    if pump_forward_W == 0 and pump_backward_W == 0:
        raise ValueError(
            "expected at least one pump to be on, got both at 0 W -- pass "
            "pump_forward_W= for co-propagating pumping, pump_backward_W= "
            "for counter-propagating, or both for bidirectional")

    nu_s = SPEED_OF_LIGHT / (wavelength_signal_nm * 1e-9)
    nu_p = SPEED_OF_LIGHT / (wavelength_pump_nm * 1e-9)
    shift = nu_p - nu_s
    if shift <= 0:
        raise ValueError(
            f"expected the pump above the signal in frequency, got a Stokes "
            f"shift of {shift / 1e12:.3g} THz -- the pump wavelength must be "
            f"the shorter one")

    gain = gain_peak_W_km
    if spectrum is not None:
        gain = gain_peak_W_km * float(spectrum.shape(np.asarray(shift)))
        if gain <= 0:
            raise ValueError(
                f"the {spectrum.standard} spectrum gives no gain at a "
                f"{shift / 1e12:.3g} THz shift -- move the pump closer to "
                f"the {spectrum.peak_shift_THz:.3g} THz peak")

    alpha_s = alpha_signal_dB_km / (10 / np.log(10))       # 1/km
    alpha_p = alpha_pump_dB_km / (10 / np.log(10))
    spontaneous = (2 * PLANCK * nu_s * bandwidth_Hz
                   * _photon_occupancy(shift, temperature_K))

    def rhs(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        signal, forward, backward, ase = y
        pump = forward + backward
        transfer = (nu_p / nu_s) * gain * signal
        return np.vstack([
            (gain * pump - alpha_s) * signal,
            -(transfer + alpha_p) * forward,
            +(transfer + alpha_p) * backward,
            (gain * pump - alpha_s) * ase + spontaneous * gain * pump,
        ])

    z = np.linspace(0.0, length_km, n_nodes)
    if pump_backward_W == 0.0:
        # everything is known at z = 0: an initial value problem, which is
        # exact and cannot fail to converge
        solved = solve_ivp(lambda zz, yy: rhs(np.atleast_1d(zz),
                                              yy[:, None]).ravel(),
                           (0.0, length_km),
                           [signal_W, pump_forward_W, 0.0, 0.0],
                           t_eval=z, rtol=tol, atol=tol * 1e-9,
                           dense_output=False)
        if not solved.success:
            raise ValueError(
                f"the co-propagating Raman integration failed: "
                f"{solved.message} -- reduce the pump power or the span "
                f"length, or loosen tol")
        profile = solved.y
    else:
        def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.array([left[0] - signal_W,          # signal at z = 0
                             left[1] - pump_forward_W,    # co pump at z = 0
                             right[2] - pump_backward_W,  # counter at z = L
                             left[3]])                    # no ASE at the input
        # undepleted profiles as the initial guess: solve_bvp converges from
        # here even under heavy depletion, and diverges from a flat guess
        guess = np.vstack([
            signal_W * np.exp(-alpha_s * z),
            pump_forward_W * np.exp(-alpha_p * z),
            pump_backward_W * np.exp(-alpha_p * (length_km - z)),
            np.zeros_like(z),
        ])
        solved = solve_bvp(rhs, boundary, z, guess, tol=tol, max_nodes=200_000)
        if solved.status != 0:
            raise ValueError(
                f"the bidirectional Raman boundary value problem did not "
                f"converge (status {solved.status}: {solved.message}) -- an "
                f"unconverged mesh is not returned because it looks like a "
                f"result; reduce the pump power, shorten the span, or raise "
                f"n_nodes")
        profile = solved.sol(z)

    return RamanSolution(
        z_km=z,
        signal_W=profile[0],
        pump_forward_W=profile[1],
        pump_backward_W=profile[2],
        ase_W=profile[3],
        loss_only_W=signal_W * np.exp(-alpha_s * z),
        bandwidth_Hz=bandwidth_Hz,
        frequency_signal_Hz=nu_s,
        _pump_transmission=float(np.exp(-alpha_p * length_km)),
    )
