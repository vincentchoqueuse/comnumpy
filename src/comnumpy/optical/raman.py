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
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = ["RamanGainSpectrum", "RamanSolution", "solve_raman",
           "get_gain_spectrum", "register_gain_spectrum",
           "available_gain_spectra"]

PLANCK = 6.62607015e-34          # J s
BOLTZMANN = 1.380649e-23         # J/K
SPEED_OF_LIGHT = 2.99792458e8    # m/s


def _check_scale(name: str, value: float, low: float, high: float,
                 unit: str, slips: Dict[str, float]) -> None:
    """Reject a value that is off by orders of magnitude, and say why.

    Every parameter below is a physical quantity in a stated unit, and
    the failure that matters is not a wrong value -- it is the *right*
    value in the wrong unit. Those slips are large: seconds for
    femtoseconds is 1e15, hertz for terahertz is 1e12. Positivity checks
    let all of them through, and the model then runs and produces a
    plausible-looking curve, which is the worst outcome available.

    The window is deliberately wide: it is a unit check, not a physics
    check, so any real glass or fibre must fit inside it. ``slips`` maps
    a candidate unit to the factor that converts it *to* the expected
    one, so the message can name the mistake instead of only the bound.
    """
    if low <= value <= high:
        return
    for candidate, factor in slips.items():
        converted = value * factor
        if low <= converted <= high:
            raise ValueError(
                f"{name} = {value:g} is far outside the plausible "
                f"{low:g} to {high:g} {unit}, but reading it as {candidate} "
                f"gives {converted:g} {unit}, which is in range -- the "
                f"parameter is expected in {unit}")
    raise ValueError(
        f"{name} = {value:g} is far outside the plausible {low:g} to "
        f"{high:g} {unit}; this is a unit check rather than a physical "
        f"one, so a value outside it is almost always a scale mistake")


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
        constant or the peak shift is not positive, or if any of them is
        off by orders of magnitude from its stated unit. The last check
        is there because the units differ between parameterizations --
        femtoseconds here, terahertz there, hertz in the table -- and a
        slip is silent otherwise: ``triangular=13.2e12`` is positive and
        valid-looking, and yields a spectrum that rises linearly across
        the whole band instead of peaking at 13.2 THz. The bounds are
        wide enough that no real glass or fibre reaches them, so what
        they reject is a scale mistake rather than an unusual design.

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
    tabulated: Optional[tuple[np.ndarray, np.ndarray]] = field(
        default=None, kw_only=True)
    quoted_at: Optional[tuple[float, float, float]] = field(
        default=None, kw_only=True)
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        given = [name for name, value in (("lorentzian", self.lorentzian),
                                          ("triangular", self.triangular),
                                          ("tabulated", self.tabulated))
                 if value is not None]
        if len(given) != 1:
            raise ValueError(
                f"expected exactly one parameterization, got "
                f"{given or 'none'} -- pass lorentzian=(tau1_fs, tau2_fs), "
                f"triangular=peak_shift_THz, or "
                f"tabulated=(shift_Hz, gain)")
        if self.tabulated is not None:
            shift, gain = (np.asarray(self.tabulated[0], dtype=float),
                           np.asarray(self.tabulated[1], dtype=float))
            if shift.ndim != 1 or shift.shape != gain.shape:
                raise ValueError(
                    f"expected two 1D arrays of the same length, got "
                    f"{shift.shape} and {gain.shape}")
            if shift.size < 2:
                raise ValueError(
                    f"a measured spectrum needs at least two points, got "
                    f"{shift.size}")
            if np.any(np.diff(shift) <= 0):
                raise ValueError(
                    "expected the Stokes shifts to be strictly increasing; "
                    "sort the table before passing it, so that the "
                    "interpolation cannot silently fold back on itself")
            if np.any(shift < 0):
                raise ValueError(
                    "expected non-negative Stokes shifts: the table gives "
                    "the gain a pump grants a signal below it, and the "
                    "other side is fixed by the model, not measured")
            if np.max(gain) <= 0:
                raise ValueError(
                    f"expected a positive peak gain, got {np.max(gain)}")
            # the shifts are the axis a unit slip hides in: a table given
            # in THz is still increasing, still positive, and still
            # interpolates -- it simply puts the whole Raman band inside
            # the first 40 Hz, where no pump-signal pair ever lands, and
            # the model then reports no gain at all rather than an error
            _check_scale("the largest tabulated Stokes shift",
                         float(np.max(shift)), 1e11, 1e15, "Hz",
                         {"terahertz": 1e12, "gigahertz": 1e9})
            object.__setattr__(self, "tabulated", (shift, gain))
            self._check_quoted_at()
            return
        if self.lorentzian is not None:
            tau1, tau2 = self.lorentzian
            if tau1 <= 0 or tau2 <= 0:
                raise ValueError(
                    f"expected positive time constants in fs, got "
                    f"({tau1}, {tau2})")
            for label, tau in (("tau1", tau1), ("tau2", tau2)):
                _check_scale(label, float(tau), 0.1, 1e6, "fs",
                             {"seconds": 1e15, "picoseconds": 1e3})
            object.__setattr__(self, "lorentzian", (float(tau1), float(tau2)))
        else:
            if self.triangular is None or self.triangular <= 0:
                raise ValueError(
                    f"expected a positive peak shift in THz, got "
                    f"{self.triangular}")
            _check_scale("triangular", float(self.triangular), 0.1, 1e3, "THz",
                         {"hertz": 1e-12, "gigahertz": 1e-3})
            object.__setattr__(self, "triangular", float(self.triangular))
        self._check_quoted_at()

    def _check_quoted_at(self) -> None:
        if self.quoted_at is None:
            return
        try:
            wavelength_nm, area_um2, radius_um = self.quoted_at
        except (TypeError, ValueError) as error:
            raise ValueError(
                "expected quoted_at=(wavelength_nm, effective_area_um2, "
                "core_radius_um) -- the three travel together because the "
                "scaling is meaningless with any one of them missing"
            ) from error
        if min(wavelength_nm, area_um2, radius_um) <= 0:
            raise ValueError(
                f"expected three positive numbers in quoted_at, got "
                f"({wavelength_nm}, {area_um2}, {radius_um})")
        _check_scale("the quoted wavelength", float(wavelength_nm),
                     100.0, 1e5, "nm", {"metres": 1e9, "micrometres": 1e3})
        _check_scale("the quoted effective area", float(area_um2),
                     1.0, 1e4, "um^2", {"square metres": 1e12})
        _check_scale("the quoted core radius", float(radius_um),
                     0.1, 1e3, "um", {"metres": 1e6, "nanometres": 1e-3})
        object.__setattr__(self, "quoted_at",
                           (float(wavelength_nm), float(area_um2),
                            float(radius_um)))

    def pair_scaling(self, frequency_Hz: np.ndarray) -> np.ndarray:
        r"""Waveguide correction of the gain, per (receiver, partner) pair.

        The gain :meth:`shape` is a property of the glass and depends on
        the Stokes shift alone. The coefficient that multiplies
        :math:`P_i P_j` in a power equation is not that shape but
        :math:`g_R / A_{\mathrm{eff}}`, and the effective area is a
        property of the waveguide: the mode spreads as the wavelength
        grows, so the same Stokes shift buys less gain further into the
        infrared. With a Gaussian mode this reduces to a one-parameter
        law, :math:`A_{\mathrm{eff}}(\nu) = A_0 k / (\ln(\nu/\nu_0) + k)`
        with :math:`k = \pi a^2 / A_0`. A pair is charged the mean of the
        two areas, and the gain is referred back to the frequency the
        table was quoted at.

        Across one band this is a few per cent; across C+L+S it is the
        difference between a flat prediction and a wrong tilt.

        Axes: *returns a matrix* ``(n, n)``, one factor per ordered pair,
        with ``j`` the partner that feeds ``i``.

        Parameters
        ----------
        frequency_Hz : array_like
            Frequencies of the waves, in Hz.

        Returns
        -------
        np.ndarray
            Scaling factors, all 1 when the spectrum does not record
            where it was quoted.

        References
        ----------
        The Gaussian-mode form :math:`w = a/\sqrt{\ln V}` and the
        arithmetic-mean overlap are transcribed from GNPy
        (``gnpy/core/parameters.py``, ``effective_area_scaling`` and
        ``effective_area_overlap``, BSD-3-Clause), which does not give a
        source for them; the one-parameter law above is that model with
        the index contrast eliminated, and is derived in
        ``validation/optical_raman_gnpy.py`` rather than taken from a
        paper. The gain profile it is applied to comes from A. D'Amico
        et al., J. Lightwave Technol., vol. 40, pp. 3499-3511, 2022,
        Section III.D.

        The confirmation that matters is empirical rather than
        bibliographic: with this law the coupling coefficients match
        GNPy's element by element on the gain side, ratio 1.0000.

        Examples
        --------
        >>> spectrum = get_gain_spectrum("blow-wood")
        >>> float(np.max(np.abs(spectrum.pair_scaling(
        ...     np.array([193e12, 206e12])) - 1.0)))
        0.0
        """
        frequency = np.asarray(frequency_Hz, dtype=float)
        if self.quoted_at is None:
            return np.ones((frequency.size, frequency.size))
        wavelength_nm, area_um2, radius_um = self.quoted_at
        reference_Hz = SPEED_OF_LIGHT / (wavelength_nm * 1e-9)
        k = np.pi * (radius_um ** 2) / area_um2
        area = k / (np.log(frequency / reference_Hz) + k)     # in units of A_0
        overlap = (area[:, None] + area[None, :]) / 2
        return (1.0 / overlap) * (frequency[None, :] / reference_Hz)

    # -- the shape --------------------------------------------------------
    def _raw(self, shift_Hz: np.ndarray) -> np.ndarray:
        """Unnormalized gain, for the shift in Hz (may be negative)."""
        shift = np.asarray(shift_Hz, dtype=float)
        if self.tabulated is not None:
            grid, gain = self.tabulated
            # Outside the measured range the gain is zero rather than
            # held at the last sample: a table stops where the
            # measurement stopped, and extrapolating a Raman spectrum
            # off the end of the data is how a tilt gets invented.
            return np.where(shift > 0,
                            np.interp(shift, grid, gain, left=0.0, right=0.0),
                            0.0)
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

    The registry is process-wide, so an example that adds to it has to
    take it back out: a doctest that leaves "LAB" behind makes every
    later test that iterates the catalog see an entry that only exists
    when the doctests ran first, which is a failure that depends on
    collection order.

    >>> _ = _SPECTRUM_REGISTRY.pop("LAB")
    >>> "LAB" in available_gain_spectra()
    False
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


# -- the wave table -------------------------------------------------------
def _as_group(value: npt.ArrayLike, name: str) -> np.ndarray:
    """One entry per member of a group, as a 1-D float array."""
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.ndim != 1:
        raise ValueError(
            f"{name} has {array.ndim} dimensions, expected a scalar or a "
            f"1-D sequence with one entry per wave")
    return array


def _broadcast_group(arrays: Dict[str, np.ndarray],
                     group: str) -> tuple[int, Dict[str, np.ndarray]]:
    """Bring every member of a group to the same length.

    A scalar is shared by every wave of the group (one launch power for
    a flat comb, one loss for every pump); anything else must already
    have one entry per wave. Two different lengths are always a
    mistake, never a broadcast.
    """
    count = max(array.size for array in arrays.values())
    for name, array in arrays.items():
        if array.size not in (1, count):
            raise ValueError(
                f"{name} has {array.size} entries while the {group} set has "
                f"{count}; expected either a single value shared by every "
                f"{group} or one value per {group}")
    return count, {name: np.broadcast_to(array, (count,)).astype(float)
                   for name, array in arrays.items()}


def _photon_occupancy(shift_Hz: npt.ArrayLike,
                      temperature_K: float) -> np.ndarray:
    r"""Phonon occupancy :math:`1 + \eta` of the Raman ASE.

    .. math::

        \eta(\Delta \nu, T) =
            \left[\exp\!\left(\frac{h \Delta \nu}{k_B T}\right) - 1\right]^{-1}

    At room temperature and a 13 THz shift this is a few per cent, so
    the spontaneous emission is close to -- but not equal to -- the
    zero-temperature value. It grows without bound as the shift goes to
    zero, which is why the closely spaced pairs of a WDM comb are noisy
    per unit of gain even though they exchange very little power.
    """
    shift = np.asarray(shift_Hz, dtype=float)
    if temperature_K <= 0:
        return np.ones_like(shift)
    with np.errstate(divide="ignore", over="ignore"):
        occupancy = 1.0 + 1.0 / np.expm1(
            PLANCK * np.abs(shift) / (BOLTZMANN * temperature_K))
    return np.where(shift > 0, occupancy, 1.0)


def _coupling_matrix(frequency_Hz: np.ndarray, gain_peak_W_km: float,
                     spectrum: Optional[RamanGainSpectrum]) -> np.ndarray:
    r"""Pairwise Raman coupling :math:`C_{ij}` of a set of waves.

    ``C[i, j]`` is the coefficient multiplying :math:`P_i P_j` in the
    equation of wave :math:`i`: positive when :math:`j` is the bluer of
    the pair and feeds :math:`i`, negative when :math:`i` feeds
    :math:`j`, and the two are tied by photon-number conservation

    .. math::

        C_{ji} = -\frac{\nu_j}{\nu_i} \, C_{ij}, \qquad \nu_j < \nu_i

    which is what makes the total photon flux conserved in the lossless
    limit whatever the number of waves.
    """
    shift = frequency_Hz[None, :] - frequency_Hz[:, None]
    if spectrum is None:
        # every pair at the peak coefficient: only meaningful for one
        # pump-signal pair, which solve_raman enforces
        gain = gain_peak_W_km * (shift > 0).astype(float)
    else:
        gain = gain_peak_W_km * spectrum.shape(shift)     # zero where shift < 0
        gain = gain * spectrum.pair_scaling(frequency_Hz)
    ratio = frequency_Hz[:, None] / frequency_Hz[None, :]
    return gain - ratio * gain.T


# -- the solver -----------------------------------------------------------
@dataclass(frozen=True)
class RamanSolution:
    r"""Power profiles along the span, and what is read off them.

    Signal Model
    ------------
    All profiles are functions of the distance :math:`z` from the span
    input. Per signal :math:`i`, the figures of merit are

    .. math::

        G_\mathrm{on\text{-}off} =
            \frac{P_{s,i}(L)}{P_{s,i}(L)\big|_{P_p = 0}}
          = \frac{P_{s,i}(L)}{P_{s,i}(0) e^{-\alpha_{s,i} L}},
        \qquad
        G_\mathrm{net} = \frac{P_{s,i}(L)}{P_{s,i}(0)}

    -- the on-off gain is what the pumps buy, the net gain is what comes
    out of the span -- and the effective noise figure

    .. math::

        F_\mathrm{eff} = \frac{1}{G_\mathrm{net}}
            \left(1 + \frac{P_\mathrm{ASE}(L)}{h \nu_s B}\right)

    which is *effective* because it refers the noise of the whole
    distributed span to a hypothetical discrete amplifier; it may be
    below 0 dB, and that is the reason distributed amplification is
    used at all.

    **Shapes.** Each profile carries one row per wave, ``(n, n_z)``,
    *except* when its group holds a single wave: a solve with one signal
    and one pump wavelength returns the plain ``(n_z,)`` curves, so the
    single-channel case reads exactly as it did before the multi-wave
    generalization. The figures of merit follow the same rule: a float
    for one signal, an array of one value per signal otherwise.

    Attributes
    ----------
    z_km : np.ndarray
        Distance grid, in km, from 0 to the span length.
    signal_W : np.ndarray
        Signal powers :math:`P_{s,i}(z)` in W, one row per signal.
    pump_forward_W, pump_backward_W : np.ndarray
        Pump powers :math:`P_p^{+}(z)` and :math:`P_p^{-}(z)` in W, one
        row per pump wavelength. Both are given on the same :math:`z`
        grid; the backward pumps are launched at :math:`z = L`. A pump
        that is off in one direction is stored as a zero row, so the two
        arrays always have the same shape.
    ase_W : np.ndarray
        Amplified spontaneous emission :math:`P_\mathrm{ASE}(z)` in W,
        one row per signal, in the reference bandwidth of the solve.
    bandwidth_Hz : float
        Reference bandwidth :math:`B` the ASE was integrated over.
    frequency_signal_Hz : np.ndarray or float
        Signal frequencies :math:`\nu_{s,i}`, needed to turn the ASE
        into a noise figure.
    loss_only_W : np.ndarray
        The signal profiles the same fibre would give with the pumps
        off, :math:`P_{s,i}(0) e^{-\alpha_{s,i} z}`. Carrying them makes
        the on-off gain a ratio of two stored curves rather than a
        recomputation.

    Examples
    --------
    >>> solution = solve_raman(length_km=80.0, gain_peak_W_km=0.4,
    ...                        pump_backward_W=0.2, signal_W=1e-3)
    >>> round(solution.on_off_gain_dB, 2)
    5.97
    """

    z_km: np.ndarray
    # the two the ASE decomposition needs, in the solver's own wave order
    # (signals first): S[i, j] is what multiplies P_j in the spontaneous
    # source of signal i, and wave_power_W[j] is that P_j
    wave_power_W: np.ndarray
    spontaneous_per_wave: np.ndarray
    signal_W: np.ndarray
    pump_forward_W: np.ndarray
    pump_backward_W: np.ndarray
    ase_W: np.ndarray
    loss_only_W: np.ndarray
    bandwidth_Hz: float
    frequency_signal_Hz: Union[float, np.ndarray]

    def _ase_from(self, columns: np.ndarray) -> np.ndarray:
        r"""ASE seeded by a subset of the waves, by superposition.

        The ASE equation is *linear* in the ASE, with a source that is a
        sum over the waves above the channel:

        .. math::

            \frac{\mathrm{d}A_i}{\mathrm{d}z} =
                g_i(z) A_i + \sum_j S_{ij} P_j(z)

        so :math:`A_i = \sum_j A_i^{(j)}` exactly, each term solving the
        same equation with one source kept. The decomposition is
        therefore not an approximation or a re-run: it is arithmetic on
        the profiles already computed.

        The integrating factor is free. :math:`g_i` is the *same* net
        gain the signal obeys, so :math:`\exp \int_0^z g_i` is simply
        :math:`P_{s,i}(z)/P_{s,i}(0)` -- the signal's own profile.
        """
        from scipy.integrate import cumulative_trapezoid

        signal = np.atleast_2d(self.signal_W)
        loss = signal / signal[:, :1]
        source = (np.atleast_2d(self.spontaneous_per_wave)[:, columns]
                  @ np.atleast_2d(self.wave_power_W)[columns])
        integral = cumulative_trapezoid(source / loss, self.z_km,
                                        initial=0.0, axis=-1)
        contribution = loss * integral
        return contribution[0] if contribution.shape[0] == 1 else contribution

    @property
    def ase_from_pumps_W(self) -> np.ndarray:
        """ASE the pumps seed, which is what a Raman amplifier is charged for."""
        columns = np.arange(self.n_signals, np.atleast_2d(self.wave_power_W).shape[0])
        return self._ase_from(columns)

    @property
    def ase_from_signals_W(self) -> np.ndarray:
        r"""ASE the channels seed in each other, which pump-only models omit.

        Every channel sits below some of its neighbours, so the comb
        seeds itself. The shifts involved are small -- a few terahertz
        across a band -- where the Bose occupancy :math:`1/(e^{h \Delta
        \nu / k T} - 1)` is of order one rather than negligible, so the
        term is thermally enhanced. Planning tools commonly drop it; on
        a 96-channel C-band comb it is worth about 0.17 dB of ASE.
        """
        return self._ase_from(np.arange(self.n_signals))

    @property
    def n_signals(self) -> int:
        """Number of signal waves the solve carried."""
        return np.atleast_2d(self.signal_W).shape[0]

    @property
    def n_pumps(self) -> int:
        """Number of pump *wavelengths* (a bidirectional pump counts once)."""
        return np.atleast_2d(self.pump_forward_W).shape[0]

    def _per_signal(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """One value per signal, or the value itself when there is one."""
        return float(values[0]) if values.size == 1 else values

    @property
    def on_off_gain_dB(self) -> Union[float, np.ndarray]:
        """Gain the pumps buy, in dB: output with pumps over output without."""
        return self._per_signal(10 * np.log10(
            np.atleast_2d(self.signal_W)[:, -1]
            / np.atleast_2d(self.loss_only_W)[:, -1]))

    @property
    def net_gain_dB(self) -> Union[float, np.ndarray]:
        """Span output over span input, in dB. Negative for a lossy span."""
        signal = np.atleast_2d(self.signal_W)
        return self._per_signal(10 * np.log10(signal[:, -1] / signal[:, 0]))

    @property
    def gain_profile_dB(self) -> np.ndarray:
        r"""On-off gain accumulated up to each :math:`z`, in dB.

        This is the curve the split-step loop of
        :class:`~comnumpy.optical.links.FiberLink` consumes -- one row
        per signal, so a WDM link applies its own tilted gain to each
        channel of the axis D44 gives it.
        """
        return 10 * np.log10(self.signal_W / self.loss_only_W)

    @property
    def tilt_dB(self) -> float:
        """Spread of the on-off gain across the signals, in dB.

        Zero for a single signal. This is the number a multi-pump design
        exists to minimize: one pump amplifies the red end of the comb
        and starves the blue end, two well-placed pumps flatten it.
        """
        gains = np.atleast_1d(np.asarray(self.on_off_gain_dB, dtype=float))
        return float(np.max(gains) - np.min(gains))

    @property
    def noise_figure_dB(self) -> Union[float, np.ndarray]:
        """Effective noise figure of the span, in dB; may be negative."""
        frequency = np.atleast_1d(np.asarray(self.frequency_signal_Hz,
                                             dtype=float))
        photon = PLANCK * frequency * self.bandwidth_Hz
        signal = np.atleast_2d(self.signal_W)
        net_gain = signal[:, -1] / signal[:, 0]
        ase = np.atleast_2d(self.ase_W)[:, -1]
        return self._per_signal(10 * np.log10((1 + ase / photon) / net_gain))

    @property
    def pump_depletion(self) -> float:
        """Fraction of the launched pump power the signals took away.

        Zero in the undepleted regime the analytic gain formula
        assumes; a large value is the warning that it no longer holds.
        Summed over the pumps: a per-pump breakdown is in the profiles.
        """
        forward = np.atleast_2d(self.pump_forward_W)
        backward = np.atleast_2d(self.pump_backward_W)
        transmission = np.atleast_1d(np.asarray(self._pump_transmission,
                                                dtype=float))
        launched = float(np.sum(forward[:, 0]) + np.sum(backward[:, -1]))
        remaining = float(np.sum(forward[:, -1]) + np.sum(backward[:, 0]))
        no_raman = float(np.sum((forward[:, 0] + backward[:, -1])
                                * transmission))
        if launched <= 0:
            return 0.0
        return max(no_raman - remaining, 0.0) / launched

    def __repr__(self) -> str:
        """Compact view: the arrays are long, the figures of merit are not."""
        gains = np.atleast_1d(np.asarray(self.on_off_gain_dB, dtype=float))
        span = f"{self.z_km[-1]:.4g} km"
        if gains.size == 1:
            gain = f"on-off gain {gains[0]:.2f} dB"
        else:
            gain = (f"on-off gain {gains.min():.2f}..{gains.max():.2f} dB "
                    f"(tilt {self.tilt_dB:.2f} dB)")
        return (f"{type(self).__name__}({self.n_signals} signal(s), "
                f"{self.n_pumps} pump(s), {span}, {gain}, "
                f"depletion {100 * self.pump_depletion:.1f}%)")

    _pump_transmission: Union[float, np.ndarray] = field(default=1.0,
                                                         repr=False)


def solve_raman(*, length_km: float, gain_peak_W_km: float,
                signal_W: npt.ArrayLike = 1e-3,
                pump_forward_W: npt.ArrayLike = 0.0,
                pump_backward_W: npt.ArrayLike = 0.0,
                alpha_signal_dB_km: npt.ArrayLike = 0.2,
                alpha_pump_dB_km: npt.ArrayLike = 0.25,
                wavelength_signal_nm: npt.ArrayLike = 1550.0,
                wavelength_pump_nm: npt.ArrayLike = 1455.0,
                spectrum: Optional[RamanGainSpectrum] = None,
                bandwidth_Hz: float = 12.5e9,
                temperature_K: float = 300.0,
                n_nodes: int = 401,
                tol: float = 1e-8) -> RamanSolution:
    r"""Solve the coupled Raman power equations along one span.

    Signal Model
    ------------
    Every wave -- each signal channel, each pump, in each direction --
    is one power :math:`P_i(z)` written in the :math:`+z` coordinate,
    with a direction :math:`d_i = +1` for a co-propagating wave and
    :math:`-1` for a counter-propagating one. They obey

    .. math::

        \frac{\mathrm{d} P_i}{\mathrm{d} z} =
            d_i \left[\sum_{j} C_{ij} P_j - \alpha_i \right] P_i

    where the pairwise coupling is set by the gain shape at the shift
    separating the two waves,

    .. math::

        C_{ij} = \frac{g_R}{A_\mathrm{eff}}\,
                 \tilde{g}(\nu_j - \nu_i) \quad (\nu_j > \nu_i),
        \qquad
        C_{ji} = -\frac{\nu_j}{\nu_i} C_{ij}

    A pair therefore always exchanges *photons* one for one, the energy
    difference going into a phonon, and that holds for every pair: pump
    to signal, pump to pump, and -- the reason a WDM comb tilts -- the
    blue channels of the comb to its own red ones.

    Spontaneous emission adds, for each signal,

    .. math::

        \frac{\mathrm{d} P_{\mathrm{ASE},i}}{\mathrm{d}z} =
            d_i \left[\left(\sum_j C_{ij} P_j - \alpha_i\right)
            P_{\mathrm{ASE},i}
            + 2 h \nu_i B \sum_{j:\, C_{ij} > 0} C_{ij} P_j
              \left(1 + \eta(\nu_j - \nu_i, T)\right)\right]

    the last bracket being the phonon occupancy, which is why a Raman
    amplifier is quieter when it is cold. The ASE is amplified but does
    not itself deplete the pumps.

    **Boundary conditions.** Co-propagating waves are known at the span
    input, counter-propagating ones at its output:

    .. math::

        P_i(0) = P_i^\mathrm{in} \ (d_i = +1), \quad
        P_i(L) = P_i^\mathrm{in} \ (d_i = -1), \quad
        P_\mathrm{ASE}(0) = 0

    With no counter-propagating pump everything is known at :math:`z=0`
    and the system is an initial value problem, integrated with
    ``scipy.integrate.solve_ivp``. Otherwise it is a genuine two-point
    boundary value problem and ``scipy.integrate.solve_bvp`` is used,
    started from the undepleted-pump profiles -- a poor initial guess
    is the usual reason that solver fails to converge.

    **Undepleted limit.** For one signal and one pump, when the signal
    takes a negligible part of the pump, the pump decays exponentially
    and the on-off gain has the closed form

    .. math::

        G_\mathrm{on\text{-}off} = \exp\left(g P_p L_\mathrm{eff}\right),
        \qquad
        L_\mathrm{eff} = \frac{1 - e^{-\alpha_p L}}{\alpha_p}

    which is what ``validation/optical_raman.py`` checks the solver
    against, along with photon-number conservation in the lossless
    limit.

    **Multi-wave solves.** Every argument describing a signal accepts
    either a scalar -- shared by the whole comb -- or one value per
    channel, and likewise for the pumps; a scalar in gives a scalar
    out. Since a shift now separates every pair, ``spectrum=`` becomes
    *required* as soon as there is more than one signal or more than one
    pump wavelength: its default meaning, "the pair sits at the gain
    peak", is only defensible for a single pair.

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
    signal_W : float or array_like, optional, keyword-only
        Launched signal powers :math:`P_\mathrm{in}`, in W. A scalar is
        the same launch power on every channel.
    pump_forward_W : float or array_like, optional, keyword-only
        Co-propagating pump powers :math:`P_f` launched at :math:`z=0`,
        in W, one per pump wavelength.
    pump_backward_W : float or array_like, optional, keyword-only
        Counter-propagating pump powers :math:`P_b` launched at
        :math:`z = L`, in W, one per pump wavelength. A pump may be on
        in both directions. There is no ``direction`` argument: which
        pumps are on *is* the configuration (D41).
    alpha_signal_dB_km, alpha_pump_dB_km : float or array_like, optional, keyword-only
        Fibre attenuation :math:`\alpha_s` and :math:`\alpha_p` at the
        signal and pump wavelengths, in dB/km.
    wavelength_signal_nm, wavelength_pump_nm : float or array_like, optional, keyword-only
        Wavelengths setting :math:`\nu_s`, :math:`\nu_p` and hence every
        Stokes shift :math:`\Delta \nu`. The lengths of the signal
        arguments fix the number of channels, those of the pump
        arguments the number of pumps.
    spectrum : RamanGainSpectrum, optional, keyword-only
        Gain shape used to scale ``gain_peak_W_km`` at each pair's
        shift. ``None`` (default) uses the peak coefficient as given,
        i.e. assumes the pump sits at the gain peak, and is only
        accepted for a single signal and a single pump wavelength.
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
        If no pump is on, if a power or a length is negative, if a pump
        does not sit above every signal in frequency, if a group's
        arguments disagree on the number of waves, if several waves need
        a ``spectrum=`` and none was given, or if the boundary value
        problem fails to converge -- an unconverged mesh is never
        returned, because it looks exactly like a plausible result.

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

    Three C-band channels under a single pump: the gain follows the
    shape of the spectrum, so the comb comes out tilted.

    >>> comb = [1530.0, 1545.0, 1560.0]
    >>> blow_wood = get_gain_spectrum("blow-wood")
    >>> one = solve_raman(length_km=80.0, gain_peak_W_km=0.4,
    ...                   wavelength_signal_nm=comb, signal_W=1e-3,
    ...                   pump_backward_W=0.4, wavelength_pump_nm=1450.0,
    ...                   spectrum=blow_wood, bandwidth_Hz=0.0)
    >>> print(np.round(one.on_off_gain_dB, 2))
    [ 9.74 11.84 10.89]

    The same total pump power split between two wavelengths halves the
    tilt -- which is the whole reason multi-pump Raman exists:

    >>> two = solve_raman(length_km=80.0, gain_peak_W_km=0.4,
    ...                   wavelength_signal_nm=comb, signal_W=1e-3,
    ...                   pump_backward_W=[0.15, 0.25],
    ...                   wavelength_pump_nm=[1425.0, 1455.0],
    ...                   spectrum=blow_wood, bandwidth_Hz=0.0)
    >>> two.n_signals, two.n_pumps
    (3, 2)
    >>> print(round(one.tilt_dB, 2), round(two.tilt_dB, 2))
    2.1 0.98
    """
    from scipy.integrate import solve_bvp, solve_ivp      # local import (D36)

    if length_km <= 0:
        raise ValueError(f"expected a positive span length, got {length_km} km")

    n_signals, signals = _broadcast_group(
        {"signal_W": _as_group(signal_W, "signal_W"),
         "wavelength_signal_nm": _as_group(wavelength_signal_nm,
                                           "wavelength_signal_nm"),
         "alpha_signal_dB_km": _as_group(alpha_signal_dB_km,
                                         "alpha_signal_dB_km")},
        "signal")
    n_pumps, pumps = _broadcast_group(
        {"pump_forward_W": _as_group(pump_forward_W, "pump_forward_W"),
         "pump_backward_W": _as_group(pump_backward_W, "pump_backward_W"),
         "wavelength_pump_nm": _as_group(wavelength_pump_nm,
                                         "wavelength_pump_nm"),
         "alpha_pump_dB_km": _as_group(alpha_pump_dB_km, "alpha_pump_dB_km")},
        "pump")

    if np.any(signals["signal_W"] <= 0):
        raise ValueError(
            f"expected positive signal powers, got {signals['signal_W']} W")
    if np.any(pumps["pump_forward_W"] < 0) or np.any(pumps["pump_backward_W"] < 0):
        raise ValueError(
            f"expected non-negative pump powers, got forward "
            f"{pumps['pump_forward_W']} W and backward "
            f"{pumps['pump_backward_W']} W")
    if not np.any(pumps["pump_forward_W"] > 0) and not np.any(pumps["pump_backward_W"] > 0):
        raise ValueError(
            "expected at least one pump to be on, got every pump at 0 W -- "
            "pass pump_forward_W= for co-propagating pumping, "
            "pump_backward_W= for counter-propagating, or both for "
            "bidirectional")

    nu_signal = SPEED_OF_LIGHT / (signals["wavelength_signal_nm"] * 1e-9)
    nu_pump = SPEED_OF_LIGHT / (pumps["wavelength_pump_nm"] * 1e-9)
    worst = float(np.min(nu_pump[:, None] - nu_signal[None, :]))
    if worst <= 0:
        raise ValueError(
            f"expected every pump above every signal in frequency, got a "
            f"Stokes shift of {worst / 1e12:.3g} THz for the closest pair -- "
            f"the pump wavelengths must be the shorter ones")

    if spectrum is None and (n_signals > 1 or n_pumps > 1):
        raise ValueError(
            f"spectrum= is required for a multi-wave solve, got None with "
            f"{n_signals} signal(s) and {n_pumps} pump(s); spectrum=None "
            f"means every pair sits at the gain peak, which is only "
            f"defensible for a single pump-signal pair -- pass "
            f"spectrum=get_gain_spectrum('blow-wood') so each pair is "
            f"coupled at its own frequency shift")

    # -- assemble the waves: signals first, then the pumps that are on
    frequency = [nu_signal]
    alpha = [signals["alpha_signal_dB_km"] / (10 / np.log(10))]     # 1/km
    direction = [np.ones(n_signals)]
    launched = [signals["signal_W"]]
    alpha_pump = pumps["alpha_pump_dB_km"] / (10 / np.log(10))
    index_forward = np.full(n_pumps, -1)
    index_backward = np.full(n_pumps, -1)
    position = n_signals
    for sign, powers, index in ((+1.0, pumps["pump_forward_W"], index_forward),
                                (-1.0, pumps["pump_backward_W"], index_backward)):
        for pump in np.flatnonzero(powers > 0):
            frequency.append(nu_pump[pump: pump + 1])
            alpha.append(alpha_pump[pump: pump + 1])
            direction.append(np.array([sign]))
            launched.append(powers[pump: pump + 1])
            index[pump] = position
            position += 1

    nu = np.concatenate(frequency)
    alpha_wave = np.concatenate(alpha)
    direction_wave = np.concatenate(direction)
    launched_wave = np.concatenate(launched)
    n_waves = nu.size

    coupling = _coupling_matrix(nu, gain_peak_W_km, spectrum)
    if not np.any(coupling[:n_signals, n_signals:] > 0):
        shifts = (nu[None, n_signals:] - nu[:n_signals, None]) / 1e12
        raise ValueError(
            f"the pumps buy no gain at all: the shifts "
            f"{np.round(shifts.ravel(), 2)} THz fall outside the "
            f"{spectrum.standard if spectrum else 'flat'} gain spectrum -- "
            f"move the pumps closer to the signals in frequency")

    # spontaneous source, one row per signal: 2 h nu B C_ij (1 + eta_ij)
    shift = nu[None, :] - nu[:n_signals, None]
    spontaneous = np.where(
        coupling[:n_signals] > 0,
        (2 * PLANCK * nu[:n_signals, None] * bandwidth_Hz
         * coupling[:n_signals] * _photon_occupancy(shift, temperature_K)),
        0.0)

    def rhs(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        power = y[:n_waves]
        ase = y[n_waves:]
        net = coupling @ power - alpha_wave[:, None]
        return np.vstack([
            direction_wave[:, None] * net * power,
            direction_wave[:n_signals, None] * (net[:n_signals] * ase
                                                + spontaneous @ power),
        ])

    z = np.linspace(0.0, length_km, n_nodes)
    forward_only = bool(np.all(direction_wave > 0))
    if forward_only:
        # everything is known at z = 0: an initial value problem, which is
        # exact and cannot fail to converge
        start = np.concatenate([launched_wave, np.zeros(n_signals)])
        solved = solve_ivp(lambda zz, yy: rhs(np.atleast_1d(zz),
                                              yy[:, None]).ravel(),
                           (0.0, length_km), start,
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
            # each wave is pinned at the end it is launched from
            edge = np.where(direction_wave > 0, left[:n_waves], right[:n_waves])
            return np.concatenate([edge - launched_wave,
                                   left[n_waves:]])      # no ASE at the input
        # undepleted profiles as the initial guess: solve_bvp converges from
        # here even under heavy depletion, and diverges from a flat guess
        travelled = np.where(direction_wave[:, None] > 0, z[None, :],
                             length_km - z[None, :])
        guess = np.vstack([
            launched_wave[:, None] * np.exp(-alpha_wave[:, None] * travelled),
            np.zeros((n_signals, z.size)),
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

    # -- reassemble: one row per signal, one row per pump wavelength
    pump_profile = []
    for index in (index_forward, index_backward):
        rows = np.zeros((n_pumps, z.size))
        for pump, position in enumerate(index):
            if position >= 0:
                rows[pump] = profile[position]
        pump_profile.append(rows)

    def fold(rows: np.ndarray) -> np.ndarray:
        """A single wave keeps the plain 1-D profile it always had."""
        return rows[0] if rows.shape[0] == 1 else rows

    loss_only = (signals["signal_W"][:, None]
                 * np.exp(-alpha_wave[:n_signals, None] * z[None, :]))
    transmission = np.exp(-alpha_pump * length_km)
    return RamanSolution(
        z_km=z,
        wave_power_W=profile[:n_waves],
        spontaneous_per_wave=spontaneous,
        signal_W=fold(profile[:n_signals]),
        pump_forward_W=fold(pump_profile[0]),
        pump_backward_W=fold(pump_profile[1]),
        ase_W=fold(profile[n_waves:]),
        loss_only_W=fold(loss_only),
        bandwidth_Hz=bandwidth_Hz,
        frequency_signal_Hz=(float(nu_signal[0]) if n_signals == 1
                             else nu_signal),
        _pump_transmission=(float(transmission[0]) if n_pumps == 1
                            else transmission),
    )
