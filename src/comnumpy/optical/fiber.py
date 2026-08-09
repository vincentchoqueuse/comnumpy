r"""What the fibre is, separated from how it is simulated (decision D46).

:class:`FiberLink` used to take twenty-one constructor arguments, mixing
two things that change for entirely different reasons: the **glass** --
its loss, its Kerr coefficient, its dispersion -- and the **numerics**
-- how many split-steps, which step distribution, what sampling rate. A
user swapping SMF for a dispersion-compensating fibre had to retype four
unrelated numbers in the middle of a list of solver settings, and
nothing tied them together or said where they came from.

:class:`FiberSpec` is the fourth instance of the pattern D15/D43/D45
established on the frequency, delay and Raman-shift axes: a frozen value
object carrying the numbers **and their provenance**, a registry of
standard entries, and a self-check at construction against a figure
published alongside them.

It also removes a latent incoherence. The link took ``lamb`` *and*
``nu`` as independent arguments, so a chain set to 1310 nm computed its
dispersion at 1310 nm and its ASE photon energy at 1550: the two agreed
only because the two defaults happened to. Here :math:`\nu = c / \lambda`
is derived, and cannot disagree with itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .constants import (CD_COEFFICIENT, FIBER_LOSS, KERR_COEFFICIENT,
                        SPEED_OF_LIGHT, WAVELENGTH)
from .utils import compute_beta2

__all__ = ["FiberSpec", "get_fiber", "register_fiber", "available_fibers"]


@dataclass(frozen=True)
class FiberSpec:
    r"""Physical parameters of an optical fibre, with their provenance.

    Signal Model
    ------------
    The three coefficients that drive the nonlinear Schrödinger equation
    the split-step method integrates:

    .. math::

        \frac{\partial A}{\partial z} =
            \underbrace{-\frac{\alpha}{2} A}_{\text{loss}}
            \underbrace{- j \frac{\beta_2}{2}
              \frac{\partial^2 A}{\partial t^2}}_{\text{dispersion}}
            + \underbrace{j \gamma |A|^2 A}_{\text{Kerr}}

    The dispersion is given as the engineering coefficient
    :math:`D` in ps/nm/km, which is what data sheets quote, and
    converted to the group-velocity dispersion the equation uses:

    .. math::

        \beta_2 = -\frac{D \lambda^2}{2 \pi c}

    The carrier frequency is **derived**, :math:`\nu = c / \lambda`, and
    is not a separate parameter: two independent arguments that must
    agree are the defect D41 exists to prevent.

    Parameters
    ----------
    alpha_dB : float, optional
        Attenuation :math:`\alpha` in dB/km.
    gamma : float, optional, keyword-only
        Kerr coefficient :math:`\gamma` in rad/W/km.
    cd_coefficient : float, optional, keyword-only
        Dispersion coefficient :math:`D` in ps/nm/km, quoted at
        ``wavelength_nm``. Negative for a dispersion-compensating fibre.
    wavelength_nm : float, optional, keyword-only
        Wavelength :math:`\lambda` in nm the coefficients are quoted at,
        and which fixes the carrier frequency.
    raman_gain_W_km : float, optional, keyword-only
        Peak Raman gain :math:`g_R / A_\mathrm{eff}` in
        :math:`\mathrm{W}^{-1}\mathrm{km}^{-1}`. A fibre property, not a
        property of the glass spectrum, which is why it lives here and
        not in :class:`~comnumpy.optical.raman.RamanGainSpectrum`.
        ``None`` when the entry does not quote one.
    standard : str, optional, keyword-only
        Name of the fibre, e.g. ``"SMF"``.
    reference : str, optional, keyword-only
        Where the numbers come from.

    Raises
    ------
    ValueError
        If the attenuation is negative, if the Kerr coefficient is
        negative, or if the wavelength is not positive. The dispersion
        may be of either sign; a Raman gain may not be negative.

    References
    ----------
    G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
    2013, Sections 1.2 and 2.3.

    Examples
    --------
    >>> fiber = get_fiber("SMF")
    >>> round(fiber.beta2, 3), round(fiber.carrier_frequency_Hz / 1e12, 4)
    (-21.683, 193.4145)
    >>> round(fiber.effective_length_km(80.0), 3)
    21.169
    """

    alpha_dB: float = FIBER_LOSS
    gamma: float = field(default=KERR_COEFFICIENT, kw_only=True)
    cd_coefficient: float = field(default=CD_COEFFICIENT, kw_only=True)
    wavelength_nm: float = field(default=WAVELENGTH, kw_only=True)
    raman_gain_W_km: Optional[float] = field(default=None, kw_only=True)
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if self.alpha_dB < 0:
            raise ValueError(
                f"expected a non-negative attenuation in dB/km, got "
                f"{self.alpha_dB} -- a fibre with gain is a Raman span, "
                f"see comnumpy.optical.raman")
        if self.gamma < 0:
            raise ValueError(
                f"expected a non-negative Kerr coefficient, got {self.gamma}")
        if self.wavelength_nm <= 0:
            raise ValueError(
                f"expected a positive wavelength in nm, got "
                f"{self.wavelength_nm}")
        if self.raman_gain_W_km is not None and self.raman_gain_W_km < 0:
            raise ValueError(
                f"expected a non-negative Raman gain, got "
                f"{self.raman_gain_W_km}")
        self._check_plausible()

    # Plausibility bounds, deliberately loose: they are not laws of
    # physics, they are the width of a unit mistake. A value a thousand
    # times too large is the signature of dB/km read as dB/m, ps/nm/km as
    # s/m/m, or nm as um -- the failure mode this class was written after
    # making. Anything a real fibre does fits inside them with room to
    # spare, so a rejection here is a unit, not a research question.
    _PLAUSIBLE = {
        "alpha_dB": (100.0, "dB/km", "0.2 for silica at 1550 nm"),
        "gamma": (1e3, "rad/W/km", "1.3 for SMF, ~5 for DCF"),
        "cd_coefficient": (1e4, "ps/nm/km", "17 for SMF, -100 for DCF"),
        "wavelength_nm": (1e5, "nm", "1550 in the C band"),
        "raman_gain_W_km": (1e3, "1/(W km)", "~0.4 for SMF"),
    }

    def _check_plausible(self) -> None:
        for name, (bound, unit, typical) in self._PLAUSIBLE.items():
            value = getattr(self, name)
            if value is not None and abs(value) > bound:
                raise ValueError(
                    f"expected {name} in {unit} (typically {typical}), got "
                    f"{value:g}, which is beyond the plausible bound "
                    f"{bound:g} -- this is the size of a unit mistake, not "
                    f"of a fibre. Pass FiberSpec(..., {name}=...) in {unit}.")
        if self.wavelength_nm < 100:
            raise ValueError(
                f"expected wavelength_nm in nm (1550 in the C band), got "
                f"{self.wavelength_nm:g} -- micrometres, perhaps?")

    # -- derived quantities ------------------------------------------------
    @property
    def beta2(self) -> float:
        r"""Group-velocity dispersion :math:`\beta_2` in ps²/km.

        Delegates to :func:`~comnumpy.optical.utils.compute_beta2`: one
        formula, one place. Writing the conversion a second time here
        got the unit scaling wrong by a factor of a thousand, and the
        self-check below is what caught it.
        """
        return compute_beta2(self.wavelength_nm, self.cd_coefficient,
                             SPEED_OF_LIGHT)

    @property
    def carrier_frequency_Hz(self) -> float:
        r"""Optical carrier :math:`\nu = c / \lambda`, in Hz."""
        return SPEED_OF_LIGHT / (self.wavelength_nm * 1e-9)

    @property
    def alpha_per_km(self) -> float:
        r"""Attenuation :math:`\alpha` in 1/km, the equation's unit."""
        return self.alpha_dB / (10 / math.log(10))

    def effective_length_km(self, length_km: float) -> float:
        r"""Nonlinear effective length :math:`(1 - e^{-\alpha L})/\alpha`.

        The length a lossless fibre would need to accumulate the same
        Kerr phase. Equal to ``length_km`` for a lossless fibre.

        Parameters
        ----------
        length_km : float
            Physical span length :math:`L` in km.

        Returns
        -------
        float
            Effective length in km.

        Examples
        --------
        >>> round(get_fiber("SMF").effective_length_km(80.0), 3)
        21.169
        >>> round(FiberSpec(0.0).effective_length_km(80.0), 3)
        80.0
        """
        alpha = self.alpha_per_km
        if alpha == 0:
            return length_km
        return (1 - math.exp(-alpha * length_km)) / alpha

    def loss_dB(self, length_km: float) -> float:
        """Total attenuation of a span of ``length_km``, in dB."""
        return self.alpha_dB * length_km

    def __repr__(self) -> str:
        raman = ("" if self.raman_gain_W_km is None
                 else f", g_R/A_eff {self.raman_gain_W_km:.4g} /W/km")
        return (f"FiberSpec({self.standard} at {self.wavelength_nm:.0f} nm)\n"
                f"  alpha {self.alpha_dB:.4g} dB/km, gamma {self.gamma:.4g} "
                f"rad/W/km, D {self.cd_coefficient:.4g} ps/nm/km"
                f"  -> beta2 {self.beta2:.4g} ps^2/km{raman}"
                + (f"\n  {self.reference}" if self.reference else ""))


# -- catalog ---------------------------------------------------------------
_FIBER_REGISTRY: dict[str, Callable[..., FiberSpec]] = {}


def register_fiber(name: str):
    """Register a catalog entry; users can add their own fibres.

    Parameters
    ----------
    name : str
        Name the factory answers to in :func:`get_fiber`.

    Examples
    --------
    >>> @register_fiber("LAB")
    ... def _lab():
    ...     return FiberSpec(0.19, gamma=1.2, standard="LAB",
    ...                      reference="bench")
    >>> get_fiber("LAB").alpha_dB
    0.19
    """
    def decorator(func: Callable[..., FiberSpec]):
        _FIBER_REGISTRY[name] = func
        return func
    return decorator


def available_fibers() -> list[str]:
    """Names accepted by :func:`get_fiber`.

    Returns
    -------
    list of str
        Registered names, sorted.

    Examples
    --------
    >>> available_fibers()
    ['DCF', 'NZDSF', 'SMF']
    """
    return sorted(_FIBER_REGISTRY)


def get_fiber(standard: str, **kwargs: Any) -> FiberSpec:
    """Return a catalog fibre by name.

    Parameters
    ----------
    standard : str
        Entry name; see :func:`available_fibers`.
    **kwargs
        Forwarded to the entry factory.

    Raises
    ------
    KeyError
        If the name is not in the catalog.

    Examples
    --------
    >>> get_fiber("SMF").cd_coefficient
    17.0
    """
    if standard not in _FIBER_REGISTRY:
        raise KeyError(
            f"unknown fibre {standard!r}; available: {available_fibers()} -- "
            f"register your own with @register_fiber, or build a FiberSpec "
            f"directly")
    return _FIBER_REGISTRY[standard](**kwargs)


def _check_expect(fiber: FiberSpec,
                  expect: Optional[dict[str, float]]) -> FiberSpec:
    """Verify a catalog entry against a figure published with it (D20)."""
    if not expect:
        return fiber
    checks = {"beta2": fiber.beta2,
              "carrier_frequency_THz": fiber.carrier_frequency_Hz / 1e12}
    for key, wanted in expect.items():
        if key not in checks:
            raise KeyError(f"unknown self-check {key!r}; known: {sorted(checks)}")
        got = checks[key]
        if abs(got - wanted) > max(0.05, 5e-3 * abs(wanted)):
            raise ValueError(
                f"{fiber.standard}: expected {key} = {wanted} from "
                f"{fiber.reference or 'the source'}, these coefficients give "
                f"{got:.4f} -- one of the two is wrong.")
    return fiber


@register_fiber("SMF")
def _smf() -> FiberSpec:
    r"""Standard single-mode fibre at 1550 nm.

    The textbook working point, and the library's historical defaults.
    The self-check is the one independent handle these numbers offer:
    :math:`D = 17` ps/nm/km at 1550 nm must give
    :math:`\beta_2 \approx -21.7` ps²/km, which is quoted separately in
    the same references. It comes out at -21.683.

    These are round textbook values, not a data sheet: no manufacturer's
    figures have been transcribed here.
    """
    return _check_expect(
        FiberSpec(0.2, gamma=1.3, cd_coefficient=17.0, wavelength_nm=1550.0,
                  raman_gain_W_km=0.4, standard="SMF",
                  reference="Agrawal, Nonlinear Fiber Optics, 5th ed., "
                            "sections 1.2 and 2.3 (textbook values)"),
        {"beta2": -21.7})


@register_fiber("NZDSF")
def _nzdsf() -> FiberSpec:
    """Non-zero dispersion-shifted fibre at 1550 nm.

    Low dispersion and a smaller effective area, hence a larger Kerr
    coefficient than SMF. Round textbook values.
    """
    return FiberSpec(0.2, gamma=1.7, cd_coefficient=4.0,
                     wavelength_nm=1550.0, standard="NZDSF",
                     reference="Agrawal, Fiber-Optic Communication Systems, "
                               "4th ed., chapter 2 (textbook values)")


@register_fiber("DCF")
def _dcf() -> FiberSpec:
    """Dispersion-compensating fibre at 1550 nm.

    Large negative dispersion, higher loss and a much smaller effective
    area. The sign of ``cd_coefficient`` is what makes it compensating,
    and it is the reason the constructor accepts either sign.
    """
    return FiberSpec(0.5, gamma=5.0, cd_coefficient=-100.0,
                     wavelength_nm=1550.0, standard="DCF",
                     reference="Agrawal, Fiber-Optic Communication Systems, "
                               "4th ed., chapter 8 (textbook values)")
