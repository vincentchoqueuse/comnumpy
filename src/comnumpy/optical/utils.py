import numpy as np
from comnumpy._backend import fft, ifft, fftfreq  # cupy-compatible (D3)
from typing import Optional

from comnumpy.exceptions import ShapeError

from .constants import PLANCK_CONSTANT, OPTICAL_CARRIER_FREQUENCY

__all__ = [
    "compute_beta2", "apply_chromatic_dispersion", "apply_kerr_nonlinearity",
    "compute_erbium_doped_fiber_amplifier_gain",
    "compute_erbium_doped_fiber_N_ase", "get_linear_step_size",
    "get_logarithmic_step_size", "itu_grid_frequency", "is_polarization_pair",
    "manakov_kerr", "dbm_to_watt", "watt_to_dbm", "launch_amplitude",
]


def compute_beta2(lamb: float, cd_coefficient: float,
                  speed_of_light: float) -> float:
    r"""
    Compute the Chromatic Dispersion coefficient β₂ in ps²/km

    Parameters
    ----------
    lamb : float
        Wavelength (nm)
    cd_coefficient: float
        Chromatic Dispersion coefficient (ps/nm/km)
    speed_of_light: float
        Speed of light (m/s)

    The formula is given by :

    .. math ::

        \beta_2 = -\frac{10^3 \cdot D \cdot \lambda^2}{2\pi c}

    .. WARNING::

        - All input values must be in the units specified above.
        - Output β₂ is given in picosecond squared per kilometer (ps²/km).


    Returns
    -------
    beta2: float
        Group velocity dispersion β₂ (in ps²/km)


    Example 1
    ---------

    >>> lamb = 1550         # in nm
    >>> cd_coefficient = 17 # in ps/nm/km
    >>> c = 299792458       # in m/s
    >>> beta2 = compute_beta2(lamb, cd_coefficient, c)
    >>> print(beta2)
    -21.682619391414896


    Notes
    -----

    - The formula is derived from Eq. (4) and (5) of [Savory, 2008].

    References
    ----------
    * [1] Savory, Seb J. "Digital filters for coherent optical receivers." Optics express 16.2 (2008): 804-817.
    * [2] Eghbali, Amir, et al. "Optimal least-squares FIR digital filters for compensation of chromatic dispersion in digital coherent
      optical receivers." Journal of lightwave technology 32.8 (2014): 1449-1456.

    """
    beta2 = -((10**3) * cd_coefficient * (lamb**2)) / (2*np.pi*speed_of_light)  # see [1] eq 4 and 5
    # cd_coefficient: chromatic dispersion coefficient in ps/(nm·km)
    # lamb: wavelength in nm
    # speed_of_light: speed of light in m/s
    # beta2: chromatic dispersion parameter in ps²/km

    # numerator unit: cd_coefficient * (lamb**2)) in ps.nm/km (ps*10^-12)
    # denominator unit: 2*np.pi*speed_of_light in m/s (m/ps * 10^-12)
    # division: ps^2/m
    # The factor 10^3 converts from per meter to per kilometer,
    return beta2


def apply_chromatic_dispersion(x: np.ndarray, z: float, beta2: float,
                               alpha_dB: Optional[float] = None,
                               fs: float = 1, direction: int = 1) -> np.ndarray:
    """
    Apply chromatic dispersion effects in optical fiber communications.

    Apply chromatic dispersion effects in the frequency domain for
    fiber-optic communication systems. It applies a dispersion-induced phase shift
    to the input signal in the frequency domain and considers signal attenuation [1].

    Parameters
    ----------
    x: numpy array
        Complex signal
    z : float
        Step length in meters (km).
    beta2: float
        coefficient in ps**2/km
    alpha_dB: float, optional
        gain in dB / km (defaut: None)
    fs : float, optional
        Sampling frequency in hertz (Hz).
    direction : int, optional
        Propagation direction, 1 for forward and -1 for backward. Defaults to 1.

    Returns
    -------
    y: numpy array
        Complex signal after chromatic dispersion

    References
    ----------
    * [1] Savory, Seb J. "Digital filters for coherent optical receivers." Optics express 16.2 (2008): 804-817.
    * [2] Eghbali, Amir, et al. "Optimal least-squares FIR digital filters for compensation of chromatic dispersion in digital coherent
      optical receivers." Journal of lightwave technology 32.8 (2014): 1449-1456.

    """
    if alpha_dB:
        alpha = (np.log(10)/10) * alpha_dB  # convert dB to linear factor
        gain = np.exp(-(alpha/2) * z * direction)  # see text before equation 6 in https://arxiv.org/pdf/2010.14258.pdf
    else:
        gain = 1

    beta2_s2_per_km = ((10**-12)**2) * beta2  # convert into s^2/km
    NFFT = x.shape[-1]      # the field may carry a polarization axis
    w = (2*np.pi*fs)*fftfreq(NFFT, d=1, like=x)
    H = np.exp(1j * (beta2_s2_per_km/2) * z * (w**2) * direction)  # see equation 4
    fftx = fft(x)
    ffty = H * fftx
    y = gain * ifft(ffty)
    return y


def apply_kerr_nonlinearity(x: np.ndarray, z: float, gamma: float,
                            gain: float = 1, direction: int = 1,
                            intensity: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    Apply Kerr nonlinearity phase rotation to a signal.

    The rotation is :math:`\exp(j \gamma z I)` where :math:`I` is the
    intensity driving it. For a single field that is :math:`|x|^2`; for
    the Manakov model of two polarizations it is the *total* intensity
    :math:`|E_x|^2 + |E_y|^2` shared by both, which is what ``intensity``
    is for.

    Parameters
    ----------
    x : np.ndarray
        Input complex signal.
    z : float
        Fiber length in km.
    gamma : float
        Kerr coefficient in rad/W/km. The Manakov model passes
        :math:`8\gamma/9` here: the factor belongs to the model, not to
        the fibre.
    gain : float, optional
        Gain factor. Default is 1.
    direction : int, optional
        Propagation direction (1=forward, -1=backward). Default is 1.
    intensity : np.ndarray, optional
        Intensity driving the rotation, broadcastable against ``x``.
        Default (None) uses ``|x|**2``, the scalar NLSE.

    Returns
    -------
    np.ndarray
        Signal after Kerr nonlinear phase rotation.
    """
    nl_param = direction * gamma * z
    power = np.abs(x)**2 if intensity is None else intensity
    return gain * x * np.exp(1j*nl_param*power)


def compute_erbium_doped_fiber_amplifier_gain(alpha_dB: float,
                                              L_span: float) -> float:
    """
    Compute the amplitude gain of an EDFA that compensates for fiber loss.

    Parameters
    ----------
    alpha_dB : float
        Fiber loss in dB/km.
    L_span : float
        Span length in km.

    Returns
    -------
    float
        Amplitude gain factor.
    """
    G = 10**(alpha_dB*L_span/10)
    gain = np.sqrt(G)
    return gain


def compute_erbium_doped_fiber_N_ase(alpha_dB: float, L_span: float,
                                     NF_dB: float,
                                     h: float = PLANCK_CONSTANT,
                                     nu: float = OPTICAL_CARRIER_FREQUENCY) -> float:
    r"""
    Compute ASENoise params

    Parameters
    ----------
    alpha_dB : float
        Fiber loss (dB/km)
    L_span : float
        Length of the link (in km)
    NF_dB: float
        Noise Figure (dB)


    The formula is given by :

    .. math ::

        N_{ASE} = (e^{\alpha L}-1) h \nu n_{sp}

    where


    .. math ::

        \alpha =\alpha_{dB}/10 \log_{10}(e)

    References
    ----------
    * [1] Essiambre, René-Jean, Gerhard Kramer, Peter J. Winzer, Gerard J. Foschini, and Bernhard Goebel.
      "Capacity limits of optical fiber networks." Journal of Lightwave technology 28, no. 4 (2010): 662-701.

    """
    # see equation 54 of the paper use a term
    # G = e^{alpha L}
    # where
    # alpha = alpha_dB/(10*np.log10(np.e))
    #
    # Using simplification, it can be checked that
    # G = e^(alpha_dB*L/10log10(e)) = exp(alpha_dB*L/(10/ln(10))) = exp(alpha_dB*L ln(10)/10)
    # by using the fact exp(a ln(b)) = b^a, we obtain
    # G = 10 ^(alpha_dB*L /10)
    G = 10**(alpha_dB*L_span/10)
    if G <= 1:
        # lossless span: no loss to compensate, no amplifier, no ASE noise
        # (the n_sp formula below diverges at G=1)
        return 0.0
    NF = 10**(NF_dB/10)
    n_sp = (NF/2) / (1-1/G)  # see Hager paper after equation 11
    N_ase = (G-1) * h * nu * n_sp   # see code of Hager https://github.com/chaeger/LDBP/blob/master/ldbp/ldbp.py
    return N_ase


def get_linear_step_size(L_span: float, StPS: int) -> np.ndarray:
    """
    Compute uniformly spaced step sizes for the split-step Fourier method.

    Parameters
    ----------
    L_span : float
        Span length in km.
    StPS : int
        Number of steps per span.

    Returns
    -------
    np.ndarray
        Array of step sizes, each equal to ``L_span / StPS``.
    """
    return (L_span/StPS)*np.ones(StPS)


def get_logarithmic_step_size(L_span: float, StPS: int, alpha_dB: float = 0,
                              step_log_factor: float = 0.4) -> np.ndarray:
    """
    Compute logarithmically spaced step sizes for the split-step Fourier method.

    Parameters
    ----------
    L_span : float
        Span length in km.
    StPS : int
        Number of steps per span.
    alpha_dB : float, optional
        Fiber loss in dB/km. Default is 0.
    step_log_factor : float, optional
        Logarithmic step factor. Default is 0.4.

    Returns
    -------
    np.ndarray
        Array of logarithmically spaced step sizes.

    References
    ----------
    * [1] O. V. Sinkin et al., "Optimization of the split-step Fourier method in
      modeling optical-fiber communications systems," J. Lightwave Technol., vol. 21,
      no. 1, pp. 61-68, Jan. 2003.
    """
    alpha = (np.log(10)/10) * (alpha_dB)
    alpha_adj = step_log_factor*alpha
    delta = (1-np.exp(-alpha_adj*L_span))/StPS
    n_vect = 1 + np.arange(StPS)
    z = -(1/alpha_adj)*np.log((1-n_vect*delta)/(1-(n_vect-1)*delta))
    return z


def itu_grid_frequency(n: int, m: int = 1) -> tuple[float, float]:
    r"""
    Center frequency and slot width of an ITU-T G.694.1 flexible-grid channel.

    A WDM channel is described by the integer couple :math:`(n, m)`
    (decision D19): the center frequency is
    :math:`193.1 + n \times 0.00625` THz and the slot width is
    :math:`12.5 \times m` GHz. The fixed grids are particular cases
    (e.g. the 50 GHz grid uses even :math:`n` multiples of 8 and
    :math:`m = 4`).

    Parameters
    ----------
    n : int
        Signed channel index on the 6.25 GHz granularity.
    m : int, optional
        Slot width multiplier (slot width = 12.5 * m GHz). Default is 1.

    Returns
    -------
    center_Hz : float
        Channel center frequency in Hz.
    width_Hz : float
        Slot width in Hz.

    References
    ----------
    ITU-T Recommendation G.694.1 (2020), section 7 (flexible DWDM grid).

    Examples
    --------
    >>> center, width = itu_grid_frequency(0, m=4)
    >>> print(f"{center/1e12:.4f} THz, {width/1e9:.1f} GHz")
    193.1000 THz, 50.0 GHz
    >>> center, width = itu_grid_frequency(-32, m=4)
    >>> print(f"{center/1e12:.4f} THz")
    192.9000 THz
    """
    # the annotation says int, but this is a library called from untyped
    # scripts, and a float n silently lands between two grid slots
    if not (isinstance(n, (int, np.integer))              # pyright: ignore[reportUnnecessaryIsInstance]
            and isinstance(m, (int, np.integer))):        # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"ITU G.694.1 channels are described by integers (n, m), got ({n!r}, {m!r})")
    if m < 1:
        raise ValueError(f"slot width multiplier m must be >= 1, got {m}")
    center_Hz = (193.1e12) + n * 6.25e9
    width_Hz = 12.5e9 * m
    return center_Hz, width_Hz


def is_polarization_pair(x: np.ndarray, block: str) -> bool:
    r"""Read the propagation model off the shape of the field.

    A ``(N,)`` or ``(..., 1, N)`` field is one polarization and obeys
    the scalar NLSE; a ``(..., 2, N)`` field is a polarization pair and
    obeys the Manakov equation. Anything else on the antenna axis is
    refused, because a pointwise Kerr step applied row by row would
    describe parallel fibres, not one fibre carrying several signals.

    **The leading axes are a batch.** Only ``-2`` and ``-1`` are read,
    so ``(B, P, N)`` propagates ``B`` independent realizations in one
    call and every block downstream broadcasts over them -- the same
    rule the MIMO blocks follow on their antenna axis (D2). That is
    also why the polarization axis must be written even when there is
    one of them: ``(B, N)`` is indistinguishable from ``(P, N)``, so a
    batch of single-polarization fields is ``(B, 1, N)``.

    Parameters
    ----------
    x : np.ndarray
        Field the block is about to propagate.
    block : str
        Name of the calling block, for the message.

    Returns
    -------
    bool
        True when the Manakov model applies.

    Raises
    ------
    ShapeError
        If the polarization axis has any size other than 1 or 2.
    """
    if x.ndim == 1:
        return False
    polarizations = x.shape[-2]
    if polarizations in (1, 2):
        return polarizations == 2
    raise ShapeError(
        f"{block} propagates one field (N,) or a polarization pair "
        f"(..., P, N) with P in {{1, 2}}; got {x.shape}, i.e. "
        f"{polarizations} on the polarization axis. A pointwise Kerr step "
        f"row by row would describe {polarizations} separate fibres, with "
        f"no XPM and no FWM between them. Two shapes are confused with "
        f"this one. For {polarizations} WDM channels, multiplex them into "
        f"one field first with comnumpy.optical.WDMMultiplexer (D44). For "
        f"{polarizations} independent realizations propagated at once, "
        f"write the polarization axis: "
        f"({polarizations}, 1, {x.shape[-1]}).")


def manakov_kerr(x: np.ndarray, gamma: float,
                 manakov: bool) -> tuple[float, Optional[np.ndarray]]:
    r"""Kerr coefficient and driving intensity of one split step.

    The Manakov equation shares the *total* intensity
    :math:`|E_x|^2 + |E_y|^2` between the two polarizations and scales
    the coefficient by :math:`8/9`, the average of the nonlinearity over
    the fast random birefringence of a real fibre. That factor is a
    property of the model, not of the glass, which is why it lives here
    and not in :class:`~comnumpy.optical.fiber.FiberSpec`.

    Parameters
    ----------
    x : np.ndarray
        Field at the current step.
    gamma : float
        Kerr coefficient of the fibre, in rad/W/km.
    manakov : bool
        Whether the field is a polarization pair.

    Returns
    -------
    gamma : float
        Coefficient to use in the rotation.
    intensity : np.ndarray or None
        Intensity driving it, or None for the scalar NLSE.
    """
    if not manakov:
        return gamma, None
    return 8 / 9 * gamma, np.sum(np.abs(x) ** 2, axis=-2, keepdims=True)


def dbm_to_watt(power_dBm: np.ndarray | float) -> np.ndarray | float:
    r"""Optical power from dBm to watts.

    Signal Model
    ------------
    .. math::

        P[\mathrm{W}] = 10^{-3} \times 10^{P[\mathrm{dBm}] / 10}

    Optical data sheets, standards and papers quote power in dBm while
    every equation in this package takes watts, so the conversion sits
    at the boundary of nearly every optical script. Written out by hand
    it is easy to get subtly wrong -- the factor is :math:`10^{-3}` and
    the divisor is 10, not 20, because this is a power and not an
    amplitude.

    Axes: *element-wise*.

    Parameters
    ----------
    power_dBm : float or np.ndarray
        Power in dBm, i.e. decibels relative to one milliwatt.

    Returns
    -------
    float or np.ndarray
        Power in watts.

    References
    ----------
    G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley,
    2010, Section 1.2.

    Examples
    --------
    >>> print(dbm_to_watt(0.0))
    0.001
    >>> print(round(dbm_to_watt(-3.0), 7))
    0.0005012
    >>> print(np.round(dbm_to_watt(np.array([-10.0, 0.0, 10.0])), 6))
    [0.0001 0.001  0.01  ]
    """
    result = 1e-3 * 10 ** (np.asarray(power_dBm, dtype=float) / 10)
    return result if np.ndim(power_dBm) else float(result)


def watt_to_dbm(power_W: np.ndarray | float) -> np.ndarray | float:
    r"""Optical power from watts to dBm, the inverse of :func:`dbm_to_watt`.

    Signal Model
    ------------
    .. math::

        P[\mathrm{dBm}] = 10 \log_{10}\!\left(
            \frac{P[\mathrm{W}]}{10^{-3}}\right)

    Axes: *element-wise*.

    Parameters
    ----------
    power_W : float or np.ndarray
        Power in watts. Must be positive: zero power is
        :math:`-\infty` dBm and negative power is not a power.

    Returns
    -------
    float or np.ndarray
        Power in dBm.

    Raises
    ------
    ValueError
        If any value is not strictly positive. A silent ``-inf`` or
        ``nan`` propagating into a link budget is worse than a stop.

    References
    ----------
    G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley,
    2010, Section 1.2.

    Examples
    --------
    >>> print(watt_to_dbm(1e-3))
    0.0
    >>> print(round(watt_to_dbm(5.0118723e-4), 4))
    -3.0
    >>> print(np.round(watt_to_dbm(np.array([1e-4, 1e-3, 1e-2])), 6))
    [-10.   0.  10.]
    """
    power = np.asarray(power_W, dtype=float)
    if np.any(power <= 0):
        raise ValueError(
            f"a power in dBm is a logarithm, so it needs a strictly "
            f"positive power in watts; got a minimum of {float(np.min(power))}.")
    result = 10 * np.log10(power / 1e-3)
    return result if np.ndim(power_W) else float(result)


def launch_amplitude(power_W: np.ndarray | float, *,
                     polarizations: int = 1) -> np.ndarray | float:
    r"""Field amplitude that launches a given optical power.

    Signal Model
    ------------
    An :class:`~comnumpy.core.processors.Amplifier` multiplies the
    **field**, and a power is the squared modulus of it, so a launch
    power is set through a square root:

    .. math::

        a = \sqrt{\frac{P}{P_{\mathrm{pol}}}}

    The division is the point. A dual-polarization signal carries the
    channel power split between its two polarizations, so each one is
    launched at :math:`\sqrt{P/2}` -- and that factor, written by hand
    at each call site, is the same one that makes an ASE budget or a
    nonlinear coefficient disagree by 3 dB (see
    :meth:`~comnumpy.optical.links.FiberLink.budget`).

    Axes: *element-wise* -- an array of powers gives an array of
    amplitudes, which is what a launch-power sweep needs.

    Parameters
    ----------
    power_W : float or np.ndarray
        Channel power in watts, summed over the polarizations.
    polarizations : int, optional, keyword-only
        1 (default) for a single-polarization field, 2 for a
        polarization pair.

    Returns
    -------
    float or np.ndarray
        Amplitude gain to give an ``Amplifier``.

    Raises
    ------
    ValueError
        If a power is negative, or if ``polarizations`` is neither 1
        nor 2.

    Examples
    --------
    >>> print(f"{launch_amplitude(dbm_to_watt(0.0)):.6f}")
    0.031623

    Two polarizations share the channel power, so each carries half of
    it -- 3 dB down, which is :math:`\sqrt{2}` in amplitude:

    >>> single = launch_amplitude(1e-3)
    >>> pair = launch_amplitude(1e-3, polarizations=2)
    >>> print(f"{single / pair:.6f}")
    1.414214
    """
    if polarizations not in (1, 2):
        raise ValueError(
            f"a fibre carries one or two polarizations, got {polarizations}.")
    power = np.asarray(power_W, dtype=float)
    if np.any(power < 0):
        raise ValueError(
            f"a launch power is not negative; got a minimum of "
            f"{float(np.min(power))} W.")
    result = np.sqrt(power / polarizations)
    return result if np.ndim(power_W) else float(result)
