import numpy as np
from comnumpy._backend import fft, ifft, fftfreq  # cupy-compatible (D3)
from typing import Optional

from .constants import PLANCK_CONSTANT, OPTICAL_CARRIER_FREQUENCY


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
    NFFT = len(x)
    w = (2*np.pi*fs)*fftfreq(NFFT, d=1, like=x)
    H = np.exp(1j * (beta2_s2_per_km/2) * z * (w**2) * direction)  # see equation 4
    fftx = fft(x)
    ffty = H * fftx
    y = gain * ifft(ffty)
    return y


def apply_kerr_nonlinearity(x: np.ndarray, z: float, gamma: float,
                            gain: float = 1, direction: int = 1) -> np.ndarray:
    """
    Apply Kerr nonlinearity phase rotation to a signal.

    Parameters
    ----------
    x : np.ndarray
        Input complex signal.
    z : float
        Fiber length in km.
    gamma : float
        Kerr coefficient in rad/W/km.
    gain : float, optional
        Gain factor. Default is 1.
    direction : int, optional
        Propagation direction (1=forward, -1=backward). Default is 1.

    Returns
    -------
    np.ndarray
        Signal after Kerr nonlinear phase rotation.
    """
    nl_param = direction * gamma * z
    return gain * x * np.exp(1j*nl_param*(np.abs(x)**2))


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
