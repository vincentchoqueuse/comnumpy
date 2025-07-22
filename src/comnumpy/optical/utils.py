import numpy as np
from .constants import PLANCK_CONSTANT, OPTICAL_CARRIER_FREQUENCY


def compute_beta2(lamb, cd_coefficient, speed_of_light):
    r"""
    Compute the Chromatic Dispersion coefficient β₂ in ps²/km

    Attributes
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
    * [2] Eghbali, Amir, et al. "Optimal least-squares FIR digital filters for compensation of chromatic dispersion in digital coherent optical receivers." Journal of lightwave technology 32.8 (2014): 1449-1456.

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


def apply_chromatic_dispersion(x, z, beta2, alpha_dB=None, fs=1, direction=1):
    """
    Implements chromatic dispersion effects in optical fiber communications.

    This class models the chromatic dispersion effect in the frequency domain for
    fiber-optic communication systems. It applies a dispersion-induced phase shift
    to the input signal in the frequency domain and considers signal attenuation [1].

    Attributes
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
    * [2] Eghbali, Amir, et al. "Optimal least-squares FIR digital filters for compensation of chromatic dispersion in digital coherent optical receivers." Journal of lightwave technology 32.8 (2014): 1449-1456.

    """
    if alpha_dB:
        alpha = (np.log(10)/10) * alpha_dB  # convert dB to linear factor
        gain = np.exp(-(alpha/2) * z * direction)  # see text before equation 6 in https://arxiv.org/pdf/2010.14258.pdf
    else:
        gain = 1

    beta2_s2_per_km = ((10**-12)**2) * beta2  # convert into s^2/km
    NFFT = len(x)
    w = (2*np.pi*fs)*np.fft.fftfreq(NFFT, d=1)
    H = np.exp(1j * (beta2_s2_per_km/2) * z * (w**2) * direction)  # see equation 4
    fftx = np.fft.fft(x)
    ffty = H * fftx
    y = gain * np.fft.ifft(ffty)
    return y


def apply_kerr_nonlinearity(x, z, gamma, gain=1, direction=1):
    nl_param = direction * gamma * z
    return gain * x * np.exp(1j*nl_param*(np.abs(x)**2))


def compute_erbium_doped_fiber_amplifier_gain(alpha_dB, L_span):
    G = 10**(alpha_dB*L_span/10)
    gain = np.sqrt(G)
    return gain


def compute_erbium_doped_fiber_N_ase(alpha_dB, L_span, NF_dB, h=PLANCK_CONSTANT, nu=OPTICAL_CARRIER_FREQUENCY):
    r"""
    Compute ASENoise params

    Attributes
    ----------
    alpha_dB : float
        Wavelength (nm)
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
    * [1] Essiambre, René-Jean, Gerhard Kramer, Peter J. Winzer, Gerard J. Foschini, and Bernhard Goebel. "Capacity limits of optical fiber networks." Journal of Lightwave technology 28, no. 4 (2010): 662-701.
    
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
    NF = 10**(NF_dB/10)
    n_sp = (NF/2) / (1-1/G)  # see Hager paper after equation 11
    N_ase = (G-1) * h * nu * n_sp   # see code of Hager https://github.com/chaeger/LDBP/blob/master/ldbp/ldbp.py
    return N_ase


def get_linear_step_size(L_span, StPS):
    """
    Linear Step Size
    """
    return (L_span/StPS)*np.ones(StPS)


def get_logarithmic_step_size(L_span, StPS, alpha_dB=0, step_log_factor=0.4):
    """
    Logarithmically spaced step size

    See also:

    References
    ----------
    * [1] O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems,"  in Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    """
    alpha = (np.log(10)/10) * (alpha_dB)
    alpha_adj = step_log_factor*alpha
    delta = (1-np.exp(-alpha_adj*L_span))/StPS
    n_vect = 1 + np.arange(StPS)
    z = -(1/alpha_adj)*np.log((1-n_vect*delta)/(1-(n_vect-1)*delta))
    return z
