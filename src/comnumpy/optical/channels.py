import numpy as np
from dataclasses import dataclass
from comnumpy.core import Processor
from .utils import apply_chromatic_dispersion, apply_kerr_nonlinearity, compute_beta2
from .constants import CD_COEFFICIENT, SPEED_OF_LIGHT, WAVELENGTH, KERR_COEFFICIENT, PLANCK_CONSTANT, OPTICAL_CARRIER_FREQUENCY


@dataclass
class PhaseNoise(Processor):
    r"""
    A class representing a Phase Noise channel.

    This class models a phase noise effect in a channel, where the phase of the signal
    is altered by a random process. The phase noise is characterized by a variance
    specified by sigma squared (sigma2).

    Attributes
    ----------
    sigma2 : float
        The variance of the phase noise.
    name : str
        Name of the channel instance. Default is "phase noise".
    """
    sigma2: float
    name: str = "phase noise"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def noise_rvs(self, X):
        N = len(X)
        sigma2 = self.sigma2
        scale = np.sqrt(sigma2)
        noise = self.rng.normal(loc=0, scale=scale, size=N)
        self._b = np.cumsum(noise)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.rvs(x)
        y = x * np.exp(1j*self._b)
        return y


@dataclass
class ChromaticDispersion(Processor):
    r"""
    Implements chromatic dispersion effects in optical fiber communications.

    This class models the chromatic dispersion effect in the frequency domain for
    fiber-optic communication systems. It applies a dispersion-induced phase shift
    to the input signal in the frequency domain and considers signal attenuation [1].

    Attributes
    ----------
    z : float
        Step length in meters (km).
    fs : float
        Sampling frequency in hertz (Hz). Default is 1 Hz.
    alpha_dB : float, optional
        Attenuation factor in decibels (dB). Default is 0 dB.
    direction : int, optional
        Propagation direction, 1 for forward and -1 for backward. Default is 1.
    name : str, optional
        Identifier for the dispersion instance. Default is "cd".
    lamb : float
        Wavelength of the signal in meters. Default is WAVELENGTH.
    D : float
        Dispersion coefficient. Default is CD_COEFFICIENT.
    c : float
        Speed of light in meters per second. Default is SPEED_OF_LIGHT.

    References
    ----------
    * [1] Shahkarami, Abtin. "Complexity reduction over bi-RNN-based Kerr nonlinearity equalization
      in dual-polarization fiber-optic communications via a CRNN-based approach."
      Dissertation, Institut polytechnique de Paris, 2022.
      URL: https://www.theses.fr/2022IPPAT034.

    Notes
    -----
    The implementation of chromatic dispersion is based on the standard fiber-optic
    communication theory, where the dispersion effect is modeled in the frequency domain
    based on the fiber parameters and the signal's wavelength. The `forward` method
    then applies this dispersion effect to an input signal. Attenuation due to
    fiber loss is also considered if `alpha_dB` is non-zero.
    """
    z: float
    fs: float = 1
    alpha_dB: float = 0
    direction: int = 1
    name: str = "cd"
    lamb: float = WAVELENGTH
    D: float = CD_COEFFICIENT
    c: float = SPEED_OF_LIGHT

    @property
    def beta2(self):
        return compute_beta2(self.lamb, self.D, self.c)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = apply_chromatic_dispersion(x, self.z, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=self.direction)
        return y

@dataclass
class KerrNonLinearity(Processor):
    r"""
    Models the Kerr nonlinearity effect in optical fibers.

    This class simulates the Kerr nonlinearity effect in fiber-optic communication systems.
    Kerr nonlinearity is a phenomenon where the refractive index of the fiber changes
    with the intensity of the light passing through it, leading to phase modulation
    of the signal. The class considers the effective length of the fiber and
    attenuation due to fiber loss [1].

    Attributes
    ----------
    z : float
        Step length in meters (km).
    direction : int, optional
        Propagation direction, 1 for forward and -1 for backward. Defaults to 1.
    name : str, optional
        Identifier for the nonlinearity instance. Defaults to "nl".
    gamma : float
        Kerr coefficient. Default is KERR_COEFFICIENT.
    gain : float
        Gain factor. Default is 1.

    References
    ----------
    * [1] Häger, Christian, and Henry D. Pfister. "Physics-based deep learning for fiber-optic
      communication systems." IEEE Journal on Selected Areas in Communications 39.1 (2020): 280-294.
      URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304

    Notes
    -----
    The Kerr nonlinearity effect is significant in high-power or long-distance fiber-optic
    systems. This implementation considers the nonlinear phase shift induced by the intensity
    of the optical signal. The `forward` method applies the nonlinear phase shift to the
    signal based on the intensity of the input signal and the Kerr coefficient (`gamma`).
    """

    z: float
    direction: int = 1
    name: str = "nl"
    gamma: float = KERR_COEFFICIENT
    gain: float = 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = apply_kerr_nonlinearity(x, self.z, self.gamma, gain=self.gain, direction=self.direction)
        return y


