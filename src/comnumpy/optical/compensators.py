import numpy as np
import numpy.linalg as LA
from comnumpy.core.generics import Processor
from scipy import signal
from scipy.linalg import toeplitz
from scipy import special
from dataclasses import dataclass, field
from typing import ClassVar, List
from .utils import compute_beta2
from .constants import WAVELENGTH, CD_COEFFICIENT, SPEED_OF_LIGHT


@dataclass
class ChromaticDispersionFIRCompensator(Processor):
    """
    FIR-based chromatic dispersion compensator using the Savory method.

    Attributes
    ----------
    z : float
        Fiber length in km.
    fs : float
        Sampling frequency in Hz.
    name : str
        Name of the processor.
    """
    lamb: ClassVar[float] = WAVELENGTH
    D: ClassVar[float] = CD_COEFFICIENT
    c: ClassVar[float] = SPEED_OF_LIGHT

    z: float  # in km
    fs: float = 1.0
    name: str = "fir cd compensator"
    h: np.ndarray = field(init=False)
    K: float = field(init=False)

    def __post_init__(self):
        beta2_ps2_per_km = compute_beta2(self.lamb, self.D, self.c)
        beta2 = ((10**-12)**2)*beta2_ps2_per_km  # convert into s^2/km
        K = - beta2 * self.z * (self.fs**2) / 2
        N = int(2 * np.floor(2 * K * np.pi) + 1)
        bound = int(np.floor(N / 2))
        n_vect = np.arange(-bound, bound + 1)
        coef = np.sqrt(1j / (4 * K * np.pi))
        self.h = coef * np.exp(-1j * (n_vect**2) / (4 * K))
        self.K = K

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = signal.convolve(x, self.h, mode='full')
        return y


@dataclass
class ChromaticDispersionLSFIRCompensator(Processor):
    """
    Least-Squares FIR Compensator for Chromatic Dispersion.

    Attributes
    ----------
    z : float
        Fiber length (km).
    N : int
        Filter length (must be odd).
    fs : float
        Sampling rate (arbitrary units).
    w_vect : List[float]
        Frequency band [Ω₁, Ω₂] for the least-squares design.
    name : str
        Name identifier for the block.

    References
    ----------
    * [1] Optimal Least-Squares FIR Digital Filters for Compensation of Chromatic Dispersion
    """
    lamb: ClassVar[float] = WAVELENGTH
    D: ClassVar[float] = CD_COEFFICIENT
    c: ClassVar[float] = SPEED_OF_LIGHT
    epsilon: ClassVar[float] = 1e-14

    z: float
    N: int
    fs: float = 1.0
    w_vect: List[float] = field(default_factory=lambda: [-np.pi, np.pi])
    name: str = "optimal"
    h: np.ndarray = field(init=False)
    K: float = field(init=False)

    def __post_init__(self):
        if self.N % 2 == 0:
            raise ValueError(f"The value of N must be odd (current={self.N})")

        beta2_ps2_per_km = compute_beta2(self.lamb, self.D, self.c)
        beta2 = ((10**-12)**2)*beta2_ps2_per_km  # convert into s^2/km
        K = -beta2 * self.z * (self.fs**2) / 2
        Omega_1, Omega_2 = self.w_vect

        # Construct Matrix Q
        q_row = np.zeros(self.N, dtype=complex)
        q_col = np.zeros(self.N, dtype=complex)
        q_row[0] = q_col[0] = (Omega_2 - Omega_1) / (2 * np.pi)
        for m in range(1, self.N):
            coef = 1 / (2j * np.pi * m)
            q_row[m] = coef * (np.exp(-1j * m * Omega_1) - np.exp(-1j * m * Omega_2))
        Q = toeplitz(q_col, q_row)

        # Construct vector d
        bound = self.N // 2
        n_vect = np.arange(-bound, bound + 1)
        coef1 = 1 / (4 * np.sqrt(np.pi * K))
        coef2 = np.exp(1j * 3 * np.pi / 4) / (2 * np.sqrt(K))
        d_vect = np.zeros(len(n_vect), dtype=complex)

        for idx, n in enumerate(n_vect):
            term1 = coef2 * (2 * K * np.pi - n)
            term2 = coef2 * (2 * K * np.pi + n)
            erf_term = special.erf(term1) + special.erf(term2)
            phase = np.exp(-1j * (n**2 / (4 * K) + 3 * np.pi / 4))
            d_vect[idx] = coef1 * phase * erf_term

        I_mat = np.eye(self.N)
        Q_inv = LA.inv(Q + self.epsilon * I_mat)
        self.h = Q_inv @ d_vect
        self.K = K

    def forward(self, x: np.ndarray) -> np.ndarray:
        return signal.convolve(x, self.h)
