import numpy as np
from dataclasses import dataclass
from typing import Literal
from comnumpy.core import Processor
from .constants import SPEED_OF_LIGHT, PLANCK_CONSTANT, WAVELENGTH, KERR_COEFFICIENT, FIBER_LOSS, CD_COEFFICIENT, OPTICAL_CARRIER_FREQUENCY
from .utils import compute_beta2, get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain, apply_chromatic_dispersion, apply_kerr_nonlinearity


@dataclass
class DBP(Processor):
    r"""
    A class that implements Digital Back Propagation (DBP).

    This class extends FiberLink and is used for simulating the process of digital
    back propagation in optical fiber communication systems. DBP compensates for
    signal distortions such as chromatic dispersion and Kerr nonlinearity effects.

    Attributes
    ----------
    N_spans : int
        Number of spans in the fiber link.
    L_span : float
        Length of each span in kilometers.
    StPS : int
        Steps per span.
    fs : float
        Sampling frequency in Hz.
    step_type : Literal["linear", "logarithmic"]
        Type of step size ('linear' or 'logarithmic').
    step_method : Literal["symmetric", "asymetric"]
        Method for splitting steps ('symmetric' or 'asymetric').
    use_only_linear : bool
        Flag to consider only linear effects.
    name : str
        Name of the span.
    c : float
        Speed of light in meters per second.
    h : float
        Planck constant in Joule seconds.
    gamma : float
        Kerr coefficient in rad/W/km.
    lamb : float
        Wavelength in nanometers.
    alpha_dB : float
        Fiber loss in dB/km.
    cd_coefficient : float
        Chromatic dispersion coefficient in ps/nm/km.
    nu : float
        Optical carrier frequency.
    step_log_factor : float
        Logarithmic step factor.
    gain : float
        Gain factor calculated based on attenuation coefficient and span length.

    Notes
    -----
    The `DBP` class is used in combination with other classes from the `comnumpy` package
    to simulate an optical communication system. It specifically addresses the digital
    compensation of signal impairments due to fiber transmission.

    References
    ----------
    * [1] O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber communications systems," in Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    """

    N_spans: int = 1
    L_span: float = 80
    StPS: int = 1
    fs: float = 1
    step_type: Literal["linear", "logarithmic"] = "linear"
    step_method: Literal["symmetric", "asymetric"] = "symmetric"
    use_only_linear: bool = False
    c: float = SPEED_OF_LIGHT
    h: float = PLANCK_CONSTANT
    gamma: float = KERR_COEFFICIENT
    lamb: float = WAVELENGTH
    alpha_dB: float = FIBER_LOSS
    cd_coefficient: float = CD_COEFFICIENT
    nu: float = OPTICAL_CARRIER_FREQUENCY
    step_log_factor: float = 0.4
    name: str = "dbp"

    def prepare(self, x: np.ndarray) -> np.ndarray:
        match self.step_type:
            case "linear":
                step_size = get_linear_step_size(self.L_span, self.StPS)
            case "logarithmic":
                step_size = get_logarithmic_step_size(self.L_span, self.StPS, alpha_dB=self.alpha_dB, step_log_factor=self.step_log_factor)
            case _:
                raise NotImplementedError(f"Step type {self.step_type} is not implemented")
 
        edfa_gain = compute_erbium_doped_fiber_amplifier_gain(self.alpha_dB, self.L_span)
        self.beta2 = compute_beta2(self.lamb, self.cd_coefficient, self.c)
        self.gain = 1/edfa_gain
        self.step_size = step_size[::-1]  # reverse order

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x

        for _ in range(self.N_spans):
            y = self.gain * y  # correct for edfa gain
            if self.use_only_linear:
                y = apply_chromatic_dispersion(y, self.L_span, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=-1)
            else:
                for num_step in range(self.StPS):
                    dz = self.step_size[num_step]
                    if self.step_method == "symmetric":
                        y = apply_chromatic_dispersion(y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=-1)
                        y = apply_kerr_nonlinearity(y, dz, self.gamma, direction=-1)
                        y = apply_chromatic_dispersion(y, dz/2, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=-1)

                    if self.step_method == "asymetric":
                        y = apply_chromatic_dispersion(y, dz, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=-1)
                        y = apply_kerr_nonlinearity(y, dz, self.gamma, direction=-1)
                        
        return y
