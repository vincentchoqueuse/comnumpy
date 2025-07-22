import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Dict
from comnumpy.core import Processor
from .devices import ErbiumDopedFiberAmplifier
from .constants import SPEED_OF_LIGHT, PLANCK_CONSTANT, WAVELENGTH, KERR_COEFFICIENT, FIBER_LOSS, CD_COEFFICIENT, OPTICAL_CARRIER_FREQUENCY
from .utils import compute_beta2, get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain, compute_erbium_doped_fiber_N_ase, apply_chromatic_dispersion, apply_kerr_nonlinearity

@dataclass
class FiberLink(Processor):
    """
    Represents a multi-span fiber link for optical communication systems. Each span
    in the link includes both linear (Chromatic Dispersion) and nonlinear (Kerr
    Nonlinearity) effects.

    The model simulates the propagation of light through the fiber spans, taking into
    account the dispersion, attenuation, and nonlinear effects based on the specified
    parameters. It uses the split-step Fourier method as a numerical solution.

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
    NF_dB : float
        Noise figure in dB.
    noise_scaling : float
        Scaling factor for noise.
    step_type : str
        Type of step size ('linear' or 'logarithmic').
    step_method : str
        Method for splitting steps ('symmetric' or 'asymetric').
    name : str
        Name of the span.
    use_only_linear : bool
        Flag to consider only linear effects.
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

    References
    ----------
    * [1] J. Shao, X. Liang and S. Kumar, "Comparison of Split-Step Fourier Schemes for Simulating Fiber Optic Communication Systems," in IEEE Photonics Journal, vol. 6, no. 4, pp. 1-15, Aug. 2014, Art no. 7200515, doi: 10.1109/JPHOT.2014.2340993.
    """
    N_spans: int = 1
    L_span: float = 80
    StPS: int = 1
    fs: float = 1
    NF_dB: float = 4
    noise_scaling: float = 1
    step_type: Literal["linear", "logarithmic"] = "linear"
    step_method: Literal["symmetric", "asymetric"] = "symmetric"
    use_only_linear: bool = False
    c: float = SPEED_OF_LIGHT              # in meters per second
    h: float = PLANCK_CONSTANT             # in Joule seconds
    gamma: float = KERR_COEFFICIENT        # in rad/W/km
    lamb: float = WAVELENGTH               # nm
    alpha_dB: float = FIBER_LOSS           # in dB/km
    cd_coefficient: float = CD_COEFFICIENT  # in ps/nm/km
    nu: float = OPTICAL_CARRIER_FREQUENCY  # optical carrier frequency 
    step_log_factor: float = 0.4
    name: str = "fiber link"
    callbacks: Optional[Dict[str, Callable[[np.ndarray], None]]] = field(default_factory=dict)

    def prepare(self, x: np.ndarray) -> np.ndarray:
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

