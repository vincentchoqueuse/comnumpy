import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
import subprocess
import os
from comnumpy.core import Processor
from .constants import SPEED_OF_LIGHT, WAVELENGTH


@dataclass
class Laser(Processor):
    r"""
    A class representing a laser.

    This object modelizes the envelope of a laser signal
 
    Parameters
    ----------
    P_dBm : float
        The laser power [dBm] 
    linewidth : float
        The laser linewidth [Hz]
    theta0: float 
        The initial phase [rad]
    seed : int
        The seed for theta0 and the random phase noise
    fs : float
        The sample frequency
    freq_offset : float
        The optical-carrier frequency offset [Hz] which corresponds to the relative offset from the laser wavelength simulation (laser TX)
    nb_samples : int 
        The number of desired samples

    """
    P_dBm: float = -10
    linewidth: float = 0
    theta0: float = None
    seed: int = None
    fs: float = 1e9
    freq_offset: float = 0
    name: str = "Laser"
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        
    def forward(self, nb_samples):
        if not isinstance(self.theta0, (float, int)):
            theta_0 = 2*np.pi*self.rng.standard_normal() - np.pi
        else:
            theta_0 = self.theta0

        dtheta = np.sqrt(2*np.pi*self.linewidth/self.fs)*self.rng.standard_normal(nb_samples)
        theta = np.cumsum(dtheta) 
        Ps = 10**((self.P_dBm-30)/10)
        T_s = 1/self.fs
        t = np.arange(nb_samples)*T_s
        E_laser = np.sqrt(Ps)*np.exp(1j*(2*np.pi*self.freq_offset*t+theta_0+theta))
        return E_laser


@dataclass
class Optical90HybridCircuit(Processor):
    """
    Models an optical 90-degree hybrid circuit for coherent detection in optical communication systems.

    This class represents an optical 90° hybrid circuit used for coherent detection, simulating the 
    Optic-Electro conversion process with an option for ideal or non-ideal operation. In an ideal 
    scenario, the circuit directly converts optical signals to electrical signals without modification. 
    In a non-ideal scenario, the conversion process accounts for non-linearities and sensitivity factors.

    Attributes
    ----------
    is_ideal : bool
        If True, represents an ideal Electro-Optic conversion. If False, includes non-idealities 
        such as sensitivity variations and laser defaults.
    sensitivity : float
        Sensitivity factor of the circuit, affecting the Electro-Optic conversion in non-ideal mode. 
        Represents the half-wave voltage in volts (V).
    laser_in : Objet of Laser class         
        The local laser used for coherent detection  

    Notes
    -----
    
    The optical 90-degree hybrid circuit is a key component in coherent optical communication systems, 
    allowing for the mixing of the signal with a local oscillator in a coherent receiver. The 'forward' 
    method of this class models the behavior of the circuit under different operating conditions, 
    simulating the impact of circuit sensitivity and ideal versus non-ideal conversion scenarios on the 
    signal processing.
    """
    is_ideal: bool = True
    sensitivity: float = 0.6
    laser_in: Optional[Laser] = None
    name: str = "Optical90HybridCircuit"

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.is_ideal:
            y = x
        else:
            if self.laser_in:
                E_laser = self.laser_in(len(x))
            else:
                E_laser = np.ones(len(x))

            y = self.sensitivity * x * np.conj(E_laser)
            y = np.real(y) + 1j*np.imag(y)
        return y


@dataclass
class PowerControl(Processor):
    """
    Implements a simple power control mechanism for signal processing.

    This class is designed to adjust the power level of a signal to a specified average power. 
    It can operate in two modes: 'natural' and 'dBm'. In 'natural' mode, the average power is 
    adjusted to a specified linear scale value. In 'dBm' mode, the power is adjusted to a specified 
    value in decibel-milliwatts (dBm), commonly used in telecommunications to express power levels.

    Attributes
    ----------
    P_moy : float
        Target average power level. The interpretation of this value depends on the 'Unit' parameter.
    unit : str, optional
        The unit in which 'P_moy' is specified. Can be 'natural' for linear scale or 'dBm' for 
        decibel-milliwatts. Defaults to 'natural'.

    Notes
    -----
    The power control is a fundamental aspect in various signal processing applications, especially 
    in communication systems where maintaining a specific power level is crucial for effective signal 
    transmission and reception. The 'forward' method is key in this process, allowing for dynamic 
    adjustment of signal power according to the specified target level and unit.
    """
    P_moy: float = 1
    unit: Literal["natural", "dBm"] = "natural"
    name: str = "power_control"

    def get_gain(self, P_moy_x):
        if self.unit == 'dBm':
            gain = np.sqrt(10**((self.P_moy-30)/10))/P_moy_x
        else:
            gain = self.P_moy/P_moy_x
        return gain

    def forward(self, x: np.ndarray) -> np.ndarray:
        P_moy_x = np.sqrt(np.mean(np.abs(x)**2))
        gain = self.get_gain(P_moy_x)
        y = gain*x
        return y


@dataclass
class ErbiumDopedFiberAmplifier(Processor):
    """
    Models an Erbium-Doped Fiber Amplifier (ErbiumDopedFiberAmplifier) in optical communication systems.

    This class simulates the operation of an ErbiumDopedFiberAmplifier, which is used to amplify optical signals 
    in fiber-optic communication systems. It applies a gain to the input signal to compensate 
    for the loss incurred during transmission through optical fibers. The gain is calculated 
    based on the fiber loss parameter and the span length of the fiber.

    Attributes
    ----------
    name : str, optional
        Identifier for the ErbiumDopedFiberAmplifier instance. Defaults to "ErbiumDopedFiberAmplifier".
    N_ase : float
        Noise spectral density per state of polarization
    seed : int
        Seed for the noise generator (default to None)

    References
    ----------
    * [1] https://www.sciencedirect.com/topics/engineering/spontaneous-emission-factor

    Notes
    -----
    ErbiumDopedFiberAmplifiers are crucial in long-haul fiber-optic communication systems to boost the signal strength 
    and maintain signal quality over long distances.
    """
    gain: float
    N_ase: float
    name: str = "ErbiumDopedFiberAmplifier"
    seed: int = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        scale = np.sqrt(self.N_ase/2)
        b = self.rng.normal(scale=scale, size=N) + 1j * self.rng.normal(scale=scale, size=N)
        y = self.gain * x + b
        return y


@dataclass
class MachZehnderModulator(Processor):
    """
    Mach-Zehnder Modulator (MZM) with IQ modulation.

    Parameters
    ----------
    is_ideal : bool
        If True, performs ideal linear modulation without imbalance.
    Vpi : float
        Half-wave voltage (Vπ).
    k : float
        Peak-to-peak voltage ratio relative to Vpi.
    gI : float
        Gain coefficient on the I branch.
    Phi : float
        Phase offset between I and Q branches (radians).
    laser_in : Optional[callable]
        Input laser field generator function or None.
    name : str
        Instance name.


    References
    ----------
    * [1] C. Peucheret, "Generation and Detection of Optical Modulation Format," 2012.

    Notes
    -----
    The modulator functions by varying the intensity of the optical field based on the input electrical 
    signals. The 'forward' method normalizes the input signal and applies the modulation process, 
    considering the modulator's configuration parameters. In the non-ideal mode, the modulation also 
    accounts for the gain imbalance and voltage variations in the in-phase and quadrature branches.
    """

    is_ideal: bool = False
    Vpi: float = 6
    k: float = 1
    gI: float = 1
    Phi: float = np.pi / 2
    laser_in: Optional[callable] = None
    name: str = "MachZehnderModulator"

    Vpp: float = field(init=False)
    VdcI: float = field(init=False)
    VdcQ: float = field(init=False)

    def __post_init__(self):
        self.Vpp = self.k * self.Vpi
        self.VdcI = self.Vpi
        self.VdcQ = self.Vpi

    def forward(self, x: np.ndarray) -> np.ndarray:
        m = max(np.max(np.abs(np.real(x))), np.max(np.abs(np.imag(x))))  # normalization coeff
        
        if self.laser_in is not None:
            E_laser = self.laser_in(len(x))
        else:
            E_laser = np.ones(len(x))  # ideal laser

        if self.is_ideal:
            y = -np.pi / 2 * self.Vpp / (2 * self.Vpi) * E_laser * x
        else:
            gQ = 1 - (self.gI - 1)  # Gain on Q branch
            uIt = self.VdcI + self.gI / 2 * self.Vpp / m * np.real(x)
            uQt = self.VdcQ + gQ / 2 * self.Vpp / m * np.imag(x)
            y = (np.cos(np.pi / 2 / self.Vpi * uIt) + np.exp(1j * self.Phi) * np.cos(np.pi / 2 / self.Vpi * uQt)) * E_laser
        return y
