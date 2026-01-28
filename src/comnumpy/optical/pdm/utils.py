from dataclasses import dataclass
import numpy as np
from scipy.stats import maxwell
import pathlib
import pandas as pd

from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector




def build_pmd_segments(L_km: float, D_pmd_ps_sqrt_km: float, N_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Build PMD segment parameters with Maxwellian-distributed DGDs.
    
    Generates differential group delays (DGDs) drawn from a Maxwellian
    distribution, with total rms DGD matching tau_rms_link = D_pmd * sqrt(L_km).
    
    Parameters
    ----------
    L_km : float
        Link length in kilometers
    D_pmd_ps_sqrt_km : float
        PMD parameter in ps/sqrt(km)
    N_segments : int
        Number of fiber segments
    
    Returns
    -------
    tau_k : np.ndarray
        Array of DGDs per segment (ps)
    theta_k : np.ndarray
        Array of PSP angles per segment (rad)
    """
    
    # total rms DGD of the whole link
    tau_rms_link = D_pmd_ps_sqrt_km * np.sqrt(L_km)  # [ps]

    # rms DGD per segment
    tau_rms_seg = tau_rms_link / np.sqrt(N_segments)

    # Maxwell scale parameter a
    a = tau_rms_seg / np.sqrt(3.0)

    # draw DGDs from Maxwell(a)
    tau_k = maxwell.rvs(scale=a, size=N_segments)    # [ps]

    # PSP angles, uniform in [0, pi)
    theta_k = np.random.uniform(0.0, np.pi, size=N_segments)

    return tau_k, theta_k


@dataclass
class DifferentialEncoding(Processor):
    modulation:str
    order:int
    norm:bool = True

    def get_binary_repr(self):
        pathname = pathlib.Path("src/comnumpy/core/data")
        filename = "{}/{}_{}_bin.csv".format(pathname, self.modulation, self.order)
        data = pd.read_csv(filename, dtype={'bin':str})
        bin_list = data['bin'].to_numpy().reshape(-1,1)
        return bin_list
    
    def diff_encode(self, data_bin):
        # 1. Define the Gray-coded positions for the first quadrant
        # bits (b3, b4) -> complex position
        first_quad_map = {
        (0,0): 1+1j,
        (0,1): 1+3j,
        (1,1): 3+3j,
        (1,0): 3+1j
        }

        # 2. Differential Mapping for (b1, b2) -> Quadrant Shift
        # Using Gray-like mapping for shifts too: 00=0, 01=1, 11=2, 10=3
        diff_map = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}
        symbols = np.zeros(len(data_bin), dtype=complex)
        state_q = 0 # Current quadrant state
        for i in range(len(data_bin)):
            # Get data_bin
            b_diff = tuple(map(int, data_bin[i][0:2]))

            b_gray = tuple(map(int, data_bin[i][2:]))
            # Update quadrant state (Differential)
            shift = diff_map[b_diff]
            state_q = (state_q + shift) % 4
            
            # Get base symbol (Gray)
            base_sym = first_quad_map[b_gray]
            
            # Rotate base symbol to the current quadrant
            # 0 -> *1, 1 -> *j, 2 -> *-1, 3 -> *-j
            symbols[i] = base_sym * np.exp(1j*state_q*np.pi/2)
        return symbols

    
    def forward(self, x):
        bin_repr = self.get_binary_repr()
        data_bin = bin_repr[x].ravel()
        diff_symb = self.diff_encode(data_bin)
        if self.norm:
            encoded_symb = diff_symb/np.sqrt(np.mean(np.abs(diff_symb)**2))
        return encoded_symb

@dataclass
class Differential_Decoding(Processor):
    modulation: str
    order: int

    def forward(self, x):
        x *= np.exp(1j*2.5*np.pi/2)
        prev = x[0]
        theta = 0
        decoded = np.zeros_like(x)

        for k in range(1, len(x)):
            z = x[k] * np.conj(prev)

            # 1. detect quadrant (differential)
            angle = np.angle(z)
            q = int(np.round(angle / (np.pi/2))) % 4
            phi_hat = q * (np.pi/2)

            # update cumulative estimated phase
            theta += phi_hat

            # 2. remove estimated phase
            y = x[k] * np.exp(-1j * theta)

            decoded[k] = y
            print(x[k], decoded[k], sep='\n')
            prev = x[k]

        return decoded
