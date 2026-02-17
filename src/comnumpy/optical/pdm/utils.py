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


# @dataclass
# class DifferentialEncoding(Processor):
#     modulation:str
#     order:int
#     norm:bool = True

#     def get_binary_repr(self):
#         pathname = pathlib.Path("src/comnumpy/core/data")
#         filename = "{}/{}_{}_bin.csv".format(pathname, self.modulation, self.order)
#         data = pd.read_csv(filename, dtype={'bin':str})
#         bin_list = data['bin'].to_numpy().reshape(-1,1)
#         return bin_list
    
#     def diff_encode(self, data_bin):
#         # 1. Define the Gray-coded positions for the first quadrant
#         # bits (b3, b4) -> complex position
#         first_quad_map = {
#         (0,0): 1+1j,
#         (0,1): 1+3j,
#         (1,1): 3+3j,
#         (1,0): 3+1j
#         }

#         # 2. Differential Mapping for (b1, b2) -> Quadrant Shift
#         # Using Gray-like mapping for shifts too: 00=0, 01=1, 11=2, 10=3
#         diff_map = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}
#         symbols = np.zeros(len(data_bin), dtype=complex)
#         state_q = 0 # Current quadrant state
#         for i in range(len(data_bin)):
#             # Get data_bin
#             b_diff = tuple(map(int, data_bin[i][0:2]))

#             b_gray = tuple(map(int, data_bin[i][2:]))
#             # Update quadrant state (Differential)
#             shift = diff_map[b_diff]
#             state_q = (state_q + shift) % 4
            
#             # Get base symbol (Gray)
#             base_sym = first_quad_map[b_gray]
            
#             # Rotate base symbol to the current quadrant
#             # 0 -> *1, 1 -> *j, 2 -> *-1, 3 -> *-j
#             symbols[i] = base_sym * np.exp(1j*state_q*np.pi/2)
#         return symbols

    
#     def forward(self, x):
#         bin_repr = self.get_binary_repr()
#         data_bin = bin_repr[x].ravel()
#         diff_symb = self.diff_encode(data_bin)
#         if self.norm:
#             encoded_symb = diff_symb/np.sqrt(np.mean(np.abs(diff_symb)**2))
#         print(encoded_symb[-5:])
#         return encoded_symb

# @dataclass
# class Differential_Decoding(Processor):
#     modulation: str
#     order: int
#     alphabet : np.ndarray

#     def get_bin_to_int_map(self):
#         pathname = pathlib.Path("src/comnumpy/core/data")
#         filename = "{}/{}_{}_bin.csv".format(pathname, self.modulation, self.order)
#         data = pd.read_csv(filename, dtype={'bin':str})
#         bin_array = data['bin'].to_numpy()
#         int_array = data['s'].to_numpy()
#         bin_to_int_map = {}
#         for i in range(len(bin_array)):
#             bin_to_int_map.update( {bin_array[i]:int_array[:]} )
#         return bin_to_int_map

#     def get_lsb(self, y, q):
#         _,y_projected = hard_projector(y, alphabet=self.alphabet)
#         first_quad_symb_norm = np.array([self.alphabet[9], self.alphabet[8], self.alphabet[12], self.alphabet[13]])
#         first_quad_bits = {
#             first_quad_symb_norm[0]:'00',
#             first_quad_symb_norm[1]:'01',
#             first_quad_symb_norm[2]:'11',
#             first_quad_symb_norm[3]:'10'
#         }
#         print(y_projected)
#         lsb = first_quad_bits[y_projected]
#         return lsb


#     def forward(self, x):
#         bin_to_int_map = self.get_bin_to_int_map()
#         int_symb = np.zeros_like(x)
#         bits_symb = []
#         phase_map = {0:'00', 1:'01', 2:'11', 3:'10'}
#         prev = x[0]
#         theta = 0
#         decoded = np.zeros_like(x)

#         for k in range(1, len(x)):
#             z = x[k] * np.conj(prev)
#             # 1. detect quadrant (differential)
#             angle = (np.angle(z)) % (2*np.pi)
#             q = int(np.round(angle / (np.pi/2))) % 4
#             bin_msb = phase_map[q]
#             phi_hat = q * (np.pi/2)
#             # update cumulative estimated phase
#             theta += phi_hat
#             theta %= (2*np.pi)
#             # 2. remove estimated phase
#             y = x[k] * np.exp(-1j * theta)
#             bin_lsb = self.get_lsb(y, q)
#             bits_symb.append( bin_msb+bin_lsb )
#             decoded[k] = y
#             int_symb[k] = bin_to_int_map[bits_symb[k]]
#             prev = x[k]

#         return int_symb

# @dataclass
# class DifferentialEncoding(Processor):
#     M: int
#     norm:bool = True
#     modulation = 'QAM'
    
#     def get_binary_repr(self):
#         pathname = pathlib.Path("src/comnumpy/core/data")
#         filename = "{}/{}_{}_bin.csv".format(pathname, self.modulation, self.M)
#         data = pd.read_csv(filename, dtype={'bin':str})
#         bin_list = data['bin'].to_numpy().reshape(-1,1)
#         return bin_list
    
#     def get_map(self):
#         harta = { '00':0, '01': np.pi/2, '11': np.pi, '10':3*np.pi/2}
#         return harta
    
#     def forward(self, x:np.ndarray) -> np.ndarray:
#         harta = self.get_map()
#         bin_repr = self.get_binary_repr()
#         data_bin = bin_repr[x].ravel()
#         R = 2*np.sqrt(2)
#         r = np.sqrt(2)
#         Ci_1 = R*np.exp(1j*np.pi/4)
#         Di_1 = r*np.exp(1j*np.pi/4)
#         S0 = Ci_1 + Di_1
#         Es = 2*(self.M-1)/3
#         encoded_symb = []
#         for bits_group in data_bin:
#             phi1 = harta[bits_group[:2]]
#             phi2 = harta[bits_group[2:]]
#             Ci = Ci_1 * np.exp(1j*phi1)
#             Di = Di_1 * np.exp(1j*phi2)
#             Si = Ci + Di
#             encoded_symb.append(Si)
#             Ci_1 = Ci
#             Di_1 = Di

#         y = np.array(encoded_symb) / np.sqrt(Es)
#         return y
    
# @dataclass
# class DifferentialDecoding(DifferentialEncoding):
#     def sgn(self, data):
#         if data > 0:
#             return 1
#         elif data < 0:
#             return -1
#         else:
#             return 0
        
#     def get_dibits(self, val):
#         if val == 0:
#             return '00'
#         elif val == np.pi/2:
#             return '01'
#         elif val == np.pi:
#             return '11'
#         elif val == 1.5*np.pi:
#             return '10'

#     def get_ph(self, prod, R):
#         if np.isclose(prod, R**2, atol=1e-5):
#             return 0
#         elif np.isclose(prod, 1j*R**2, atol=1e-5):
#             return np.pi/2
#         elif np.isclose(prod,-R**2, atol=1e-5):
#             return np.pi
#         elif np.isclose(prod,-1j*R**2, atol=1e-5):
#             return 1.5*np.pi



#     def forward(self, x:np.ndarray)->np.ndarray:
#         Es = 2*(self.M-1)/3
#         x *= np.sqrt(Es)
#         R = 2*np.sqrt(2)
#         r = np.sqrt(2)
#         Ci_1 = R*np.exp(1j*np.pi/4)
#         Di_1 = r*np.exp(1j*np.pi/4)
#         bits = []
#         decoded_symb = []
#         for symb in x:
#             Cpi = R/np.sqrt(2) * ( self.sgn(np.real(symb)) + 1j*self.sgn(np.imag(symb)) )
#             prod_cp = Cpi * np.conj(Ci_1)
#             phi_est1 = self.get_ph(prod_cp, R)
#             bits1 = self.get_dibits(phi_est1)
#             Dpi = r/np.sqrt(2) * ( self.sgn( np.real(symb - Cpi) ) + 1j*self.sgn( np.imag(symb-Cpi) ) )
#             prod_dp = Dpi * np.conj(Di_1)
#             phi_est2 = self.get_ph(prod_dp, r)
#             decod = Cpi + Dpi
#             decoded_symb.append(decod)
#             bits2 = self.get_dibits(phi_est2)
#             bits.append( bits1 + bits2 )
#             Ci_1 = Cpi
#             Di_1 = Dpi

#         x_est = np.array(decoded_symb) / np.sqrt(Es)
#         integers = np.array([int(b, 2) for b in bits])
#         return integers


@dataclass
class DifferentialEncoding(Processor):
    M: int
    norm: bool = True
    modulation = 'QAM'

    def get_binary_repr(self):
        pathname = pathlib.Path("src/comnumpy/core/data")
        filename = f"{pathname}/{self.modulation}_{self.M}_bin.csv"
        data = pd.read_csv(filename, dtype={'bin': str})
        return data['bin'].to_numpy()

    def get_map_array(self):
        # Vectorized mapping for 2-bit strings → angle
        mapping = {
            '00': 0.0,
            '01': np.pi/2,
            '11': np.pi,
            '10': 3*np.pi/2
        }
        keys = np.array(list(mapping.keys()))
        values = np.array(list(mapping.values()))
        return keys, values

    def forward(self, x: np.ndarray) -> np.ndarray:
        bin_repr = self.get_binary_repr()
        data_bin = bin_repr[x]  # shape: (N,)

        keys, values = self.get_map_array()
        map_dict = dict(zip(keys, values))  # needed for np.vectorize


        # Extract phi1 and phi2 for each symbol
        phi1_strs = np.array([b[:2] for b in data_bin])
        phi2_strs = np.array([b[2:] for b in data_bin])
        phi1 = np.vectorize(map_dict.get)(phi1_strs)
        phi2 = np.vectorize(map_dict.get)(phi2_strs)


        R = 2 * np.sqrt(2)
        r = np.sqrt(2)
        Es = 2 * (self.M - 1) / 3

        # Preallocate arrays
        n = len(x)
        Ci = np.zeros(n, dtype=complex)
        Di = np.zeros(n, dtype=complex)
        Si = np.zeros(n, dtype=complex)

        # Initial vectors
        Ci_prev = R * np.exp(1j * np.pi / 4)
        Di_prev = r * np.exp(1j * np.pi / 4)

        # Loop is still needed due to recurrence
        for i in range(n):
            Ci[i] = Ci_prev * np.exp(1j * phi1[i])
            Di[i] = Di_prev * np.exp(1j * phi2[i])
            Si[i] = Ci[i] + Di[i]
            Ci_prev = Ci[i]
            Di_prev = Di[i]

        return Si / np.sqrt(Es)


@dataclass
class DifferentialDecoding(DifferentialEncoding):
    def sgn_arr(self, arr):
        return np.sign(arr).astype(int)
    
    def angle_to_dibit(self, angles):
        # Map rounded angles to dibits using a lookup array
        angle_lookup = {
            0.0: '00',
            np.pi/2: '01',
            np.pi: '11',
            1.5*np.pi: '10'
        }
        return np.vectorize(angle_lookup.get)(angles)
    
    def quantize_angle(self, angles):
        # Round angle to nearest quadrant (0, pi/2, pi, 3pi/2)
        angles = np.angle(angles) % (2*np.pi)
        idx = np.round(angles / (np.pi/2)) % 4
        return idx * (np.pi/2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        Es = 2 * (self.M - 1) / 3
        x_scaled = x * np.sqrt(Es)

        R = 2 * np.sqrt(2)
        r = np.sqrt(2)

        # Initialize memory
        n = len(x)
        Ci_1 = np.full(n, R*np.exp(1j*np.pi/4), dtype=complex)
        Di_1 = np.full(n, r*np.exp(1j*np.pi/4), dtype=complex)

        # Pre-allocate output
        Cpi = np.zeros(n, dtype=complex)
        Dpi = np.zeros(n, dtype=complex)
        decoded = np.zeros(n, dtype=complex)
        phi1_list = np.zeros(n)
        phi2_list = np.zeros(n)

        # Vectorized loop (loop kept, but minimal logic inside)
        for i in range(n):
            symb = x_scaled[i]
            # Estimate quadrant center
            re, im = np.real(symb), np.imag(symb)
            Cpi[i] = R / np.sqrt(2) * (self.sgn_arr(re) + 1j * self.sgn_arr(im))
            prod_cp = Cpi[i] * np.conj(Ci_1[i])
            phi1_list[i] = self.quantize_angle(prod_cp)

            # Estimate displacement
            residual = symb - Cpi[i]
            re_d, im_d = np.real(residual), np.imag(residual)
            Dpi[i] = r / np.sqrt(2) * (self.sgn_arr(re_d) + 1j * self.sgn_arr(im_d))
            prod_dp = Dpi[i] * np.conj(Di_1[i])
            phi2_list[i] = self.quantize_angle(prod_dp)

            decoded[i] = Cpi[i] + Dpi[i]

            if i + 1 < n:
                Ci_1[i+1] = Cpi[i]
                Di_1[i+1] = Dpi[i]

        bits1 = self.angle_to_dibit(phi1_list)
        bits2 = self.angle_to_dibit(phi2_list)
        bit_strings = np.char.add(bits1, bits2)

        integers = np.array([int(b, 2) for b in bit_strings])
        return integers


@dataclass
class MCMA_(Processor):
    alphabet: np.ndarray
    L:int
    mu: float
    os:int = 2
    name:str="MCMA"

    def __post_init__(self):
        self.RpR,self.RpI = self.compute_mcma_radii()

    def compute_mcma_radii(self):
        aR = np.real(self.alphabet)
        aI = np.imag(self.alphabet)

        RpR = np.mean(aR**4) / np.mean(aR**2)
        RpI = np.mean(aI**4) / np.mean(aI**2)
        return RpR, RpI


    def reset(self):
        self.h11 = np.zeros(self.L, dtype=complex)
        self.h12 = np.zeros(self.L, dtype=complex)
        self.h21 = np.zeros(self.L, dtype=complex)
        self.h22 = np.zeros(self.L, dtype=complex)
        self.h11[0] = 1
        self.h22[0] = 1
        return self.h11,self.h12,self.h21,self.h22

    def grad(self, x, y):
        # y: shape (2,)  -> [y1, y2]
        # x: shape (2, L)

        y1, y2 = y
        x1, x2 = x

        e1 = (
            np.real(y1) * (np.real(y1)**2 - self.RpR)
            + 1j * np.imag(y1) * (np.imag(y1)**2 - self.RpI)
        )

        e2 = (
            np.real(y2) * (np.real(y2)**2 - self.RpR)
            + 1j * np.imag(y2) * (np.imag(y2)**2 - self.RpI)
        )

        grad = np.zeros((4, self.L), dtype=complex)

        grad[0] = e1 * np.conj(x1)  # h11
        grad[1] = e1 * np.conj(x2)  # h12
        grad[2] = e2 * np.conj(x1)  # h21
        grad[3] = e2 * np.conj(x2)  # h22

        return grad


    def forward(self, X):
        #print("PDL MCMA:",10*np.log10(np.mean(np.abs(X[0,:])**2)/np.mean(np.abs(X[1,:])**2)))
        Y = np.zeros_like(X)
        N = X.shape[1]
        self.h11,self.h12,self.h21,self.h22 = self.reset()
        for n in range(self.L + 1, N):
            # if n < self.os * 500_000:
            #     mu = self.mu * 5
            # else:
            mu = self.mu
            input = X[:, n : n - self.L : -1] # X[:, n-self.L+1:n+1][:, ::-1]
            x_1 = input[0, :]
            x_2 = input[1, :]
            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])
            if (n % self.os) == 0:
                grad = self.grad(input, output)
                self.h11 = self.h11 - mu * grad[0, :]
                self.h22 = self.h22 - mu * grad[3, :]
                self.h12 = self.h12 - mu * grad[1, :]
                self.h21 = self.h21 - mu * grad[2, :]

            Y[:, n] = output
        self.Y = Y
        return Y

    def get_data(self):
        return self.Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    


@dataclass
class MCMA_SoftContinuation(Processor):
    """
    MCMA → soft-decision MCMA continuation

    - Pure MCMA before `switch`
    - After `switch`, blends MCMA gradient with soft-decision gradient
    """

    alphabet: np.ndarray
    L: int
    mu1: float
    mu2: float
    switch: int
    alpha: float = 0.1        # soft-decision weight
    sigma2: float = 0.5       # softness (≈ noise variance)
    os: int = 2
    name: str = "MCMA_Soft"

    def __post_init__(self):
        self.RpR, self.RpI = self.compute_mcma_radii()

    # -------------------------------------------------
    # MCMA radii
    # -------------------------------------------------
    def compute_mcma_radii(self):
        aR = np.real(self.alphabet)
        aI = np.imag(self.alphabet)
        RpR = np.mean(aR**4) / np.mean(aR**2)
        RpI = np.mean(aI**4) / np.mean(aI**2)
        return RpR, RpI

    # -------------------------------------------------
    # Reset taps
    # -------------------------------------------------
    def reset(self):
        self.h11 = np.zeros(self.L, dtype=complex)
        self.h12 = np.zeros(self.L, dtype=complex)
        self.h21 = np.zeros(self.L, dtype=complex)
        self.h22 = np.zeros(self.L, dtype=complex)
        self.h11[0] = 1
        self.h22[0] = 1
        return self.h11, self.h12, self.h21, self.h22

    # -------------------------------------------------
    # Soft symbol estimator
    # -------------------------------------------------
    def soft_symbol(self, y):
        # y shape: (N,)
        d2 = np.abs(y[:, None] - self.alphabet[None, :])**2
        w = np.exp(-d2 / self.sigma2)
        w /= np.sum(w, axis=1, keepdims=True)
        return w @ self.alphabet

    # -------------------------------------------------
    # Gradient
    # -------------------------------------------------
    def grad(self, x, y, use_soft=False):
        y1, y2 = y
        x1, x2 = x

        # --- MCMA error ---
        e1_mcma = (
            np.real(y1) * (np.real(y1)**2 - self.RpR)
            + 1j * np.imag(y1) * (np.imag(y1)**2 - self.RpI)
        )
        e2_mcma = (
            np.real(y2) * (np.real(y2)**2 - self.RpR)
            + 1j * np.imag(y2) * (np.imag(y2)**2 - self.RpI)
        )

        if not use_soft:
            e1, e2 = e1_mcma, e2_mcma
        else:
            # --- Soft-decision error ---
            s1 = self.soft_symbol(np.array([y1]))[0]
            s2 = self.soft_symbol(np.array([y2]))[0]

            e1_sd = y1 - s1
            e2_sd = y2 - s2

            # --- Blend ---
            e1 = self.alpha * e1_mcma + (1-self.alpha) * e1_sd
            e2 = self.alpha * e2_mcma + (1-self.alpha) * e2_sd

        grad = np.zeros((4, self.L), dtype=complex)
        grad[0] = e1 * np.conj(x1)
        grad[1] = e1 * np.conj(x2)
        grad[2] = e2 * np.conj(x1)
        grad[3] = e2 * np.conj(x2)

        return grad

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, X):
        Y = np.zeros_like(X)
        N = X.shape[1]
        self.h11, self.h12, self.h21, self.h22 = self.reset()

        for n in range(self.L + 1, N):
            mu = self.mu1 if n < self.switch else self.mu2

            input = X[:, n : n - self.L : -1]
            x1, x2 = input[0], input[1]

            y1 = np.dot(self.h11, x1) + np.dot(self.h12, x2)
            y2 = np.dot(self.h21, x1) + np.dot(self.h22, x2)
            output = np.array([y1, y2])

            if (n % self.os) == 0:
                use_soft = n <= self.switch
                grad = self.grad(input, output, use_soft=use_soft)

                self.h11 -= mu * grad[0]
                self.h12 -= mu * grad[1]
                self.h21 -= mu * grad[2]
                self.h22 -= mu * grad[3]

            Y[:, n] = output

        self.Y = Y
        return Y

    def get_data(self):
        return self.Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


from dataclasses import dataclass
import numpy as np

from dataclasses import dataclass
import numpy as np

@dataclass
class GenieAidedPolResolverInteger:
    """
    Genie-aided polarization ambiguity resolution
    operating on integer symbol indices.

    This is evaluation-only.
    Applied AFTER differential decoding.
    """

    name: str = "GenieAidedPolResolverInteger"

    def resolve(self, D_hat: np.ndarray, D_true: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        D_hat : ndarray (2, N)
            Detected integer symbols (after differential decoding)

        D_true : ndarray (2, N)
            True integer symbols (after differential decoding)

        Returns
        -------
        D_best : ndarray (2, N)
            Ambiguity-resolved symbols
        """

        if D_hat.shape != D_true.shape:
            raise ValueError("Shape mismatch between detected and true symbols.")

        # Candidate 1: no swap
        cand_identity = D_hat

        # Candidate 2: polarization swap
        cand_swap = D_hat[::-1]

        # Compute SER for both
        err_identity = np.mean(cand_identity != D_true)
        err_swap = np.mean(cand_swap != D_true)

        if err_identity <= err_swap:
            return cand_identity
        else:
            return cand_swap


class PolarizationPowerNormalizer(Processor):
    """
    Simple per-polarization power normalization.

    Ensures each polarization has unit average power:
        y_i <- y_i / sqrt(E[|y_i|^2])

    This is NOT whitening.
    This is a scalar normalization per polarization.
    """

    eps: float = 1e-12
    name: str = "PolPowerNorm"


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : ndarray (2, N)
            Equalized complex symbols

        Returns
        -------
        X_norm : ndarray (2, N)
            Power-normalized symbols
        """

        if X.shape[0] != 2:
            raise ValueError("Expected dual-polarization signal (2, N)")

        X_norm = X.copy()

        for pol in range(2):
            power = np.mean(np.abs(X[pol])**2)
            gain = np.sqrt(power + self.eps)
            X_norm[pol] /= gain

        return X_norm