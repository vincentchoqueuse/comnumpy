import numpy as np
import itertools
import numpy.linalg as LA
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector, soft_projector, zf_estimator, mmse_estimator


def validate_H(H):
    if H is None:
        raise ValueError("Channel H is not set.")
    elif not isinstance(H, np.ndarray):
        raise TypeError("Channel H must be a NumPy array.")


def validate_sigma2(sigma2):
    if sigma2 is None:
        raise ValueError("Noise variance sigma2 is not set.")
    elif sigma2 < 0:
        raise TypeError("Noise variance sigma2 should be greater than 0.")


@dataclass
class MaximumLikelihoodDetector(Processor):
    r"""
    Implements the ML (Maximum Likelihood) Detector for white Gaussian noise in a MIMO (Multiple-Input, Multiple-Output) communication system.

    Algorithm
    ---------

    The ML Detector is designed to estimate the transmitted signal from the received signal in the presence of noise, based on the Maximum Likelihood criterion.

    .. math ::

        \widehat{\mathbf{x}}[n] = \arg \min_{\mathbf{x}\in \mathcal{M}^{N_t}}\|\mathbf{y}[n] - \mathbf{H}\mathbf{x}\|^2

    .. WARNING::

        The ML detector can be computationally expensive for large number of transmit antennas or high order constellation.

    Attributes
    ----------
    alphabet : numpy.ndarray
        Internal storage for the symbol constellation.
    H : numpy.ndarray
        Internal storage for the channel matrix.
    name : str
        Name of the detector.

    References
    ----------
    * Larsson, Erik G., Petre Stoica, and Girish Ganesan. Space-time block coding for wireless communications. Cambridge university press, 2003.
    """
    
    alphabet: np.ndarray
    H: Optional[np.ndarray] = None
    is_mimo: bool = True
    name: str = "ML Detector"

    def get_nb_candidates(self):
        _, N_t = self.H.shape
        return len(self.alphabet) ** N_t
    
    def get_candidates(self, alphabet, N_t):
        symbols = np.arange(len(alphabet))
        input_list = [p for p in itertools.product(symbols, repeat=N_t)]

        # preallocation of memory
        X = np.zeros((N_t, len(input_list)), dtype=complex)
        S = np.zeros((N_t, len(input_list)))

        for indice in range(len(input_list)):
            input = np.array(input_list[indice])
            x = self.alphabet[input]
            X[:, indice] = x
            S[:, indice] = input

        return S, X

    def forward(self, Y):

        validate_H(self.H)

        H = self.H
        _, N_t = H.shape
        _, N = Y.shape
        S = np.zeros((N_t, N), dtype=int)
        alphabet = self.alphabet

        S_candidates, X_candidates = self.get_candidates(alphabet, N_t)
        Y_candidates = np.matmul(
            H, X_candidates
        )  # compute all combinaison of received data

        for n in range(N):
            y = np.transpose(np.atleast_2d(Y[:, n]))
            index_min = np.argmin(np.sum(np.abs(y - Y_candidates) ** 2, axis=0))
            S[:, n] = S_candidates[:, index_min]

        self.S = S
        return S


@dataclass
class LinearDetector(Processor):
    r"""
    Implements a Linear MIMO detector.

    Algorithm
    ---------

    A linear detector is a two-step detector

    1. Channel Equalization (using zero forcing or MMSE equalization)
    2. Symbol Detection

    Attributes
    ----------
    alphabet : numpy.ndarray
        Symbol constellation.
    H : numpy.ndarray
        Internal storage for the channel matrix.
    method: zf or mmse
        linear estimation technique (default: zf)
    sigma2: float
        noise variance (used for the mmse equalizer only)
    name : str
        Name of the detector.

    References
    ----------
    * Larsson, Erik G., Petre Stoica, and Girish Ganesan. Space-time block coding for wireless communications. Cambridge university press, 2003.
    """
    alphabet: np.ndarray
    H: Optional[np.ndarray] = None
    method: Literal["zf", "mmse"] = "zf"
    sigma2: float = None
    is_mimo: bool = True
    name: str = "ZF Detector"

    def linear_estimator(self, Y):
        r"""
        Perform Zero Forcing or MMSE linear equalization 
        """
        match self.method:
            case "zf":
                output = zf_estimator(Y, self.H)
            case "mmse":
                output = mmse_estimator(Y, self.H, self.sigma2 )
        return output

    def forward(self, Y):
        validate_H(self.H)
        Z = self.linear_estimator(Y)
        S, _ = hard_projector(Z, self.alphabet)
        return S


@dataclass
class OrderedSuccessiveInterferenceCancellationDetector(Processor):
    """
    Ordered Successive Interference Cancellation (OSIC) detector.

    Parameters
    ----------
    alphabet : np.ndarray
        Modulation alphabet (e.g. 16-QAM points)
    osic_type : str
        Ordering strategy: 'sinr', 'colnorm', or 'snr'
    H : Optional[np.ndarray]
        Channel matrix (NR x NT)
    sigma2 : Optional[float]
        Noise variance
    name : str
        Component name

    Reference
    ---------
    * Cho, Yong Soo, et al. MIMO-OFDM wireless communications with MATLAB. John Wiley & Sons, 2010.
    """
    alphabet: np.ndarray
    osic_type: str = "sinr"  # 'sinr', 'colnorm', or 'snr'
    H: Optional[np.ndarray] = None
    method: Literal["zf", "mmse"] = "zf"
    sigma2: Optional[float] = None
    name: str = "OSIC Detector"

    def __post_init__(self):
        if self.osic_type == "sinr":
            self.method = "mmse"
        elif self.osic_type in ("colnorm", "snr"):
            self.method = "zf"
        else:
            raise ValueError("osic_type must be 'sinr', 'colnorm', or 'snr'")

    def ordering(self, H: np.ndarray) -> int:
        NT = H.shape[1]

        match self.osic_type:

            case "sinr":
                W = LA.inv(H.conj().T @ H + self.sigma2 * np.eye(NT)) @ H.conj().T
                WH = W @ H

                diag_WH2 = np.abs(np.diag(WH))**2                         # |WH[i,i]|^2
                WH2 = np.abs(WH)**2                                       # |WH[i,j]|^2
                interference = np.sum(WH2, axis=1) - diag_WH2             # somme des interférences
                noise_term = self.sigma2 * np.sum(np.abs(W)**2, axis=1)   # bruit résiduel
                denominator = interference + noise_term
                sinr = diag_WH2 / denominator
                return int(np.argmax(sinr))

            case "colnorm":
                return np.argmax(np.linalg.norm(H, axis=0))

            case "snr":
                G = LA.inv(H.conj().T @ H) @ H.conj().T
                return int(np.argmin(LA.norm(G, axis=1)))

            case _:
                raise ValueError("osic_type must be 'sinr', 'colnorm', or 'snr'.")

    def forward(self, Y: np.ndarray) -> np.ndarray:
        if self.H is None or (self.sigma2 is None and self.osic_type == "sinr"):
            raise ValueError("H and sigma2 must be set before calling forward().")

        Y_temp = Y.copy()
        NT = self.H.shape[1]
        S_hat = np.zeros((NT, Y.shape[1]), dtype=int)
        order = []
        remaining_idx = list(range(NT))

        for stage in range(NT):
            H_temp = self.H[:, remaining_idx]
            idx_local = self.ordering(H_temp)
            best_current_idx = remaining_idx[idx_local]
            order.append(best_current_idx)

            # perform estimation
            match self.method:
                case "zf":
                    Z = zf_estimator(Y_temp, H_temp)
                case "mmse":
                    Z = mmse_estimator(Y_temp, H_temp, self.sigma2 )
            
            # perform detection
            S, _ = hard_projector(Z, self.alphabet)
            s_est = S[idx_local, :]
            x_est = self.alphabet[s_est]

            # update Y, S and remaining_ix
            Y_temp = Y_temp - H_temp[:, idx_local][:, np.newaxis] * x_est
            S_hat[best_current_idx, :] = s_est
            del remaining_idx[idx_local]

        return S_hat


@dataclass
class ApproximateMessagePassingDetector(Processor):
    """
    Implements the AMP (Approximate Message Passing) MIMO detector.

    Attributes
    ----------
    alphabet : numpy.ndarray
        Symbol constellation.
    H : numpy.ndarray
        Internal storage for the channel matrix.
    sigma2 : float
        Noise variance.
    N_it : int
        Number of iterations.
    alpha : float
        Damping factor.

    name : str
        Name of the detector.
    """
    alphabet: np.ndarray
    H: Optional[np.ndarray] = None
    sigma2: float = None
    alpha: float = 1
    N_it: int = 100
    is_mimo: bool = True
    name: str = "AMP Detector"

    def fit(self, y):
        # see Algorithm 2
        H = self.H
        N_r, N_t = H.shape
        x_t = np.zeros(N_t)
        r_t = y - np.matmul(H, x_t)
        H_H = np.transpose(np.conjugate(H))
        beta = N_t / N_r  # system ratio (below equation 1)
        tau_2 = beta * 1 / self.sigma2

        for _ in range(self.N_it):
            z_t = x_t + np.matmul(H_H, r_t)
            sigma2_t = self.sigma2 * (1 + tau_2)
            x_t = soft_projector(
                z_t, self.alphabet, sigma2_t
            )  # F function in the original publication
            kernel = np.abs(self.alphabet.reshape(1, -1) - x_t.reshape(-1, 1)) ** 2
            G = soft_projector(z_t, self.alphabet, tau_2, kernel)
            tau_2_old = tau_2
            tau_2 = (beta / (self.sigma2)) * np.mean(G)
            term1 = tau_2 / (1 + tau_2_old)
            r_t = y - np.matmul(self.H, x_t) + term1 * r_t

        return x_t

    def forward(self, Y):
        validate_H(self.H)
        validate_sigma2(self.sigma2)

        H = self.H
        _, N = Y.shape
        _, N_t = H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:, n] = self.fit(Y[:, n])

        S, _ = hard_projector(X, self.alphabet)
        return S


@dataclass
class OrthogonalApproximateMessagePassingDetector(Processor):
    """
    Implements the OAMP (Orthogonal AMP) MIMO detector.

    Attributes
    ----------
    alphabet : numpy.ndarray
        Symbol constellation.
    H : numpy.ndarray
        Internal storage for the channel matrix.
    sigma2 : float
        Noise variance.
    type : str
        Type of linear estimator.
    N_it : int
        Number of iterations.
    alpha : float
        Damping factor.
    name : str
        Name of the detector.
    """
    
    alphabet: np.ndarray
    H: Optional[np.ndarray] = None
    sigma2: float = None
    alpha: float = 1
    N_it: int = 100
    type: Literal["H", "pinv", "MMSE"] = "MMSE"
    is_mimo: bool = True
    name: str = "OAMP Detector"

    def get_W(self, vt_2=0):
        H = self.H

        match self.type:

            case "H":
                H_H = np.transpose(np.conjugate(H))
                W = H_H

            case "pinv":
                W = LA.pinv(H)

            case "MMSE":
                N_r, _ = H.shape
                H_H = np.transpose(np.conjugate(H))
                term1 = vt_2 * np.matmul(H, H_H) + self.sigma2 * np.eye(N_r)
                W = vt_2 * np.matmul(H_H, LA.inv(term1))

            case _:
                raise ValueError(f"Unknown type: {self.type}")

        return W

    def get_vt_2(self, error, epsilon=0.001):
        H = self.H
        N_r, _ = H.shape
        R = self.sigma2 * np.eye(N_r)
        H_H = np.conjugate(np.transpose(H))
        num = np.sum(np.abs(error) ** 2) - np.trace(R)
        den = np.trace(np.matmul(H_H, H))
        return max(num / den, epsilon)

    def get_tau_2(self, B, W, vt_2):
        N_r, N_t = self.H.shape
        R = self.sigma2 * np.eye(N_r)
        W_H = np.conjugate(np.transpose(W))
        B_H = np.conjugate(np.transpose(B))
        term1 = (vt_2 / N_t) * np.trace(np.matmul(B, B_H))
        term2 = (1 / N_t) * np.trace(np.matmul(W, np.matmul(R, W_H)))
        tau_2 = term1 + term2
        return tau_2

    def fit(self, y):
        tau_2, vt_2 = 1, 1
        _, N_t = self.H.shape
        x_t = np.zeros(N_t)

        for _ in range(self.N_it):
            W = self.get_W(vt_2)
            B = np.eye(N_t) - np.matmul(W, self.H)
            error = y - np.matmul(self.H, x_t)
            z_t = x_t + np.matmul(W, error)
            x_t = soft_projector(z_t, self.alphabet, tau_2)
            vt_2 = self.get_vt_2(error)
            tau_2 = self.get_tau_2(B, W, vt_2)

        return x_t

    def forward(self, Y):
        validate_H(self.H)
        validate_sigma2(self.sigma2)


        _, N = Y.shape
        _, N_t = self.H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:, n] = self.fit(Y[:, n])

        S, _ = hard_projector(X, self.alphabet)
        return S
