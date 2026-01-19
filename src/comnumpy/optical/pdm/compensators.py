import numpy as np
from comnumpy.core.processors import Processor
from comnumpy.core.utils import get_alphabet, hard_projector
import matplotlib.pyplot as plt
from scipy import signal
from typing import Literal, Optional
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import os


@dataclass
class CMA(Processor):
    """
    CMA blind equalizer (dual-pol) from:
    Faruk, Md Saifuddin, and Seb J. Savory. "Digital signal processing for coherent transceivers employing multilevel formats."
    Journal of Lightwave Technology 35.5 (2017): 1125-1141.
    """

    L: int = 7
    alphabet: np.ndarray = None
    mu: float = 0.00001
    oversampling: int = 1
    norm: bool = True
    debug: bool = False
    mix: bool = True
    name: str = "cma"

    # internal / derived fields
    err_CMA1: list = field(init=False, repr=False)
    err_CMA2: list = field(init=False, repr=False)
    save_loss: bool = field(default=True, repr=False)
    sampled_output: list = field(init=False, repr=False)
    R: float = field(init=False)
    h11: np.ndarray = field(init=False, repr=False)
    h12: np.ndarray = field(init=False, repr=False)
    h21: np.ndarray = field(init=False, repr=False)
    h22: np.ndarray = field(init=False, repr=False)
    Y: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # compute R and initialize containers/filters
        self.err_CMA1 = []
        self.err_CMA2 = []
        self.sampled_output = []
        self.R = np.mean(np.abs(self.alphabet) ** 4) / np.mean(np.abs(self.alphabet) ** 2)
        self.h11 = np.zeros(self.L, dtype=complex)
        self.h12 = np.zeros(self.L, dtype=complex)
        self.h21 = np.zeros(self.L, dtype=complex)
        self.h22 = np.zeros(self.L, dtype=complex)
        self.Y = None

    def reset(self):
        self.h11[:] = 0
        self.h12[:] = 0
        self.h21[:] = 0
        self.h22[:] = 0
        self.err_CMA1 = []
        self.err_CMA2 = []
        self.sampled_output = []
        self.Y = None

    def prepare(self, X: np.ndarray):
        if self.norm:
            self.h11[0] = np.sqrt(np.mean(np.abs(X[0, :]) ** 2))
            self.h22[0] = np.sqrt(np.mean(np.abs(X[1, :]) ** 2))
        else:
            self.h11[0] = 1
            self.h22[0] = 1

    def grad(self, input: np.ndarray, output: np.ndarray, target=None) -> np.ndarray:
        N = len(input[0])
        x_1 = input[0]
        x_2 = input[1]

        radius_1 = np.abs(output[0]) ** 2
        error_1 = self.R - radius_1
        radius_2 = np.abs(output[1]) ** 2
        error_2 = self.R - radius_2

        grad = np.zeros((4, N), dtype=complex)
        grad[0, :] = -error_1 * output[0] * np.conj(x_1)
        grad[1, :] = -error_1 * output[0] * np.conj(x_2)
        grad[2, :] = -error_2 * output[1] * np.conj(x_1)
        grad[3, :] = -error_2 * output[1] * np.conj(x_2)
        return grad

    def accumulate_loss(self, output: np.ndarray):
        radius_1 = np.abs(output[0]) ** 2
        error_1 = self.R - radius_1
        radius_2 = np.abs(output[1]) ** 2
        error_2 = self.R - radius_2

        self.err_CMA1.append(np.atleast_1d(error_1 ** 2))
        self.err_CMA2.append(np.atleast_1d(error_2 ** 2))

    def get_loss(self):
        return self.err_CMA1, self.err_CMA2

    def forward(self, X: np.ndarray) -> np.ndarray:
        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N), dtype=complex)
        self.prepare(X)

        for n in range(L + 1, N):
            input = X[:, n : n - L : -1]
            x_1 = input[0, :]
            x_2 = input[1, :]
            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])
            if (n % os) == 0:
                grad = self.grad(input, output)
                self.h11 = self.h11 - self.mu * grad[0, :]
                self.h22 = self.h22 - self.mu * grad[3, :]

                if self.mix:
                    self.h12 = self.h12 - self.mu * grad[1, :]
                    self.h21 = self.h21 - self.mu * grad[2, :]

            Y[:, n] = output
        self.Y = Y
        return Y

    def get_data(self):
        return self.Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

@dataclass
class RDE(CMA):
    """
    RDE: Radius-Directed Equalizer derived from.

    Inherits the blind equalizer structure from CMA and changes the error
    computation to use the nearest constellation radius (RDE logic).
    """

    # internal lists and buffers (initialized in __post_init__)
    err1_RDE: list = field(init=False, repr=False)
    err2_RDE: list = field(init=False, repr=False)
    sampled_output: list = field(init=False, repr=False)

    def __post_init__(self):
        # initialize parent fields
        super().__post_init__()
        # initialize RDE-specific containers
        self.err1_RDE = []
        self.err2_RDE = []
        self.sampled_output = []

    def reset(self):
        # call parent reset to ensure consistent state
        super().reset()

    def grad(self, input: np.ndarray, output: np.ndarray, target=None) -> np.ndarray:
        x_1 = input[0]
        x_2 = input[1]
        radius_list = np.unique(np.abs(self.alphabet)**2)

        radius_1 = np.abs(output[0]) ** 2
        radius_2 = np.abs(output[1]) ** 2
        index_1 = np.argmin((radius_1 - radius_list)**2)
        index_2 = np.argmin((radius_2 - radius_list)**2)
        error_1 = radius_list[index_1] - radius_1
        error_2 = radius_list[index_2] - radius_2

        grad = np.zeros((4, len(x_1)), dtype=complex)
        grad[0, :] = -error_1 * output[0] * np.conj(x_1)
        grad[1, :] = -error_1 * output[0] * np.conj(x_2)
        grad[2, :] = -error_2 * output[1] * np.conj(x_1)
        grad[3, :] = -error_2 * output[1] * np.conj(x_2)
        return grad

    def accumulate_loss(self, output: np.ndarray):
        radius_list = np.unique(np.abs(self.alphabet)**2)
        radius_1 = np.abs(output[0]) ** 2
        radius_2 = np.abs(output[1]) ** 2
        index_1 = np.argmin((radius_1 - radius_list) ** 2)
        index_2 = np.argmin((radius_2 - radius_list) ** 2)
        error_1 = radius_list[index_1] - radius_1
        error_2 = radius_list[index_2] - radius_2

        self.err1_RDE.append(np.atleast_1d(error_1 ** 2))
        self.err2_RDE.append(np.atleast_1d(error_2 ** 2))

    def get_loss(self):
        return self.err1_RDE, self.err2_RDE

    def forward(self, X: np.ndarray) -> np.ndarray:
        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N), dtype=complex)
        self.prepare(X)

        for n in range(L + 1, N):
            input = X[:, n : n - L : -1]

            x_1 = input[0, :]
            x_2 = input[1, :]
            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])

            if (n % os) == 0:
                grad = self.grad(input, output)
                self.h11 -= self.mu * grad[0, :]
                self.h22 -= self.mu * grad[3, :]

                if self.mix:
                    self.h12 -= self.mu * grad[1, :]
                    self.h21 -= self.mu * grad[2, :]

            Y[:, n] = output
        self.Y = Y
        return Y

    def get_data(self) -> Any:
        return self.Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


from dataclasses import dataclass, field
import numpy as np

@dataclass
class DDLMS(CMA):
    """
    DD_LMS derived from CMA.
    """

    err1_DD: list = field(init=False, repr=False)
    err2_DD: list = field(init=False, repr=False)
    sampled_output: list = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.err1_DD = []
        self.err2_DD = []
        self.sampled_output = []

    def reset(self):
        self.h11[:] = 0
        self.h12[:] = 0
        self.h21[:] = 0
        self.h22[:] = 0
        self.err1_DD = []
        self.err2_DD = []
        self.sampled_output = []
        self.Y = None

    def grad(self, input, output, target=None):
        x_1 = input[0]
        x_2 = input[1]

        y_1 = output[0]
        y_2 = output[1]

        det_1 = hard_projector(np.array([y_1]), self.alphabet)[1]
        det_2 = hard_projector(np.array([y_2]), self.alphabet)[1]

        error_1 = det_1 - y_1
        error_2 = det_2 - y_2

        grad = np.zeros((4, len(x_1)), dtype=complex)
        grad[0, :] = -error_1 * np.conj(x_1)
        grad[1, :] = -error_1 * np.conj(x_2)
        grad[2, :] = -error_2 * np.conj(x_1)
        grad[3, :] = -error_2 * np.conj(x_2)
        return grad

    def accumulate_loss(self, output):
        y_1 = output[0]
        y_2 = output[1]
        det_1 = hard_projector(np.array([y_1]), self.alphabet)[1]
        det_2 = hard_projector(np.array([y_2]), self.alphabet)[1]
        error_1 = det_1 - y_1
        error_2 = det_2 - y_2
        self.err1_DD.append(np.abs(error_1) ** 2)
        self.err2_DD.append(np.abs(error_2) ** 2)

    def get_loss(self):
        return self.err1_DD, self.err2_DD

    def forward(self, X):
        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N), dtype=complex)
        self.prepare(X)

        for n in range(L + 1, N):
            input = X[:, n : n - L : -1]

            x_1 = input[0, :]
            x_2 = input[1, :]
            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])

            if (n % os) == 0:
                self.accumulate_loss(output)
                grad = self.grad(input, output)
                self.h11 -= self.mu * grad[0, :]
                self.h22 -= self.mu * grad[3, :]

                if self.mix:
                    self.h12 -= self.mu * grad[1, :]
                    self.h21 -= self.mu * grad[2, :]

            Y[:, n] = output

        self.Y = Y
        return Y

    def get_data(self):
        return self.Y

    def __call__(self, X):
        return self.forward(X)
    

@dataclass
class AdaptiveChannel(Processor):
    r"""Adaptive channel equalizer with multiple modes (CMA, RDE, DD).
    
    This processor implements adaptive equalization for dual-polarization signals
    using various algorithms: Constant Modulus Algorithm (CMA), Radius Directed
    Equalization (RDE), or Decision-Directed (DD) mode.
    
    Parameters
    ----------
    L : int
        Filter length (number of taps)
    alphabet : np.ndarray
        Constellation alphabet for the modulation scheme
    mu : float, optional
        Step size for gradient descent (default: 0.00001)
    oversampling : int, optional
        Oversampling factor (default: 1)
    norm : bool, optional
        Whether to normalize initial filter coefficients (default: True)
    debug : bool, optional
        Enable debug mode (default: False)
    mix : bool, optional
        Enable cross-polarization mixing (default: True)
    name : str, optional
        Processor name (default: "adaptive")
    mode : Literal["cma", "rde", "dd"], optional
        Equalization mode (default: "cma")
    """
    
    L: int = 7
    alphabet: np.ndarray = None
    mu: float = 0.00001
    oversampling: int = 1
    norm: bool = True
    debug: bool = False
    mix: bool = True
    name: str = "adaptive"
    mode: Literal["cma", "rde", "dd"] = "cma"
    
    # Computed attributes
    err_CMA1: list = field(default_factory=list, init=False)
    err_CMA2: list = field(default_factory=list, init=False)
    R: float = field(init=False)
    radius: np.ndarray = field(init=False)
    reset_called: bool = field(default=False, init=False)
    
    # Filter coefficients (initialized in reset)
    h11: np.ndarray = field(init=False, repr=False)
    h12: np.ndarray = field(init=False, repr=False)
    h21: np.ndarray = field(init=False, repr=False)
    h22: np.ndarray = field(init=False, repr=False)
    Y: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize computed parameters."""
        super().__init__()
        self.R = np.mean(np.abs(self.alphabet)**4) / np.mean(np.abs(self.alphabet)**2)
        self.radius = np.unique(np.abs(self.alphabet)**2)

    def reset(self):
        """Reset filter coefficients to zero."""
        self.h11 = np.zeros(self.L, dtype=complex)
        self.h12 = np.zeros(self.L, dtype=complex)
        self.h21 = np.zeros(self.L, dtype=complex)
        self.h22 = np.zeros(self.L, dtype=complex)
        self.Y = None
        self.mode = "cma"
        self.reset_called = True

    def prepare(self, X):
        """Initialize filter coefficients based on input signal power.
        
        Parameters
        ----------
        X : np.ndarray
            Input signal array with shape (2, N)
        """
        if not self.reset_called:
            self.reset()
        if self.norm:
            self.h11[0] = np.sqrt(np.mean(np.abs(X[0, :]) ** 2))
            self.h22[0] = np.sqrt(np.mean(np.abs(X[1, :]) ** 2))
        else:
            self.h11[0] = 1
            self.h22[0] = 1
    
    def grad(self, input, output, target=None):
        """Compute gradient for filter coefficient update.
        
        Parameters
        ----------
        input : np.ndarray
            Input signal segment
        output : np.ndarray
            Filter output
        target : np.ndarray, optional
            Target signal (unused)
            
        Returns
        -------
        np.ndarray
            Gradient array with shape (4, N)
        """
        # compute loss
        N = len(input[0])
        x_1 = input[0]
        x_2 = input[1]

        if self.mode == 'cma':
            radius_1 = np.abs(output[0]) ** 2
            error_1 = self.R - radius_1
            radius_2 = np.abs(output[1]) ** 2
            error_2 = self.R - radius_2

        if self.mode == 'rde':
            radius_1 = np.abs(output[0]) ** 2
            radius_2 = np.abs(output[1]) ** 2
            index_1 = np.argmin((radius_1 - self.radius)**2)
            index_2 = np.argmin((radius_2 - self.radius)**2)
            error_1 = self.radius[index_1] - radius_1
            error_2 = self.radius[index_2] - radius_2

        if self.mode == 'dd':
            y_1 = output[0]
            y_2 = output[1]
            det_1 = hard_projector(np.array([y_1]), self.alphabet)[1]
            det_2 = hard_projector(np.array([y_2]), self.alphabet)[1]
            error_1 = det_1 - y_1
            error_2 = det_2 - y_2

        # compute grad with respect to h11, h12, h12 and h22
        grad = np.zeros((4, N), dtype=complex)
        grad[0, :] = -error_1 * output[0] * np.conj(x_1)
        grad[1, :] = -error_1 * output[0] * np.conj(x_2)
        grad[2, :] = -error_2 * output[1] * np.conj(x_1)
        grad[3, :] = -error_2 * output[1] * np.conj(x_2)
        return grad
    
    def iteration(self, n, Y):
        """Hook for custom per-iteration processing. """

        pass

    def forward(self, X):
        """Process input signal through adaptive equalizer."""

        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N), dtype=complex)
        self.prepare(X)

        for n in range(L + 1, N):
            input = X[:, n : n - L : -1]  

            # compute output
            x_1 = input[0, :]
            x_2 = input[1, :]

            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])

            if (n % os) == 0:
                grad = self.grad(input, output)  # compute grad
                # update parameter
                self.h11 = self.h11 - self.mu * grad[0, :]
                self.h22 = self.h22 - self.mu * grad[3, :]

                if self.mix == True:
                    self.h12 = self.h12 - self.mu * grad[1, :]
                    self.h21 = self.h21 - self.mu * grad[2, :]
            
            self.iteration(n, output)
            Y[:, n] = output  

        self.Y = Y
        return Y
    
    def get_data(self):
        return self.Y
    
    def __call__(self, X):
        return self.forward(X)
    
@dataclass
class Switch(AdaptiveChannel):
    r"""Adaptive channel equalizer with automatic mode switching.
    
    This processor extends :class:`Adaptive_Channel` by automatically switching
    from CMA to RDE mode after a specified number of samples. This progressive
    equalization strategy starts with blind CMA convergence and then refines
    using RDE for better performance.
    """
    name: str = "adaptive_channel"
    tx_before_CMA: Optional[np.ndarray] = None
    commute: int = 100_000
    
    phase_corrector: Optional[object] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        super().__post_init__()

    def iteration(self, step, Y):
        if (self.mode == "cma") and (step >= self.commute):
            self.mode = "rde"
         # if (self.mode == "rde") and (step >= self.commute[1]):
        #     # self.phase_corrector = Phase_Correction_2case(N=self.L, tx_before_CMA=self.tx_before_CMA, alphabet=self.alphabet)
        #     # Y_signal = self.get_data()
        #     # Y_corrected = self.phase_corrector(Y_signal)
        #     # self.Y = Y_corrected
        #     self.mode = "dd"


# @dataclass
# class SOP_compensator(Processor):
#     """
#     Decision-directed SOP tracker (2x2 Jones) for dual-pol signals.
#     - Update once per symbol: if (n % oversampling) == phase
#     - Compensation applied to all samples: x_hat[n] = U^H y[n]
#     - Place after SRRCFilter (rx), before Downsampler / Demapper

#     Update (at symbol instants):
#       J <- J + mu * (y - J s) s^H
#       U <- unitary projection of J via SVD (periodic)
#     """
#     alphabet: np.ndarray
#     mu: float = 3e-3
#     oversampling: int = 2
#     phase: int = 0
#     proj_period_sym: int = 1
#     name: str = "sop_compensator"

#     # internal state
#     J: np.ndarray = field(init=False, repr=False)
#     U: np.ndarray = field(init=False, repr=False)
#     _sym_updates: int = field(init=False, repr=False, default=0)

#     def __post_init__(self):
#         self.alphabet = np.asarray(self.alphabet)
#         self.J = np.eye(2, dtype=np.complex128)
#         self.U = np.eye(2, dtype=np.complex128)
#         self._sym_updates = 0

#     def reset(self):
#         self.J[:, :] = 0
#         self.J[0, 0] = 1
#         self.J[1, 1] = 1
#         self.U[:, :] = 0
#         self.U[0, 0] = 1
#         self.U[1, 1] = 1
#         self._sym_updates = 0

#     @staticmethod
#     def _project_unitary(J: np.ndarray) -> np.ndarray:
#         U, _, Vh = np.linalg.svd(J)
#         return (U @ Vh).astype(np.complex128)

#     def forward(self, Y: np.ndarray) -> np.ndarray:
#         """
#         Y: np.ndarray, shape (2, N), complex
#         Returns Xhat: compensated signal (2, N)
#         """
#         assert Y.ndim == 2 and Y.shape[0] == 2, "SOP_compensator expects input of shape (2, N)."
#         N = Y.shape[1]
#         Xhat = np.empty_like(Y, dtype=np.complex128)

#         for n in range(N):
#             y = Y[:, n]
#             # Apply current unitary compensation to every sample
#             xhat = self.U.conj().T @ y
#             Xhat[:, n] = xhat

#             # Update at symbol instants only
#             if (n % self.oversampling) == self.phase:
#                 # Decision-directed reference (per-pol hard decisions)
#                 _, s_vec = hard_projector(xhat, self.alphabet)  # shape (2,)
#                 resid = y - (self.J @ s_vec)
#                 self.J = self.J + self.mu * np.outer(resid, np.conjugate(s_vec))

#                 # Periodic unitary projection for stability
#                 self._sym_updates += 1
#                 if (self._sym_updates % self.proj_period_sym) == 0:
#                     self.U = self._project_unitary(self.J)

#         return Xhat

#     def __call__(self, X: np.ndarray) -> np.ndarray:
#         return self.forward(X)





'''
def mcma_targets(alphabet: np.ndarray):
    Re = np.real(alphabet)
    Im = np.imag(alphabet)

    R1 = np.mean(Re**4) / np.mean(Re**2)
    R2 = np.mean(Im**4) / np.mean(Im**2)

    return float(R1), float(R2)


@dataclass
class MCMA:
    alphabet: np.ndarray
    mu: float = 1e-3
    oversampling: int = 2
    phase: int = 0
    unitary_projection: bool = True
    name: str = "mcma_2x2"
    W: np.ndarray = field(init=False, repr=False)
    R1: float = field(init=False)
    R2: float = field(init=False)
    _sym_updates: int = field(init=False, repr=False)

    def __post_init__(self):
        self.R1, self.R2 = mcma_targets(self.alphabet)
        self.W = np.zeros((2, 2), dtype=complex)
        self.reset()

    def reset(self):
        self.W[:] = 0
        self.W[0, 0] = 1
        self.W[1, 1] = 1
        self._sym_updates = 0

    def project_unitary(self):

        U, _, Vh = np.linalg.svd(self.W)
        self.W = (U @ Vh).astype(complex)

    def forward(self, X: np.ndarray):

        assert X.ndim == 2 and X.shape[0] == 2, "MCMA--intrare de formă (2, N)"

        N = X.shape[1]
        Y = np.empty_like(X, dtype=complex)

        for n in range(N):
            x = X[:, n]            # eșantion intrare
            y = self.W @ x         # egalizare
            Y[:, n] = y

            # update doar pe simbol
            if (n % self.oversampling) == self.phase:

                yR = np.real(y)
                yI = np.imag(y)

                e = (self.R1 - yR**2) * yR + 1j * (self.R2 - yI**2) * yI

                self.W += self.mu * np.outer(e, np.conj(x))

                self._sym_updates += 1
                # if self.unitary_projection:
                #     self.project_unitary()

        return Y

    def __call__(self, X: np.ndarray):
        return self.forward(X)
'''


'''
def mcma_targets(alphabet, p=2):
    aR = np.real(alphabet)
    aI = np.imag(alphabet)

    RpR = np.mean(np.abs(aR)**(2*p)) / np.mean(np.abs(aR)**p)
    RpI = np.mean(np.abs(aI)**(2*p)) / np.mean(np.abs(aI)**p)
    return RpR, RpI


@dataclass
class MCMA(Processor):
    alphabet: np.ndarray
    mu: float = 1e-3
    p: int = 2
    name: str = "MCMA"

    W: np.ndarray = field(init=False)
    RpR: float = field(init=False)
    RpI: float = field(init=False)

    def __post_init__(self):
        self.RpR, self.RpI = mcma_targets(self.alphabet, self.p)
        self.reset()

    def reset(self):
        self.W = np.eye(2, dtype=complex)

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.shape[0] == 2
        Y = np.zeros_like(X)

        for k in range(X.shape[1]):
            x = X[:, k]
            y = self.W @ x
            Y[:, k] = y

            yR = np.real(y)
            yI = np.imag(y)

            # MCMA error (eq. 13)
            eR = yR * (np.abs(yR)**self.p - self.RpR)
            eI = yI * (np.abs(yI)**self.p - self.RpI)
            e = eR + 1j * eI

            # Stochastic gradient update (eq. 12)
            self.W -= self.mu * np.outer(e, np.conj(x))

        return Y
'''


import numpy as np
from dataclasses import dataclass, field

def mcma_targets(alphabet: np.ndarray, p: int = 2):
    # Rp,R = E[|a_R|^{2p}] / E[|a_R|^p],  Rp,I = E[|a_I|^{2p}] / E[|a_I|^p]
    aR = np.real(alphabet)
    aI = np.imag(alphabet)
    RpR = np.mean(np.abs(aR) ** (2 * p)) / np.mean(np.abs(aR) ** p)
    RpI = np.mean(np.abs(aI) ** (2 * p)) / np.mean(np.abs(aI) ** p)
    return float(RpR), float(RpI)

@dataclass
class MCMA:
    alphabet: np.ndarray
    mu: float = 1e-3
    p: int = 2  # p ≥ 2
    W: np.ndarray = field(init=False, repr=False)
    RpR: float = field(init=False)
    RpI: float = field(init=False)
    name: str = "MCMA"

    def __post_init__(self):
        assert self.p >= 2, "p>= 2"
        self.RpR, self.RpI = mcma_targets(self.alphabet, self.p)
        self.reset()

    def reset(self):
        self.W = np.eye(2, dtype=complex)

    def forward(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2 and X.shape[0] == 2, "MCMA expects input of shape (2, N)"
        N = X.shape[1]
        Y = np.zeros_like(X, dtype=complex)

        for k in range(N):
            x = X[:, k]
            y = self.W @ x
            Y[:, k] = y


            yR = np.real(y)
            yI = np.imag(y)
            abs_yR = np.abs(yR)
            abs_yI = np.abs(yI)

            eR = yR * (abs_yR ** (self.p - 2)) * (abs_yR ** self.p - self.RpR)
            eI = yI * (abs_yI ** (self.p - 2)) * (abs_yI ** self.p - self.RpI)
            e = eR + 1j * eI

            self.W -= self.mu * np.outer(e, np.conj(x))

        return Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)