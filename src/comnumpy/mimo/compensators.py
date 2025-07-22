import numpy as np
from dataclasses import dataclass
from typing import Literal
from comnumpy.core import Processor
from comnumpy.core.utils import hard_projector


@dataclass
class BlindDualMIMOCompensator(Processor):
    r"""
    BlindDualMIMOCompensator for 2x2 MIMO channels.

    This class implements a blind dual MIMO compensator designed for 2x2 MIMO channels. It uses various loss functions to estimate filter weights without requiring a training sequence. The compensator supports different modes of operation, including Constant Modulus Algorithm (CMA), Radius Directed Equalization (RDE), and Decision Directed (DD) algorithms.

    Signal Model
    ------------
    The signal model is defined as:

    .. math::

        \mathbf{y}[n] = \mathbf{H}^H[n] \tilde{\mathbf{x}}[n]

    where:
    
    * :math:`\mathbf{H}[n]` is a matrix of size :math:`2 \times (2(2L+1))` containing the filter weights at step :math:`n`.
    * :math:`\mathbf{y}[n]` is a vector of size 2 containing the two polarisation data
    * :math:`\tilde{\mathbf{x}}[n]` is a :math:`2(2L+1)` vector containing the :math:`L^{th}` previous transmitted data on the two polarizations. This vector is obtained by stacking vertically the two polarizations as follows:

    .. math::

        \tilde{\mathbf{x}}[n] = \begin{bmatrix}
        x_0[n+L]\\
        \vdots\\
        x_0[n]\\
        \vdots\\
        x_0[n-L]\\
        x_1[n+L]\\
        \vdots\\
        x_1[n]\\
        \vdots\\
        x_1[n-L]\\
        \end{bmatrix}

    The filter weights are estimated using one of the following loss functions:

    - **Constant Modulus Algorithm (CMA)**: Minimizes the metric:

      .. math::

          \mathcal{L}_{CMA}(y[n]) = | R - |y[n]|^2 |^2

      where :math:`R = \frac{E[|s|^4]}{E[|s|^2]}` is derived from the alphabet.

    - **Radius Directed Equalization (RDE)**: Minimizes the metric:

      .. math::

          \mathcal{L}_{RDE}(y[n]) = | \mathcal{P}_{rad}^2(|y[n]|) - |y[n]|^2 |^2

      where :math:`\mathcal{P}_{rad}(|y[n]|)` is the orthogonal projector into the list of radius alphabet.

    - **Decision Directed (DD)**: Minimizes the metric:

      .. math::

          \mathcal{L}_{DD}(y[n]) = | \mathcal{P}_{\mathcal{M}}(y[n]) - y[n] |^2

      where :math:`\mathcal{P}_{\mathcal{M}}(y[n])` is the orthogonal projector into the alphabet.

    Attributes
    ----------
    L : int
        Length of the filter.
    alphabet : np.ndarray
        Alphabet used for modulation.
    mu : float, optional
        Step size for the update (default is 1e-4).
    oversampling : int, optional
        Oversampling factor (default is 1). When the oversampling is greater than one, the algorithm implements a fractionaly spaced equalizer
    norm : bool, optional
        Flag to normalize the filter weights (default is True).
    mode : Literal["cma", "rde", "dd"], optional
        Mode of operation (default is "cma").
    sub_block_length : int, optional
        Length of sub-blocks for processing (default is 20).
    name : str, optional
        Name of the processor (default is "mimo filter").

    References
    ----------
    * Faruk, Md Saifuddin, and Seb J. Savory. "Digital signal processing for coherent transceivers employing multilevel formats." Journal of Lightwave Technology 35.5 (2017): 1125-1141.
    """
    L: int = 10
    alphabet: np.ndarray = None
    mu: float = 1e-4
    oversampling: int = 1
    norm: bool = True
    mode: Literal["cma", "rde", "dd"] = "cma"
    sub_block_length = 20
    name: str = "mimo filter"

    def __post_init__(self):
        """
        Prepare the filter coefficients.
        """
        self.initialize_H()
        self.radius_cma = np.mean(np.abs(self.alphabet)**4) / np.mean(np.abs(self.alphabet)**2)
        self.radius_list = np.unique(np.abs(self.alphabet))

    def initialize_H(self):
        H = np.zeros((2, 2*(2*self.L+1)), dtype=complex)
        H[0, self.L] = 1
        H[1, (2*self.L+1)+self.L] = 1
        self.H = H

    def grad(self, input: np.ndarray, output: np.ndarray, target=None) -> np.ndarray:

        if self.mode == "cma":
            # see equation 19/20
            error = self.radius_cma - np.abs(output)**2
            term1 = (error * np.conjugate(output))
            grad = (term1.reshape(-1, 1)) * input
        
        if self.mode == "rde":
            _, radius_est = hard_projector(np.abs(output), self.radius_list)
            error = radius_est**2 - np.abs(output)**2
            term1 = (error * np.conjugate(output))
            grad = (term1.reshape(-1, 1)) * input

        if self.mode == "dd":
            _, output_est = hard_projector(output, self.alphabet)
            error = output_est - output
            term1 = np.conjugate(error)
            grad = (term1.reshape(-1, 1)) * input

        return grad
    
    def process_after_iteration(self, n, Y_sub):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:

        if X.shape[0] != 2:
            raise ValueError(f"Blind Dual MIMO Compensator only works for dual polarization signals (X shape={X.shape})")

        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N//os), dtype=complex)

        for n in range(2*L + 1, N, os):
            x_sub = np.ravel(X[:, n:n-(2*L+1):-1])
            y_sub = np.matmul(np.conjugate(self.H), x_sub)  # filter output
            grad = self.grad(x_sub, y_sub)
            self.H += self.mu*grad  # implement equation in matrix form directly
            Y[:, n//os] = y_sub

            # perform process after_iteration
            self.process_after_iteration(n//os, Y[:, n//os-1::-100])

        return Y
