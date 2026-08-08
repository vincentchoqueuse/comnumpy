import numpy as np
import itertools
from typing import Optional
from dataclasses import dataclass, field
from scipy.fft import fft, ifft, ifftshift
from comnumpy.core import Processor
from .metrics import compute_PAPR

@dataclass(slots=True)
class HardClipper(Processor):
    """
    Implements a hard clipping method to reduce the Peak-to-Average Power Ratio (PAPR) of a signal.

    The HardClipper class applies a hard clipping technique to a signal, which reduces its PAPR.
    Hard clipping limits the amplitude of the signal to a certain threshold, reducing the peaks of the signal
    while keeping the average power relatively unchanged.

    Attributes
    ----------
    cr_dB : float
        Clipping ratio in decibels (dB). Determines the threshold for clipping.
    name : str
        Name of the processor.

    Reference
    ---------
    * [1] Y. Rahmatallah et S. Mohan, « Peak-To-Average Power Ratio Reduction in OFDM Systems: A Survey And Taxonomy »,
      IEEE Commun. Surv. Tutorials, vol. 15, no 4, p. 1567 1592, 2013, doi: 10.1109/SURV.2013.021313.00164.
    """
    cr_dB: float
    name: str = field(default="hard_clipping", kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    cr: float = field(init=False, repr=False)

    def __post_init__(self):
        self.cr = 10 ** (self.cr_dB / 20)

    def forward(self, x: np.ndarray) -> np.ndarray:
        Pmoy = np.mean(np.abs(x) ** 2)
        Tm = self.cr * np.sqrt(Pmoy)
        if np.iscomplexobj(x):
            y = np.where(np.abs(x) > Tm, Tm * np.exp(1j * np.angle(x)), x)
        else:
            y = np.where(np.abs(x) > Tm, Tm * np.sign(x), x)
        return y


@dataclass(slots=True)
class IctPaprReductor(Processor):
    """
    Implements the Iterative Clipping and Filtering (ICT) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    The IctPaprReductor class reduces the PAPR of an OFDM signal using the ICT method.
    This method involves iteratively clipping and filtering the signal to achieve a target PAPR level.

    Attributes
    ----------
    PAPR_max_dB : float
        Target maximum PAPR in decibels (dB).
    filter_weight : float
        Weighting factor applied during the filtering step.
    N_it : int
        Number of iterations for the clipping and filtering process. Default is 16.
    shift : bool
        Whether to apply an inverse FFT shift to the processed signal. Default is False.
    norm : str
        Normalization mode for FFT and IFFT operations. Default is "ortho".
    name : str
        Name of the processor. Default is "ICT".

    Reference
    ---------
    * [1] Wang, Y-C., and Z-Q. Luo. "Optimized iterative clipping and filtering for PAPR reduction of OFDM signals."
      IEEE Transactions on communications 59.1 (2010): 33-37.
    """
    PAPR_max_dB: float
    filter_weight: float
    N_it: int = field(default=16, kw_only=True)
    shift: bool = field(default=False, kw_only=True)
    norm: str = field(default="ortho", kw_only=True)
    name: str = field(default="ICT", kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    cr: float = field(init=False, repr=False)

    def __post_init__(self):
        PAPR_max = 10 ** (self.PAPR_max_dB / 10)
        self.cr = np.sqrt(PAPR_max)

    def clip(self, x: np.ndarray) -> np.ndarray:
        # per-block RMS amplitude, computed along the block content axis
        Pmoy = np.sqrt(np.mean(np.abs(x) ** 2, axis=-1, keepdims=True))
        Tm = self.cr * Pmoy  # see equation 7
        y = np.where(np.abs(x) > Tm, Tm * np.exp(1j * np.angle(x)), x)
        return y

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Block layout (..., T, F): every block is processed along axis -1
        Y_preprocessed = X
        for _ in range(self.N_it):
            y = ifft(Y_preprocessed, norm="ortho", axis=-1)
            y = self.clip(y)
            Y_preprocessed = fft(y, norm="ortho", axis=-1)
            Y_preprocessed = self.filter_weight * Y_preprocessed  # out-of-band filtering

        if self.shift:
            Y_preprocessed = ifftshift(Y_preprocessed, axes=-1)
        Y = ifft(Y_preprocessed, norm=self.norm, axis=-1)

        return Y


@dataclass(slots=True)
class PtsPaprReductor(Processor):
    """
    Implements the Partial Transmit Sequences (PTS) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    The PtsPaprReductor class reduces the PAPR of an OFDM signal.
    It employs the PTS method, which involves dividing the signal into sub-blocks, applying different phase factors to each block,
    and then selecting the combination of phase factors that minimizes the PAPR.

    Attributes
    ----------
    phase_alphabet : list
        List of phase factors to be used in the PTS method.
    N_sub : int
        Number of sub-blocks the OFDM signal is divided into.
    name : str
        Name of the processor. Default is "PTS".

    Reference
    ---------
    * [1] L. J. Cimini and N. R. Sollenberger, "Peak-to-average power ratio reduction of an OFDM signal using partial transmit sequences,"
      1999 IEEE International Conference on Communications (Cat. No. 99CH36311), Vancouver, BC, Canada, 1999, pp. 511-515 vol.1,
      doi: 10.1109/ICC.1999.767992.
    """
    phase_alphabet: Optional[list]
    N_sub: int = field(default=16, kw_only=True)
    name: str = field(default="PTS", kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    combinations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.combinations = np.array(list(itertools.product(self.phase_alphabet, repeat=self.N_sub)))
        if self.phase_alphabet is None:
            self.phase_alphabet = [1, -1]

    def get_subblocks(self, X: np.ndarray) -> np.ndarray:
        # Adjacent partition: blocks consist of a contiguous set of subcarriers and are of equal size
        N = len(X)
        if N % self.N_sub != 0:
            raise ValueError("N_sc must be divisible by N_sub.")

        L = int(N / self.N_sub)
        X_m_array = np.zeros((N, self.N_sub), dtype=X.dtype)

        for m in range(self.N_sub):
            X_m_array[m * L:(m + 1) * L, m] = X[m * L:(m + 1) * L]

        return X_m_array

    def find_optimal_combination(self, x_m_array: np.ndarray) -> tuple:
        papr_list = np.zeros(len(self.combinations))

        for index, combination in enumerate(self.combinations):
            x_m_temp = np.dot(x_m_array, combination)
            papr_list[index] = compute_PAPR(x_m_temp)

        index_min = np.argmin(papr_list)
        combination = self.combinations[index_min]
        x_m = np.dot(x_m_array, combination)
        return x_m, combination

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Block layout (T, F): one optimal phase combination per block
        Y = np.zeros(X.shape, dtype=X.dtype)
        combination_list = []

        for t_index in range(X.shape[-2]):
            X_m_array = self.get_subblocks(X[t_index, :])
            x_m_array_ifft = ifft(X_m_array, norm="ortho", axis=0)
            x_m_ifft, combination = self.find_optimal_combination(x_m_array_ifft)
            Y[t_index, :] = x_m_ifft
            combination_list.append(combination)

        return Y
