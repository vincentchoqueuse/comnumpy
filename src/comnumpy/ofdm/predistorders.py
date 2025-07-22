import numpy as np
import itertools
from dataclasses import dataclass, field
from scipy.fft import fft, ifft, fftshift, ifftshift
from comnumpy.core import Processor
from .metrics import compute_PAPR

@dataclass
class Hard_Clipper(Processor):
    """
    Implements a hard clipping method to reduce the Peak-to-Average Power Ratio (PAPR) of a signal.

    The Hard_Clipper class is designed to apply a hard clipping technique to a signal, which reduces its PAPR.
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
    * [1] Y. Rahmatallah et S. Mohan, « Peak-To-Average Power Ratio Reduction in OFDM Systems: A Survey And Taxonomy », IEEE Commun. Surv. Tutorials, vol. 15, no 4, p. 1567 1592, 2013, doi: 10.1109/SURV.2013.021313.00164.
    """
    cr_dB: float
    name: str = "hard_clipping"

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


@dataclass
class ICT_PAPR_Reductor(Processor):
    """
    Implements the Iterative Clipping and Filtering (ICT) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    The ICT_PAPR_Reductor class is designed to reduce the PAPR of an Orthogonal Frequency Division Multiplexing (OFDM) signal using the ICT method.
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
        Whether to apply an inverse FFT shift to the processed signal. Default is True.
    norm : str
        Normalization mode for FFT and IFFT operations. Default is "ortho".
    name : str
        Name of the processor. Default is "ICT".

    Reference
    ---------
    * [1] Wang, Y-C., and Z-Q. Luo. "Optimized iterative clipping and filtering for PAPR reduction of OFDM signals." IEEE Transactions on communications 59.1 (2010): 33-37.
    """
    PAPR_max_dB: float
    filter_weight: float
    N_it: int = 16
    shift: bool = False
    axis: int = 0
    norm: str = "ortho"
    name: str = "ICT"

    def __post_init__(self):
        PAPR_max = 10 ** (self.PAPR_max_dB / 10)
        self.cr = np.sqrt(PAPR_max)

    def clip(self, x: np.ndarray) -> np.ndarray:
        Pmoy = np.sqrt(np.mean(np.abs(x) ** 2))
        Tm = self.cr * Pmoy  # see equation 7
        y = np.where(np.abs(x) > Tm, Tm * np.exp(1j * np.angle(x)), x)
        return y

    def forward(self, X: np.ndarray) -> np.ndarray:
        N_sc, L = X.shape
        Y_preprocessed = np.zeros((N_sc, L), dtype=X.dtype)

        for l in range(L):
            X_l = X[:, l]
            for _ in range(self.N_it):
                x_l = ifft(X_l, norm="ortho")
                x_l = self.clip(x_l)
                X_l = fft(x_l, norm="ortho")
                X_l = self.filter_weight * X_l  # out-of-band filtering
            Y_preprocessed[:, l] = X_l

        if self.shift:
            Y_preprocessed = ifftshift(Y_preprocessed, axes=self.axis)
        Y = ifft(Y_preprocessed, norm=self.norm, axis=self.axis)

        return Y


@dataclass
class PTS_PAPR_Reductor(Processor):
    """
    Implements the Partial Transmit Sequences (PTS) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    The PTS_PAPR_Reductor class is designed to reduce the PAPR of an Orthogonal Frequency Division Multiplexing (OFDM) signal.
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
    * [1] L. J. Cimini and N. R. Sollenberger, "Peak-to-average power ratio reduction of an OFDM signal using partial transmit sequences," 1999 IEEE International Conference on Communications (Cat. No. 99CH36311), Vancouver, BC, Canada, 1999, pp. 511-515 vol.1, doi: 10.1109/ICC.1999.767992.
    """
    phase_alphabet: None
    N_sub: int = 16
    name: str = "PTS"
    combinations: np.ndarray = field(init=False)

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
        _, L = X.shape
        Y = np.zeros(X.shape, dtype=X.dtype)
        combination_list = []

        for l in range(L):
            X_m_array = self.get_subblocks(X[:, l])
            x_m_array_ifft = ifft(X_m_array, norm="ortho", axis=0)
            x_m_ifft, combination = self.find_optimal_combination(x_m_array_ifft)
            Y[:, l] = x_m_ifft
            combination_list.append(combination)

        return Y
