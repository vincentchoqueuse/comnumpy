import numpy as np
import itertools
from typing import Optional
from dataclasses import dataclass, field
from comnumpy._backend import fft, ifft, ifftshift  # cupy-compatible (D3)
from comnumpy.core import Processor
from .metrics import compute_papr

__all__ = ["HardClipper", "IctPaprReductor", "PtsPaprReductor"]

@dataclass(slots=True)
class HardClipper(Processor):
    r"""
    Implements a hard clipping method to reduce the Peak-to-Average Power Ratio (PAPR) of a signal.

    Signal Model
    ------------
    Hard clipping limits the amplitude of the signal to a threshold
    :math:`T_m` while preserving the phase, reducing the peaks of the
    signal while keeping the average power relatively unchanged:

    .. math::
        y[n] = \left\{\begin{array}{cl}
        x[n] & \text{if } |x[n]| \le T_m,\\
        T_m \, e^{i \angle x[n]} & \text{if } |x[n]| > T_m,
        \end{array}\right.
        \qquad T_m = \gamma \sqrt{P_x}

    where:

    * :math:`x[n]` is the input signal and :math:`y[n]` the clipped output,
    * :math:`P_x = \mathbb{E}\left[|x[n]|^2\right]` is the mean input power,
    * :math:`\gamma = 10^{\mathrm{CR_{dB}}/20}` is the clipping ratio.

    For real-valued inputs, :math:`T_m \, \mathrm{sign}(x[n])` replaces the
    polar form.

    Axes: *axis -1* -- the clipping is applied pointwise, but the
    threshold :math:`T_m` is computed from the mean power **along the
    last axis**, so each row of a batch is clipped against its own
    power -- one transmitter per trial (D51).

    Parameters
    ----------
    cr_dB : float
        Clipping ratio :math:`\mathrm{CR_{dB}}` in decibels (dB).
        Determines the threshold for clipping.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"hard_clipping"``.

    References
    ----------
    Y. Rahmatallah, S. Mohan, "Peak-To-Average Power Ratio Reduction in
    OFDM Systems: A Survey And Taxonomy", IEEE Communications Surveys &
    Tutorials, vol. 15, no. 4, pp. 1567-1592, 2013,
    doi: 10.1109/SURV.2013.021313.00164.

    Examples
    --------
    >>> x = np.array([0.5, 2.0, -3.0])
    >>> clipper = HardClipper(cr_dB=0)
    >>> print(np.round(clipper(x), 3))
    [ 0.5    2.    -2.102]
    """
    cr_dB: float
    name: str = field(default="hard_clipping", kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    cr: float = field(init=False, repr=False)

    def __post_init__(self):
        self.cr = 10 ** (self.cr_dB / 20)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # per-row power: each trial of a batch clips against its own
        # RMS, not the RMS of the whole stack (D51)
        Pmoy = np.mean(np.abs(x) ** 2, axis=-1, keepdims=True)
        Tm = self.cr * np.sqrt(Pmoy)
        if np.iscomplexobj(x):
            y = np.where(np.abs(x) > Tm, Tm * np.exp(1j * np.angle(x)), x)
        else:
            y = np.where(np.abs(x) > Tm, Tm * np.sign(x), x)
        return y


@dataclass(slots=True)
class IctPaprReductor(Processor):
    r"""
    Implements the Iterative Clipping and Filtering (ICT) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    Signal Model
    ------------
    The frequency-domain input is iteratively transformed to the time
    domain, clipped, transformed back and filtered. Starting from
    :math:`X^{(0)}[k] = X[k]`, each iteration :math:`i = 1, \dots, N_{it}`
    computes:

    .. math::
        s^{(i)}[n] = \mathrm{IDFT}\{X^{(i-1)}\}[n], \qquad
        X^{(i)}[k] = w \, \mathrm{DFT}\{\mathrm{clip}(s^{(i)})\}[k]

    where :math:`\mathrm{clip}(\cdot)` is the phase-preserving amplitude
    clipping of :class:`HardClipper` with the per-block threshold:

    .. math::
        T_m = \gamma \sqrt{P_s}, \qquad
        \gamma = \sqrt{10^{\mathrm{PAPR_{max}}/10}},

    and where:

    * :math:`X[k]` is the frequency-domain input at subcarrier :math:`k`,
    * :math:`P_s` is the mean power of the current time-domain block :math:`s^{(i)}[n]`,
    * :math:`\mathrm{PAPR_{max}}` is the target maximum PAPR in dB,
    * :math:`w` is the filtering weight applied in the frequency domain.

    The output is the time-domain signal
    :math:`y[n] = \mathrm{IDFT}\{X^{(N_{it})}\}[n]`.

    Axes: *axis -1* -- each block of the Block layout ``(..., T, F)`` is
    processed along the block content axis.

    Parameters
    ----------
    PAPR_max_dB : float
        Target maximum PAPR :math:`\mathrm{PAPR_{max}}` in decibels (dB).
    filter_weight : float
        Weighting factor :math:`w` applied during the filtering step.
    N_it : int, optional, keyword-only
        Number of iterations :math:`N_{it}` for the clipping and filtering process. Default is 16.
    shift : bool, optional, keyword-only
        Whether to apply an inverse FFT shift before the final IFFT. Default is False.
    norm : str, optional, keyword-only
        Normalization mode for the final IFFT. Default is "ortho".
    name : str, optional, keyword-only
        Name of the processor. Default is "ICT".

    References
    ----------
    Y.-C. Wang, Z.-Q. Luo, "Optimized iterative clipping and filtering for
    PAPR reduction of OFDM signals", IEEE Transactions on Communications,
    vol. 59, no. 1, pp. 33-37, 2011.

    Examples
    --------
    >>> X = np.array([[1, 1, -1, 1, 1, -1, -1, 1]], dtype=complex)
    >>> reductor = IctPaprReductor(PAPR_max_dB=2.0, filter_weight=1.0)
    >>> y = reductor(X)
    >>> print(round(float(compute_papr(y[0], unit="dB")), 2))
    2.0
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
    r"""
    Implements the Partial Transmit Sequences (PTS) method for Peak-to-Average Power Ratio (PAPR) reduction in OFDM signals.

    Signal Model
    ------------
    Each frequency-domain block :math:`X[k]` of length :math:`N` is
    partitioned into :math:`M` disjoint adjacent sub-blocks
    :math:`X_m[k]` (each keeping its :math:`N/M` contiguous subcarriers
    and zeros elsewhere). The transmitted time-domain block is the
    phase-rotated combination:

    .. math::
        y[n] = \sum_{m=1}^{M} \hat{b}_m \, \mathrm{IDFT}\{X_m\}[n]

    where the phase factors are selected by exhaustive search over the
    phase alphabet :math:`\mathcal{B}` to minimize the PAPR:

    .. math::
        (\hat{b}_1, \dots, \hat{b}_M) =
        \arg\min_{b_m \in \mathcal{B}}
        \mathrm{PAPR}\left(\sum_{m=1}^{M} b_m \, \mathrm{IDFT}\{X_m\}\right)

    and where:

    * :math:`X[k]` is the frequency-domain input at subcarrier :math:`k`,
    * :math:`M` is the number of sub-blocks,
    * :math:`\mathcal{B}` is the phase factor alphabet,
    * :math:`y[n]` is the time-domain output block.

    Axes: *declared axis* -- expects the 2-D Block layout ``(T, F)``,
    validated in ``prepare()``; one phase combination is optimized per
    block and the IDFT runs along the block content axis.

    Parameters
    ----------
    phase_alphabet : list, optional
        Phase factor alphabet :math:`\mathcal{B}` used in the PTS
        method. Default is ``[1, -1]``.
    N_sub : int, optional, keyword-only
        Number of sub-blocks :math:`M` the OFDM signal is divided into. Default is 16.
    name : str, optional, keyword-only
        Name of the processor. Default is "PTS".

    References
    ----------
    L. J. Cimini, N. R. Sollenberger, "Peak-to-average power ratio
    reduction of an OFDM signal using partial transmit sequences", 1999
    IEEE International Conference on Communications, Vancouver, BC,
    Canada, 1999, pp. 511-515 vol. 1, doi: 10.1109/ICC.1999.767992.

    Examples
    --------
    >>> reductor = PtsPaprReductor([1, -1], N_sub=2)
    >>> Y = reductor(np.ones((1, 4), dtype=complex))
    >>> print(np.round(Y, 3) + 0.0)
    [[0.+0.j 1.+1.j 0.+0.j 1.-1.j]]
    >>> print(round(float(compute_papr(Y[0], unit="dB")), 2))
    3.01
    """
    phase_alphabet: Optional[list[complex]] = None
    N_sub: int = field(default=16, kw_only=True)
    name: str = field(default="PTS", kw_only=True)
    # internal state (declared for slots, D40a): always assigned in __post_init__
    combinations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # the default has to be applied *before* it is used: the previous
        # order built the combinations from None and only then substituted
        # [1, -1], so the documented default raised TypeError
        if self.phase_alphabet is None:
            self.phase_alphabet = [1, -1]
        self.combinations = np.array(
            list(itertools.product(self.phase_alphabet, repeat=self.N_sub)))

    def prepare(self, X: np.ndarray) -> None:
        from comnumpy.exceptions import ShapeError  # local import (D36)
        if np.ndim(X) != 2:
            raise ShapeError(
                f"PtsPaprReductor expects the 2-D Block layout (T, F), "
                f"got shape {np.shape(X)} -- reshape with "
                f"Serial2Parallel first.")

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

    def find_optimal_combination(self, x_m_array: np.ndarray) -> tuple[np.ndarray, float]:
        papr_list = np.zeros(len(self.combinations))

        for index, combination in enumerate(self.combinations):
            x_m_temp = np.dot(x_m_array, combination)
            papr_list[index] = compute_papr(x_m_temp)

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
