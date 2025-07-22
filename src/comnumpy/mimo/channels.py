import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from comnumpy.core import Processor
from comnumpy.core.utils import compute_sigma2
from .validators import validate_input


def apply_correlation(H, Rx=None, Ry=None):
    if Rx:
        Rx_sqrt = np.linalg.cholesky(Rx)
        H = np.matmul(H, Rx_sqrt)

    if Ry:
        Ry_sqrt = np.linalg.cholesky(Ry)
        H = np.matmul(H, Ry_sqrt)
    return H


@dataclass
class MIMOChannel(Processor):

    r"""
    A base class for modeling Multiple-Input Multiple-Output (MIMO) channels.

    This class provides a framework for simulating MIMO communication channels,
    including methods for setting channel matrices, configuring signal-to-noise
    ratios (SNR), generating noise, and processing input signals.

    Signal Model
    ------------

    .. math ::
    
        \mathbf{y}[n] = \sum_{l=0}^{L}\mathbf{H}[l]x[n-l] + \mathbf{b}[n]

    where

    * :math:`\mathbf{H}[l]` is a channel matrix of size :math:`N_r \times N_t` corresponding the :math:`l^{th}` channel tap,
    * :math:`\mathbf{x}[n]` is a :math:`N_t` vector containing the transmitted data,
    * :math:`\mathbf{b}[n]\sim \mathcal{N}_c(\mathbf{0},\sigma^2\mathbf{I}_{N_r})` is a :math:`N_r` vector containing the additive white Gaussian noise.        

    Attributes
    ----------
    P : float
        Transmit power.
    sigma2s : float, optional
        Signal power (default=1)
    noise_value: float, optional
        Noise value (default = 1)
    noise_unit : str, optional
        Noise unit (default="sigma2")
    estimate_sigma2s : bool
        Flag to estimate signal power.
    H_list : Optional[np.array]
        List of channel matrices for each tap. Each matrice should have equal dimension.
    mode : Literal["naive", "optimized"]
        Processing mode, either "naive" or "optimized" (defaut="optimized"). The optimized mode compute the channel output using a single matrix multiplication while the naive implementation uses for loops.
    extend : bool
        Flag to extend the input signal.
    seed : int
        Seed for random number generation.
    name : str
        Name of the processor.
    """

    sigma2s: float = 1
    noise_value: float = 1.
    noise_unit: Literal["sigma2", "snr", "snr_dB", "snr_dBm"] = "sigma2"
    estimate_sigma2s: bool = False
    H_list: Optional[np.array] = None
    mode: Literal["naive", "optimized"] = "optimized"
    extend: bool = True
    seed: int = None
    name: str = "mimo channel"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        if isinstance(self.H_list, list):
            self.N_r, self.N_t = self.H_list[0].shape
            self.N_tap = len(self.H_list)

            for n in range(1, self.N_tap):
                if self.H_list[n].shape != self.H_list[0].shape:
                    raise ValueError("Channel dimension does not match")

        self.set_channel_matrix()

    @property
    def H(self):
        return self.H_list

    @property
    def sigma2n(self):
        return compute_sigma2(self.noise_value, self.noise_unit, self.sigma2s)
        
    def set_channel_matrix(self):
        H_matrix = np.zeros((self.N_r, self.N_tap * self.N_t), dtype=complex)
        for indice in range(self.N_tap):
            H_matrix[:, indice * self.N_t:(indice + 1) * self.N_t] = self.H_list[indice]

        self.H_matrix = H_matrix

    def update_sigma2s(self, X):
        self.sigma2 = np.sum(np.abs(X)**2) / np.prod(X.shape)

    def noise_rvs(self, N, X=None):

        if self.estimate_sigma2s:
            self.update_sigma2s(X)

        sigma2n = self.sigma2n
        scale = np.sqrt(sigma2n/2)
        size = (self.N_r, N)
        B_r = self.rng.normal(scale=scale, size=size)
        B_i = self.rng.normal(scale=scale, size=size)
        B = B_r + 1j*B_i
        self._B = B

    def info(self):
        print(f"* MIMO Channel ({self.N_tap} tap(s)):")
        for index in range(self.N_tap):
            H = self.H_list[index]
            print(f"tap {index}:\n{H}")
            condition_number = np.linalg.cond(H)
            _, S, _ = np.linalg.svd(H)
            norm = np.linalg.norm(H)
            print("Condition Number=", condition_number)
            print(f"singular value={S}")
            print(f"norm={norm}")

    def forward(self, X: np.ndarray) -> np.ndarray:

        validate_input(X, self.N_t)
        L = self.N_tap
        N = X.shape[1]

        if self.mode == "naive":
            N = X.shape[1]
            size = N + self.extend*(self.N_tap-1)
            Y = np.zeros((self.N_r, size), dtype=complex)

            for n in range(size):
                for m in range(self.N_tap):
                    if ((n - m) >= 0) and ((n - m) < N):
                        H = self.H_list[m]
                        Y[:, n] += np.matmul(H, X[:, n - m])

        if self.mode == "optimized":
            # create X_matrix
            if self.extend:
                X_matrix = np.zeros((L*self.N_t, N + L-1), dtype=complex)
                for l_index in range(L):
                    X_matrix[l_index*self.N_t:(l_index+1)*self.N_t, l_index:l_index+N] = X
            else:
                X_matrix = np.zeros((L*self.N_t, N), dtype=complex)
                for l_index in range(L):
                    X_matrix[l_index*self.N_t:(l_index+1)*self.N_t, l_index:] = X[:, :N-l_index]

            Y = np.matmul(self.H_matrix, X_matrix)

        if self.sigma2n > 0:
            N = Y.shape[1]
            self.noise_rvs(N, X=X)
            Y += self._B

        return Y


@dataclass
class FlatFadingRayleighChannel(MIMOChannel):
    r"""
    A processor for simulating flat fading Rayleigh channels.

    This class extends the MIMOChannel class to model flat fading Rayleigh channels,
    where the channel matrix is generated based on Gaussian noise with optional
    correlation matrices.

    Signal Model
    ------------

    .. math ::
    
        \mathbf{y}[n] = \mathbf{H}x[n] + \mathbf{b}[n]

    where

    * :math:`\mathbf{H}` is a channel matrix of size :math:`N_r \times N_t`,
    * :math:`\mathbf{x}[n]` is a :math:`N_t` vector containing the transmitted data,
    * :math:`\mathbf{b}[n]` is a :math:`N_r` vector containing the additive white Gaussian noise with distribution :math:`\mathcal{N}_c(\mathbf{0},\sigma^2\mathbf{I}_{N_r})`.


    Attributes
    ----------
    N_r : int
        Number of receive antennas.
    N_t : int
        Number of transmit antennas.
    noise_value: float, optional
        Noise value (default = 1)
    noise_unit : str, optional
        Noise unit (default="sigma2")
    scale : float
        Scaling factor for the channel matrix.
    Rx : Optional[np.ndarray]
        Receive correlation matrix.
    Ry : Optional[np.ndarray]
        Transmit correlation matrix.
    name : str
        Name of the processor.

    """
    N_r: int = 2
    N_t: int = 2
    sigma2_s: float = 1
    scale: float = 1
    Rx: Optional[np.ndarray] = None
    Ry: Optional[np.ndarray] = None
    name: str = "flat_fading_rayleight_channel"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.N_tap = 1
        self.channel_matrix_rvs()

    @property
    def H(self):
        return self.H_list[0]

    def channel_matrix_rvs(self):
        r"""
        Generate the channel matrix according to the following model

        .. math ::

            \mathbf{H} = \frac{1}{\sqrt{N_t}}\mathbf{R}_y^{1/2}\mathbf{W}\mathbf{R}_x^{1/2}

        where

        * :math:`\mathbf{W}` is a channel matrix of size :math:`N_r \times N_t` whose elements :math:`\mathbf{w}_{uv}[l]` follow a Gaussian distribution :math:`\mathcal{N}_c(\mathbf{0},\mathbf{I}_{N_r})`.
        * :math:`\mathbf{R}_x` is the transmitter covariance matrix,
        * :math:`\mathbf{R}_y` is the receiver covariance matrix.

        Reference
        ---------
        * Dupuy, Florian, and Philippe Loubaton. "On the capacity achieving covariance matrix for frequency selective MIMO channels using the asymptotic approach." IEEE Transactions on Information Theory 57.9 (2011): 5737-5753.
        """
        size = (self.N_r, self.N_t)
        H = (self.rng.normal(scale=self.scale, size=size) + 1j * self.rng.normal(scale=self.scale, size=size))
        H = np.sqrt(self.sigma2_s/self.N_t) * apply_correlation(H, self.Rx, self.Ry)
        self.H_list = [H]
        self.set_channel_matrix()


@dataclass
class SelectiveRayleighChannel(MIMOChannel):
    r"""
    A processor for simulating selective Rayleigh channels.

    This class extends the MIMOChannel class to model selective Rayleigh channels,
    where multiple taps are considered, each with its own channel matrix. The
    channel matrices can be generated with optional correlation matrices.

    Signal Model
    ------------

    .. math ::
    
        \mathbf{y}[n] = \sum_{l=0}^{L}\mathbf{H}[l]x[n-l] + \mathbf{b}[n]

    where

    * :math:`\mathbf{H}[l]` is a channel matrix of size :math:`N_r \times N_t` corresponding the :math:`l^{th}` channel tap,
    * :math:`\mathbf{x}[n]` is a :math:`N_t` vector containing the transmitted data,
    * :math:`\mathbf{b}[n]\sim \mathcal{N}_c(\mathbf{0},\sigma^2\mathbf{I}_{N_r})` is a :math:`N_r` vector containing the additive white Gaussian noise.        

    Attributes
    ----------
    N_r : int
        Number of receive antennas.
    N_t : int
        Number of transmit antennas.
    N_tap : int
        Number of taps.
    noise_value: float, optional
        Noise value (default = 1)
    noise_unit : str, optional
        Noise unit ["sigma2", "snr", "snr_dB", "snr_dBm"] (default="sigma2")
    scale : float
        Scaling factor for the channel matrices.
    Rx : Optional[np.ndarray]
        Receive correlation matrix.
    Ry : Optional[np.ndarray]
        Transmit correlation matrix.
    name : str
        Name of the processor.
    """
    N_r: int = 2
    N_t: int = 2
    N_tap: int = 2
    sigma2_s: float = 1
    scale: float = 1
    Rx: Optional[np.ndarray] = None
    Ry: Optional[np.ndarray] = None
    name: str = "selective_rayleight_channel"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.channel_matrix_rvs()

    def channel_matrix_rvs(self):
        r"""
        Generate the channel matrix according to the following model

        .. math ::

            \mathbf{H}[l] = \frac{1}{\sqrt{N_t}}\mathbf{R}_y^{1/2}\mathbf{W}[l]\mathbf{R}_x^{1/2}

        where

        * :math:`\mathbf{W}[l]` is a channel matrix of size :math:`N_r \times N_t` whose elements :math:`\mathbf{w}_{uv}[l]` follow a Gaussian distribution :math:`\mathcal{N}_c(\mathbf{0},\mathbf{I}_{N_t})`.
        * :math:`\mathbf{R}_x` is the transmitter covariance matrix,
        * :math:`\mathbf{R}_y` is the receiver covariance matrix.

        Reference
        ---------
        * Dupuy, Florian, and Philippe Loubaton. "On the capacity achieving covariance matrix for frequency selective MIMO channels using the asymptotic approach." IEEE Transactions on Information Theory 57.9 (2011): 5737-5753.
        """
        H_list = []
        size = (self.N_t, self.N_t)
        for indice in range(self.N_tap):
            H_temp = self.rng.normal(scale=self.scale, size=size) + 1j * self.rng.normal(scale=self.scale, size=size)
            H_temp = np.sqrt(self.sigma2_s/self.N_t) * apply_correlation(H_temp, self.Rx, self.Ry)  # see equation (3)
            H_list.append(H_temp)

        self.H_list = H_list
        self.set_channel_matrix()

