import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass

from comnumpy.core.processors import Processor


@dataclass
class PhaseNoise(Processor):
    r"""
    A class representing a Phase Noise channel.

    This class models a phase noise effect in a channel, where the phase of the signal
    is altered by a random process. The phase noise is characterized by a variance
    specified by sigma squared (sigma2).

    .. math ::

        y[n] = x[n]e^{j \phi[n]}

    where :

    .. math ::

        \phi[n] = \sum_{k=0}^{n} \phi_k

    with :math:`\phi_k \sim \mathcal{N}(0, \sigma^2)`.

    Attributes
    ----------

    sigma2 : float
        The variance of the phase noise.
    name : str
        Name of the channel instance. Default is "phase noise".
    
    """
    sigma2: float
    name: str = "phase noise"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def noise_rvs(self, X):
        N = len(X)
        sigma2 = self.sigma2
        scale = np.sqrt(sigma2)
        noise = self.rng.normal(loc=0, scale=scale, size=N)
        self._b = np.cumsum(noise)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # generate phase noise samples
        self.noise_rvs(x)
        y = x * np.exp(1j*self._b)
        return y

@dataclass
class PMD(Processor):
    r"""
    A class representing a Polarization Mode Dispersion (PMD) channel.

    This class models the Polarization Mode Dispersion effect in dual-polarization optical signals.
    PMD occurs due to differential group delay (DGD) between the two principal polarization axes
    of an optical fiber. The PMD effect is computed in the frequency domain using rotation matrices.

    .. math ::

        \mathbf{X}_{out} = \mathbf{H}_2(\theta) \cdot \mathbf{J}_{PMD} \cdot \mathbf{H}_1(\theta) \cdot \mathbf{X}_{in}

    where :math:`\mathbf{H}_1` and :math:`\mathbf{H}_2` are rotation matrices that transform between
    the observed and principal polarization axes, and :math:`\mathbf{J}_{PMD}` applies the differential
    phase delay in the frequency domain.

    Attributes
    ----------
    t_dgd : float
        Differential group delay (DGD) between the two polarization axes [seconds].
    fs : float
        Sampling frequency [Hz].
    theta : float
        Rotation angle of the principal axes relative to the observed axes [radians]. Default is 0.0.
    name : str
        Name of the channel instance. Default is "PMD".

    References
    ----------
    * [1] Ip, Ezra, and Joseph M. Kahn. "Digital equalization of chromatic dispersion and
      polarization mode dispersion." Journal of Lightwave Technology 25.8 (2007): 2033-2043.

    """
    t_dgd: float
    fs: float
    theta: float = 0.0
    name: str = "PMD"

    def rotation_matrix(self, theta: float) -> np.ndarray:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ], dtype=np.complex128)

    def forward(self, X_in: np.ndarray, update_params: bool = False) -> np.ndarray:
        N = X_in.shape[1]
        w = 2 * np.pi * np.linspace(-self.fs / 2, self.fs / 2, N, endpoint=False)

        X_in_freq = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X_in, axes=1), axis=1), axes=1)

        # rotation into principal axes
        H1 = self.rotation_matrix(self.theta)
        X_rot1 = H1 @ X_in_freq

        J_PMD = np.vstack((
            np.exp(1j * w * self.t_dgd / 2),
            np.exp(-1j * w * self.t_dgd / 2)
        ))
        X_bir_freq = J_PMD * X_rot1

        # rotation back to observed axes
        H2 = self.rotation_matrix(-self.theta)
        X_rot2 = H2 @ X_bir_freq

        X_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_rot2, axes=1), axis=1), axes=1)

        return X_out


@dataclass
class PMD_(Processor):
    r"""
    A class representing a Polarization Mode Dispersion (PMD) channel.

    This class models the Polarization Mode Dispersion effect in dual-polarization optical signals.
    PMD occurs due to differential group delay (DGD) between the two principal polarization axes
    of an optical fiber. The PMD effect is computed in the frequency domain using rotation matrices.

    .. math ::

        \mathbf{X}_{out} = \mathbf{H}_2(\theta) \cdot \mathbf{J}_{PMD} \cdot \mathbf{H}_1(\theta) \cdot \mathbf{X}_{in}

    where :math:`\mathbf{H}_1` and :math:`\mathbf{H}_2` are rotation matrices that transform between
    the observed and principal polarization axes, and :math:`\mathbf{J}_{PMD}` applies the differential
    phase delay in the frequency domain.

    Attributes
    ----------
    t_dgd : float
        Differential group delay (DGD) between the two polarization axes [seconds].
    fs : float
        Sampling frequency [Hz].
    theta : float
        Rotation angle of the principal axes relative to the observed axes [radians]. Default is 0.0.
    name : str
        Name of the channel instance. Default is "PMD".

    References
    ----------
    * [1] Ip, Ezra, and Joseph M. Kahn. "Digital equalization of chromatic dispersion and
      polarization mode dispersion." Journal of Lightwave Technology 25.8 (2007): 2033-2043.

    """
    t_dgd: float
    fs: float
    name: str = "PMD"

    def forward(self, X_in: np.ndarray, update_params: bool = False) -> np.ndarray:
        N = X_in.shape[1]
        w = 2 * np.pi * np.linspace(-self.fs / 2, self.fs / 2, N, endpoint=False)

        X_in_freq = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(X_in, axes=1), axis=1), axes=1)

        J_PMD = np.vstack((
            np.exp(1j * w * self.t_dgd / 2),
            np.exp(-1j * w * self.t_dgd / 2)
        ))
        X_bir_freq = J_PMD * X_in_freq

        X_out = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_bir_freq, axes=1), axis=1), axes=1)

        return X_out

    
@dataclass
class PDL(Processor):
    r"""
    A class representing Polarization-Dependent Loss (PDL) in optical communication systems.

    This class models the Polarization-Dependent Loss effect on a dual-polarization signal.
    PDL causes differential attenuation between the two principal polarization states of light.
    The effect is computed using rotation matrices to align the signal with the principal PDL axes.

    .. math ::

        \mathbf{X}_{out} = \mathbf{H}(\theta) \cdot \mathbf{L} \cdot \mathbf{H}^{-1}(\theta) \cdot \mathbf{X}_{in}

    where :math:`\mathbf{H}(\theta)` is a rotation matrix, :math:`\mathbf{L}` is a diagonal matrix
    representing the differential loss between the two polarization states, and :math:`\mathbf{X}_{in}`
    is the input polarization state vector.

    Attributes
    ----------
    gamma_db : float
        Polarization-Dependent Loss in decibels [dB].
    theta : float
        Rotation angle of the principal PDL axes relative to the observed axes [radians].
    name : str
        Name of the PDL instance. Default is "PDL".
    """
    gamma_db: float
    theta: float
    name: str = "PDL"

    def __post_init__(self):
        self.gamma = (10 ** (self.gamma_db / 10) - 1) / (10 ** (self.gamma_db / 10) + 1)
    
    def rot_mat(self, theta: float) -> np.ndarray:
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        rot_mat = self.rot_mat(self.theta)
        rot_mat_inv = self.rot_mat(-self.theta)
        pdl_vect = np.array([[(1 + self.gamma) ** 0.5], [(1 - self.gamma) ** 0.5]])
        
        H1 = np.matmul(rot_mat_inv, X_in)
        H2 = pdl_vect * H1
        X_out = np.matmul(rot_mat, H2)
        return X_out

@dataclass
class PDL_(Processor):
    r"""
    A class representing Polarization-Dependent Loss (PDL) in optical communication systems.

    This class models the Polarization-Dependent Loss effect on a dual-polarization signal.
    PDL causes differential attenuation between the two principal polarization states of light.
    The effect is computed using rotation matrices to align the signal with the principal PDL axes.

    .. math ::

        \mathbf{X}_{out} = \mathbf{H}(\theta) \cdot \mathbf{L} \cdot \mathbf{H}^{-1}(\theta) \cdot \mathbf{X}_{in}

    where :math:`\mathbf{H}(\theta)` is a rotation matrix, :math:`\mathbf{L}` is a diagonal matrix
    representing the differential loss between the two polarization states, and :math:`\mathbf{X}_{in}`
    is the input polarization state vector.

    Attributes
    ----------
    gamma_db : float
        Polarization-Dependent Loss in decibels [dB].
    theta : float
        Rotation angle of the principal PDL axes relative to the observed axes [radians].
    name : str
        Name of the PDL instance. Default is "PDL".
    """
    gamma_db: float
    name: str = "PDL"

    def __post_init__(self):
        self.gamma = (10 ** (self.gamma_db / 10) - 1) / (10 ** (self.gamma_db / 10) + 1)

    def forward(self, X_in: np.ndarray) -> np.ndarray:
        pdl_vect = np.array([[(1 + self.gamma) ** 0.5], [(1 - self.gamma) ** 0.5]])
        X_out = pdl_vect * X_in
        return X_out

@dataclass
class SOP(Processor):
    r"""
    A class representing State of Polarization (SOP) Drift in optical communication systems.

    This class models the random drift of the polarization state due to birefringence variations
    in the optical fiber, typically caused by temperature fluctuations and mechanical stress.
    The SOP drift is simulated using random rotations in the polarization space via the 
    matrix exponential of skew-Hermitian matrices constructed from Pauli matrices.

    .. math ::

        \mathbf{Y}[:,i] = \text{expm}\left(-j \sum_{k=1}^{3} \sigma_k(i) \mathbf{P}_k\right) \cdot \mathbf{X}[:,i]

    where :math:`\sigma_k(i)` are random variables drawn from a normal distribution with variance
    :math:`\sigma^2 = 2\pi \cdot \text{linewidth} \cdot T_{\text{symb}}`, and :math:`\mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3`
    are the three Pauli matrices.

    Attributes
    ----------
    T_symb : float
        Symbol period (time between symbols) [seconds].
    linewidth : float
        Polarization linewidth [Hz].
    name : str
        Name of the SOP drift instance. Default is "SOP_Drift".

    """
    T_symb: float
    linewidth: float
    name: str = "SOP_Drift"

    def get_pauli_matrices(self) -> tuple:
        """Return the three Pauli matrices."""
        p1 = np.array([[1, 0], [0, -1]])
        p2 = np.array([[0, 1], [1, 0]])
        p3 = np.array([[0, -1j], [1j, 0]])
        return p1, p2, p3

    def forward(self, X: np.ndarray) -> np.ndarray:
        sigma = 2 * np.pi * self.linewidth * self.T_symb
        _, N = X.shape
        Y = np.zeros_like(X)

        sigma_vect = np.random.normal(loc=0, scale=np.sqrt(sigma), size=(3, N))
        p1, p2, p3 = self.get_pauli_matrices()
        
        rot_mat = np.eye(2)
        for i in range(N):
            matrix_to_exponentiate = -1j * (sigma_vect[0, i] * p1 + sigma_vect[1, i] * p2 + sigma_vect[2, i] * p3)
            rot_mat = np.matmul(expm(matrix_to_exponentiate), rot_mat)
            Y[:, i] = np.matmul(rot_mat, X[:, i])

        return Y



@dataclass
class SOP_(Processor):
    """
    Stateful SOP drift model (paper-correct, optimized).

    J_{k+1} = exp(-j * alpha_k · sigma) J_k
    alpha_k ~ N(0, 2π Δp T_symb I_3)
    """

    T_symb: float
    linewidth: float
    segments: int         
    name: str = "SOP_Drift"

    def __post_init__(self):
        # Wiener variance (Eq. 5)
        self.sigma2 = 2 * np.pi * (self.linewidth/self.segments) * self.T_symb

        # Pauli matrices
        self.p1 = np.array([[1, 0], [0, -1]], dtype=complex)
        self.p2 = np.array([[0, 1], [1, 0]], dtype=complex)
        self.p3 = np.array([[0, -1j], [1j, 0]], dtype=complex)

        # Persistent Jones matrix (J_0)
        self.J = np.eye(2, dtype=complex)

    def _su2_expm(self, alpha: np.ndarray) -> np.ndarray:
        """
        Closed-form exp(-j * alpha · Pauli)
        """
        ax, ay, az = alpha
        theta = np.sqrt(ax**2 + ay**2 + az**2)

        if theta < 1e-14:
            return np.eye(2, dtype=complex)

        nx, ny, nz = alpha / theta
        P = nx*self.p1 + ny*self.p2 + nz*self.p3

        return (
            np.cos(theta) * np.eye(2)
            - 1j * np.sin(theta) * P
        )

    def _step(self):
        """
        One SOP Wiener increment.
        """
        alpha = np.random.normal(
            scale=np.sqrt(self.sigma2),
            size=3
        )
        U = self._su2_expm(alpha)
        self.J = U @ self.J

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply SOP drift symbol-by-symbol.
        """
        Y = np.empty_like(X)
        _, N = X.shape

        for k in range(N):
            self._step()
            Y[:, k] = self.J @ X[:, k]

        return Y

