import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Union, Optional, Literal
from scipy import signal
from scipy.linalg import toeplitz
from scipy.optimize import least_squares
from comnumpy.core import Processor, Recorder
from .utils import hard_projector
from .processors import Amplifier
from .validators import validate_data


@dataclass
class TrainedBasedMixin():
    def __post_init__(self):
        validate_data(self.target_data)

    def get_target_data(self):
        """
        Retrieve the target data associated with the model.

        Returns:
            numpy.ndarray or array-like: The target data.

        Raises:
            TypeError: If `target_data` is neither a numpy array nor
            an instance of Recorder.
        """
        if isinstance(self.target_data, np.ndarray):
            return self.target_data
        elif isinstance(self.target_data, Recorder):
            return self.target_data.get_data()
        else:
            raise TypeError("target_data must be a numpy array or Recorder.")


@dataclass
class DCCorrector(Processor):
    r"""
    A class for correcting the mean of a dataset along a specified axis.

    This class adjusts the input data so that its mean along the specified axis matches a target value.
    This is useful for normalizing data or removing bias.


    Signal Model
    ------------
    
    .. math::

       y[n] = x[n] + \alpha

    where the coefficient :math:`\alpha` is adjusted to meet the DC constraint

    Attributes
    ----------
    value : float
        The target mean value to which the data should be adjusted (default: 0)
    axis : int
        The axis along which to compute the mean.
    name : str
        Name of the mean corrector instance.

    """
    value: float = 0.0
    axis: int = 0
    name: str = "mean_corrector"

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_mean = np.mean(x, axis=self.axis)
        y = x - x_mean + self.value
        return y


@dataclass
class Normalizer(Amplifier):
    r"""
    A class for normalizing data based on the specified normalization type.

    Signal Model
    ------------

    .. math::

       y[n] = \alpha x[n]

    where the normalization coefficient :math:`\alpha` depends on the normalization technique.

    - **'amp'**: Scales the signal by a constant factor :math:`\alpha = \text{value}`.

      .. math::

         \alpha = \text{value}

    - **'abs'**: Normalizes the signal by the maximum absolute value.

      .. math::

         \alpha = \frac{\text{value}}{\max(|x[n]|)}

    - **'var'**: Normalizes the signal based on its variance.

      .. math::

         \alpha = \sqrt{\frac{\text{value}}{\sigma_x^2}}

    - **'max'**: Normalizes the signal by the maximum absolute value of the real and imaginary parts.

      .. math::

         \alpha = \frac{\text{value}}{\max(\max(|\text{Re}(x[n])|), \max(|\text{Im}(x[n])|))}


    Attributes
    ----------
    method : str
        Type of normalization to be applied. Supported types are 'amp' for scaling coefficient, 'max' for maximum value normalization, 'var' for variance-based normalization, and 'abs' for absolute maximum value normalization.
    value : float, optional
        The target value for the normalization type. Default is 1.0.


    Example
    -------
    >>> normalizer = Normalizer(method='max', value=2.0)
    >>> X = np.array([1, 2, 3, 4])
    >>> Y = normalizer.forward(X)
    >>> print(Y)
    [0.5 1.  1.5 2. ]
    """
    method: Literal['amp', 'abs', 'var', 'max'] = "amp"
    value: float = 1.
    gain: float = 1
    
    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("The value for normalization must be positive.")
        self.gain = 1

    def prepare(self, X):
        match self.method:
            case "amp":
                gain = self.value
            case "abs":
                gain = self.value/np.max(np.abs(x))
            case "var":
                variance = np.var(x)
                gain = np.sqrt(self.value)/np.sqrt(variance)
            case "max":
                max_value = np.max([np.max(np.abs(np.real(x))), np.max(np.abs(np.imag(x)))])
                gain = self.value/max_value
            case _:
                gain = 1

        self.gain = gain


@dataclass
class BlindIQCompensator(Processor):

    r"""
    Blind IQ compensator based on the diagonalisation of the augmented covariance matrix.
    This compensation assumes the circularity of compensated signal

    Signal Model
    ------------

    This class implements the following transformation 

    .. math ::

        y[n] = \alpha \Re e(x[n]) + \beta \Im m(x[n])
    
    Algorithm
    ---------

    such as :

    .. math ::

        E[\Re e^2(y[n])] &= E[\Im m^2(y[n])] = 1\\
        E[\Re e(y[n])\Im m(y[n])] & = 0

    """
    should_fit: bool = True
    coef: float = 1
    name: str = "iq_compensator"

    def __post_init__(self):
        self.alpha = 1
        self.beta = 0

    def fit(self, x):
        N = len(x)
        X = np.vstack([x.real, x.imag])

        # compute covariance matrix
        R = (1/N) * np.matmul(X, np.transpose(X))

        # perform eigenvalue decomposition
        V, U = LA.eig(R)

        # perform whitening
        D = np.diag(1/np.sqrt(V))
        M = np.matmul(D, np.transpose(U))

        self.alpha = M[0, 0] + 1j * M[1, 0]
        self.beta = M[0, 1] + 1j * M[1, 1]

    """
    def fit(self, x: np.ndarray) -> np.ndarray:
        # implementation of the gram schmit orthogonalization
        # Reference
        # ---------
        # * [1] I. Fatadin, S. J. Savory and D. Ives,  "Compensation of Quadrature Imbalance in an Optical QPSK Coherent Receiver," 
        # in IEEE Photonics Technology Letters, vol. 20, no. 20, pp. 1733-1735, Oct.15, 2008, doi: 10.1109/LPT.2008.2004630.
        x_r = np.real(x)
        r_11 = np.mean(x_r**2)
        y_r = x_r / np.sqrt(r_11)

        x_i = np.imag(x)
        r_12 = np.mean(x_r * x_i)
        x_ii = x_i - r_12*x_r/r_11
        r_22 = np.mean(x_ii**2)
        y_i = x_ii / np.sqrt(r_22)
        y = y_r + 1j*y_i
        return y
    """

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.should_fit:
            self.fit(x)

        coef = np.sqrt(self.coef/2)
        y = coef*(self.alpha * x.real + self.beta * x.imag)
        return y


@dataclass
class BlindCFOCompensator(Processor):
    r"""
    Blind CFO compensator based on the maximisation of the periodogram
    of 4th order statistic.

    Signal Model
    ------------

    .. math::
        y[n] = x[n]e^{-j\widehat{\omega}_0 n}

    Algorithm
    ---------

    .. math::
        \widehat{\omega}_0 = \frac{1}{4} \arg \max_{\omega} \left|\sum_{n=0}^{N-1} x^4[n]e^{-j\omega n}\right|^2

    The maximisation is performed using the Newton Algorithm

    Attributes
    ----------
    w0_init : float
        Initialisation in rad/samples
    N_iter : int
        Number of iterations
    method : str
        method used for maximisation

    """
    w0_init: float = 0.0
    N_iter: int = 3
    should_fit: bool = True
    grid_search: bool = True
    save_history: bool = False
    method: Literal["grad", "newton"] = "newton"
    step_size: float = 1e-8
    grid_search_tuple: tuple = (-0.1, 0.1, 0.0001)
    name: str = "cfo_compensator"

    def __post_init__(self):
        self.grid_search_array = np.arange(self.grid_search_tuple[0], self.grid_search_tuple[1], self.grid_search_tuple[2])
        self.history = []

    def loss(self, x, w):
        N = len(x)
        x4 = x**4
        dtft = self.compute_dtft(x4, w)
        return (np.abs(dtft)**2)/N

    def compute_dtft(self, x, w):
        N = len(x)
        N_vect = np.arange(N)
        dtft = np.sum(x*np.exp(-1j*w*N_vect))
        return dtft
    
    def callback(self, intermediate_result):
        if self.save_history:
            self.history.append(intermediate_result)

    def fit(self, x, w0):
        w = 4*w0
        N = len(x)
        x4 = x**4
        N_vect = np.arange(N)
        step_size = self.step_size

        if self.grid_search:
            w_vect = 4*self.grid_search_array
            cost_vect = np.zeros(len(w_vect))

            for index, w in enumerate(w_vect):
                cost_vect[index] = self.loss(x, w)

            index_max = np.argmax(cost_vect)
            w = w_vect[index_max]
            self.callback(4*w)

        for _ in range(self.N_iter):

            if self.method == "grad":
                dtft = self.compute_dtft(x4, w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                h = step_size * grad.real

            if self.method == "newton":
                dtft = self.compute_dtft(x4, w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                dtft_diff2 = self.compute_dtft(-(N_vect**2)*x4, w)
                grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                J = (2/N) * (np.real(dtft_diff2*np.conj(dtft)) + np.abs(dtft_diff)**2)
                h = -grad.real/J.real

            w = w + h
            self.callback(4*w)

        self.w0 = np.real(w)/4

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        N_vect = np.arange(N)

        if self.should_fit:
            self.fit(x, self.w0_init)

        x = x*np.exp(-1j*self.w0*N_vect)
        return x


@dataclass
class BlindPhaseCompensation(Processor):
    """
    A class to perform blind phase compensation on a given signal.

    Attributes:
        alphabet (np.ndarray): The alphabet used for phase compensation.
        theta (float): Initial phase angle. Default is 0.
        name (str): Name of the processor. Default is "phase".
    """
    alphabet: np.ndarray
    theta: float = 0.0
    should_fit: bool = True
    name: str = "phase correction"

    def cost(self, theta: float, x: np.ndarray) -> np.ndarray:
        y = x * np.exp(1j * theta)
        s, y_est = hard_projector(y, self.alphabet)
        error = y - y_est
        error_real = np.hstack([np.real(error), np.imag(error)])
        return error_real

    def fit(self, X: np.ndarray):
        res = least_squares(self.cost, self.theta, args=(X,))
        self.theta = res.x[0]

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.should_fit:
            self.fit(X)
        Y = X * np.exp(1j * self.theta)
        return Y


@dataclass
class DataAidedFIRCompensator(Processor):
    """
    Data_Aided_FIR()

    Data aided estimation of a FIR filter using ZF estimation.
    """

    h: np.array
    target_data = Union[np.array, Recorder]
    should_fit: bool = True
    name: str = "data_aided_fir"

    def fit(self, x, x_target):
        L = len(x)
        first_col = x_target
        H = toeplitz(first_col, np.zeros(L))
        self.h = np.matmul(LA.pinv(H), x)

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.should_fit:
            x_target = self.get_target_data()
            self.fit(x, x_target)

        L = len(self.h)
        y, _ = signal.deconvolve(np.hstack([x, np.zeros(L-1)]), self.h)
        return y


@dataclass
class TrainedBasedPhaseCompensator(TrainedBasedMixin, Processor):

    target_data: Union[np.array, Recorder]
    name: str = "data_aided_phase"

    def __post__init__(self):
        self.theta = 0

    def fit(self, x, x_target):
        self.theta = np.angle(np.sum(np.conj(x)*x_target))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_target = self.get_target_data()
        self.fit(x, x_target)
        return x*np.exp(1j*self.theta)


@dataclass
class TrainedBasedComplexGainCompensator(TrainedBasedMixin, Processor):
    """This class performs the complex-gain channel estimation and compensation

    Attributes
    ----------
    recorder_preamble : ndarray
        The recorded preamble used to compute the complex gain of the channel.
    position : int
        The place of this recorded preamble in the input signal.
    """
    target_data = Union[np.array, Recorder]
    position: int = 0
    name: str = "complex_gain_compensator"

    def forward(self, x: np.ndarray) -> np.ndarray:
        ref_preamble = self.get_target_data()
        N_preamble = len(ref_preamble)

        if N_preamble > 0:
            x_preamble = x[self.position:self.position+N_preamble]
            x_preamble_resized = np.resize(x_preamble, (N_preamble, 1))
            x_preamble_pinv = np.linalg.pinv(x_preamble_resized)
            complex_correction = np.dot(x_preamble_pinv, ref_preamble)
        else:
            complex_correction = 1

        y = x*complex_correction
        return y


@dataclass
class TrainedBasedSimpleSynchronizer(TrainedBasedMixin, Processor):
    """ Implements a simple synchronizer using cross-correlation to determine time delay and scaling between signals.

        Attributes
        ----------
        target_data : ndarray
            The reference preamble signal to which the input signals will
            be synchronized.
        scale_correction : bool, optional
            If True, applies a scaling correction based on the peak of the
            cross-correlation. Default is True.
        save_cross_corr : bool, optional
            If True, saves the computed cross-correlation and the associated
            lag vector. Default is True.
        signal_len : integer, optional
            Truncates the signal to the given length after synchronization
        name : str, optional
            Name of the synchronizer instance. Default is "synchronizer".
    """
    target_data = Union[np.array, Recorder]
    scale_correction: bool = True
    save_cross_correlation: bool = True
    signal_len: Optional[int] = None
    name: str = "synchronizer"

    def __post_init__(self):
        self.delay = None
        self.scale = 1
        self.cross_corr = None
        self.n_vect = None

    def fit(self, x, x_preamble):
        N = len(x)
        N_preamble = len(x_preamble)

        x_preamble_padded = np.zeros(N, dtype=x.dtype)
        x_preamble_padded[:N_preamble] = x_preamble

        # compute cross correlation
        cross_corr = np.correlate(x, x_preamble_padded, mode='full')
        cross_corr *= (1/N_preamble)
        n_vect = np.arange(len(cross_corr)) - (N - 1)

        # Find the time delay: the index of the maximum cross-correlation
        # minus the length of x minus 1
        index_max = np.argmax(np.abs(cross_corr)**2)
        value_max = cross_corr[index_max]

        self.delay = n_vect[index_max]
        if self.scale_correction:
            self.scale = value_max

        # save correlation if needed
        if self.save_cross_correlation:
            self.cross_corr = cross_corr
            self.n_vect = n_vect

    def plot(self):
        plt.figure()
        plt.plot(self.n_vect, np.abs(self.cross_corr))
        plt.title('Cross-correlation magnitude')
        plt.xlabel('Lag')
        plt.ylabel('Magnitude')
        plt.grid(True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_preamble = self.get_target_data()

        self.fit(x, x_preamble)
        if self.signal_len:
            y = self.scale*x[self.delay:self.delay+self.signal_len]
        else:
            y = self.scale*x[self.delay:]
        return y


@dataclass
class TrainedBasedFineSynchronizer(TrainedBasedMixin, Processor):

    """ Implements a simple synchronizer using cross-correlation to determine time delay and scaling between signals.

        Attributes
        ----------
        target_data : ndarray
            The reference preamble signal to which the input signals will
            be synchronized.
        up_factor : int
            The upsampling factor made before cross-correlation
        scale_correction : bool, optional
            If True, applies a scaling correction based on the peak of the
            cross-correlation. Default is True.
        save_cross_corr : bool, optional
            If True, saves the computed cross-correlation and the associated
            lag vector. Default is True.
        signal_len : integer, optional
            Truncates the signal to the given length after synchronization
        d_max :integer, optional
            The maximum expected delay [number of samples of x]
        name : str, optional
            Name of the synchronizer instance. Default is "synchronizer".
    """
    target_data = Union[np.array, Recorder]
    scale_correction: bool = True
    save_cross_correlation: bool = True
    signal_len: Optional[int] = None
    d_max: Optional[int] = None
    name: str = "synchronizer"

    def __post_init__(self):
        self.delay = None
        self.scale = 1
        self.cross_corr = None
        self.n_vect = None

    def fit(self, x, x_preamble):
        N = len(x)
        N_preamble = len(x_preamble)
        x_preamble_padded = np.zeros(N, dtype=x.dtype)
        x_preamble_padded[:N_preamble] = x_preamble

        # compute cross correlation
        cross_corr = np.correlate(x,  x_preamble_padded, mode='full')
        cross_corr *= (1/N_preamble)
        n_vect = np.arange(len(cross_corr)) - (N - 1)

        # Find the time delay: the index of the maximum cross-correlation
        # minus the length of x minus 1
        index_max = np.argmax(np.abs(cross_corr)**2)
        value_max = cross_corr[index_max]

        self.delay = n_vect[index_max]
        # print(self.delay)
        self.delay = np.max([self.delay, 0])
        if self.scale_correction:
            self.scale = value_max

        # save correlation if needed
        if self.save_cross_correlation:
            self.cross_corr = cross_corr
            self.n_vect = n_vect

    def plot(self):
        plt.figure()
        plt.plot(self.n_vect, np.abs(self.cross_corr))
        plt.title('Cross-correlation magnitude')
        plt.xlabel('Lag')
        plt.ylabel('Magnitude')
        plt.grid(True)


    def forward(self, x: np.ndarray) -> np.ndarray:
        x_preamble = self.get_target_data()

        # upsampling
        x_up = signal.resample_poly(x, self.up_factor, 1)
        x_preamble_up = signal.resample_poly(x_preamble, self.up_factor, 1)

        if not self.d_max:
            Nmax = len(x_up)
        else:
            Nmax = min(len(x_up), len(x_preamble_up)+int((self.d_max+1)*self.up_factor))

        self.fit(x_up[:Nmax], x_preamble_up)

        y_up = x_up[self.delay:]
        # downsampling
        y = signal.resample_poly(y_up, 1, self.up_factor)

        y = self.scale*y
        if self.signal_len:
            y = y[:self.signal_len]

        return y
