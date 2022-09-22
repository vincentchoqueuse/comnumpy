import numpy as np
import numpy.linalg as LA
from .core import Processor

class Normalizer(Processor):

    def __init__(self, type="energy", name="normalizer"):
        self.type = "energy"
        self.name = name 

    def forward(self, x):
        if self.type == "energy":
            energy = np.mean(np.abs(x)**2)
            y = x/np.sqrt(energy)
        else:
            y = x 
        return y


class Blind_IQ(Processor):

    """
    Blind_IQ()
    Blind IQ estimator based on the diagonalisation of the augmented covariance matrix. This compensation assumes the circularity of compensated signal
    Parameters
    ----------
    None
    Returns
    -------
    Compensated signal
    """

    def __init__(self, name="iq_compensator"):
        self.name = name

    def forward(self, x):

        N = len(x)
        X = np.vstack([np.real(x), np.imag(x)])

        # compute covariance matrix
        R = (1/N)*np.matmul(X, np.transpose(X))
        # perform eigenvalue decomposition
        V, U = LA.eig(R)

        # perform whitening
        D = np.diag(1/np.sqrt(V))
        M = np.matmul(D, np.transpose(U))
        Y = np.matmul(M, X)
        x = Y[0, :] + 1j*Y[1, :]
        return x


class Blind_CFO(Processor):

    """
    Blind_CFO()
    Blind CFO estimator based on the maximisatio of the periodogram of 4th order statistic
    .. math::
        \\widehat{\\omega} = \\frac{1}{4} \\arg \\max_{\\omega} |\\sum_{n=0}^{N-1}x^4[n]e^{-j\\omega n}|^2
    The maximisation is performed using the Newton Algorithm
    Parameters
    ----------
    w0 : float
        Initialisation in rad/samples
    N_iter : int
        Number of iterations
    method : str
    Returns
    -------
    Compensated signal
    """

    def __init__(self, w0=0, N_iter=10, training=True, method="newton", step_size=10**(-5), name="cfo"):
        self.name = name
        self.w_init = w0
        self.N_iter = N_iter
        self.method = method
        self.step_size = step_size
        self._training = training

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

    def fit(self, x, w0):
        w = w0
        N = len(x)
        x4 = x**4
        N_vect = np.arange(N)
        step_size = self.step_size

        if self.method == "grid-search":
            w_vect = 4*np.arange(0.0045, 0.0055, 0.00001)
            cost_vect = np.zeros(len(w_vect))
            for index, w in enumerate(w_vect):
                cost_vect[index] = self.loss(x, w)
            index_max = np.argmax(cost_vect)
            w = w_vect[index_max]

        else:
            for n in range(self.N_iter):
                if self.method == "grad":
                    dtft = self.compute_dtft(x4, w)
                    dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                    grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                    h = step_size * grad

                if self.method == "newton":
                    dtft = self.compute_dtft(x4, w)
                    dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                    dtft_diff2 = self.compute_dtft(-(N_vect**2)*x4, w)
                    grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                    J = (2/N) * (np.real(dtft_diff2*np.conj(dtft)) + (np.abs(dtft_diff)**2))
                    h = -grad/J

                w = w + h

        w0 = np.real(w)/4
        return w0

    def forward(self, x):

        N = len(x)
        N_vect = np.arange(N)
        if self._training:
            w0 = self.fit(x, 4*self.w_init)
        x = x*np.exp(-1j*w0*N_vect)
        return x


class Data_Aided_FIR(Processor):

    """
    Data_Aided_FIR()
    Data aided estimation of a FIR filter using ZF estimation.
    """

    def __init__(self, h, name="data_aided_fir"):
        self._h = h
        self.name = name

    def fit(self, y_set):
        # Estimation of the channel coefficient
        # x = Hy = Yh
        index_pilots = y_set[0]
        y_target = y_set[1]
        N = len(index_pilots)
        L = len(self._h)
        x_trunc = self._x[index_pilots]

        first_col = np.zeros(N, dtype=np.complex)
        first_col = y_target
        H = toeplitz(first_col, np.zeros(L))
        self._h = np.matmul(LA.pinv(H), x_trunc)

    def forward(self, x):
        L = len(self._h)
        y, _ = deconvolve(np.hstack([x, np.zeros(L-1)]), self._h)
        return y


class Data_Aided_Phase(Processor):

    def __init__(self, name="data_aided_phase"):
        self._theta = 0
        self.name = name

    def fit(self, x, x_target):
        self._theta = np.angle(np.sum(np.conj(x)*x_target))

    def forward(self, x):
        return x*np.exp(1j*self._theta)