import numpy as np
import itertools
import numpy.linalg as LA
from .core import Processor
from .functional import hard_projector, soft_projector


class ML(Processor):

    """Implements the ML Detector for white Gaussian noise.
    Parameters
    ----------
    H : numpy array
        Channel matrix
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.
    """

    def __init__(self, H, alphabet, name = "ML"):
        super().__init__()
        self._H = H
        self._alphabet = alphabet
        self.name = name

    def get_nb_candidates(self):
        N_r, N_t = self._H.shape
        alphabet = self._alphabet
        return len(alphabet)**N_t

    def get_candidates(self, alphabet, N_t):
        symbols = np.arange(len(alphabet))
        input_list = [p for p in itertools.product(symbols, repeat=N_t)]

        # preallocation of memory
        X = np.zeros((N_t,len(input_list)),dtype=complex)
        S = np.zeros((N_t,len(input_list)))

        for indice in range(len(input_list)):
            input = np.array(input_list[indice])  # store combinaison
            x = self._alphabet[input]             # transmitted data
            X[:,indice] = x
            S[:,indice] = input

        return S, X

    def forward(self,Y):
        """ performs detection using the received samples :math:`\mathbf{Y}`."""
        N_r, N_t = self._H.shape
        N_r, N = Y.shape
        S = np.zeros((N_t,N),dtype=int)
        alphabet = self._alphabet
        
        S_candidates, X_candidates = self.get_candidates(alphabet,N_t)
        Y_candidates = np.matmul(self._H,X_candidates)  # compute all combinaison of received data

        for n in range(N):
            y = np.transpose(np.atleast_2d(Y[:,n]))
            index_min = np.argmin(np.sum(np.abs(y-Y_candidates)**2,axis=0))
            S[:,n] = S_candidates[:,index_min]

        self.S = S
        return S


class ZF(Processor):

    """Implements the Zero-Forcing (ZF) MIMO detector.
    Parameters
    ----------
    H : numpy array
        Channel matrix
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.
    """

    def __init__(self, H, alphabet, name = "ZF"):
        self._H = H
        self.alphabet = alphabet
        self.name = name

    def linear_estimator(self,Y):
        H_inv = LA.pinv(self._H)
        X_est = np.matmul(H_inv,Y)
        return X_est

    def forward(self,Y):
        """ perform detection using the received samples :math:`\mathbf{Y}`."""
        X0 = self.linear_estimator(Y)
        S, X = hard_projector(X0, self.alphabet)
        return S


class MMSE(Processor):
    
    """Implements the MMSE MIMO detector.
    Parameters
    ----------
    H : numpy array
        Channel matrix
    sigma2: float
        noise variance
    alphabet : numpy array
        symbol constellation 
    output : str, optional 
        specify if the forward function should output the symbols or the index.
    """

    def __init__(self,H, sigma2, alphabet, output="symbols",name = "MMSE"):
        super().__init__(H,alphabet,output=output,name = name)
        self._sigma2 = sigma2

    def linear_estimator(self,Y):
        N_r,N_t = self._H.shape
        H_H = np.conjugate(np.transpose(self._H))
        A = np.matmul(LA.inv(np.matmul(H_H,self._H)+self._sigma2*np.eye(N_t)),H_H)
        X_est = np.matmul(A,Y)
        return X_est

    def forward(self,Y):
        """ perform detection using the received samples :math:`\mathbf{Y}`."""
        X0 = self.linear_estimator(Y)
        S, X = hard_projector(X0, self.alphabet)
        return S


class AMP(Processor):

    # see reference: https://www.csl.cornell.edu/~studer/papers/15ISIT-lama.pdf

    def __init__(self, H, alphabet, alpha=1, N_it = 100, sigma2=None, name = "AMP"):
        self.H = H
        self.sigma2 = sigma2
        self.N_it = N_it
        self.alpha = alpha
        self.alphabet = alphabet
        self.name = name

    def fit(self, y):
        # see Algorithm 2
        N_r, N_t = self.H.shape
        x_t = np.zeros(N_t)
        r_t = y - np.matmul(self.H, x_t)
        H_H = np.transpose(np.conjugate(self.H))
        beta = N_t / N_r  # system ratio (below equation 1)
        tau_2 = beta*1 / self.sigma2

        for _ in range(self.N_it):
            z_t = x_t + np.matmul(H_H, r_t)
            sigma2_t = self.sigma2 * (1+ tau_2)
            x_t = soft_projector(z_t, self.alphabet, sigma2_t)  # F function in the original publication
            kernel = np.abs(self.alphabet.reshape(1, -1) - x_t.reshape(-1, 1))**2
            G = soft_projector(z_t, self.alphabet, tau_2, kernel)
            tau_2_old = tau_2
            tau_2 = (beta / (self.sigma2)) * np.mean(G)
            term1 = tau_2/( 1 + tau_2_old)
            r_t = y - np.matmul(self.H, x_t) + term1 * r_t

        return x_t
 
    def forward(self, Y):
        N_r, N = Y.shape
        N_r, N_t = self.H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:,n] = self.fit(Y[:,n])

        S, X = hard_projector(X, self.alphabet)
        return S


class OAMP(Processor):

    # see the nice reference here: https://www.politesi.polimi.it/bitstream/10589/181815/3/Executive_Summary_long_version.pdf

    def __init__(self, H, alphabet, alpha=1, N_it = 100, sigma2=None, type="MMSE", name = "OAMP"):
        self.H = H
        self.sigma2 = sigma2
        self.type = type
        self.N_it = N_it
        self.alpha = alpha
        self.alphabet = alphabet
        self.name = name

    def get_W(self, vt_2=0):

        if self.type == "H":
            H_H = np.transpose(np.conjugate(self.H))
            W = H_H
        if self.type == "pinv":
            W = LA.pinv(self.H)
        if self.type == "MMSE":
            N_r, N_t = self.H.shape
            H = self.H
            H_H = np.transpose(np.conjugate(H))
            term1 = vt_2 * np.matmul(H,H_H) + self.sigma2*np.eye(N_r)
            W = vt_2 * np.matmul(H_H, LA.inv(term1))
        return W

    def get_vt_2(self, error, epsilon = 0.001):
        N_r, N_t = self.H.shape
        R = self.sigma2*np.eye(N_r)
        H_H = np.conjugate(np.transpose(self.H))
        num = np.sum(np.abs(error)**2) - np.trace(R)
        den = np.trace(np.matmul(H_H, self.H))
        return max(num/den, epsilon) 

    def get_tau_2(self, B , W, vt_2):
        N_r, N_t = self.H.shape
        R = self.sigma2*np.eye(N_r)
        W_H = np.conjugate(np.transpose(W))
        B_H = np.conjugate(np.transpose(B))
        term1 = (vt_2/N_t) * np.trace(np.matmul(B, B_H))
        term2 = (1/N_t) * np.trace(np.matmul(W, np.matmul(R,W_H)))
        tau_2 = term1 + term2
        return tau_2

    def fit(self, y):
        tau_2, vt_2 = 1, 1
        N_r, N_t = self.H.shape
        x_t = np.zeros(N_t)

        for _ in range(self.N_it):
            W = self.get_W(vt_2)
            B = np.eye(N_t) - np.matmul(W,self.H)
            error = y - np.matmul(self.H, x_t)
            z_t = x_t + np.matmul(W, error)
            x_t = soft_projector(z_t, self.alphabet, tau_2)
            vt_2 = self.get_vt_2(error)
            tau_2 = self.get_tau_2(B, W, vt_2)

        return x_t

    def forward(self, Y):
        _, N = Y.shape
        _, N_t = self.H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:,n] = self.fit(Y[:,n])

        S, X = hard_projector(X, self.alphabet)
        return S
