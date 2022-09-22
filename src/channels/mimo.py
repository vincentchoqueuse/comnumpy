from numpy.random import randn
import numpy as np
from .core import Channel

class Random_Channel(Channel):

    def __init__(self, N_r, N_t, norm=False):
        self.N_r = N_r 
        self.N_t = N_t
        self.norm = norm
        self.rvs()

    def rvs(self):
        H_r = randn(self.N_r, self.N_t)
        H_i = randn(self.N_r, self.N_t)
        H = np.sqrt((1/self.N_t)/2) * (H_r + 1j*H_i)
        
        if self.norm : 
            energy = np.sum(np.abs(H)**2,axis=0)
            coef = np.sqrt(1/energy).reshape(1,-1)
            H = coef*H
        self.H = H

    def forward(self,X):
        Y = np.matmul(self.H,X)
        return Y


class AWGN(Channel):

    def __init__(self, sigma2):
        self.sigma2 = sigma2

    def rvs(self, N_r, N):
        B_r = randn(N_r, N)
        B_i = randn(N_r, N)
        coef = np.sqrt(self.sigma2/2)
        B = coef * ( B_r + 1j * B_i)
        return B
        
    def forward(self,X):
        N_r, N = X.shape
        self._B = self.rvs(N_r, N)
        Y = X + self._B
        return Y


class Selective_Channel(Channel):

    # Blind joint MIMO channel and data estimation based on regularized ML

    def __init__(self, H_list):
        self.H_list = H_list 

    def forward(self, X):
        L = len(self.H_list)
        N = X.shape[1]
        N_tot = N+L-1
        H_0 = self.H_list[0]
        N_r, N_t = H_0.shape
        Y = np.zeros((N_r, N_tot), dtype=complex)

        for n in range(N):
            for m in range(L):
                if n-m >= 0:
                    H = self.H_list[m]
                    Y[:,n] += np.matmul(H, X[:,n-m])

        return Y
