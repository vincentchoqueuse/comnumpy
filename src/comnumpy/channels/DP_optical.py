import optical
import numpy as np

class Chromatic_Dispersion(optical.CD):
    """ Chromatic Dispersion Channel 
    """

    def forward(self, X):
        N_t, NFFT = X.shape
        w = (2*np.pi*self.F_s)*np.fft.fftfreq(NFFT, d=1)
        H = np.exp(1j * (self.beta2/2) * self.z * (w**2) * self.direction)
        fft_X0 = np.fft.fft(X[:, 0])
        fft_X1 = np.fft.fft(X[:, 1])

        Y = np.zeros((N_t, NFFT))
        Y[:, 0] = self.gain * np.fft.ifft(H * fft_X0)
        Y[:, 1] = self.gain * np.fft.ifft(H * fft_X1)
        return Y

class NonLinearity(optical.NonLinearity):

    def forward(self, X):
        N_t, N = X.shape
        abs_X0 = np.abs(X[:, 0])
        abs_X1 = np.abs(X[:, 1])

        Y = np.zeros((N_t, N))
        Y[:, 0] = X[:, 0]*np.exp(1j*self.nl_param*(abs_X0**2 + (2/3)*abs_X1**2))
        Y[:, 1] = X[:, 1]*np.exp(1j*self.nl_param*(abs_X1**2 + (2/3)*abs_X0**2))
        return Y

class EDFA(optical.EDFA):

    def forward(self,X):
        Y = self.gain*X
        return Y

class ASE(optical.ASE):

    def forward(self,X):
        N_t, N = X.shape
        noise = np.sqrt(self.p_ase/2) * (np.random.randn((N_t, N)) + 1j * np.random.randn((N_t, N)))
        Y = X + noise 
        return Y

class PMD():

    # equation 101
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7174950

    def __init__(self, gamma_min, gamma_max):

    def get_D(self, w):
        D = np.exp(-1j * (self.beta2/2) * self.z * (w**2))
        D_tot = np.zeros((len(w), 2, 2))
        D_tot[:, 0, 0] = D 
        D_tot[:, 1, 1] = D 
        return D

    def get_U(self, w):
        D_tot = np.zeros((len(w), 2, 2))
        D_tot[:, 0, 0] = np.exp(1j*w*self.delta_T/2)
        D_tot[:, 1, 1] = np.exp(-1j*w*self.delta_T/2)
        R1_H = np.conjugate(np.transpose(self.R1))
        U = np.matmul(np.matmul(R1_H, D_tot), self.R1)
        return U

    def get_K(self, w):
        # global PDL K
        R2_H = np.conjugate(np.transpose(self.R2))
        D = np.diag([np.sqrt(self.gamma_max), np.sqrt(self.gamma_min)])
        K = np.matmul(np.matmul(R2_H, D), self.R2)
        # frequency independent matrix
        K_tot = np.tile(K,[len(w),1,1])
        return K_tot

    def get_H(self,w):
        D_tot = self.get_D(w)
        U_tot = self.get_U(w)
        K_tot = self.get_K(w)
        T_tot = np.tile(self.T,[len(w),1,1])
        H = np.matmul(D_tot,np.matmul(U_tot, np.matmul(K_tot, T_tot)))
        return H

    def forward(self, x):
        NFFT = len(x)
        w = (2*np.pi*self.F_s)*np.fft.fftfreq(NFFT, d=1)
        H = self.get_H(w)
        fftx = np.fft.fft(x)
        ffty = H * fftx
        y = self.gain * np.fft.ifft(ffty)
        return y

