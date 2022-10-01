from scipy.stats import randint
from .core import Processor
from core import Sequential
from channels.optical import EDFA, Fiber_Link
import numpy as np
from scipy import signal
from scipy.linalg import toeplitz, norm
from scipy import special
from scipy.optimize import least_squares
from analysers.logger import Power_Reporter
import numpy.linalg as LA
import copy


class FIR_CD_Compensator(Processor):

    def __init__(self, z, D=17*(10**-6), lamb=1.55*10**-6, F_s=1, name="savory"):
        self.F_s = F_s
        self.z = z  # in m
        self.lamb = lamb 
        self.D = D 
        self.c = 3*(10**8)
        self.name = name
    
    def compute_beta2(self):
        return -self.D * self.lamb**2/(2*np.pi*self.c)

    def get_N(self):
        # savory non-aliased choice
        K = self.K()
        N = int(2*np.floor(2*K*np.pi) + 1)
        return N

    def K(self):
        T = 1/self.F_s
        beta2 = self.compute_beta2()
        return -beta2*self.z*(self.F_s**2)/2

    def h(self):
        K = self.K()
        N = self.get_N()
        bound = int(np.floor(N/2))
        n_vect = np.arange(-bound, bound+1)
        coef = np.sqrt(1j / (4*K*np.pi))
        h = coef * np.exp(-1j * (n_vect**2) / (4*K))
        return h
        
    def forward(self,x):
        h = self.h()
        y = signal.convolve(x, h)
        return y


class LS_FIR_CD_Compensator(FIR_CD_Compensator):
    #Optimal Least-Squares FIR Digital Filters for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers

    def __init__(self, z, N, D=17*(10**-6), lamb=1.55*10**-6, F_s=1, epsilon=10**(-14), w_vect = [-np.pi,np.pi], name="optimal"):
        self.F_s = F_s
        self.z = z  # in m
        self.lamb = lamb  
        self.D = D  
        self.c = 3*(10**8)
        self.epsilon = epsilon
        self.w_vect = w_vect
        self.N = N
        self.name = name

    def h(self):
        K = self.K()
        N = self.N
        Omega_1 = self.w_vect[0]
        Omega_2 = self.w_vect[1]

        # compute Q
        q_row = np.zeros(N, dtype=complex)
        q_col = np.zeros(N, dtype=complex)
        q_row[0] = (Omega_2 - Omega_1)/ (2*np.pi)
        q_col[0] = q_row[0]
        for m in range(1,N):
            coef = 1/(2j*np.pi*m)
            q_row[m] = coef * (np.exp(-1j*m*Omega_1) - np.exp(-1j*m*Omega_2)) # eq 10
        Q = toeplitz(q_col, q_row)

        # compute d
        bound = int(np.floor(N/2))
        n_vect = np.arange(-bound, bound+1)
        d_vect = np.zeros(len(n_vect), dtype=complex)
        
        # compute d vector
        coef1 = 1/(4*np.sqrt(np.pi*K))
        coef2 = np.exp(1j*3*np.pi/4) / (2*np.sqrt(K))
        for indice, n in enumerate(n_vect):
            term1 = coef2*(2*K*np.pi - n)
            term2 = coef2*(2*K*np.pi + n)
            term_temp = special.erf(term1) + special.erf(term2)  # eq 13
            d_vect[indice] = coef1 * np.exp(-1j*((n**2/(4*K)) + 3*np.pi/4)) * term_temp

        # compute h
        I_mat = np.eye(N)
        Q_inv = LA.inv(Q + self.epsilon*I_mat)
        h = np.matmul(Q_inv, d_vect)
        return h


class GSOP(Processor):

    # Reference: I. Fatadin, S. J. Savory and D. Ives, 
    # "Compensation of Quadrature Imbalance in an Optical QPSK Coherent Receiver," in IEEE Photonics Technology Letters, vol. 20, no. 20, pp. 1733-1735, Oct.15, 2008, doi: 10.1109/LPT.2008.2004630.

    def __init__(self, name="gsop"):
        self.name = name 

    def forward(self, x):
        # implementation of the gram schmit orthogonalization
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


class CMA(Processor):
    """
    Faruk, Md Saifuddin, and Seb J. Savory. "Digital signal processing for coherent transceivers employing multilevel formats." 
    Journal of Lightwave Technology 35.5 (2017): 1125-1141.
    """

    def __init__(self, L, alphabet, mu=0.00001, oversampling=1, norm=True, debug=False, mix=True, name="cma"):
        self.mu = mu
        self.L = L
        self.alphabet = alphabet
        self.oversampling = oversampling
        self.mix = mix
        self.name = name
        self.norm = norm
        
    def prepare(self, X):
        self.h11 = np.zeros(self.L, dtype=complex)
        self.h12 = np.zeros(self.L, dtype=complex)
        self.h21 = np.zeros(self.L, dtype=complex)
        self.h22 = np.zeros(self.L, dtype=complex)

        if self.norm:
            self.h11[0] = np.sqrt(np.mean(np.abs(X[0,:])**2))
            self.h22[0] = np.sqrt(np.mean(np.abs(X[1,:])**2))
        else:
            self.h11[0] = 1
            self.h22[0] = 1

    def grad(self, input, output, target=None):
        # compute loss
        N = len(input[0])
        x_1 = input[0]
        x_2 = input[1]
        radius_list = np.unique(np.abs(self.alphabet)**2)

        # compute loss
        radius_1 = np.abs(output[0])**2
        radius_2 = np.abs(output[1])**2
        index_1 = np.argmin((radius_1 - radius_list)**2)
        index_2 = np.argmin((radius_2 - radius_list)**2)
        error_1 = radius_list[index_1] - radius_1
        error_2 = radius_list[index_2] - radius_2

        # compute grad with respect to h11, h12, h12 and h22
        grad = np.zeros((4,N), dtype=complex)
        grad[0,:] = -error_1*output[0]*np.conj(x_1)
        grad[1,:] = -error_1*output[0]*np.conj(x_2)
        grad[2,:] = -error_2*output[1]*np.conj(x_1)
        grad[3,:] = -error_2*output[1]*np.conj(x_2)
        return grad

    def forward(self, X):
        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M,N), dtype=complex)

        self.prepare(X)
       
        for n in range(L+1, N):
            input = X[:,n:n-L:-1]  # we need to begin a L+1 to allow this syntax

            # compute output
            x_1 = input[0,:]
            x_2 = input[1,:]
            y_1 = np.dot(self.h11, x_1) + np.dot(self.h12, x_2)
            y_2 = np.dot(self.h21, x_1) + np.dot(self.h22, x_2)
            output = np.array([y_1, y_2])
            
            if (n % os) == 0:
                grad = self.grad(input, output)  # compute grad
  
                # update parameter
                self.h11 = self.h11 - self.mu * grad[0,:]
                self.h22 = self.h22 - self.mu * grad[3,:]

                if self.mix == True:
                    self.h12 = self.h12 - self.mu * grad[1,:]
                    self.h21 = self.h21 - self.mu * grad[2,:]

                
            Y[:,n] = output  # store data

        # downsampling
        Y_sub = Y[:,::os]
        return Y_sub


class Blind_Phase_Compensation(Processor):

    def __init__(self, alphabet, theta=0, name="phase"):
        self.alphabet = alphabet
        self.name = name
        self.theta = 0

    def cost(self, theta):
        y = self.x * np.exp(1j*theta)
        error_tot = y.reshape(-1, 1) - self.alphabet.reshape(1, -1)
        index = np.argmin(np.abs(error_tot)**2, axis=1)
        error = y - self.alphabet[index]
        error_real = np.hstack([np.real(error),np.imag(error)])
        return error_real

    def forward(self, x):
        self.x = x
        theta = self.theta
        res = least_squares(self.cost, theta)
        self.theta = res.x
        y = self.x * np.exp(1j*self.theta)
        return y


class DBP(Fiber_Link):

    direction = -1

    def __init__(self, N_span, StPS, L_span, gamma=1.3*1e-3, lamb=1.55 * 10**-6, c=3*(10**8), alpha_dB= 0.2*1e-3, F_s=1, step_structure="symmetric", step_type="linear", step_log_factor=0.4, step_scheme=1, include_edfa=False, name="DBP"):
        self.N_span = N_span
        self.StPS = StPS
        self.L_span = L_span
        self.gamma = gamma 
        self.lamb = lamb
        self.c = c
        self.alpha_dB = alpha_dB
        self.F_s = F_s 
        self.step_structure = step_structure
        self.step_type = step_type
        self.step_log_factor = step_log_factor
        self.step_scheme = step_scheme # 0: no gain, 1: gain in CD, 2: gain in nl
        self.name = name 
        self.include_edfa = include_edfa

        self.prepare()

    def get_pre_span_module_list(self):
        """
        return the module list before a span
        """
        edfa = EDFA(self.alpha_dB, L_span = self.L_span, direction=self.direction)
        return [edfa]

    def get_step_module_list(self, dz):
        """
        return the module list for a single step  of length dz
        """
        nl = self.get_non_linear_model(dz)
        cd = self.get_CD(dz)
        return [cd, nl]

    def get_post_span_module_list(self):
        """
        return the module list after a span
        """
        return []


