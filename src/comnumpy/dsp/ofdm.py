import numpy as np
import numpy.linalg as LA
from .core import Processor
from comnumpy.metrics.ofdm import compute_PAPR
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import itertools


class Cyclic_Prefix_Adder(Processor):

    """
    Processor for adding a cyclic prefix to the input data to combat multi-path interference.

    Parameters
    ----------
    N_cp : int
        The length of the cyclic prefix to be added. Must be a non-negative integer.

    """

    def __init__(self, N_cp, name="cp adder"):
        if not (isinstance(N_cp, int) and N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")
        
        self.N_cp = N_cp
        self.name = name

    def forward(self, X):
        N_sc, M = X.shape
        N_cp = self.N_cp

        Y = np.zeros((N_sc+N_cp, M), dtype=X.dtype)
        Y[:N_cp, :] = X[-N_cp:, :]
        Y[N_cp:, :] = X
        return Y
    
class Cyclic_Prefix_Remover(Processor):

    """
    Processor for removing a cyclic prefix to the input data

    Parameters
    ----------
    N_cp : int
        The length of the cyclic prefix to be added. Must be a non-negative integer.

    """

    def __init__(self, N_cp, name="cp remover"):
        if not (isinstance(N_cp, int) and N_cp >= 0):
            raise ValueError("N_cp must be a positive integer.")
        
        self.N_cp = N_cp
        self.name = name

    def forward(self, X):
        Y = X[self.N_cp:, :]    
        return Y


class FFT_Processor(Processor):

    """
    Processor for performing Fast Fourier Transform (FFT) on the input data.

    Parameters
    ----------
    apply_fftshift : bool, optional
        If True, applies the FFT shift which swaps the low and high frequency components.
        Default is False.

    norm : {"ortho", None}, optional
        Normalization mode for FFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
    """

    def __init__(self, apply_fftshift=True, norm="ortho", name="fft"):
        self.norm = norm
        self.apply_fftshift = apply_fftshift
        self.name = name

    def forward(self, X):
        Y = fft(X, norm=self.norm, axis=0)
        if self.apply_fftshift:
            Y = fftshift(Y, axes=0)
        return Y


class IFFT_Processor(Processor):

    """
    Processor for performing Inverse Fast Fourier Transform (FFT) on the input data.

    Parameters
    ----------
    apply_ifftshift : bool, optional
        If True, applies the IFFT shift which swaps the low and high frequency components.
        Default is False.

    norm : {"ortho", None}, optional
        Normalization mode for IFFT. "ortho" means orthonormal FFT is computed.
        None means no normalization is applied. Default is "ortho".
    """

    def __init__(self, apply_ifftshift=True, norm="ortho", name="ifft"):
        self.apply_ifftshift = apply_ifftshift
        self.norm = norm
        self.name = name

    def forward(self, X):
        if self.apply_ifftshift:
            X = ifftshift(X, axes=0)
        Y = ifft(X, norm=self.norm, axis=0)
        return Y



def get_standard_carrier_allocation(config_name, os=1, hermitian_sym=None):

    ofdm_config_dict = { #OFDM_config_name, [N_block, N_sc_nulled_DC, N_sc_nulled_left, N_sc_nulled_rigth, pilot_position(start = 0)]
                    'IQtools_128': [128, 3, 6, 5, [16, 28, 40, 52, 76, 88, 100, 112] ], 
                    '802.11ah_32': [32, 1, 3, 2, [9, 23] ], 
                    '802.11ah_64': [64, 1, 4, 3, [11, 25, 39, 53] ], 
                    '802.11ah_128': [128, 3, 6, 5, [11, 39, 53, 75, 89, 117] ], 
                    '802.11ah_256': [256, 3, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231] ], 
                    '802.11ah_512': [512, 11, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231, 281, 309, 345, 373, 395, 423, 459, 487] ], 
                    'NoPilot_16': [16, 3, 6, 5, [] ],
                    'NoPilot_32': [32, 3, 6, 5, [] ],
                    'NoPilot_64': [64, 3, 6, 5, [] ],
                    'NoPilot_128': [128, 3, 6, 5, [] ],
                    'NoPilot_256': [256, 3, 6, 5, [] ],
                    'NoPilot_512': [512, 3, 6, 5, [] ],
                    'NoPilot_1024': [1024, 3, 6, 5, [] ],
                    'NoPilot_2048': [2048, 3, 6, 5, [] ],
                    'NoPilot_4096': [4096, 3, 6, 5, [] ],
                    'NoPilot_8192': [8192, 3, 6, 5, [] ],
                    'NoPilot_16384': [16384, 3, 6, 5, [] ]
                    }
    
    ofdm_config = ofdm_config_dict[config_name]
    N_block = ofdm_config[0]
    N_sc_nulled_right = ofdm_config[3]
    pilot_position = ofdm_config[4]
    
    # preprocess some attributes
    if hermitian_sym:
        N_sc_nulled_DC= (ofdm_config[1]//2)*2+1 #N_sc_nulled_DC must be odd
        N_sc_nulled_left = np.max([1, ofdm_config[2]]) # the 1st left and the central subcarriers are nulled,
    else:
        N_sc_nulled_DC = ofdm_config[1]
        N_sc_nulled_left = ofdm_config[2]
        
    # number of data subcarrier
    N_data = N_block-N_sc_nulled_DC-N_sc_nulled_left-N_sc_nulled_right-len(pilot_position)
    if hermitian_sym:
        N_data = N_data//2
    else:
        N_data = N_data

    # allocate subcarriers
    subcarrier_type = np.full(N_block, 1) # initialize with data

    if len(pilot_position) > 0:
        subcarrier_type[pilot_position] = 2  # reserved for pilot

    subcarrier_type[:N_sc_nulled_left] = 0  #reserved for null carriers
    subcarrier_type[-N_sc_nulled_right:] = 0 #reserved for null carriers

    # add null carriers à DC
    middle = N_block//2
    width = (N_sc_nulled_DC//2)
    subcarrier_type[ middle-width:middle+width+1] = 0

    # oversampling
    N_sc_nulled_os = N_block*(os-1)
    subcarrier_type = np.insert(subcarrier_type, N_block, np.zeros(N_sc_nulled_os//2))
    subcarrier_type = np.insert(subcarrier_type, 0, np.zeros(N_sc_nulled_os//2)) 

    N_size = len(subcarrier_type)
    
    if hermitian_sym:
        subcarrier_type[-1:-N_size//2:-1] = -1 # hermitian symmetry 

    return subcarrier_type


class Carrier_Allocator(Processor):

    def __init__(self, subcarrier_type, pilots=[], name="carrier allocator"):

        N_pilots = len(np.where(subcarrier_type==2)[0])
        if N_pilots != len(pilots):
            raise ValueError("Incompatible number of pilots ({} needed, {} provided)".format(N_pilots, len(pilots)))
        
        self.subcarrier_type = subcarrier_type
        self.pilots = pilots
        self.name = name

    def forward(self, X):

        N_subcarriers = len(self.subcarrier_type)
        N_data = len(np.where(self.subcarrier_type==1)[0])
        N_pilots = len(np.where(self.subcarrier_type==2)[0])

        N, L = X.shape
        if N_data != N:
            raise ValueError("Incompatible number of data subcarriers ({} needed, {} provided)".format(N_subcarriers, N))

        Y = np.zeros((N_subcarriers, L), dtype=X.dtype)

        # add data
        Y[self.subcarrier_type==1, :] = X

        # add pilot if needed
        if N_pilots>0:
            pilots_reshaped = self.pilots[:, np.newaxis]
            Y[self.subcarrier_type==2, :] = pilots_reshaped

        return Y


class Carrier_Extractor(Processor):

    def __init__(self, subcarrier_type, pilot_recorder=None, name="carrier extractor"):
        self.subcarrier_type = subcarrier_type
        self.pilot_recorder = pilot_recorder
        self.name = name

    def forward(self, X):
        X_data = X[self.subcarrier_type==1, :]
        X_pilots = X[self.subcarrier_type==2, :]

        if self.pilot_recorder:
            self.pilot_recorder(X_pilots)

        return X_data


class Frequency_Domain_Equalizer(Processor):

    def __init__(self, h, apply_fftshift=True, name="equalizer"):
        self.h = h
        self.apply_fftshift = apply_fftshift
        self.name = name

    def get_weight(self, nb_subcarriers):
        H = fft(self.h, n=nb_subcarriers)
        weight = 1./H 
        if self.apply_fftshift:
            weight = fftshift(weight)
        return weight

    def forward(self, X):
        nb_subcarriers,_ = X.shape
        weight = self.get_weight(nb_subcarriers)
        weight = weight[:, np.newaxis] 
        Y = weight*X
        return Y

        
class Hard_Clipper(Processor):
    """Reduce signal PAPR by hard clipping

    Parameters
    ----------
    cr_dB : real, int
        Clipping ratio in dB

    Methods
    -------
    forward(x):
            Processes the input data vector through hard clipping.
            
    [1] Y. Rahmatallah et S. Mohan, « Peak-To-Average Power Ratio Reduction in OFDM Systems: A Survey And Taxonomy », 
    IEEE Commun. Surv. Tutorials, vol. 15, no 4, p. 1567 1592, 2013, doi: 10.1109/SURV.2013.021313.00164.
    """

    def __init__(self, cr_dB, name="hard_clipping"):
        self.cr = 10**(cr_dB/20)
        self.name = name

    def forward(self, x):
        Pmoy = np.mean(np.abs(x)**2)
        Tm = self.cr*np.sqrt(Pmoy)
        y = np.where(np.abs(x) > Tm, Tm*np.exp(1j*np.angle(x)), x)
        return y


class ICT_PAPR_Reductor(Processor):

    """PTS_PAPR_Reductor


    Parameters
    ----------
    PAPR_max_dB : real, int
        PAPR_max in dB
    
    Reference
    ---------
    * [1] Wang, Y-C., and Z-Q. Luo. "Optimized iterative clipping and filtering for PAPR reduction of OFDM signals." IEEE Transactions on communications 59.1 (2010): 33-37.
    """

    def __init__(self, PAPR_max_dB, filter_weight, N_it=16, apply_ifftshift=True, norm="ortho"):

        self.apply_ifftshift = apply_ifftshift
        self.filter_weight = filter_weight
        
        PAPR_max = 10**(PAPR_max_dB/10)
        self.cr = np.sqrt(PAPR_max)
        self.N_it = N_it
        self.norm = norm

    def clip(self, x):
        Pmoy = np.sqrt(np.mean(np.abs(x)**2))
        Tm = self.cr*Pmoy # see equation 7
        y = np.where(np.abs(x) > Tm, Tm*np.exp(1j*np.angle(x)), x)
        return y

    def forward(self, X):
        N_sc, L = X.shape 
        Y_preprocessed = np.zeros((N_sc, L), dtype=X.dtype)
        
        for l in range(L):
            X_l = X[:, l]

            for m in range(self.N_it):
                x_l = ifft(X_l, norm="ortho")
                x_l = self.clip(x_l)
                X_l = fft(x_l, norm="ortho")
                X_l = self.filter_weight*X_l  # out-of-band filtering

            Y_preprocessed[:, l] = X_l

        if self.apply_ifftshift:
            Y_preprocessed = ifftshift(Y_preprocessed, axes=0)
        Y = ifft(Y_preprocessed, norm=self.norm, axis=0)
        
        return Y



class PTS_PAPR_Reductor(Processor):

    """PTS_PAPR_Reductor
    
    Reference
    ---------
    * [1] L. J. Cimini and N. R. Sollenberger, "Peak-to-average power ratio reduction of an OFDM signal using partial transmit sequences," 1999 IEEE International Conference on Communications (Cat. No. 99CH36311), Vancouver, BC, Canada, 1999, pp. 511-515 vol.1, doi: 10.1109/ICC.1999.767992.
    
    """

    def __init__(self,  phase_alphabet=[1, -1], N_sub=16):
        self.phase_alphabet = phase_alphabet
        self.N_sub = N_sub
        self.combinations = np.array(list(itertools.product(phase_alphabet, repeat=N_sub)))

    def get_subblocks(self, X):
        # adjacent partition: blocks consists of a contiglious set of subcarriers and are of equal size
        N = len(X)
        N_sub = self.N_sub  # number of partitions
        X_m_array = np.zeros((N, N_sub), dtype=X.dtype)
        
        if N % N_sub != 0:
            raise ValueError("N_sc must be divisible by N_sub.")
        
        L = int(N/N_sub)
        for m in range(N_sub):
            X_m_array[m*L:(m+1)*L, m] = X[m*L:(m+1)*L]

        return X_m_array
    
    def find_optimal_combination(self, x_m_array):

        papr_list = np.zeros(len(self.combinations))

        for index, combination in enumerate(self.combinations):
            x_m_temp = np.dot(x_m_array, combination)
            papr_list[index] = compute_PAPR(x_m_temp)

        index_min = np.argmin(papr_list)
        combination = self.combinations[index_min]
        x_m = np.dot(x_m_array, combination)
        return x_m, combination

    def forward(self,X):
        _, L = X.shape 

        Y = np.zeros(X.shape, dtype=X.dtype)
        combination_list = []
        for l in range(L):
            X_m_array = self.get_subblocks(X[:, l])
            x_m_array_ifft = ifft(X_m_array, norm="ortho", axis=0)
            x_m_ifft,combination = self.find_optimal_combination(x_m_array_ifft)
            
            Y[:, l] = x_m_ifft
            combination_list.append(combination)

        return Y








    
