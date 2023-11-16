import numpy as np
from .core import Channel

class Chromatic_Dispersion(Channel):
    """ 
    This module applies the chromatic dispersion effect in the frequency domain. 

    Parameters
    ----------
    z : float
        Step length [km].
    F_s : float
        Sampling frequency [Hz].
    D : float
        Fiber dispersion parameter [ps/nm/km].
    lamb : float
        Wavelength [nm].
    c : float
        Speed of light [km/s].
    alpha_dB : float
        Attenuation factor in dB.

    References
    ----------
    
    * [1] Shahkarami, Abtin. Complexity reduction over bi-RNN-based Kerr nonlinearity equalization in dual-polarization fiber-optic communications via a CRNN-based approach. Diss. Institut polytechnique de Paris, 2022 (https://www.theses.fr/2022IPPAT034).

    """

    def __init__(self, z, D=17*(10**-6), lamb=1553*(10**-9), c=3*(10**8), F_s=1, alpha_dB=0, direction=1, name="cd"):
        self.z = z  #  in m
        self.lamb = lamb  # in m
        self.D = D # in ps/m/m
        self.c = c # in m/s
        self.F_s = F_s  # sampling frequency [Hz]
        self.alpha_dB = alpha_dB
        self.direction = direction
        self.name = name

        self.prepare()

    def prepare(self):
        # second-order dispersion coefficient
        # see equation 3 in https://www.hindawi.com/journals/jfs/2022/8316404/
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304

        beta2 = - self.D * self.lamb**2 / (2*np.pi*self.c)

        if self.alpha_dB != 0:
            alpha = (np.log(10)/10)*(self.alpha_dB)
            gain = np.exp(-(alpha/2)*self.z*self.direction)
        else:
            gain = 1
        self.gain = gain 
        self.beta2 = beta2

    def get_H(self, w):
        return np.exp(1j * (self.beta2/2) * self.z * (w**2) * self.direction)

    def forward(self, x):
        NFFT = len(x)
        w = (2*np.pi*self.F_s)*np.fft.fftfreq(NFFT, d=1)
        H = self.get_H(w)
        fftx = np.fft.fft(x)
        ffty = H * fftx
        y = self.gain * np.fft.ifft(ffty)
        return y

    def to_dict(self):
        param = (self.beta2/2) * self.z * self.direction
        return {"param": param, "gain": self.gain, "z": self.z}


class Kerr_NonLinearity(Channel):
    """
    This module implements the phase nonlinearity.

    Parameters
    ----------
    z : float
        Step length [m].
    gamma : float
        Kerr coefficient [rad/W/m].
    alpha_dB : float
        Fiber loss parameter [dB/m].

    References
    ----------

    * [1] Häger, Christian, and Henry D. Pfister. "Physics-based deep learning for fiber-optic communication systems." IEEE Journal on Selected Areas in Communications 39.1 (2020): 280-294 (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304)
    """
    def __init__(self, z, gamma=1.3*10**-3, alpha_dB=0, direction=1, name="nl"):
        self.z = z  # in m
        self.gamma = gamma  # in rad/W/m
        self.alpha_dB = alpha_dB
        self.direction = direction
        self.name = name

        self.prepare()

    def prepare(self):

        if self.alpha_dB != 0:
            alpha = self.alpha_dB * (np.log(10)/10)
            gain = np.exp(-(alpha/2)*self.z*self.direction)
            L_eff = self.direction*(1-np.exp(-alpha*self.z))/alpha
        else:
            gain = 1
            L_eff = self.direction*self.z
            
        self.gain = gain
        self.nl_param = self.gamma*L_eff

    def forward(self, x):
        y = x * self.gain * np.exp(1j*self.nl_param*(np.abs(x)**2))
        return y 

    def to_dict(self):
        return {"param": self.nl_param, "gain": self.gain, "z": self.z}
   

class ASE(Channel):

    """
    This class adds Amplified Spontaneous Emission (ASE) to a signal

    Parameters
    ----------
    alpha_dB : float
        Fiber loss parameter [dB/km].
    L_span : float
        Span length [km].
    NF_dB : float
        Noise figure in dB.
    scaling : int
        Scaling factor m (1 or 2).
    h : float
        Planck’s constant [Js].
    nu : float
        Carrier frequency [Hz].

    Notes
    -----
    The ASE is modeled by an AWGN noise with the power spectral density

    .. math::

        N_{\\text{ase}} = m n_{\\text{sp}} (G-1)h \\nu_s B

    where

    - :math:`B`: optical noise bandwidth,
    - :math:`G=e^{\\alpha L_A}`,
    - :math:`\\alpha` : fiber loss coefficient given by :math:`\\alpha = \\frac{\\alpha_{\\text{dB}}}{10 \log_{10}(e)}`,
    - :math:`n_{\\text{sp}}` : spontaneous-emission factor.

    References
    ----------
    * [1] Shieh, William, and Ivan B. Djordjevic. OFDM for optical communications. Academic press, 2009.
    
    """

    def __init__(self, alpha_dB=0.2*10**-3, L_span=80*10**3, F_s=1, NF_dB=5, scaling=2, h=6.626*(10**-34), nu=1.946*(10**14), name="ASE"):
        self.alpha_dB = alpha_dB  # [dB/m] 
        self.L_span  = L_span   # segment length [m]    
        self.NF_dB = NF_dB  # noise_figure [dB]
        self.h = h 
        self.nu_s = nu
        self.F_s = F_s  #  sampling frequency
        self.scaling = scaling
        self.name = name 

        self.prepare()

    def prepare(self):
        # see table 1 in https://arxiv.org/pdf/2010.14258.pdf
        alpha_dB_tot = self.alpha_dB*self.L_span 
        G = 10**(alpha_dB_tot/10)
        NF = 10**(self.NF_dB/10)
        n_sp = (NF/2)/(1-1/G) # spontaneous emission factor see table1
        B = self.F_s
        p_ase  =  self.scaling * n_sp * (G-1) * self.h * self.nu_s * B
        self.p_ase = p_ase
        
    def forward(self,x):
        N = len(x)
        noise = np.sqrt(self.p_ase/2) * (np.random.randn(N) + 1j * np.random.randn(N))
        y = x + noise 
        return y

    def to_dict(self):
        return {"power": self.p_ase}


class EDFA(Channel):

    """
    This module applies a gain to the input signal to compensate for the fiber loss.

    Parameters
    ----------
    alpha_dB : float
        Fiber loss parameter [dB/km].
    L_span : float
        Span length [km].

    References
    ----------

    * [1] https://www.sciencedirect.com/topics/engineering/spontaneous-emission-factor
    """

    def __init__(self, alpha_dB=0.2*10**-3, L_span=80*10**3, direction=1, name = "EDFA"):
        self.alpha_dB = alpha_dB  # [dB/m] 
        self.L_span  = L_span   # segment length [m]    
        self.direction = direction
        self.name = name
        self.prepare()

    def prepare(self):
        alpha_dB_tot = self.direction*self.alpha_dB*self.L_span 
        G = 10**(alpha_dB_tot/10)
        self.gain = np.sqrt(G)

    def forward(self,x):
        y = self.gain*x
        return y

    def to_dict(self):
        return {"gain": self.gain}


