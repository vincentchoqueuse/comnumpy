import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from comnumpy import Sequential
from comnumpy.channels.optical import ASE, Kerr_NonLinearity
from comnumpy.channels.ssfm import Base_Fiber_Link, Linear_Step_Size_Mixin, Logarithmic_Step_Size_Mixin
from comnumpy.dsp.frontend import Upsampler, Downsampler, SRRC_filter, BW_filter
from comnumpy.dsp.dbp import Base_DBP
from comnumpy.dsp.compensator import Data_Aided_Phase
from comnumpy.metrics.functional import compute_effective_SNR

# define new modules for non linearity 
# I use exactly the same expressions as described by Hager (https://github.com/chaeger/LDBP/blob/fd5b6f7c3f2a3409f1c30f1725b1474a4dff9662/ldbp/ldbp.py#L431)
# https://arxiv.org/pdf/2010.14258.pdf


class Kerr_NonLinearity_Hager(Kerr_NonLinearity):

    """
    This module implements the phase nonlinearity

    :param alpha_dB: fiber loss parameter [dB/m]
    :param z: step length [m]
    :param gamma: kerr coefficient [rad/W/m]

    See also:

    * Physics-Based Deep Learning for Fiber-Optic Communication Systems
    * https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304
    """

    def __init__(self, z, gamma, gain_pre=1, alpha_dB=0, direction=1, name="nl"):
        self.z = z  # in m
        self.gamma = gamma  # rad/W/m
        self.alpha_dB = alpha_dB # in dB/m
        self.direction = direction
        self.gain_pre = gain_pre
        self.name = name
        self.prepare()

    def prepare(self):
        """add gain and nl_parameter. In the Hager implementation the gain is accumulated in the nonlinearity"""
        if self.direction == 1:
            L_eff = self.z
        else:
            alpha = (np.log(10)/10) * (self.alpha_dB)
            L_eff = ((1-np.exp(-alpha*self.z))/alpha)

        self.gain = 1
        self.nl_param = self.direction*self.gamma * L_eff * (self.gain_pre**2)

# Customize Fiber Link and DBP
class Fiber_Link(Logarithmic_Step_Size_Mixin, Base_Fiber_Link):

    """
    A Multi_Span_Link link composed of N spans. Each span is composed of StPS step. Each step is composed of 
    linear (CD) and non-linear (Kerr Nonlinearity) sections.

    :param N_span: Number of spans
    :param StPS: Number of step by span
    :param lamb: wavelength [m]
    :param L_span: Length of span [m]
    :param alpha_dB: fiber loss parameter [dB/m]
    :param gamma: kerr coefficient [rad/W/m]
    :param c: speed of light [m/s]
    :param h: Planck’s constant [Js]
    :param nu: carrier frequency [Hz]


    See Also:

    * https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6860304
    """

    def __init__(self, N_span, StPS, L_span, gamma=1.3*(10**-3), lamb=1550*(10**-9), c=299792458, h=6.6260657*10**-34, nu=193414489032258.06, alpha_dB=0.2*10**-3, F_s=1, NF_dB = 4, name="span"):
        self.N_span = N_span
        self.StPS = StPS
        self.L_span = L_span # in km
        self.gamma = gamma  # in rad/W/m
        self.alpha_dB = alpha_dB # in dB/m
        self.c = c # in m/s
        self.h = h  # in Js
        self.lamb = lamb # in m 
        self.nu_s = nu # in Hz
        self.F_s = F_s # in Hz
        self.NF_dB = NF_dB
        self.noise_scaling = 2
        self.name = name 
        self.prepare()

    def get_linear_module(self, dz):
        return Chromatic_Dispersion(dz, alpha_dB=0, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=1)

    def get_non_linear_module(self, dz, gain_pre=1):
        return Kerr_NonLinearity_Hager(dz, self.gamma, gain_pre = gain_pre, direction=1)

    def get_span_module_list(self):
        """
        return the module list for a single span
        """
        module_list = []
        alpha_lin = (np.log(10)/10)*(self.alpha_dB)
        gain = 1
        dz_rest = 0

        z = self.get_step_size()
        for num_step in range(self.StPS):
            dz = z[num_step]
            dz_temp = dz/2 + dz_rest
            gain = gain*np.exp(-(alpha_lin/2)*dz_temp*self.direction)
            cd = self.get_linear_module(dz_temp)
            nl = self.get_non_linear_module(dz, gain_pre=gain)
            dz_rest = dz/2
            module_list.extend([cd, nl])

        # last step
        cd = self.get_linear_module(dz_rest)
        module_list.append(cd)

        # end of span
        ase = ASE(self.alpha_dB, L_span = self.L_span, h=self.h, nu = self.nu_s, F_s=self.F_s, NF_dB=self.NF_dB, scaling=self.noise_scaling)
        module_list.append(ase)
        return module_list


class DBP(Linear_Step_Size_Mixin, Base_DBP):

    def __init__(self, N_span, StPS, L_span, gamma=1.3, lamb=1.55*10**-6, c=299792458, alpha_dB= 0.2, F_s=1, name="DBP"):
        self.N_span = N_span  # number of spans
        self.StPS = StPS
        self.L_span = L_span
        self.gamma = gamma 
        self.lamb = lamb
        self.c = c
        self.alpha_dB = alpha_dB
        self.F_s = F_s 
        self.name = name 
        self.prepare()

    def get_pre_span_module_list(self):
        return []

    def get_span_module_list(self):
        z = self.get_step_size()
        alpha_lin = (np.log(10)/10)*(self.alpha_dB)
        gain = np.prod(np.exp(-(alpha_lin/2)*z[1:]))

        module_list = []
        for num_step in range(self.StPS):
            dz = z[num_step]
            nl = Kerr_NonLinearity_Hager(dz, self.gamma, alpha_dB=self.alpha_dB, gain_pre=gain, direction=-1)
            cd = Chromatic_Dispersion(dz, alpha_dB=0, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=-1)
            gain = gain*np.exp(-(alpha_lin/2)*dz*self.direction)
            module_list.extend([cd, nl])

        return module_list


## parameters
system = 0
sigma2_s = 1
alpha_dB =  0.2*(10**-3)
gamma = 1.3*(10**-3)
D = 17
lamb=1.55*10**-6
c = 299792458
h = 6.6260657*10**-34
nu = 193414489032258.06
N_s = 2**9  # number of symbols
oversampling_sim = 6  #number of samples/symbol for simulated channel
oversampling_dsp = 2  #number of samples/symbol for dsp processing
NF_dB = 5 # noise figure in dB
rolloff = 0.1
StPS = 50
export_params = True
DEBUG = False

if system == 0:
    R_s = 10.7*(10**9)  # baud rate
    L_span = 80*10**3 # in m
    N_span = 25
    dBm_list = np.arange(-10, 6)
    y_lim = [13, 25]
    x_lim = [-10, 5]
    
if system == 1:
    R_s = 32*(10**9)  # baud rate
    L_span = 100*10**3 # in m
    N_span = 10
    dBm_list = np.arange(-6, 9)
    y_lim = [15.5, 29.5]

        
N = N_s*oversampling_sim   # signal length in number of samples
F_s = R_s*oversampling_sim
oversampling_ratio = int(oversampling_sim/oversampling_dsp)
F_s2 = R_s*oversampling_dsp

# Export parameters for SSFM Link and DBP
ssfm = Fiber_Link(N_span, StPS, L_span, gamma=gamma, lamb=lamb, c=c, h=h, nu=nu, alpha_dB=alpha_dB, NF_dB=NF_dB, F_s=F_s)
dbp = DBP(N_span, 1, L_span, gamma=gamma, lamb=lamb, c=c, alpha_dB=alpha_dB, F_s=F_s/oversampling_ratio, name="DBP")

if export_params:
    ssfm.to_json(path="ssfm_hager_{}.json".format(system))
    dbp.to_json(path="dbp_hager_{}.json".format(system))

# reference curve (no non-linearity)
chain_linear = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRC_filter(rolloff, oversampling_sim, method="fft"),
                Fiber_Link(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, c=c, h=h, nu=nu, NF_dB=NF_dB, F_s=F_s),
                BW_filter(1/oversampling_sim),
                Downsampler(oversampling_ratio),
                DBP(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, c=c, F_s=F_s/oversampling_ratio, name="DBP"),
                SRRC_filter(rolloff, oversampling_dsp, method="fft", scale=1/np.sqrt(oversampling_dsp)),
                Downsampler(oversampling_dsp)
                ])

# SSFM (generate the signal output after the communication chain)
chain_non_linear = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRC_filter(rolloff, oversampling_sim, method="fft"),
                Fiber_Link(N_span, StPS, L_span, gamma=gamma, alpha_dB=alpha_dB, c=c, h=h, nu=nu, NF_dB=NF_dB, F_s=F_s),
                BW_filter(1/oversampling_sim),
                Downsampler(oversampling_ratio)
                ])


# construct rx chain with different DBP configuration
DBP_list = [{"name": "linear", "N_step": 1, "gamma":0}, 
            {"name": "DBP1", "N_step": 1, "gamma":gamma},
            {"name": "DBP2", "N_step": 2, "gamma":gamma}
            ]

chain_rx_list = []
for index, DBP_temp in enumerate(DBP_list):
    N_step_temp = DBP_temp["N_step"]
    gamma_temp = DBP_temp["gamma"]
    # create a receiver processing
    chain_dbp = Sequential([
                DBP(N_span, N_step_temp, L_span, gamma=gamma_temp, lamb=lamb, c=c, alpha_dB=alpha_dB, F_s=F_s/oversampling_ratio, name="DBP"),
                SRRC_filter(rolloff, oversampling_dsp, method="fft", scale=1/np.sqrt(oversampling_dsp)),
                Downsampler(oversampling_dsp),
                ])
    chain_rx_list.append(chain_dbp)

# perform monte carlo simulation
N_trial = 20
N_curves = 1+len(DBP_list)
N_dBm = len(dBm_list)
snr_array = np.zeros((N_dBm, N_curves))
phase_compensator = Data_Aided_Phase()

for index in tqdm(range(N_dBm)):

    dBm = dBm_list[index]
    Po = (1e-3)*10**(0.1*dBm)
    amp = np.sqrt(Po)

    for num_trial in range(N_trial):

        # transmitted symbols
        x = np.sqrt(sigma2_s/2)*(np.random.randn(N_s)+ 1j*np.random.randn(N_s))

        # reference chain
        x_amp = amp*x
        y_output = chain_linear(x_amp)

        # post process amp+phase
        x_est = (1/amp)*y_output
        phase_compensator.fit(x_est, x) 
        x_est2 = phase_compensator(x_est)

        snr = compute_effective_SNR(x, x_est2)
        snr_array[index, 0] += (1/N_trial) * snr

        # non linear channel output
        z_output = chain_non_linear(amp*x)
        for index_chain, chain_rx in enumerate(chain_rx_list):
            y_output = chain_rx(z_output)

            # post process amp+phase
            x_est = (1/amp)*y_output
            phase_compensator.fit(x_est, x) 
            x_est2 = phase_compensator(x_est)
            snr = compute_effective_SNR(x, x_est2)
            
            snr_array[index, index_chain+1] += (1/N_trial) * snr

# construct legend name
legend_names = ["gamma=0"]
for index, DBP_temp in enumerate(DBP_list):
    legend_names.append(DBP_temp["name"])
    

print(10*np.log10(snr_array))
plt.figure()
plt.plot(dBm_list, 10*np.log10(snr_array))
plt.ylabel("Effective SNR (dB)")
plt.legend(legend_names)
plt.xlabel("dBm")
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.grid()
plt.show()
