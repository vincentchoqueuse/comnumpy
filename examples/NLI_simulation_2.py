import numpy as np
import sys

sys.path.insert(0, '../src')

from core import Sequential
from channels.optical import ASE, NonLinearity, Fiber_Link, Symmetric_Step_Mixin, Logarithmic_Step_Size_Mixin
from dsp.frontend import Upsampler, Downsampler, SRRC_filter
from dsp.optical import DBP
from dsp.compensator import Data_Aided_Phase
from metrics.functional import compute_effective_SNR
import matplotlib.pyplot as plt
from tqdm import tqdm

# define new modules for non linearity 
# I use exactly the same expressions as described by Hager (https://github.com/chaeger/LDBP/blob/fd5b6f7c3f2a3409f1c30f1725b1474a4dff9662/ldbp/ldbp.py#L431)
# https://arxiv.org/pdf/2010.14258.pdf

class NonLinearity_Hager(NonLinearity):

    def __init__(self, z, gamma, gain_pre=1, alpha_dB=0, direction=1, name="nl"):
        self.z = z  # [m]
        self.gamma = gamma  # nonlinear parameter [1/(W*m)]
        self.alpha_dB = alpha_dB
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
        self.nl_param = self.direction * self.gamma * L_eff * (self.gain_pre**2)

# Customize Fiber Link and DBP

class  Fiber_Link_Hager(Symmetric_Step_Mixin, Logarithmic_Step_Size_Mixin, Fiber_Link):

    def __init__(self, N_span, StPS, L_span, gamma=1.3*1e-3, lamb=1.55 * 10**-6, c=3*(10**8), h=6.626*(10**-34), nu=1.946*(10**14), alpha_dB= 0.2*1e-3, F_s=1, NF_dB = 4, name="span"):
        self.N_span = N_span
        self.StPS = StPS
        self.L_span = L_span
        self.gamma = gamma 
        self.alpha_dB = alpha_dB
        self.c = c
        self.h = h 
        self.lamb = lamb
        self.nu_s = nu
        self.F_s = F_s 
        self.NF_dB = NF_dB
        self.noise_scaling = 2
        self.name = name 
        self.prepare()

    def get_non_linear_module(self, dz, gain_pre=1):
        return NonLinearity_Hager(dz, self.gamma, gain_pre = gain_pre, direction=1)

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
            cd = self.get_cd_module(dz_temp)
            nl = self.get_non_linear_module(dz, gain_pre=gain)
            dz_rest = dz/2
            module_list.extend([cd, nl])

        # last step
        cd = self.get_cd_module(dz_rest)
        module_list.append(cd)

        # end of span
        ase = ASE(self.alpha_dB, L_span = self.L_span, h=self.h, nu = self.nu_s, F_s=self.F_s, NF_dB=self.NF_dB, scaling=self.noise_scaling)
        module_list.append(ase)
        return module_list


class DBP_Hager(DBP):

    def __init__(self, N_span, StPS, L_span, gamma=1.3*1e-3, lamb=1.55 * 10**-6, c=3*(10**8), alpha_dB= 0.2*1e-3, F_s=1, name="DBP"):
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

    def get_non_linear_module(self, dz, gain_pre=1):
        return NonLinearity_Hager(dz, self.gamma, alpha_dB=self.alpha_dB, gain_pre=gain_pre, direction=-1)

    def get_span_module_list(self):
        z = self.get_step_size()
        alpha_lin = (np.log(10)/10)*(self.alpha_dB)
        gain = np.prod(np.exp(-(alpha_lin/2)*z[1:]))
        #gain = np.exp(-(alpha_lin/2)*self.L_span)
    
        module_list = []
        for num_step in range(self.StPS):
            dz = z[num_step]
            cd = self.get_cd_module(dz)
            nl = self.get_non_linear_module(dz, gain_pre=gain)
            gain = gain*np.exp(-(alpha_lin/2)*dz*self.direction)
            module_list.extend([cd, nl])

        return module_list


## parameters
system = 1
sigma2_s = 1
alpha_dB =  0.2*1e-3
lamb = 1.55 * 10**-6
gamma = 1.3*(10**-3)
D = 17*(10**-6)
N_s = 2**8  # number of symbols
oversampling_sim = 6  #number of samples/symbol for simulated channel
oversampling_dsp = 6  #number of samples/symbol for dsp processing
NF_dB = 5 # noise figure in dB
rolloff = 0.1
StPS = 50
c = 299792458
h = 6.6260657*10**-34
nu = 193414489032258.06

if system == 0:
    R_s = 10.7*(10**9)  # baud rate (number of symbols / second)
    L_span = 80 * (10**3)
    N_span = 25
    dBm_list = np.arange(-12, 3)
    
if system == 1:
    R_s = 32*(10**9)  # baud rate (number of symbols / second)
    L_span = 100 * (10**3)
    N_span = 10
    dBm_list = np.arange(-6, 9)
        
N = N_s * oversampling_sim   # signal length in number of samples
F_s = R_s*oversampling_sim
oversampling_ratio = int(oversampling_sim/oversampling_dsp)
F_s2 = R_s*oversampling_dsp

# Export parameters for SSFM Link and DBP
ssfm = Fiber_Link_Hager(N_span, StPS, L_span, gamma=gamma, alpha_dB=alpha_dB, c=c, h=h, nu=nu, NF_dB=NF_dB, F_s=F_s)
ssfm.to_json(path="ssfm_hager.json")

dbp = DBP_Hager(N_span, 3, L_span, gamma=gamma, alpha_dB=alpha_dB, c=c, F_s=F_s/3, name="DBP")
dbp.to_json(path="dbp_hager.json")


# reference curve (no non-linearity)
chain_linear = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRC_filter(rolloff, oversampling_sim, method="fft"),
                Fiber_Link_Hager(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, c=c, h=h, nu=nu, NF_dB=NF_dB, F_s=F_s),
                SRRC_filter(rolloff, oversampling_sim, method="fft", scale=1/np.sqrt(oversampling_sim)),
                Downsampler(oversampling_ratio),
                DBP_Hager(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, c=c, F_s=F_s/oversampling_ratio, name="DBP"),
                Downsampler(oversampling_dsp)
                ])

# SSFM (generate the signal output after the communication chain)
chain_non_linear = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRC_filter(rolloff, oversampling_sim, method="fft"),
                Fiber_Link_Hager(N_span, StPS, L_span, gamma=gamma, alpha_dB=alpha_dB, c=c, h=h, nu=nu, NF_dB=NF_dB, F_s=F_s),
                SRRC_filter(rolloff, oversampling_sim, method="fft", scale=1/np.sqrt(oversampling_sim)),
                Downsampler(oversampling_ratio)])


# construct rx chain with different DBP configuration
DBP_list = [{"name": "linear", "N_step": 1, "gamma":0}, 
            {"name": "DBP2", "N_step": 2, "gamma":gamma},
            {"name": "DBP4", "N_step": 4, "gamma":gamma},
            {"name": "DBP500", "N_step": 500, "gamma":gamma}]

chain_rx_list = []
for index, DBP_temp in enumerate(DBP_list):
    N_step_temp = DBP_temp["N_step"]
    gamma_temp = DBP_temp["gamma"]
    # create a receiver processing
    chain_dbp = Sequential([
                DBP_Hager(N_span, N_step_temp, L_span, gamma=gamma_temp, alpha_dB=alpha_dB, c=c, F_s=F_s/oversampling_ratio, name="DBP"),
                Downsampler(oversampling_dsp)
                ])
    chain_rx_list.append(chain_dbp)

# perform monte carlo simulation
N_trial = 10
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
        y_output = chain_linear(amp*x)

        # post process amp+phase
        x_est = (1/amp)*y_output
        phase_compensator.fit(x_est, x) 
        x_est2 = phase_compensator(x_est)
        snr = compute_effective_SNR(x, x_est2, unit="dB")
        snr_array[index, 0] += (1/N_trial) * snr

        # non linear channel output
        z_output = chain_non_linear(amp*x)

        for index_chain, chain_rx in enumerate(chain_rx_list):
            y_output = chain_rx(z_output)

            # post process amp+phase
            x_est = (1/amp)*y_output
            phase_compensator.fit(x_est, x) 
            x_est2 = phase_compensator(x_est)
            snr = compute_effective_SNR(x, x_est2, unit="dB")
            snr_array[index, index_chain+1] += (1/N_trial) * snr


# construct legend name
legend_names = ["gamma=0"]
for index, DBP_temp in enumerate(DBP_list):
    legend_names.append(DBP_temp["name"])
    
plt.figure()
plt.plot(dBm_list, snr_array)
plt.ylabel("Effective SNR (dB)")
plt.legend(legend_names)
plt.xlabel("dBm")
plt.show()
