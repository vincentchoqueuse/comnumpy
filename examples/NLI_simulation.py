import numpy as np
import sys

sys.path.insert(0, '../src')

from core import Sequential
from channels.optical import EDFA, Fiber_Link, Symmetric_Step_Mixin, Scheme_I_Step_Mixin
from dsp.frontend import Upsampler, Downsampler, SRRC_filter
from dsp.optical import DBP
from dsp.compensator import Data_Aided_Phase
from metrics.functional import compute_effective_SNR
import matplotlib.pyplot as plt
from tqdm import tqdm


# define new classes for SSFM et DBP

class Symmetric_Fiber_Link_I(Symmetric_Step_Mixin, Scheme_I_Step_Mixin, Fiber_Link):
    name = "symmetric_fiber_link"

class Symmetric_DBP(Symmetric_Step_Mixin, Scheme_I_Step_Mixin, DBP):
    name = "symmetric_DBP"

class Only_Linear_DBP(Scheme_I_Step_Mixin, DBP):
    name = "only_linear"

    def get_span_module_list(self):
        """
        return the module list for a single span
        """
        module_list = self.get_pre_span_module_list()
        cd = self.get_cd_module(self.L_span)
        module_list.append(cd)
        return module_list 


## parameters
system = 0
sigma2_s = 1
alpha_dB =  0.2*1e-3
lamb = 1.55 * 10**-6
gamma = 1.3*(10**-3)
D = 17*(10**-6)
N_s = 2**8  # number of symbols
oversampling_sim = 6  #number of samples/symbol for simulated channel
oversampling_dsp = 6  #number of samples/symbol for dsp processing
NF_dB = 4 # noise figure in dB
rolloff = 0.1
StPS = 50
noise_scaling = 1


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

DBP_list = [
            {"name": "linear", "N_step": 1, "technique": Only_Linear_DBP},
            {"name": "DBP_1", "N_step": 1, "technique": Symmetric_DBP},
            {"name": "DBP_2", "N_step": 2, "technique": Symmetric_DBP},
            {"name": "DBP_3", "N_step": 3, "technique": Symmetric_DBP},
            {"name": "DBP_50", "N_step": 50, "technique": Symmetric_DBP}]

chain_ref = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRC_filter(rolloff, oversampling_sim, method="fft"),
                Symmetric_Fiber_Link_I(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, NF_dB=NF_dB, F_s=F_s, noise_scaling=noise_scaling),
                SRRC_filter(rolloff, oversampling_sim, method="fft", scale=1/np.sqrt(oversampling_sim)),
                Downsampler(oversampling_ratio),
                Symmetric_DBP(N_span, StPS, L_span, gamma=0, alpha_dB=alpha_dB, F_s=F_s/oversampling_ratio),
                Downsampler(oversampling_dsp)
                ])

chain_tx = Sequential([
            Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
            SRRC_filter(rolloff, oversampling_sim, method="fft"),
            Symmetric_Fiber_Link_I(N_span, StPS, L_span, gamma=gamma, alpha_dB=alpha_dB, NF_dB=NF_dB, F_s=F_s, noise_scaling = noise_scaling),
            SRRC_filter(rolloff, oversampling_sim, method="fft", scale=1/np.sqrt(oversampling_sim)),
            Downsampler(oversampling_ratio)
            ])

# construct list of receiver chain
chain_rx_list = []
for index_DBP, DBP_temp in enumerate(DBP_list):
    N_step = DBP_temp["N_step"]
    name = DBP_temp["name"]
    technique = DBP_temp["technique"]
    chain_rx_temp = Sequential([
        technique(N_span, N_step, L_span, gamma=gamma, alpha_dB=alpha_dB, F_s=F_s/oversampling_ratio),
        Downsampler(oversampling_dsp),
        ])
    chain_rx_temp.name = name
    chain_rx_list.append(chain_rx_temp)


N_trial = 10
N_curves = len(chain_rx_list)+1
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
        y_output = chain_ref(amp*x)

        # post process amp+phase
        x_est = (1/amp)*y_output
        phase_compensator.fit(x_est, x) 
        x_est2 = phase_compensator(x_est)
        snr = compute_effective_SNR(x, x_est2, unit="dB")
        snr_array[index, 0] += (1/N_trial) * snr

        # channel simulation + DBP
        z = chain_tx(amp*x)

        for index_chain, chain_rx in enumerate(chain_rx_list):
            
            y_output = chain_rx(z)
            # post process amp+phase
            x_est = (1/amp)*y_output
            phase_compensator.fit(x_est, x) 
            x_est2 = phase_compensator(x_est)
            snr = compute_effective_SNR(x, x_est2, unit="dB")
            snr_array[index, 1+index_chain] += (1/N_trial) * snr


# construct legend name
legend_names = ["gamma=0"]
for chain in chain_rx_list:
    legend_names.append(chain.name)

# plot figures
plt.figure()
plt.plot(dBm_list, snr_array)
plt.ylabel("Effective SNR (dB)")
plt.legend(legend_names)
plt.xlabel("dBm")
plt.show()