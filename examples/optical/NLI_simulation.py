import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from comnumpy.core import Sequential
from comnumpy.core.generators import GaussianGenerator
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.core.filters import SRRCFilter, BWFilter
from comnumpy.core.metrics import compute_effective_SNR
from comnumpy.optical.links import FiberLink
from comnumpy.optical.dbp import DBP

# This script uses exactly the same setup as described by Hager (https://github.com/chaeger/LDBP/blob/fd5b6f7c3f2a3409f1c30f1725b1474a4dff9662/ldbp/ldbp.py#L431)
# https://arxiv.org/pdf/2010.14258.pdf
# this simulation try to reproduce the curves of the publication using classicial SSFM and DBP layers.
#
# Be aware: The default configuration (N_trial=10) results in an execution time of approximately 30 minutes.
# parameters

system = 1
N_s = 2**9  # number of symbols
oversampling_sim = 6  # number of samples/symbol for simulated channel
oversampling_dsp = 2  # number of samples/symbol for dsp processing
NF_dB = 5  # noise figure in dB
rolloff = 0.1
StPS = 500
export_params = True
DEBUG = False

if system == 0:
    R_s = 10.7*(10**9)  # baud rate
    L_span = 80  # in km
    N_span = 25
    dBm_list = np.arange(-10, 6)
    y_lim = [13, 25]
    x_lim = [-10, 5]
    
if system == 1:
    R_s = 32*(10**9)  # baud rate
    L_span = 100  # in km
    N_span = 10
    dBm_list = np.arange(-6, 9)
    y_lim = [16.5, 29.5]
    x_lim = [-6, 8]

N = N_s*oversampling_sim   # signal length in number of samples
fs = R_s*oversampling_sim
oversampling_ratio = int(oversampling_sim/oversampling_dsp)
fs2 = R_s*oversampling_dsp

# create signal generator
generator = GaussianGenerator()

# reference curve (wihtout nonlinearity)
linear_channel = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRCFilter(rolloff, oversampling_sim, method="fft"),
                FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB, fs=fs, use_only_linear=True, name="link"),
                BWFilter(1/oversampling_sim),
                Downsampler(oversampling_ratio),
                ])

# SSFM (generate the signal output after the communication chain)
non_linear_channel = Sequential([
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRCFilter(rolloff, oversampling_sim, method="fft"),
                FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB, fs=fs, name="link"),
                BWFilter(1/oversampling_sim),
                Downsampler(oversampling_ratio),
                ])

# construct receivers
receiver_config_list = [
            {"name": "linear link", "StPS": 1, "use_only_linear": True, "step_type": "linear"},
            {"name": "linear", "StPS": 1, "use_only_linear": True, "step_type": "linear"},
            {"name": "DBP1", "StPS": 1, "use_only_linear": False, "step_type": "linear"},
            {"name": "DBP2 (log)", "StPS": 2, "use_only_linear": False, "step_type": "logarithmic"},
            {"name": "DBP4 (log)", "StPS": 4, "use_only_linear": False, "step_type": "logarithmic"},
            {"name": "DBP500", "StPS": 500, "use_only_linear": False, "step_type": "linear"}
            ]
receiver_list = []
for index, scenario_temp in enumerate(receiver_config_list):
    receiver = Sequential([
                DBP(N_span, L_span, scenario_temp["StPS"], step_type=scenario_temp["step_type"], fs=fs/oversampling_ratio, use_only_linear=scenario_temp["use_only_linear"], name="dbp"),
                SRRCFilter(rolloff, oversampling_dsp, method="fft", scale=1/np.sqrt(oversampling_dsp)),
                Downsampler(oversampling_dsp),
                ])
    receiver_list.append(receiver)


# perform monte carlo simulation
N_trial = 10
N_dBm = len(dBm_list)

start_time = time.time()

receiver_list = receiver_list
N_curves = len(receiver_list)
snr_array = np.zeros((N_dBm, N_curves))

for index in tqdm(range(N_dBm)):

    dBm = dBm_list[index]
    Po = (1e-3) * (10**(0.1*dBm))
    amp = np.sqrt(Po)

    for num_trial in range(N_trial):

        x = generator(N_s)
        x_scaled = amp * x

        for indice in range(len(receiver_list)):

            if indice == 0:  # linear channel
                z = linear_channel(x_scaled)
            else:
                z = non_linear_channel(x_scaled)

            # get receiver
            receiver = receiver_list[indice]

            # dsp non-linear compensation + scale and phase correction
            y = receiver(z)
            theta_est = np.angle(np.sum(np.conj(y)*x_scaled))   # estimate phase
            coef = (1/amp)*np.exp(1j*theta_est)                 # compute estimated phase + deterministic amplitude correction
            x_est = coef * y                                    # compensate signal
        
            # compute metric
            snr = compute_effective_SNR(x, x_est)
            snr_array[index, indice] += (1/N_trial) * snr

stop_time = time.time()
delta_time = stop_time - start_time
print(f"execution time: {delta_time}")

legend_names = [config["name"] for config in receiver_config_list]
plt.figure()
plt.plot(dBm_list, 10*np.log10(snr_array))
plt.ylabel("Effective SNR (dB)")
plt.legend(legend_names)
plt.xlabel("dBm")
plt.ylim(y_lim)
plt.xlim(x_lim)
plt.grid()
plt.show()
