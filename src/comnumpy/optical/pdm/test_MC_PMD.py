# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import pandas as pd
# import seaborn as sns
# from scipy import signal
# import matplotlib.patches as mpatches
# import os
# from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, LogFormatterMathtext
# import matplotlib.ticker as ticker

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# # import sys
# # sys.path.insert(0, r"D:\comnumpy\src\comnumpy")

# from comnumpy.core import Sequential
# from comnumpy.core.channels import AWGN
# from comnumpy.core.generators import SymbolGenerator
# from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
# from comnumpy.core.utils import get_alphabet, hard_projector
# from comnumpy.optical.pdm.sinks import IQ_Scope, IQ_Scope_PostProcessing
# from comnumpy.optical.pdm.channels import PhaseNoise
# from comnumpy.core.filters import SRRCFilter
# from comnumpy.core.processors import Upsampler, Downsampler
# from comnumpy.optical.pdm.trail_remover import TrailRemover
# from comnumpy.core.monitors import Recorder
# from comnumpy.core.metrics import compute_ser, compute_ber

# from comnumpy.optical.pdm.generics import PDMWrapper, ChannelWrapper
# from comnumpy.optical.pdm.channels import PMD, SOP, PDL
# from comnumpy.optical.pdm.compensators import CMA, RDE, DDLMS, AdaptiveChannel, Switch
# # from comnumpy.optical.pdm.compensators import CMA
# from comnumpy.optical.pdm.utils import *

# # parameters
# type, M = "QAM", 16
# N = 1_200_000
# alphabet = get_alphabet(type, M)

# Fs = 56e9 
# Ts = 1 / 28e9

# Fiber_length = 500 # Km
# D_pmd = 0.1e-12 # ps/sqrt(km), for aggresive cases, choose 0.1
# seg = 1 # number of fiber segments

# t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
# pmd_params = [t_dgd_k, rot_angle_k]
# # PDL #
# pdl_db = 3
# pdl_theta = np.pi/4

# # SRRC filter
# roll_off = 0.35
# srrc_taps = 30

# #PDL parameters
# pdl_db = 3
# pdl_theta = np.pi/4

# # Oversampling 
# oversampling = 2

# ### Impairments parameters ###
# # PMD #
# t_dgd = 1e-12
# rot_angle = np.pi/4

# # SOP drift
# pol_linewidth = None

# # PN
# laser_linewidth = 1e5
# sigma2_pn = 2 * np.pi * laser_linewidth * Ts

# # CMA #
# cma_taps = 7

# # CMA convergence #
# conv = 500_000

# tx = Sequential( [
#             Upsampler(oversampling),
#             SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
# ] )


# channel = Sequential( [
#             # PDL(pdl_db, pdl_theta),
#             PMD(t_dgd, Fs, theta=rot_angle),
#             PDL(pdl_db, pdl_theta),
# ] )

# rx = Sequential( [ #nu mai e necesar Trail Remover
#             Downsampler(oversampling),
# ] )

# P_total = 5 # in dB
# p_seg = P_total / np.sqrt(seg)
# pdl_params = np.random.uniform(0, np.sqrt(3)*p_seg, seg )

# channel_params = [ pol_linewidth, pmd_params, pdl_params ]
# step_LPN = 1e-2
# step_CMA = 1e-3
# # step_CMA_list = [1e-3]
# step_RDE = 1e-3
# step_DD = 1e-4
# step_PMD = 1e-4
# # step_CMA_list = np.linspace(1e-5, 1e-2, num=2)
# step_CMA_list = np.logspace(-5, -2, num = 5)

# recorder_before_CMA = Recorder(name='data_before_CMA')
# recorder_emision = Recorder(name='tx_symbols_emision')
# recorder_real_symb = Recorder(name='tx_symbols')

# recorder_reception = Recorder(name='rx_symbols')

# SNR = 20 
# chain = Sequential([
#             SymbolGenerator(M),
#             Recorder(name='data_tx'),
#             PDMWrapper(SymbolMapper(alphabet)),
#             # Recorder(name='tx_symbols'),
#             recorder_real_symb,
#             PDMWrapper(tx, name='tx'),
#             # Recorder(name='tx_symbols_emision'),
#             recorder_emision,
#             ChannelWrapper(seq_obj=channel, L=seg, params=channel_params),
#             # Selective_Channel(N_r, N_t, N_tap, coef=0.3),
#             # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
#             # AWGN: folosim parametrul 'value' (SNR) si 'unit' pentru tipul valorii
#             # Inlocuire vechi: AWGN(20, method='SNR_dB') -> AWGN(value=20, unit='snr_dB')
#             PDMWrapper( AWGN(value=20, unit='snr_dB'), name='noise'),
#             # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
#             PDMWrapper( SRRCFilter(roll_off, oversampling, srrc_taps, method='fft') ),
#             # Recorder(name='data_before_CMA'),
#             recorder_before_CMA,
#             PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#             # CMA(L=cma_taps, alphabet=alphabet, mu=step_CMA, oversampling=oversampling, name='CMA'),
#             # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#                  # Use keyword args to avoid positional misassignment; ensure L is int and mu is float
#             Switch(L=cma_taps, alphabet=alphabet, mu=step_PMD, oversampling=oversampling, tx_before_CMA=recorder_before_CMA, name='adaptive_channel'),
#             # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#             # Recorder(name='after_phase_correction'),
#             # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#             # CMA(cma_taps, alphabet, step_CMA, oversampling, name='CMA'),
#             # RDE(cma_taps, alphabet, step_RDE, oversampling, name='rde'),
#             # DD_LMS(cma_taps, alphabet, step_RDE, oversampling, name='dd'),
#             PDMWrapper(rx, name='rx'),
#             recorder_reception,
#             PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#             # MSE(recorder_real_symb, alphabet, name='mse'),
#             # PhUnGrid(None, name='PhUnCor'),
#             # PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
#             # PDMWrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
#             PDMWrapper(SymbolDemapper(alphabet)),
#             Recorder(name='data_rx'),
#             ])

# # Monte Carlo loops
# ber_vs_step = []
# nr_repetitions = 5
# avg_ber_per_step = []

# for mu in step_CMA_list:
#     # Reset equalizer state and set new step size for this sweep point
#     # chain["adaptive_channel"].reset()
#     chain["adaptive_channel"].mu = mu
#     ber_runs = []
#     for i in range(nr_repetitions):
#         chain["adaptive_channel"].reset()
#         # chain["adaptive_channel"].mu = float(mu)
#         #chain["CMA"].reset()
#         # refresh ChannelWrapper to reset internal PMD generators
#         for idx, mod in enumerate(chain.module_list):
#             if isinstance(mod, ChannelWrapper):
#                 chain.module_list[idx] = ChannelWrapper(seq_obj=channel, L=seg, params=channel_params)
#                 break
#         y = chain(N)[:, conv:]
#         data_tx = chain['data_tx'].get_data()
#         data_tx1 = np.reshape(data_tx, (2, -1))[0, conv:]
#         data_tx2 = np.reshape(data_tx, (2, -1))[1, conv:]

#         ser1  = compute_ser(data_tx1, y[0,:])
#         ber1 = compute_ber( data_tx1, y[0,:], width=int(np.log2(M)) )
#         ser2  = compute_ser(data_tx2, y[1,:])
#         ber2 = compute_ber( data_tx2, y[1,:], width=int(np.log2(M)) )
#         ser = np.mean( [ser1, ser2] )
#         ber = np.mean( [ber1, ber2] )

#         # prints for each run
#         print(f"mu={mu:.2e} run={i+1}/{nr_repetitions} -> SNR={SNR}: SER pol0: {ser1:.4e}\tBER pol0: {ber1:.4e}")
#         print(f"mu={mu:.2e} run={i+1}/{nr_repetitions} -> SNR={SNR}: SER pol1: {ser2:.4e}\tBER pol1: {ber2:.4e}")

#         M = 1000
#         h = np.ones(M) / M
#         # ber_runs.append(ber)
#         output = chain["adaptive_channel"].get_data()
#         output = output[:, ::oversampling]
#         output = output[:, :-srrc_taps * oversampling]

#         # CMA
#         R = np.mean(np.abs(alphabet)**4) / np.mean(np.abs(alphabet)**2)
#         radius_1 = np.abs(output[0])**2
#         radius_2 = np.abs(output[1])**2
#         error1 = (radius_1 - R) ** 2
#         error2 = (radius_2 - R) ** 2

#         # RDE
#         err1_rde, err2_rde = [], []
#         radius_list = np.unique(np.abs(alphabet)**2)
#         for i in range(cma_taps + 1, len(output[0])):
#             r1 = np.abs(output[0][i]) ** 2
#             r2 = np.abs(output[1][i]) ** 2
#             idx1 = np.argmin((radius_list - r1) ** 2)
#             idx2 = np.argmin((radius_list - r2) ** 2)
#             err1_rde.append((radius_list[idx1] - r1) ** 2)
#             err2_rde.append((radius_list[idx2] - r2) ** 2)

#         plt.figure()
#         length_cma = 50_000
#         end_rde = 600_000
#         rde_len = end_rde - length_cma

#         # CMA
#         plt.plot(np.arange(0, length_cma),
#                     signal.convolve(error1[:length_cma], h, mode='same'),
#                     label="CMA - Pol X")
#         plt.plot(np.arange(0, length_cma),
#                     signal.convolve(error2[:length_cma], h, mode='same'),
#                     label="CMA - Pol Y")

#         # RDE
#         plt.plot(np.arange(length_cma, end_rde),
#                     signal.convolve(err1_rde[:rde_len], h, mode='same'),
#                     label="RDE - Pol X")
#         plt.plot(np.arange(length_cma, end_rde),
#                     signal.convolve(err2_rde[:rde_len], h, mode='same'),
#                     label="RDE - Pol Y")

#         plt.axvline(x=length_cma, color='black', linestyle='--', label='Switch to RDE')
#         plt.title(f'Loss function for {step_PMD:e}')
#         plt.xlabel('n'); plt.ylabel('Error'); plt.legend(); plt.grid(True); plt.tight_layout()

#     avg_ber = np.mean(ber_runs)
#     avg_ber_per_step.append(avg_ber)
    
#     #csv
#     ber_vs_step.append({
#         "step": mu,
#         "BER":avg_ber
#     })
#     print(f"mu={mu:.2e} -> BER(avg over {nr_repetitions}) = {avg_ber:.4e}")

# #save the datas in a csv file
# df = pd.DataFrame(ber_vs_step)
# df.to_csv(f"dates_{seg}_seg_PMD_PDL_Switch_16qam.csv", index=False)


# plt.figure(figsize=(7, 5))
# plt.loglog(step_CMA_list, avg_ber_per_step, marker='o', linewidth=2)

# plt.xlabel("Switch step (μ)")
# plt.ylabel("Average BER over Monte Carlo")
# plt.title(f"BER vs STEP (SNR={SNR} dB)")

# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# ax = plt.gca()
# ax.xaxis.set_major_formatter(LogFormatterMathtext())
# ax.xaxis.set_minor_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(LogFormatterMathtext())
# ax.yaxis.set_minor_formatter(NullFormatter())

# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
from scipy import signal
import matplotlib.patches as mpatches
import os
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, LogFormatterMathtext
import matplotlib.ticker as ticker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# import sys
# sys.path.insert(0, r"D:\comnumpy\src\comnumpy")

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.optical.pdm.sinks import IQ_Scope, IQ_Scope_PostProcessing
from comnumpy.optical.pdm.channels import PhaseNoise
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.optical.pdm.trail_remover import TrailRemover
from comnumpy.core.monitors import Recorder
from comnumpy.core.metrics import compute_ser, compute_ber

from comnumpy.optical.pdm.generics import PDMWrapper, ChannelWrapper
from comnumpy.optical.pdm.channels import PMD, SOP, PDL
from comnumpy.optical.pdm.compensators import CMA, RDE, DDLMS, AdaptiveChannel, Switch
# from comnumpy.optical.pdm.compensators import CMA
from comnumpy.optical.pdm.utils import *

# parameters
type, M = "QAM", 16
N = 1_200_000
alphabet = get_alphabet(type, M)

Fs = 56e9
Ts = 1 / 28e9

Fiber_length = 500 # Km
D_pmd = 0.1e-12 # ps/sqrt(km), for aggresive cases, choose 0.1
seg = 1 # number of fiber segments

t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
pmd_params = [t_dgd_k, rot_angle_k]
# PDL #
pdl_db = 3
pdl_theta = np.pi/4

# SRRC filter
roll_off = 0.35
srrc_taps = 30

#PDL parameters
pdl_db = 3
pdl_theta = np.pi/4

# Oversampling
oversampling = 2

### Impairments parameters ###
# PMD #
t_dgd = 1e-12
rot_angle = np.pi/4

# SOP drift
pol_linewidth = None

# PN
laser_linewidth = 1e5
sigma2_pn = 2 * np.pi * laser_linewidth * Ts

# CMA #
cma_taps = 7

# CMA convergence #
conv = 500_000

tx = Sequential([
    Upsampler(oversampling),
    SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
])

channel = Sequential([
    # PDL(pdl_db, pdl_theta),
    PMD(t_dgd, Fs, theta=rot_angle),
    PDL(pdl_db, pdl_theta),
])

rx = Sequential([ # nu mai e necesar Trail Remover
    Downsampler(oversampling),
])

P_total = 5 # in dB
p_seg = P_total / np.sqrt(seg)
pdl_params = np.random.uniform(0, np.sqrt(3)*p_seg, seg)

channel_params = [pol_linewidth, pmd_params, pdl_params]
step_LPN = 1e-2
step_CMA = 1e-3
# step_CMA_list = [1e-3]
step_RDE = 1e-3
step_DD = 1e-4
step_PMD = 1e-4
# step_CMA_list = np.linspace(1e-5, 1e-2, num=2)
step_CMA_list = np.logspace(-5, -2, num=5)

recorder_before_CMA = Recorder(name='data_before_CMA')
recorder_emision = Recorder(name='tx_symbols_emision')
recorder_real_symb = Recorder(name='tx_symbols')

recorder_reception = Recorder(name='rx_symbols')

SNR = 20
chain = Sequential([
    SymbolGenerator(M),
    Recorder(name='data_tx'),
    PDMWrapper(SymbolMapper(alphabet)),
    # Recorder(name='tx_symbols'),
    recorder_real_symb,
    PDMWrapper(tx, name='tx'),
    # Recorder(name='tx_symbols_emision'),
    recorder_emision,
    ChannelWrapper(seq_obj=channel, L=seg, params=channel_params),
    # Selective_Channel(N_r, N_t, N_tap, coef=0.3),
    # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
    # AWGN: folosim parametrul 'value' (SNR) si 'unit' pentru tipul valorii
    # Inlocuire vechi: AWGN(20, method='SNR_dB') -> AWGN(value=20, unit='snr_dB')
    PDMWrapper(AWGN(value=20, unit='snr_dB'), name='noise'),
    # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
    PDMWrapper(SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')),
    # Recorder(name='data_before_CMA'),
    recorder_before_CMA,
    PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # CMA(L=cma_taps, alphabet=alphabet, mu=step_CMA, oversampling=oversampling, name='CMA'),
    # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # Use keyword args to avoid positional misassignment; ensure L is int and mu is float
    Switch(L=cma_taps, alphabet=alphabet, mu=step_PMD, oversampling=oversampling, tx_before_CMA=recorder_before_CMA, name='adaptive_channel'),
    # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # Recorder(name='after_phase_correction'),
    # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # CMA(cma_taps, alphabet, step_CMA, oversampling, name='CMA'),
    # RDE(cma_taps, alphabet, step_RDE, oversampling, name='rde'),
    # DD_LMS(cma_taps, alphabet, step_RDE, oversampling, name='dd'),
    PDMWrapper(rx, name='rx'),
    recorder_reception,
    PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # MSE(recorder_real_symb, alphabet, name='mse'),
    # PhUnGrid(None, name='PhUnCor'),
    # PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
    # PDMWrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
    PDMWrapper(SymbolDemapper(alphabet)),
    Recorder(name='data_rx'),
])

# Monte Carlo loops
ber_vs_step = []
nr_repetitions = 5
avg_ber_per_step = []

for mu in step_CMA_list:
    # Reset equalizer state and set new step size for this sweep point
    chain["adaptive_channel"].mu = mu
    ber_runs = []
    for i in range(nr_repetitions):
        chain["adaptive_channel"].reset()
        # refresh ChannelWrapper to reset internal PMD generators
        for idx, mod in enumerate(chain.module_list):
            if isinstance(mod, ChannelWrapper):
                chain.module_list[idx] = ChannelWrapper(seq_obj=channel, L=seg, params=channel_params)
                break
        y = chain(N)[:, conv:]
        data_tx = chain['data_tx'].get_data()
        data_tx1 = np.reshape(data_tx, (2, -1))[0, conv:]
        data_tx2 = np.reshape(data_tx, (2, -1))[1, conv:]

        ser1  = compute_ser(data_tx1, y[0,:])
        ber1 = compute_ber(data_tx1, y[0,:], width=int(np.log2(M)))
        ser2  = compute_ser(data_tx2, y[1,:])
        ber2 = compute_ber(data_tx2, y[1,:], width=int(np.log2(M)))
        ser = np.mean([ser1, ser2])
        ber = np.mean([ber1, ber2])

        # prints for each run
        print(f"mu={mu:.2e} run={i+1}/{nr_repetitions} -> SNR={SNR}: SER pol0: {ser1:.4e}\tBER pol0: {ber1:.4e}")
        print(f"mu={mu:.2e} run={i+1}/{nr_repetitions} -> SNR={SNR}: SER pol1: {ser2:.4e}\tBER pol1: {ber2:.4e}")

        # collect BER for averaging
        ber_runs.append(float(ber))

        # loss plotting
        win_len = 1000
        h = np.ones(win_len) / win_len
        output = chain["adaptive_channel"].get_data()
        output = output[:, ::oversampling]
        output = output[:, :-srrc_taps * oversampling]

        # CMA
        R = np.mean(np.abs(alphabet)**4) / np.mean(np.abs(alphabet)**2)
        radius_1 = np.abs(output[0])**2
        radius_2 = np.abs(output[1])**2
        error1 = (radius_1 - R) ** 2
        error2 = (radius_2 - R) ** 2

        # RDE
        err1_rde, err2_rde = [], []
        radius_list = np.unique(np.abs(alphabet)**2)
        for k in range(cma_taps + 1, len(output[0])):
            r1 = np.abs(output[0][k]) ** 2
            r2 = np.abs(output[1][k]) ** 2
            idx1 = np.argmin((radius_list - r1) ** 2)
            idx2 = np.argmin((radius_list - r2) ** 2)
            err1_rde.append((radius_list[idx1] - r1) ** 2)
            err2_rde.append((radius_list[idx2] - r2) ** 2)

        plt.figure()
        length_cma = 100_000
        end_rde = 600_000
        rde_len = end_rde - length_cma

        # CMA
        plt.plot(np.arange(0, length_cma),
                 signal.convolve(error1[:length_cma], h, mode='same'),
                 label="CMA - Pol X")
        plt.plot(np.arange(0, length_cma),
                 signal.convolve(error2[:length_cma], h, mode='same'),
                 label="CMA - Pol Y")

        # RDE
        plt.plot(np.arange(length_cma, end_rde),
                 signal.convolve(err1_rde[:rde_len], h, mode='same'),
                 label="RDE - Pol X")
        plt.plot(np.arange(length_cma, end_rde),
                 signal.convolve(err2_rde[:rde_len], h, mode='same'),
                 label="RDE - Pol Y")

        plt.axvline(x=length_cma, color='black', linestyle='--', label='Switch to RDE')
        plt.title(f'Loss function for {step_PMD:e}')
        plt.xlabel('n'); plt.ylabel('Error'); plt.legend(); plt.grid(True); plt.tight_layout()

    # average BER across repetitions
    avg_ber = float(np.mean(ber_runs))
    avg_ber_per_step.append(avg_ber)

    # csv
    ber_vs_step.append({
        "step": mu,
        "BER": avg_ber
    })
    print(f"mu={mu:.2e} -> BER(avg over {nr_repetitions}) = {avg_ber:.4e}")

# save the datas in a csv file
df = pd.DataFrame(ber_vs_step)
df.to_csv(f"dates_{seg}_seg_PMD_PDL_Switch_16qam.csv", index=False)

plt.figure(figsize=(7, 5))
plt.loglog(step_CMA_list, avg_ber_per_step, marker='o', linewidth=2)

plt.xlabel("Switch step (μ)")
plt.ylabel("Average BER over Monte Carlo")
plt.title(f"BER vs STEP (SNR={SNR} dB)")

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()
plt.show()