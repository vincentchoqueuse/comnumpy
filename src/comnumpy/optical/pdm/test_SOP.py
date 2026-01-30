import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
from scipy import signal
import matplotlib.patches as mpatches
import os
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter, LogFormatterMathtext

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
from comnumpy.optical.pdm.compensators import CMA, RDE, DDLMS, AdaptiveChannel, Switch, MCMA
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
seg = 20 # number of fiber segments

t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
pmd_params = [t_dgd_k, rot_angle_k]


# SRRC filter
roll_off = 0.35
srrc_taps = 30

# Oversampling 
oversampling = 2

### Impairments parameters ###
# PMD #
t_dgd = 1e-12
rot_angle = np.pi/4

# SOP drift
pol_linewidth = 1e3

# PN
laser_linewidth = 1e5
sigma2_pn = 2 * np.pi * laser_linewidth * Ts

# CMA #
cma_taps = 7


#PDL parameters
pdl_db = 3
pdl_theta = np.pi/4

# CMA convergence #
conv = 500_000

tx = Sequential( [
            Upsampler(oversampling),
            SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
] )


channel = Sequential( [
            # PMD(t_dgd, Fs, theta=rot_angle),
            # PDL(pdl_db, pdl_theta),
            SOP(T_symb=Ts, linewidth=pol_linewidth),
] )

rx = Sequential( [
            Downsampler(oversampling),
] )

P_total = 5 # in dB
p_seg = P_total / np.sqrt(seg)
pdl_params = np.random.uniform(0, np.sqrt(3)*p_seg, seg )

channel_params = [ pol_linewidth, pmd_params, pdl_params ]
step_LPN = 1e-2
step_CMA = 1e-3
step_CMA_list = [1e-3]
step_RDE = 1e-3
step_DD = 1e-4
step_PMD = 1e-4
step_MCMA = 1e-3
# step_CMA_list = np.linspace(1e-5, 1e-2, num=2)
step_MIMO_list = np.logspace(-5, -2, num = 5)

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
            PDMWrapper(tx, 'tx'),
            # Recorder(name='tx_symbols_emision'),
            recorder_emision,
            ChannelWrapper(seq_obj=channel, L=seg, params=channel_params),
            # Selective_Channel(N_r, N_t, N_tap, coef=0.3),
            # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
            # AWGN: folosim parametrul 'value' (SNR) si 'unit' pentru tipul valorii
            # Inlocuire vechi: AWGN(20, method='SNR_dB') -> AWGN(value=20, unit='snr_dB')
            PDMWrapper( AWGN(value=20, unit='snr_dB'), name='noise'),
            # PDM_Wrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
            PDMWrapper( SRRCFilter(roll_off, oversampling, srrc_taps, method='fft') ),
            # Recorder(name='data_before_CMA'),
            recorder_before_CMA,
            PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # SOP_compensator(alphabet=alphabet, mu=1e-3, oversampling=oversampling, phase=0, proj_period_sym=1, name='sop_compensator'),
            MCMA(alphabet=alphabet, mu=step_MCMA, p=2, name="MCMA"),
            # CMA(L=cma_taps, alphabet=alphabet, mu=step_CMA, oversampling=oversampling, name='CMA'),
            # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # Switch(cma_taps, alphabet, step_MIMO_list, oversampling, tx_before_CMA=recorder_before_CMA, name='adaptive_channel'),
            #PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # Recorder(name='after_phase_correction'),
            # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # CMA(cma_taps, alphabet, step_CMA, oversampling, name='CMA'),
            # RDE(cma_taps, alphabet, step_RDE, oversampling, name='rde'),
            # DD_LMS(cma_taps, alphabet, step_RDE, oversampling, name='dd'),
            PDMWrapper(rx,'rx'),
            recorder_reception,
            # MSE(recorder_real_symb, alphabet, name='mse'),
            # PhUnGrid(None, name='PhUnCor'),
            PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # PDMWrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
            PDMWrapper(SymbolDemapper(alphabet)),
            Recorder(name='data_rx'),
            ])


####### 1 iteration ######
'''
y = chain(N)[:,conv:]
data_tx = chain['data_tx'].get_data()
data_tx1 = np.reshape(data_tx, (2,-1))[0,conv:]
data_tx2 = np.reshape(data_tx, (2,-1))[1,conv:]

ser1  = compute_ser(data_tx1, y[0,:])
ber1 = compute_ber( data_tx1, y[0,:], width=int(np.log2(M)) )
ser2  = compute_ser(data_tx2, y[1,:])
ber2 = compute_ber( data_tx2, y[1,:], width=int(np.log2(M)) )
ser = np.mean( [ser1, ser2] )
ber = np.mean( [ber1, ber2] )
print(f'SNR={SNR}:', 'SER:', ser1, '\tBER:', ber1)
print(f'SNR={SNR}:', 'SER:', ser2, '\tBER:', ber2)
plt.show()
'''

####### Monte Carlo ######
ber_vs_step = []
nr_repetitions = 1
avg_ber_per_step = []

for mu in step_MIMO_list:
    # chain["CMA"].mu = mu
    # chain["sop_compensator"].mu = mu
    chain["MCMA"].mu = mu
    ber_runs = []

    for _ in range(nr_repetitions):
        # chain["CMA"].reset()
        # chain["sop_compensator"].reset()
        chain["MCMA"].reset()

        # reset ChannelWrapper
        for idx, mod in enumerate(chain.module_list):
            if isinstance(mod, ChannelWrapper):
                chain.module_list[idx] = ChannelWrapper(
                    seq_obj=channel,
                    L=seg,
                    params=channel_params
                )
                break

        y = chain(N)[:, conv:]
        data_tx = chain['data_tx'].get_data()

        tx1 = np.reshape(data_tx, (2, -1))[0, conv:]
        tx2 = np.reshape(data_tx, (2, -1))[1, conv:]

        ber1 = compute_ber(tx1, y[0], width=int(np.log2(M)))
        ber2 = compute_ber(tx2, y[1], width=int(np.log2(M)))

        ber_runs.append(np.mean([ber1, ber2]))

    avg_ber = np.mean(ber_runs)
    avg_ber_per_step.append(avg_ber)
    
    #csv
    ber_vs_step.append({
        "step": mu,
        "BER":avg_ber
    })
    print(f"mu={mu:.3e} → BER={avg_ber_per_step[-1]:.3e}")


#save the datas in a csv file
df = pd.DataFrame(ber_vs_step)
df.to_csv(f"dates_{seg}_seg_{nr_repetitions}reps_2_SOP_MCMA_16qam.csv", index=False)

plt.figure(figsize=(7, 5))
plt.loglog(step_MIMO_list, avg_ber_per_step, marker='o', linewidth=2)

plt.xlabel("SOP compensator step (μ)")
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
