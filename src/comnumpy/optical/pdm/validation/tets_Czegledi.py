import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.ticker import NullFormatter, LogFormatterMathtext

import sys
sys.path.insert(0, "src\\")

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.optical.pdm.sinks import IQ_Scope, IQ_Scope_PostProcessing
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.core.monitors import Recorder
from comnumpy.core.metrics import compute_ser, compute_ber, compute_evm

from comnumpy.optical.pdm.generics import PDMWrapper, ChannelWrapper
from comnumpy.optical.pdm.channels import SOP, PDL
from comnumpy.optical.pdm.compensators import CMA, RDE, DDLMS, AdaptiveChannel, Switch, MCMA, DD_Czegledi
from comnumpy.optical.pdm.utils import *

# parameters
type, M = "QAM", 16
N = 1_200_000
alphabet = get_alphabet(type, M, type='bin')

Fs = 56e9 
Ts = 1 / 28e9

seg = 20 # number of fiber segments

# SRRC filter
roll_off = 0.35
srrc_taps = 30

# Oversampling 
oversampling = 2

### Impairments parameters ###
# SOP drift
pol_linewidth = 1.4e3

#PDL parameters
pdl_db = 3
pdl_theta = np.pi/4

Fiber_length = 500 # Km
D_pmd = 0.1e-12 # ps/sqrt(km), for aggresive cases, choose 0.1
t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
pmd_params = [t_dgd_k, rot_angle_k]

P_total = 0 # in dB
p_seg = P_total / np.sqrt(seg)
pdl_params = np.random.uniform(0, np.sqrt(3)*p_seg, seg )
# step_MIMO_list = np.logspace(-5, -2, num = 5)
step_MIMO_list = [1e-3]


# CMA #
cma_taps = 16

# CMA convergence #
conv = 500_000

tx = Sequential( [
            Upsampler(oversampling),
            SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
] )


channel = Sequential( [
            #PDL(pdl_db, pdl_theta),
            SOP(T_symb=Ts, linewidth=pol_linewidth),
] )

rx = Sequential( [
            Downsampler(oversampling),
] )

channel_params = [ pol_linewidth, pmd_params, pdl_params ]
step_CMA_list = np.logspace(-3, -3, num = 1)

recorder_before_CMA = Recorder(name='data_before_CMA')
recorder_emision = Recorder(name='tx_symbols_emision')
recorder_real_symb = Recorder(name='tx_symbols')
recorder_reception = Recorder(name='rx_symbols')

SNR = 18
chain = Sequential([
            SymbolGenerator(M),
            Recorder(name='data_tx'),
            #PDMWrapper(SymbolMapper(alphabet)),
            PDMWrapper(DifferentialEncoding(M)),
            PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            recorder_real_symb,
            PDMWrapper(tx, 'tx'),
            recorder_emision,
            ChannelWrapper(seq_obj=channel, L=seg, params=channel_params),
            PDMWrapper( AWGN(value=SNR, unit='snr_dB'), name='noise'),
            PDMWrapper( SRRCFilter(roll_off, oversampling, srrc_taps, method='fft') ),
            recorder_before_CMA,
            #PDMWrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
            # MCMA(alphabet=alphabet, mu=1e-3, p=2, name="MCMA"),
            DD_Czegledi(alphabet=alphabet, mu=1e-3, P=1, name="DD_Czegledi"),
            #CMA(L=cma_taps, alphabet=alphabet, mu=1e-3, oversampling=oversampling, name='CMA'),
            # PDM_Wrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # Switch(cma_taps, alphabet, step_MIMO_list, oversampling, tx_before_CMA=recorder_before_CMA, name='adaptive_channel'),
            #PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            PDMWrapper(rx,'rx'),
            recorder_reception,
            # MSE(recorder_real_symb, alphabet, name='mse'),
            # PhUnGrid(None, name='PhUnCor'),
            PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            # PDMWrapper(IQ_Scope_PostProcessing(axis='equal', nlim=(-10000,N))),
            PDMWrapper(DifferentialDecoding(M)),
            Recorder(name='data_rx'),
            ])


####### Monte Carlo: SER vs Δp_tot·T #######
ser_vs_linewidth = []
nr_repetitions = 3
ser_list = []
dp_tot_T_list = []

linewidth_list = [1.4e1, 1.4e2, 4.2e2, 1.4e3, 4.2e3, 1.4e4, 4.2e4, 1.4e5, 4.2e5, 1.4e6, 4.2e6]
# linewidth_list = [1.4e2]

for pol_linewidth in linewidth_list:
    channel = Sequential([
        SOP(T_symb=Ts, linewidth=pol_linewidth)
    ])
    channel_params = [pol_linewidth, pmd_params, pdl_params]
    
    ser_runs = []

    for rep in range(nr_repetitions):
        chain["DD_Czegledi"].reset()

        # actualizează canalul în chain
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

        ser1 = compute_ser(tx1, y[0])
        ser2 = compute_ser(tx2, y[1])
        ser_runs.append([ser1, ser2])

        print(f"[Repetition {rep+1}/{nr_repetitions}] linewidth={pol_linewidth:.1e} Hz → SER_pol1={ser1:.3e}, SER_pol2={ser2:.3e}")

    # După toate repetările:
    ser1_avg = np.mean([s[0] for s in ser_runs])
    ser2_avg = np.mean([s[1] for s in ser_runs])
    ser_avg = 0.5 * (ser1_avg + ser2_avg)

    dp_tot_T = seg * pol_linewidth * Ts
    dp_tot_T_list.append(dp_tot_T)
    ser_list.append(ser_avg)

    ser_vs_linewidth.append({
        "pol_linewidth_Hz": pol_linewidth,
        "dp_tot_T": dp_tot_T,
        "SER": ser_avg,
        "SER_pol1": ser1_avg,
        "SER_pol2": ser2_avg
    })

    print(f"Final → linewidth={pol_linewidth:.1e} Hz → dp·T={dp_tot_T:.2e} → Mean SER={ser_avg:.3e}\n")

df = pd.DataFrame(ser_vs_linewidth)
df.to_csv(f"SER_vs_dpTotT_seg{seg}_SNR{SNR}_MCMA.csv", index=False)


plt.figure(figsize=(7, 5))
plt.loglog(dp_tot_T_list, ser_list, marker='o', linewidth=2)

plt.xlabel(r'$\Delta p_{\mathrm{tot}} \cdot T$')
plt.ylabel("Symbol Error Rate (SER)")
plt.title(f"SER vs Δp_tot·T (SNR={SNR} dB)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(LogFormatterMathtext())
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tight_layout()
plt.show()

