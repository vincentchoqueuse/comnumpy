import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.ticker import NullFormatter, LogFormatterMathtext
from time import time

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

from comnumpy.optical.pdm.generics import PDMWrapper, ChannelWrapper_
from comnumpy.optical.pdm.channels import SOP, SOP_, PDL_, PMD_, PMD
from comnumpy.optical.pdm.compensators import MCMA_to_DD_Czegledi, DD_Czegledi
from comnumpy.optical.pdm.utils import *

# parameters
type, M = "QAM", 16
N = 1_200_000
alphabet = get_alphabet(type, M, type='bin')

Ts = 1 / 28e9
seg = 20 # number of fiber segments

# SRRC filter
roll_off = 0.35
srrc_taps = 30

# Oversampling 
oversampling = 2
Fs = (1/Ts) * oversampling

### Impairments parameters ###
# SOP drift
pol_linewidth = 28e3

Fiber_length = 1000 # Km
D_pmd = 0.1e-12 # ps/sqrt(km), for aggresive cases, choose 0.1
t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
pmd_params = [t_dgd_k]

pdl_params = 0.2 # segment-wise
step_MIMO_list = [1e-3]


# Convergence #
conv = 500_000
tx = Sequential( [
            Upsampler(oversampling),
            SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
] )


channel = Sequential( [
            SOP_(T_symb=Ts, linewidth=pol_linewidth, segments=seg),
            PDL_(pdl_params),
            PMD_(t_dgd_k, Fs)
] )

rx = Sequential( [
            Downsampler(oversampling),
] )

channel_params = [ pol_linewidth, pmd_params, pdl_params ]

mcma = MCMA_(alphabet, L=7, mu1=6e-3, mu2=6e-3, switch=0, os=oversampling)
dd = DD_Czegledi(alphabet, mu=6e-3)

eq = MCMA_to_DD_Czegledi(
    mcma=mcma,
    dd=dd,
    switch=oversampling*conv,
    name="MCMA_to_DD_"
)

SNR = 20
chain = Sequential([
            SymbolGenerator(M),
            Recorder(name='data_tx'),
            PDMWrapper(DifferentialEncoding(M)),
            PDMWrapper(tx, 'tx'),
            ChannelWrapper_(seq_obj=channel, L=seg, params=channel_params),
            PDMWrapper( AWGN(value=SNR, unit='snr_dB'), name='noise'),
            PDMWrapper( SRRCFilter(roll_off, oversampling, srrc_taps, method='fft') ),
            MCMA_(alphabet, 7, 1e-3, 1e-3, 0, os=oversampling, name="MCMA"),
            PDMWrapper(rx,'rx'),
            PDMWrapper(IQ_Scope(axis='equal', nlim=(-10000,N))),
            PDMWrapper(DifferentialDecoding(M)),
            Recorder(name='data_rx'),
            ])

# start = time()
# y = chain(N)[:,conv:]
# stop = time()
# print('Ellapsed time:', stop-start)
# data_tx = chain['data_tx'].get_data()
# data_tx1 = np.reshape(data_tx, (2,-1))[0,conv:]
# data_tx2 = np.reshape(data_tx, (2,-1))[1,conv:]

# ser1  = compute_ser(data_tx1, y[0,:])
# ber1 = compute_ber( data_tx1, y[0,:], width=int(np.log2(M)) )
# ser2  = compute_ser(data_tx2, y[1,:])
# ber2 = compute_ber( data_tx2, y[1,:], width=int(np.log2(M)) )
# ser = np.mean( [ser1, ser2] )
# ber = np.mean( [ber1, ber2] )
# print(f'SNR={SNR}:', 'SER:', ser1, '\tBER:', ber1)
# print(f'SNR={SNR}:', 'SER:', ser2, '\tBER:', ber2)
# plt.show()
##### Monte Carlo: SER vs Δp_tot·T #######
start = time()
ser_vs_linewidth = []
nr_repetitions = 3
ser_list = []
dp_tot_T_list = []

linewidth_list = [28e1, 5*28e1, 28e2, 5*28e2, 28e3, 5*28e3, 28e4, 5*28e4, 28e5]

for pol_linewidth in linewidth_list:
    channel = Sequential( [
                SOP_(T_symb=Ts, linewidth=pol_linewidth, segments=seg),
                PDL_(pdl_params),
                PMD_(t_dgd_k, Fs)
    ] )
    channel_params = [pol_linewidth, pmd_params, pdl_params]
    
    ser_runs = []

    for rep in range(nr_repetitions):
        chain["MCMA"].reset()

        # actualizează canalul în chain
        for idx, mod in enumerate(chain.module_list):
            if isinstance(mod, ChannelWrapper_):
                chain.module_list[idx] = ChannelWrapper_(
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

    dp_tot_T = pol_linewidth * Ts
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
df.to_csv(f"src\\comnumpy\\optical\\pdm\\validation\\results\\SER_vs_dpTotT_seg{seg}_SNR{SNR}_{ chain["MCMA"].name}_S3.csv", index=False)


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
stop = time()
print('Ellapsed time:', stop-start)
plt.show()

