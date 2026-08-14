"""BER of two chromatic-dispersion FIR compensators, three modulations.

Reproduces figures 3 and 4 of Eghbali et al., "Optimal Least-Squares
FIR Digital Filters for Compensation of Chromatic Dispersion in Digital
Coherent Optical Receivers": the truncated closed form of Savory
against the least-squares design, on system 1 of their Table 1.
"""
import numpy as np
import matplotlib.pyplot as plt

from comnumpy import print_data, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.core.utils import Constellation
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ber
from comnumpy.core.filters import SRRCFilter
from comnumpy.optical.compensators import (
    ChromaticDispersionFIRCompensator, ChromaticDispersionLSFIRCompensator)
from comnumpy.optical.channels import ChromaticDispersion

style.use()

# parameters
N = 200000             # symbols per point; more smooths the curves
z = 4000               # km -- system 1 of Table 1
fs = 21.4e9
N_h = 1000             # SRRC filter delay, in symbols
oversampling = 2
rolloff = 0.25
k_vect = [4, 6, 8]     # bits per symbol; the chain is reconfigured per k
snr_bit_dB_list = np.arange(10, 31)
seed = 1               # one master seed reproduces the whole figure
curve_names = ("theory", "Savory", "LS")

comp_ref = ChromaticDispersionFIRCompensator(z, fs=fs)
N_filter = len(comp_ref.h)
print(f"K: {comp_ref.K}")
print(f"filter length: {N_filter}")
# total delay of the filters, compensated before sampling
total_delay = int(2 * N_h * oversampling + np.floor(N_filter / 2))

chain = Sequential([
    SymbolGenerator(M=4, name="generator"),
    SymbolMapper(Constellation("QAM", 2 ** k_vect[0]), name="mapper"),
    Upsampler(oversampling),
    SRRCFilter(rolloff, oversampling, N_h=N_h),
    ChromaticDispersion(z, fs=fs),
    AWGN(sigma2=1.0, name="noise"),
    ], observations=["generator"])

compensators = {
    "Savory": Sequential([
        ChromaticDispersionFIRCompensator(z, fs=fs),
        SRRCFilter(rolloff, oversampling, N_h=N_h),
        Downsampler(oversampling, phase=total_delay),
        SymbolDemapper(Constellation("QAM", 2 ** k_vect[0]), name="demapper"),
        ]),
    "LS": Sequential([
        ChromaticDispersionLSFIRCompensator(z, N_filter,
                                            fs=fs,
                                            w_vect=[-np.pi, np.pi]),
        SRRCFilter(rolloff, oversampling, N_h=N_h),
        Downsampler(oversampling, phase=total_delay),
        SymbolDemapper(Constellation("QAM", 2 ** k_vect[0]), name="demapper"),
        ]),
}


fig, ax = plt.subplots()
line_style = {"theory": "k-", "Savory": "--", "LS": ":"}
for k in k_vect:
    constellation = Constellation("QAM", 2 ** k)
    # Reconfigured through set_params, never by assignment (D50):
    # SymbolMapper coerces its alphabet in __post_init__, and a direct
    # write skips that.
    chain.set_params(generator__M=constellation.order,
                     mapper__alphabet=constellation)
    for compensator in compensators.values():
        compensator.set_params(demapper__alphabet=constellation)

    # --- metrics, pre-allocated --------------------------------------
    # One array per curve, indexed by name.
    ber = {}
    for name in curve_names:
        ber[name] = np.zeros(len(snr_bit_dB_list))

    # --- simulation loop ---------------------------------------------
    # One child seed per SNR point (D6/D35).
    point_seeds = np.random.SeedSequence(seed).spawn(len(snr_bit_dB_list))
    for index, snr_bit_dB in enumerate(snr_bit_dB_list):
        ber["theory"][index] = constellation.metrics(snr_bit_dB)["ber"]

        sigma2 = constellation.energy / k / 10 ** (snr_bit_dB / 10)
        chain.set_params(noise__sigma2=sigma2)
        chain.seed(int(point_seeds[index].generate_state(1)[0]))
        y = chain(N)
        s = chain.observation("generator")

        for name, compensator in compensators.items():
            s_est = compensator(y)
            ber[name][index] = compute_ber(s_est, s[:len(s_est)], width=k)

    # --- results: table and figure ---------------------------------------
    print_data({"x": snr_bit_dB_list, "curves": ber},
               xlabel="snr_bit_dB", ylabel=f"BER, QAM{2 ** k}")
    print()
    for name in curve_names:
        ax.semilogy(snr_bit_dB_list, ber[name], line_style[name],
                    label=f"{name} (QAM{2 ** k})")

ax.set_xlabel("SNR per bit (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-6, 0.2)
ax.set_xlim(10, 30)
ax.legend()
plt.show()
