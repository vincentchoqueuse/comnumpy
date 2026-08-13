"""BER of two chromatic-dispersion FIR compensators, three modulations.

Reproduces figures 3 and 4 of Eghbali et al., "Optimal Least-Squares
FIR Digital Filters for Compensation of Chromatic Dispersion in Digital
Coherent Optical Receivers": the truncated closed form of Savory
against the least-squares design, on system 1 of their Table 1.
"""
import numpy as np
import matplotlib.pyplot as plt

from comnumpy import Experiment, style
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

config = {
    "N": 200000,       # symbols per point; more smooths the curves
    "z": 4000,         # km -- system 1 of Table 1
    "fs": 21.4e9,
}
N_h = 1000             # SRRC filter delay, in symbols
oversampling = 2
rolloff = 0.25
k_vect = [4, 6, 8]     # bits per symbol; the chain is rebuilt per run

comp_ref = ChromaticDispersionFIRCompensator(config["z"], fs=config["fs"])
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
    ChromaticDispersion(config["z"], fs=config["fs"]),
    AWGN(sigma2=1.0, name="noise"),
    ], taps=["generator"])

compensators = {
    "Savory": Sequential([
        ChromaticDispersionFIRCompensator(config["z"], fs=config["fs"]),
        SRRCFilter(rolloff, oversampling, N_h=N_h),
        Downsampler(oversampling, phase=total_delay),
        SymbolDemapper(Constellation("QAM", 2 ** k_vect[0]), name="demapper"),
        ]),
    "LS": Sequential([
        ChromaticDispersionLSFIRCompensator(config["z"], N_filter,
                                            fs=config["fs"],
                                            w_vect=[-np.pi, np.pi]),
        SRRCFilter(rolloff, oversampling, N_h=N_h),
        Downsampler(oversampling, phase=total_delay),
        SymbolDemapper(Constellation("QAM", 2 ** k_vect[0]), name="demapper"),
        ]),
}


def simulate(config, seed):
    """One SNR per bit: propagate once, both compensators on the field.

    The blocks are reconfigured through set_params, never by assignment
    (D50): SymbolMapper coerces its alphabet in __post_init__, and a
    direct write skips that.
    """
    constellation = Constellation("QAM", 2 ** config["k"])
    sigma2 = (constellation.energy / config["k"]
              / 10 ** (config["snr_bit_dB"] / 10))
    chain.set_params(generator__M=constellation.order,
                     mapper__alphabet=constellation,
                     noise__sigma2=sigma2)
    chain.seed(seed)
    y = chain(config["N"])
    s = chain.tap("generator")

    observed = {"theory": float(constellation.metrics(
        config["snr_bit_dB"])["ber"])}
    for name, compensator in compensators.items():
        compensator.set_params(demapper__alphabet=constellation)
        s_est = compensator(y)
        observed[name] = compute_ber(s_est, s[:len(s_est)], width=config["k"])
    return observed


fig, ax = plt.subplots()
line_style = {"theory": "k-", "Savory": "--", "LS": ":"}
for k in k_vect:
    config["k"] = k
    experiment = Experiment(config, parameter="snr_bit_dB",
                            values=np.arange(10, 31), seed=1)
    result = experiment.run(simulate)
    result.print(ylabel=f"BER, QAM{2 ** k}")
    print()
    for name, values in result.data.items():
        ax.semilogy(result.values, values, line_style[name],
                    label=f"{name} (QAM{2 ** k})")

ax.set_xlabel("SNR per bit (dB)")
ax.set_ylabel("BER")
ax.set_ylim(1e-6, 0.2)
ax.set_xlim(10, 30)
ax.legend()
plt.show()
