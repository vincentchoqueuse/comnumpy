"""What back-propagation is worth, as a function of the number of steps.

The launch-power sweep of Hager and Pfister (arXiv:2010.14258), with one
curve per step count. Run from this directory: it writes the tutorial's
figures into ../../docs/tutorials/img/.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.filters import BWFilter, SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_effective_snr, compute_ser
from comnumpy.core.processors import Downsampler, Upsampler
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.optical.dbp import DBP
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import dbm_to_watt

img_dir = "../../docs/tutorials/img/"

M = 16
alphabet = get_alphabet("QAM", M)
N_s = 2**11               # 8192 symbols per point: the SER floor is 1e-4
oversampling_sim = 6
oversampling_dsp = 2
NF_dB = 5
rolloff = 0.1
StPS = 200                    # forward propagation, the reference
R_s = 32e9
L_span = 100
N_span = 10
dBm_list = np.arange(-6, 6, 1.5)
N_trial = 4

fs = R_s * oversampling_sim
oversampling_ratio = int(oversampling_sim / oversampling_dsp)

source = Sequential([SymbolGenerator(M, name="tx"), SymbolMapper(alphabet)],
                    taps=["tx"])
transmitter = Sequential([
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        ])
fibre = Sequential([
        FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, name="link"),
        BWFilter(1 / oversampling_sim),
        Downsampler(oversampling_ratio),
        ])

# Every receiver is the same chain with one argument changed. "amplifier
# noise only" back-propagates a link whose nonlinearity was switched off,
# which is the bound the others are trying to reach.
receivers = {
    "amplifier noise only": {"StPS": 1, "use_only_linear": True},
    "dispersion compensation": {"StPS": 1, "use_only_linear": True},
    "DBP, 1 step/span": {"StPS": 1, "use_only_linear": False},
    "DBP, 2 steps/span": {"StPS": 2, "use_only_linear": False},
    "DBP, 4 steps/span": {"StPS": 4, "use_only_linear": False},
    "DBP, 50 steps/span": {"StPS": 50, "use_only_linear": False},
}


def get_receiver(steps, linear_only):
    return Sequential([
        DBP(N_span, L_span=L_span, StPS=steps, step_type="linear",
            fs=fs / oversampling_ratio, use_only_linear=linear_only,
            name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        ])


snr = {name: np.zeros(len(dBm_list)) for name in receivers}
ser = {name: np.zeros(len(dBm_list)) for name in receivers}
elapsed = dict.fromkeys(receivers, 0.0)

for index, dBm in enumerate(dBm_list):
    amp = np.sqrt(dbm_to_watt(dBm))
    for trial in range(N_trial):
        source.seed(index * N_trial + trial)
        symbols = source(N_s)
        reference = source.tap("tx")
        launched = amp * transmitter(symbols)

        fibre.seed(trial)
        fibre.set_params(link__use_only_linear=True)
        linear = fibre(launched)
        fibre.seed(trial)
        fibre.set_params(link__use_only_linear=False)
        nonlinear = fibre(launched)

        for name, config in receivers.items():
            field = linear if name == "amplifier noise only" else nonlinear
            start = time.perf_counter()
            estimate = get_receiver(config["StPS"], config["use_only_linear"])(field)
            elapsed[name] += time.perf_counter() - start
            theta = np.angle(np.sum(np.conj(estimate) * symbols))
            corrected = np.exp(1j * theta) / amp * estimate
            detected, _ = hard_projector(corrected, alphabet)
            snr[name][index] += compute_effective_snr(symbols, corrected) / N_trial
            ser[name][index] += compute_ser(reference, detected) / N_trial

snr_dB = {name: 10 * np.log10(values) for name, values in snr.items()}

print("launch power [dBm]  " + "  ".join(f"{value:5.1f}" for value in dBm_list))
for name, values in snr_dB.items():
    print(f"{name:24s}" + "  ".join(f"{value:5.1f}" for value in values))

print("\nreceiver                  best SNR   at power    total time")
for name, values in snr_dB.items():
    best = int(np.argmax(values))
    print(f"{name:24s} {values[best]:6.2f} dB {dBm_list[best]:6.1f} dBm "
          f"{elapsed[name]:9.1f} s")

ax = plot_error_rate(dBm_list, snr_dB, xlabel="launch power [dBm]",
                     ylabel="effective SNR [dB]", yscale="linear",
                     title=f"{N_span} x {L_span} km, {M}-QAM at "
                           f"{R_s / 1e9:.0f} GBd")
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig1.png")

ax = plot_error_rate(dBm_list, ser, xlabel="launch power [dBm]", ylabel="SER",
                     title="the same sweep, in symbol error rate")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig2.png")
