"""What back-propagation is worth, as a function of the number of steps.

The launch-power sweep of Hager and Pfister (arXiv:2010.14258), with one
curve per step count. Run from this directory: it writes the tutorial's
figures into ../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.compensators import DataAidedPhaseCompensator
from comnumpy.core.filters import BWFilter, SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_effective_snr, compute_ser
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
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
oversampling_ratio = oversampling_sim // oversampling_dsp


# --- the chain, in two pieces -----------------------------------------
# The one-shot tutorial ran one chain, transmitter to decision. Here six
# receivers are compared over the *same* channel realization, at eight
# launch powers and four trials each: running the whole chain per
# receiver would re-do 192 split-step propagations instead of 32, and
# the propagation is where all the time goes. So the chain is cut where
# the physics ends and the DSP begins.
def get_unprocessed_chain():
    """Symbols to the field the receiver sees, launch power included."""
    return Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet, name="signal_tx"),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        Amplifier(1.0, name="launch"),
        FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, name="link"),
        BWFilter(1 / oversampling_sim),
        Downsampler(oversampling_ratio),
        ], taps=["data_tx", "signal_tx"])


def get_receiver(steps, linear_only, *, gain, reference):
    """Back-propagation, matched filter and data-aided phase correction.

    ``reference`` is the transmitted symbol sequence of the run being
    received: the chain that produced it is a different object here, so
    the reference is passed rather than wired.
    """
    return Sequential([
        DBP(N_span, L_span=L_span, StPS=steps, step_type="linear",
            fs=fs / oversampling_ratio, use_only_linear=linear_only,
            name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(gain),
        DataAidedPhaseCompensator(reference, name="phase"),
        ])


# Every receiver is the same chain with one argument changed. "amplifier
# noise only" receives a link whose nonlinearity was switched off, which
# is the bound the others are trying to reach.
receivers = {
    "amplifier noise only": (1, True),
    "dispersion compensation": (1, True),
    "DBP, 1 step/span": (1, False),
    "DBP, 2 steps/span": (2, False),
    "DBP, 4 steps/span": (4, False),
    "DBP, 50 steps/span": (50, False),
}

channel = get_unprocessed_chain()
snr = {}
ser = {}
elapsed = {}
for name in receivers:
    snr[name] = np.zeros(len(dBm_list))
    ser[name] = np.zeros(len(dBm_list))
    elapsed[name] = 0.0

for index, dBm in enumerate(dBm_list):
    amp = np.sqrt(dbm_to_watt(dBm))
    for trial in range(N_trial):
        # the two fields differ only by the fibre's nonlinearity, and
        # the same seed gives them the same symbols and the same noise
        fields = {}
        for use_only_linear in (True, False):
            channel.seed(index * N_trial + trial)
            channel.set_params(launch__gain=amp,
                               link__use_only_linear=use_only_linear)
            fields[use_only_linear] = channel(N_s * oversampling_sim)
        symbols, reference = channel.tap("signal_tx"), channel.tap("data_tx")

        for name, (steps, linear_only) in receivers.items():
            receiver = get_receiver(steps, linear_only, gain=1 / amp,
                                    reference=symbols)
            bound = name == "amplifier noise only"
            estimate = receiver(fields[bound])
            elapsed[name] += receiver.elapsed_
            detected, _ = hard_projector(estimate, alphabet)
            snr[name][index] += compute_effective_snr(symbols, estimate) / N_trial
            ser[name][index] += compute_ser(reference, detected) / N_trial

snr_dB = {}
for name, values in snr.items():
    snr_dB[name] = 10 * np.log10(values)

header = "launch power [dBm]  "
for value in dBm_list:
    header += f"{value:7.1f}"
print(header)
for name, values in snr_dB.items():
    line = f"{name:24s}"
    for value in values:
        line += f"{value:7.1f}"
    print(line)

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
