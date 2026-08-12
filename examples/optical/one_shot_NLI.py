"""Fibre propagation span by span, and digital back-propagation.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.filters import BWFilter, SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_effective_snr, compute_ser
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.optical.dbp import DBP
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import dbm_to_watt

img_dir = "../../docs/tutorials/img/"

M = 16
modulation = "QAM"
alphabet = get_alphabet(modulation, M)
N_s = 2**9                    # number of symbols
oversampling_sim = 6          # samples per symbol in the channel
oversampling_dsp = 2          # samples per symbol in the receiver
NF_dB = 5                     # amplifier noise figure in dB
rolloff = 0.1
StPS = 500                    # split steps per span, forward
StPS_DBP = 100                # split steps per span, backward
R_s = 10.7e9                  # baud rate
L_span = 80                   # span length in km
N_span = 25
dBm = -3

N = N_s * oversampling_sim
fs = R_s * oversampling_sim
oversampling_ratio = int(oversampling_sim / oversampling_dsp)
Po = dbm_to_watt(dBm)
amp = np.sqrt(Po)

# --- the link -------------------------------------------------------
# One span is fibre plus an amplifier that puts the power back and adds
# spontaneous-emission noise; the callback keeps the field at the output
# of each of them, which is what the degradation figure needs.
per_span = []
link = FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                 fs=fs, name="link",
                 callbacks={"post_span": lambda y, num_span: per_span.append(y)})

chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet, name="signal_tx"),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        Amplifier(amp),
        link,
        BWFilter(1 / oversampling_sim),
        Downsampler(oversampling_ratio),
        ], taps=["data_tx", "signal_tx"])

chain.seed(0)
start = time.perf_counter()
y_rx = chain(N)
print(f"{N_span} spans at {StPS} steps per span: "
      f"{time.perf_counter() - start:.1f} s")

s_tx = chain.tap("data_tx")
x_tx = chain.tap("signal_tx")

plot_iq(y_rx, title=f"received signal ({dBm} dBm, {N_span} spans)")
plt.savefig(f"{img_dir}/one_shot_nli_fig1.png")

# --- what the receiver does with it ----------------------------------


def receive(field, n_spans, steps, *, linear_only):
    """Back-propagate over ``n_spans``, then filter, decide, and score.

    The same receiver serves the two strategies of this tutorial and the
    span-by-span reading below: only the number of spans it undoes and
    whether it undoes the nonlinearity change.
    """
    receiver = Sequential([
        DBP(n_spans, L_span=L_span, StPS=steps, step_type="linear",
            use_only_linear=linear_only, fs=fs / oversampling_ratio,
            name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(1 / amp),
        ])
    estimate = receiver(field)
    # the nonlinearity leaves a common phase rotation, and the tutorial is
    # about what is left *after* an ordinary carrier recovery has removed it
    theta = np.angle(np.sum(np.conj(estimate) * x_tx))
    return estimate, np.exp(1j * theta) * estimate, theta


def score(corrected):
    """Effective SNR and symbol error rate of a corrected estimate."""
    detected, _ = hard_projector(corrected, alphabet)
    return (10 * np.log10(compute_effective_snr(x_tx, corrected)),
            compute_ser(s_tx, detected))


# --- degradation, span by span ---------------------------------------
# Each recorded field is undone with dispersion compensation alone, over
# exactly the spans it has travelled: the curve is therefore what a
# linear receiver would see if the link stopped there.
spans = np.arange(1, N_span + 1)
snr_per_span = []
for index, field in enumerate(per_span, start=1):
    reduced = Downsampler(oversampling_ratio)(BWFilter(1 / oversampling_sim)(field))
    _, corrected, _ = receive(reduced, index, 1, linear_only=True)
    snr_per_span.append(score(corrected)[0])

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, index in zip(axes, (1, N_span // 2, N_span), strict=True):
    reduced = Downsampler(oversampling_ratio)(
        BWFilter(1 / oversampling_sim)(per_span[index - 1]))
    _, corrected, _ = receive(reduced, index, 1, linear_only=True)
    plot_iq(corrected, reference=alphabet, ax=ax)
    ax.set_title(f"after {index} span{'s' if index > 1 else ''}, "
                 f"SNR {snr_per_span[index - 1]:.1f} dB")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig2.png")

ax = plot_error_rate(spans, {"dispersion compensation only":
                             np.array(snr_per_span)},
                     xlabel="spans travelled", ylabel="effective SNR [dB]",
                     yscale="linear", title=f"{dBm} dBm launch power")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig3.png")

print("\nspan   effective SNR")
for index in (1, 5, 10, 15, 20, 25):
    print(f"{index:4d} {snr_per_span[index - 1]:14.2f} dB")

# --- the two strategies ----------------------------------------------
results = {}
for label, linear_only, steps in (("dispersion compensation", True, 1),
                                  ("digital back-propagation", False,
                                   StPS_DBP)):
    start = time.perf_counter()
    raw, corrected, theta = receive(y_rx, N_span, steps, linear_only=linear_only)
    elapsed = (time.perf_counter() - start) * 1e3
    snr, ser = score(corrected)
    results[label] = (raw, corrected, snr, ser)
    print(f"{label:26s} SNR={snr:5.2f} dB  SER={ser:.4f}  "
          f"residual phase={np.rad2deg(theta):+6.1f} deg  {elapsed:.0f} ms")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for ax, (label, (raw, corrected, snr, ser)) in zip(axes, results.items(),
                                                   strict=True):
    plot_iq(raw, label="before phase correction", ax=ax)
    plot_iq(corrected, label="after phase correction", reference=alphabet,
            ax=ax)
    ax.set_title(f"{label}\nSNR {snr:.2f} dB, SER {ser:.3f}")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig4.png")
