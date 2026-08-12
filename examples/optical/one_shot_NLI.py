"""Fibre propagation span by span, and digital back-propagation.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
"""
import time

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
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.optical.dbp import DBP
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import dbm_to_watt

img_dir = "../../docs/tutorials/img/"

M = 16
alphabet = get_alphabet("QAM", M)
N_s = 2**9                    # symbols per run
oversampling_sim = 6          # samples per symbol in the channel
oversampling_dsp = 2          # samples per symbol in the receiver
NF_dB = 5                     # amplifier noise figure in dB
rolloff = 0.1
StPS = 200                    # split steps per span, forward
StPS_DBP = 100                # split steps per span, backward
R_s = 10.7e9                  # baud rate
L_span = 80                   # span length in km
N_span = 25
dBm = -3

N = N_s * oversampling_sim
fs = R_s * oversampling_sim
oversampling_ratio = oversampling_sim // oversampling_dsp
amp = np.sqrt(dbm_to_watt(dBm))


# --- the three pieces of the chain -----------------------------------
def get_transmitter():
    """Random symbols to a launched optical field."""
    return [
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet, name="signal_tx"),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        Amplifier(amp),
        ]


def get_channel(n_spans):
    """``n_spans`` of fibre and amplifier, then the receiver front end."""
    return [
        FiberLink(N_spans=n_spans, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, name="link"),
        BWFilter(1 / oversampling_sim),
        Downsampler(oversampling_ratio),
        ]


def get_receiver(n_spans, *, steps=1, linear_only=True, reference=None):
    """Back-propagation, matched filter, and a data-aided phase correction.

    The two strategies of this tutorial are one argument apart:
    ``linear_only=True`` undoes the dispersion alone, in one step per
    span; ``linear_only=False`` undoes the nonlinearity too, in ``steps``.

    ``reference`` is the transmitted symbol sequence the phase estimate
    is fitted against. Inside a full chain it is left unset and wired
    from the transmitter instead.
    """
    return [
        DBP(n_spans, L_span=L_span, StPS=steps, step_type="linear",
            use_only_linear=linear_only, fs=fs / oversampling_ratio,
            name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(1 / amp),
        DataAidedPhaseCompensator(reference, name="phase"),
        ]


def get_full_chain(n_spans, **kwargs):
    """Transmitter, link and receiver, as one chain.

    The phase compensator needs the transmitted symbols and the chain
    produces them itself, so the edge is declared with ``wiring`` rather
    than threaded through by hand: the estimate is then fitted against
    *this* run's data, whatever the chain is seeded with.
    """
    return Sequential(get_transmitter() + get_channel(n_spans)
                      + get_receiver(n_spans, **kwargs),
                      taps=["data_tx", "signal_tx"],
                      wiring={"phase.reference": "signal_tx"})


def score(reference, estimate, symbols):
    """Effective SNR in dB and symbol error rate of an estimate."""
    detected, _ = hard_projector(estimate, alphabet)
    return (10 * np.log10(compute_effective_snr(reference, estimate)),
            compute_ser(symbols, detected))


# --- 1. what the link does to the signal ------------------------------
# One call per span count, and the same seed each time: the transmitted
# symbols and the amplifier noise are identical, so the only thing that
# changes along the curve is the distance travelled.
spans = (1, 5, 10, 15, 20, 25)
estimates, snr_per_span = {}, {}
for n_spans in spans:
    chain = get_full_chain(n_spans)
    chain.seed(0)
    start = time.perf_counter()
    estimates[n_spans] = chain(N)
    elapsed = time.perf_counter() - start
    snr, ser = score(chain.tap("signal_tx"), estimates[n_spans],
                     chain.tap("data_tx"))
    snr_per_span[n_spans] = snr
    print(f"{n_spans:2d} spans: SNR {snr:5.2f} dB, SER {ser:.4f}, "
          f"phase {np.rad2deg(chain['phase'].theta_):+6.1f} deg, "
          f"{elapsed:5.1f} s")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, n_spans in zip(axes, (1, 10, 25), strict=True):
    plot_iq(estimates[n_spans], reference=alphabet, ax=ax)
    ax.set_title(f"after {n_spans} span{'s' if n_spans > 1 else ''}, "
                 f"SNR {snr_per_span[n_spans]:.1f} dB")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig1.png")

ax = plot_error_rate(np.array(spans),
                     {"dispersion compensation only":
                      np.array(list(snr_per_span.values()))},
                     xlabel="spans travelled", ylabel="effective SNR [dB]",
                     yscale="linear", title=f"{dBm} dBm launch power")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig2.png")

# Where that time goes. The chain has thirteen blocks and one of them
# is the whole cost, so the run time above is the propagation and
# nothing else.
print("\nblock                    time")
profile = get_full_chain(N_span).seed(0).profile_execution_time(N)
for block_id, elapsed in profile.items():
    print(f"{block_id:22s} {1e3 * elapsed:8.1f} ms")

# --- 2. splitting the chain -------------------------------------------
# Comparing receivers over the same channel realization with the chain
# above would re-run the split-step propagation once per receiver, and
# the table just said what that costs. The link is run once instead, and
# the receivers are applied to the field it produced.
unprocessed = Sequential(get_transmitter() + get_channel(N_span),
                         taps=["data_tx", "signal_tx"])
unprocessed.seed(0)
y_rx = unprocessed(N)
s_tx, x_tx = unprocessed.tap("data_tx"), unprocessed.tap("signal_tx")

plot_iq(y_rx, title=f"received field ({dBm} dBm, {N_span} spans)")
plt.savefig(f"{img_dir}/one_shot_nli_fig3.png")

results = {}
for label, linear_only, steps in (("dispersion compensation", True, 1),
                                  ("digital back-propagation", False,
                                   StPS_DBP)):
    receiver = Sequential(get_receiver(N_span, steps=steps,
                                       linear_only=linear_only,
                                       reference=x_tx))
    start = time.perf_counter()
    estimate = receiver(y_rx)
    elapsed = 1e3 * (time.perf_counter() - start)
    snr, ser = score(x_tx, estimate, s_tx)
    results[label] = (estimate, snr, ser)
    print(f"{label:26s} SNR={snr:5.2f} dB  SER={ser:.4f}  "
          f"residual phase={np.rad2deg(receiver['phase'].theta_):+6.1f} deg  "
          f"{elapsed:.0f} ms")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for ax, (label, (estimate, snr, ser)) in zip(axes, results.items(),
                                             strict=True):
    plot_iq(estimate, reference=alphabet, ax=ax)
    ax.set_title(f"{label}\nSNR {snr:.2f} dB, SER {ser:.3f}")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig4.png")
