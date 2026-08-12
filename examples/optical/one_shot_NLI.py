"""Fibre propagation span by span, and digital back-propagation.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
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


# --- the chain --------------------------------------------------------
def get_full_chain(n_spans, *, steps=1, linear_only=True):
    """The whole link, from the symbols to the decisions.

    Transmitter, ``n_spans`` of fibre and amplifier, and the receiver
    that undoes them. The two strategies of this tutorial are one
    argument apart: ``linear_only=True`` undoes the dispersion alone, in
    one step per span, ``linear_only=False`` undoes the nonlinearity too,
    in ``steps``.

    The last block is a data-aided phase correction, and the reference it
    needs is the transmitted symbol sequence -- which the chain produces
    itself, so the edge is declared with ``wiring`` instead of being
    passed by hand. What comes out of the chain is therefore ready to be
    compared with what went in, at any number of spans.
    """
    return Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet, name="signal_tx"),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        Amplifier(amp),
        FiberLink(N_spans=n_spans, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, name="link"),
        BWFilter(1 / oversampling_sim),
        Downsampler(oversampling_ratio, name="rx_field"),
        DBP(n_spans, L_span=L_span, StPS=steps, step_type="linear",
            use_only_linear=linear_only, fs=fs / oversampling_ratio,
            name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(1 / amp),
        DataAidedPhaseCompensator(name="phase"),
        ], taps=["data_tx", "signal_tx", "rx_field", "phase"],
        wiring={"phase.reference": "signal_tx"})


def score(chain):
    """Effective SNR in dB and symbol error rate of a finished run."""
    estimate = chain.tap("phase")
    detected, _ = hard_projector(estimate, alphabet)
    return (10 * np.log10(compute_effective_snr(chain.tap("signal_tx"),
                                                estimate)),
            compute_ser(chain.tap("data_tx"), detected))


# --- 1. what the link does to the signal ------------------------------
# One call per span count, and the same seed each time: the transmitted
# symbols and the amplifier noise are identical, so the only thing that
# changes along the curve is the distance travelled. Every chain records
# the wall time of its last pass in `elapsed_`, so the run is also the
# measurement.
spans = (1, 5, 10, 15, 20, 25)
estimates = {}
snr_per_span = {}
for n_spans in spans:
    chain = get_full_chain(n_spans).seed(0)
    estimates[n_spans] = chain(N)
    snr, ser = score(chain)
    snr_per_span[n_spans] = snr
    print(f"{n_spans:2d} spans: SNR {snr:5.2f} dB, SER {ser:.4f}, "
          f"phase {np.rad2deg(chain['phase'].theta_):+6.1f} deg, "
          f"{chain.elapsed_:5.1f} s")

# `chain` is the 25-span one the loop ended on, and its taps still hold
# that run: the field the receiver saw, before any of the DSP.
plot_iq(chain.tap("rx_field"),
        title=f"received field ({dBm} dBm, {N_span} spans)")
plt.savefig(f"{img_dir}/one_shot_nli_fig1.png")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, n_spans in zip(axes, (1, 10, 25), strict=True):
    plot_iq(estimates[n_spans], reference=alphabet, ax=ax)
    ax.set_title(f"after {n_spans} span{'s' if n_spans > 1 else ''}, "
                 f"SNR {snr_per_span[n_spans]:.1f} dB")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig2.png")

ax = plot_error_rate(np.array(spans),
                     {"dispersion compensation only":
                      np.array(list(snr_per_span.values()))},
                     xlabel="spans travelled", ylabel="effective SNR [dB]",
                     yscale="linear", title=f"{dBm} dBm launch power")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig3.png")

# Where that time goes. `profile_execution_time` runs the chain and times
# each block on the way through, so the same pass answers the question.
profile = get_full_chain(N_span).seed(0).profile_execution_time(N)
print("\nblock                    time")
for block_id, elapsed in profile.items():
    print(f"{block_id:22s} {1e3 * elapsed:8.1f} ms")

# --- 2. undoing the nonlinearity too ----------------------------------
# The same chain with the nonlinear term switched back on in the
# receiver. Everything else -- the seed, the link, the phase correction
# -- is unchanged, so the comparison is over the same realization.
back_propagated = get_full_chain(N_span, steps=StPS_DBP,
                                 linear_only=False).seed(0)
profile_dbp = back_propagated.profile_execution_time(N)
snr_dbp, ser_dbp = score(back_propagated)
print(f"\ndispersion compensation   SNR={snr_per_span[N_span]:5.2f} dB  "
      f"receiver {1e3 * profile['dbp']:7.1f} ms")
print(f"digital back-propagation  SNR={snr_dbp:5.2f} dB  SER={ser_dbp:.4f}  "
      f"receiver {1e3 * profile_dbp['dbp']:7.1f} ms  "
      f"residual phase={np.rad2deg(back_propagated['phase'].theta_):+.1f} deg")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
plot_iq(estimates[N_span], reference=alphabet, ax=axes[0])
axes[0].set_title(f"dispersion compensation\n"
                  f"SNR {snr_per_span[N_span]:.2f} dB")
plot_iq(back_propagated.tap("phase"), reference=alphabet, ax=axes[1])
axes[1].set_title(f"digital back-propagation\nSNR {snr_dbp:.2f} dB")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig4.png")
