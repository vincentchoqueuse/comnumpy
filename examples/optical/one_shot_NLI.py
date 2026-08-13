"""Fibre propagation span by span, and digital back-propagation.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy.core import Sequential, plot_iq
from comnumpy.core.compensators import DataAidedPhaseCompensator
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_effective_snr, compute_ser
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.core.utils import Constellation
from comnumpy import style
from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import dbm_to_watt, launch_amplitude

style.use()

img_dir = "../../docs/tutorials/img/"

constellation = Constellation("QAM", 16)
N_s = 2**9 * 6                # symbols per run
oversampling_sim = 6          # samples per symbol in the channel
oversampling_dsp = 2          # samples per symbol in the receiver
NF_dB = 5                     # amplifier noise figure in dB
rolloff = 0.1
StPS = 50                     # split steps per span, forward
StPS_DBP = 100                # split steps per span, backward
R_s = 10.7e9                  # baud rate
L_span = 80                   # span length in km
N_span = 25
dBm = -3
fiber = FiberSpec()           # the standard fibre, named so the budget sees it

fs = R_s * oversampling_sim
oversampling_ratio = oversampling_sim // oversampling_dsp
amp = launch_amplitude(dbm_to_watt(dBm))


# --- the chain --------------------------------------------------------
def get_chain(n_spans, *, steps=1, linear_only=True):
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
        SymbolGenerator(constellation.order, name="data_tx"),
        SymbolMapper(constellation, name="signal_tx"),
        Amplifier(amp),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        FiberLink(N_spans=n_spans, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, fiber=fiber, name="link"),
        Downsampler(oversampling_ratio, use_filter=True, name="rx_field"),
        DBP(n_spans, L_span=L_span, StPS=steps, step_type="linear",
            use_only_linear=linear_only, fs=fs / oversampling_ratio,
            fiber=fiber, name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(1 / amp),
        DataAidedPhaseCompensator(name="phase"),
        SymbolDemapper(constellation, name="data_rx"),
        ], taps=["data_tx", "signal_tx", "rx_field", "phase", "data_rx"],
        wiring={"phase.reference": "signal_tx"})


def score(chain):
    """Effective SNR in dB and symbol error rate of a finished run."""
    return (compute_effective_snr(chain.tap("signal_tx"),
                                  chain.tap("phase"), unit="dB"),
            compute_ser(chain.tap("data_tx"), chain.tap("data_rx")))


# --- 1. what the link does to the signal ------------------------------
# One call per span count, and the same seed each time: the transmitted
# symbols and the amplifier noise are identical, so the only thing that
# changes along the curve is the distance travelled. Every chain records
# the wall time of its last pass in `elapsed_`, so the run is also the
# measurement.
#
# Each span count is run twice. The second pass switches the fibre's Kerr
# term off through the chain -- same seed, same symbols, same amplifiers,
# so the *only* thing removed is the nonlinearity. What is left is the
# ASE the amplifiers have piled up, and it is the reference every number
# below is read against: a receiver cannot do better than a link with no
# nonlinearity in it. It is also the cheapest check that the chain is
# sound, because that reference has a closed form and the fibre does not.
spans = (1, 5, 10, 15, 20, 25)
estimates = {}
snr_per_span = {}
snr_ase_only = {}
print("spans   measured   ASE only   the fibre      SER     phase     time")
for n_spans in spans:
    chain = get_chain(n_spans).seed(0)
    chain(N_s)
    estimates[n_spans] = chain.tap("phase")
    snr, ser = score(chain)
    snr_per_span[n_spans] = snr
    elapsed = chain.elapsed_
    theta_deg = np.rad2deg(chain["phase"].theta_)
    # the field the receiver saw, before any of the DSP; the loop keeps
    # overwriting it, so what survives it is the longest link's
    received_field = chain.tap("rx_field")

    chain.seed(0).set_params(link__use_only_linear=True)
    chain(N_s)
    snr_ase_only[n_spans] = score(chain)[0]

    print(f"{n_spans:5d} {snr:8.2f} dB {snr_ase_only[n_spans]:7.2f} dB "
          f"{snr_ase_only[n_spans] - snr:8.2f} dB {ser:8.4f} "
          f"{theta_deg:+7.1f} deg {elapsed:6.1f} s")

# Two checks on that reference, both cheap, and worth running on any
# chain before believing a decibel it produces.
#
# The first switches the noise off as well as the nonlinearity. What is
# left is what the transmitter and the receiver do to a signal that
# travelled through nothing: the distortion floor of the DSP itself --
# pulse shaping, resampling, matched filtering. It has to sit far above
# every number in the table, or the chain is measuring its own filters
# rather than the fibre.
floor = get_chain(1).seed(0)
floor.set_params(link__use_only_linear=True, link__noise_scaling=0.0)
floor(N_s)
print(f"\ndistortion floor of the chain, no noise and no fibre: "
      f"{score(floor)[0]:.1f} dB")

# The second is the amplifier noise against its closed form. The link
# budgets its own noise: `budget` asks it how much ASE it accumulates in
# the bandwidth a matched filter keeps -- the symbol rate, not the
# simulated `fs` -- and that prediction owes the simulation nothing, it
# comes from the noise figure and the span loss alone.
print("\nspans   ASE only   P / P_ASE      gap")
for n_spans in spans:
    link = get_chain(n_spans)["link"]
    ase_W = link.budget(R_s)["ase_power_W"]
    predicted = 10 * np.log10(dbm_to_watt(dBm) / ase_W)
    print(f"{n_spans:5d} {snr_ase_only[n_spans]:8.2f} dB {predicted:8.2f} dB "
          f"{snr_ase_only[n_spans] - predicted:+8.2f} dB")

plot_iq(received_field,
        title=f"received field ({dBm} dBm, {N_span} spans)")
plt.savefig(f"{img_dir}/one_shot_nli_fig1.png")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for ax, n_spans in zip(axes, (1, 10, 25), strict=True):
    plot_iq(estimates[n_spans], reference=constellation, ax=ax)
    ax.set_title(f"after {n_spans} span{'s' if n_spans > 1 else ''}, "
                 f"SNR {snr_per_span[n_spans]:.1f} dB")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig2.png")

# the dicts are keyed by span count in sweep order, so their values
# are already the curves
measured = np.array(list(snr_per_span.values()))
ase_only = np.array(list(snr_ase_only.values()))

# An effective SNR is not an error rate: linear ordinate, two curves,
# nothing a helper would say better than the four lines that draw it.
fig3, ax = plt.subplots()
ax.plot(spans, ase_only, "o-", fillstyle="none", label="amplifier noise only")
ax.plot(spans, measured, "s-", fillstyle="none",
        label="dispersion compensation only")
ax.set_xlabel("spans travelled")
ax.set_ylabel("effective SNR [dB]")
ax.set_title(f"{dBm} dBm launch power")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig3.png")

# Where that time goes. `profile_execution_time` runs the chain and times
# each block on the way through, so the same pass answers the question.
profile = get_chain(N_span).seed(0).profile_execution_time(N_s)
print("\nblock                    time")
for block_id, elapsed in profile.items():
    print(f"{block_id:22s} {1e3 * elapsed:8.1f} ms")

# --- 2. undoing the nonlinearity too ----------------------------------
# The same chain with the nonlinear term switched back on in the
# receiver. Everything else -- the seed, the link, the phase correction
# -- is unchanged, so the comparison is over the same realization.
back_propagated = get_chain(N_span, steps=StPS_DBP,
                                 linear_only=False).seed(0)
profile_dbp = back_propagated.profile_execution_time(N_s)
snr_dbp, ser_dbp = score(back_propagated)
print(f"\ndispersion compensation   SNR={snr_per_span[N_span]:5.2f} dB  "
      f"receiver {1e3 * profile['dbp']:7.1f} ms")
print(f"digital back-propagation  SNR={snr_dbp:5.2f} dB  SER={ser_dbp:.4f}  "
      f"receiver {1e3 * profile_dbp['dbp']:7.1f} ms  "
      f"residual phase={np.rad2deg(back_propagated['phase'].theta_):+.1f} deg")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for ax, symbols, name in [
        (axes[0], estimates[N_span],
         f"dispersion compensation\nSNR {snr_per_span[N_span]:.2f} dB"),
        (axes[1], back_propagated.tap("phase"),
         f"digital back-propagation\nSNR {snr_dbp:.2f} dB")]:
    plot_iq(symbols, reference=constellation, title=name, ax=ax)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_nli_fig4.png")
