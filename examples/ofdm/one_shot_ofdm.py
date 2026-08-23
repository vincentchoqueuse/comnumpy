"""Single-carrier equalization against OFDM, on one EPA channel.

Run from this directory: it writes the tutorial's figures and diagrams
into ../../docs/tutorials/.
"""

import matplotlib.pyplot as plt
import numpy as np

from comnumpy import monte_carlo, plot_data, print_data, style
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN, FIRChannel, TappedDelayLineChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.fading import get_delay_profile
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.ofdm.chains import OFDMReceiver, OFDMTransmitter

style.use()

img_dir = "../../docs/tutorials/img/"

N = 1280
fs = 7.68e6
snr_dB = 18
constellation = Constellation("QAM", 16)

# --- the channel -----------------------------------------------------
# EPA is the 3GPP Extended Pedestrian A profile; the block draws one
# realization of it. The channel says what it is rather than being
# described from outside.
channel = TappedDelayLineChannel(get_delay_profile("EPA"), fs=fs, seed=8)
print(channel)

# Sounding the model with an impulse gives the realization as a tap
# vector -- which is what both receivers below are given.
h = channel.impulse_response()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
channel.plot("impulse", ax=axes[0])
channel.plot("frequency", scale="dB", ax=axes[1])
axes[0].set_title("one realization of EPA")
axes[1].set_title("what each frequency sees")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_ofdm_fig1.png")

gain_dB = 20 * np.log10(np.abs(np.fft.fft(h, 128)))
print(f"\n|H| spans {gain_dB.max() - gain_dB.min():.1f} dB across "
      f"{fs / 1e6:.2f} MHz")

# --- the problem -----------------------------------------------------
# Send 16-QAM straight through it and look at what arrives.
sc_chain = Sequential([
        SymbolGenerator(constellation.order, name="data_tx"),
        SymbolMapper(constellation),
        FIRChannel(h),
        AWGN(snr_dB=snr_dB, name="data_rx"),
        LinearEqualizer(h, method="zf", name="data_rx_eq"),
        SymbolDemapper(constellation)
    ], observations=["data_tx", "data_rx", "data_rx_eq"])

sc_chain.seed(1)
detected = sc_chain(N)
sc_ser = compute_ser(sc_chain.observation("data_tx"), detected)
print(f"single carrier: SER {sc_ser:.4f}, {sc_chain.elapsed_ * 1e3:.0f} ms")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for ax, observed, name in [(axes[0], "data_rx", "received"),
                      (axes[1], "data_rx_eq", "after ZF equalization")]:
    plot_iq(sc_chain.observation(observed), reference=constellation, title=name, ax=ax)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_ofdm_fig2.png")

# --- the other strategy ----------------------------------------------
N_carrier = 128
N_cp = 10
ofdm_chain = Sequential([
        SymbolGenerator(constellation.order, name="data_tx"),
        SymbolMapper(constellation),
        OFDMTransmitter(N_carrier, N_cp),
        FIRChannel(h),
        AWGN(snr_dB=snr_dB, name="data_rx"),
        OFDMReceiver(N_carrier, N_cp, h=h, name="data_rx_eq"),
        SymbolDemapper(constellation)
    ], observations=["data_tx", "data_rx_eq"])

ofdm_chain.seed(1)
detected = ofdm_chain(N)
ofdm_ser = compute_ser(ofdm_chain.observation("data_tx"), detected)
speedup = sc_chain.elapsed_ / ofdm_chain.elapsed_
print(f"OFDM          : SER {ofdm_ser:.4f}, {ofdm_chain.elapsed_ * 1e3:.2f} ms "
      f"({speedup:.0f} times faster)")

plot_iq(ofdm_chain.observation("data_rx_eq"), reference=constellation,
        title="OFDM, after one-tap equalization")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig3.png")

# --- error rate ------------------------------------------------------
# One operating point is not a conclusion, and 1280 symbols only
# resolve down to 8e-4. Instead of a Python loop over repeated runs,
# the stimulus grows a leading batch axis (D51): every block broadcasts
# over it, the noise is drawn independently per trial, and compute_ser
# pools the n_trials frames of each sweep point -- exactly the mean of
# the per-trial rates. The channel stays the same throughout, by
# construction.
snr_dB_list = np.arange(6, 22, 2)
n_trials = 2

# --- the sweep: no simulation loop -----------------------------------
curves = {}
curves["single carrier"] = monte_carlo(
    sc_chain, "data_rx.snr_dB", snr_dB_list, {"ser": compute_ser},
    (n_trials, N), reference="data_tx", seed=1)["ser"]
curves["OFDM"] = monte_carlo(
    ofdm_chain, "data_rx.snr_dB", snr_dB_list, {"ser": compute_ser},
    (n_trials, N), reference="data_tx", seed=1)["ser"]

# --- results: table and figure ---------------------------------------

ser_data = {"x": snr_dB_list, "curves": curves}
print()
print_data(ser_data, xlabel="SNR [dB]", ylabel="SER")

plot_error_rate(snr_dB_list, curves, ylabel="SER",
                title="16-QAM over one EPA realization")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig4.png")

# --- what it costs ---------------------------------------------------
# A chain records the wall time of its last pass in `elapsed_`, so the
# run is also the measurement. One loop, over the swept lengths; the
# two chains are timed explicitly inside it.
lengths = np.array([128, 256, 512, 1024])
runtime = {"single carrier": np.zeros(len(lengths)),
           "OFDM": np.zeros(len(lengths))}

for index, length in enumerate(lengths):
    sc_chain.seed(1)
    sc_chain(int(length))
    runtime["single carrier"][index] = 1e3 * sc_chain.elapsed_
    ofdm_chain.seed(1)
    ofdm_chain(int(length))
    runtime["OFDM"][index] = 1e3 * ofdm_chain.elapsed_

# --- results: table and figure ---------------------------------------

print()
print_data({"x": lengths,
            "curves": {"single carrier": runtime["single carrier"],
                       "OFDM": runtime["OFDM"],
                       "ratio": (runtime["single carrier"]
                                 / runtime["OFDM"])}},
           xlabel="block length N",
           ylabel="receiver runtime [ms], and their ratio")

# The same arrays, drawn. The ratio is dimensionless and does not
# belong on an axis in milliseconds, so the figure takes the two
# runtimes alone.
ax5 = plot_data({"x": lengths, "curves": runtime},
                xlabel="block length $N$",
                ylabel="receiver runtime [ms]",
                xscale="log", yscale="log", marker="o", fillstyle="none")
ax5.set_title("what equalization costs")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig5.png")
