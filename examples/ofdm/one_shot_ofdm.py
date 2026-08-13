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
from comnumpy.core.visualizers import plot_error_rate
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
for key, value in channel.info().items():
    print(f"{key}: {value}")

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
    ], taps=["data_tx", "data_rx", "data_rx_eq"])

sc_chain.seed(1)
detected = sc_chain(N)
sc_ser = compute_ser(sc_chain.tap("data_tx"), detected)
print(f"single carrier: SER {sc_ser:.4f}, {sc_chain.elapsed_ * 1e3:.0f} ms")

# An IQ plane is a scatter of the real part against the imaginary one;
# style.apply gives it the axis labels, the grid and the equal aspect
# ratio without which a constellation is not the constellation.
alphabet = constellation.alphabet
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for ax, tap, name in [(axes[0], "data_rx", "received"),
                      (axes[1], "data_rx_eq", "after ZF equalization")]:
    symbols = sc_chain.tap(tap)
    ax.plot(np.real(symbols), np.imag(symbols), ".", label="symbols")
    ax.plot(np.real(alphabet), np.imag(alphabet), "kx", markersize=9,
            label="alphabet")
    ax.set_title(name)
    style.apply(ax, "iq")
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
    ], taps=["data_tx", "data_rx_eq"])

ofdm_chain.seed(1)
detected = ofdm_chain(N)
ofdm_ser = compute_ser(ofdm_chain.tap("data_tx"), detected)
speedup = sc_chain.elapsed_ / ofdm_chain.elapsed_
print(f"OFDM          : SER {ofdm_ser:.4f}, {ofdm_chain.elapsed_ * 1e3:.2f} ms "
      f"({speedup:.0f} times faster)")

equalized = ofdm_chain.tap("data_rx_eq")
fig3, ax3 = plt.subplots()
ax3.plot(np.real(equalized), np.imag(equalized), ".", label="symbols")
ax3.plot(np.real(alphabet), np.imag(alphabet), "kx", markersize=9,
         label="alphabet")
ax3.set_title("OFDM, after one-tap equalization")
style.apply(ax3, "iq")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig3.png")

# --- error rate ------------------------------------------------------
# One operating point is not a conclusion. Each sweep is repeated over
# independent noise seeds because 1280 symbols only resolve down to
# 8e-4; the channel is the same throughout, by construction.
snr_list = np.arange(6, 22, 2)
measured = {}
for name, chain in (("single carrier", sc_chain), ("OFDM", ofdm_chain)):
    runs = [monte_carlo(chain, "data_rx.snr_dB", snr_list, {"ser": compute_ser}, N,
                        reference="data_tx", seed=trial)["ser"]
            for trial in range(1, 3)]
    measured[name] = np.mean(runs, axis=0)

# A swept result is an abscissa and one series per curve, which is what
# monte_carlo already returns. Written down once, it is printed and
# plotted from the same object rather than restated for each.
ser_data = {"x": snr_list, "curves": measured}
print()
print_data(ser_data, xlabel="SNR [dB]", ylabel="SER")

plot_error_rate(snr_list, measured, ylabel="SER",
                title="16-QAM over one EPA realization")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig4.png")

# --- what it costs ---------------------------------------------------
# A chain records the wall time of its last pass in `elapsed_`, so the
# run is also the measurement.
lengths = [128, 256, 512, 1024]
runtime = {"single carrier": [], "OFDM": []}
for length in lengths:
    for name, chain in (("single carrier", sc_chain), ("OFDM", ofdm_chain)):
        chain.seed(1)
        chain(length)
        runtime[name].append(1e3 * chain.elapsed_)

runtime_data = {
    "x": lengths,
    "curves": {
        "single carrier": np.array(runtime["single carrier"]),
        "OFDM": np.array(runtime["OFDM"]),
        "ratio": (np.array(runtime["single carrier"])
                  / np.array(runtime["OFDM"])),
    },
}
print()
print_data(runtime_data, xlabel="block length N",
           ylabel="receiver runtime [ms], and their ratio")

# The same dictionary, drawn. The ratio is dimensionless and does not
# belong on an axis in milliseconds, so the figure takes the two
# runtimes; everything else comes from the object the table printed.
timed = {"x": lengths, "curves": {}}
for name, values in runtime_data["curves"].items():
    if name != "ratio":
        timed["curves"][name] = values
ax5 = plot_data(timed, xlabel="block length $N$",
                ylabel="receiver runtime [ms]",
                xscale="log", yscale="log", marker="o", fillstyle="none")
ax5.set_title("what equalization costs")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig5.png")
