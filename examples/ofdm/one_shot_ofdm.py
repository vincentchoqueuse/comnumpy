"""Single-carrier equalization against OFDM, on one EPA channel.

Run from this directory: it writes the tutorial's figures and diagrams
into ../../docs/tutorials/.
"""

import matplotlib.pyplot as plt
import numpy as np

from comnumpy import monte_carlo, style
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

print("\nSNR [dB]  single carrier      OFDM")
for index, value in enumerate(snr_list):
    print(f"{value:8d} {measured['single carrier'][index]:15.4f} "
          f"{measured['OFDM'][index]:9.4f}")

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

print("\n     N   single carrier      OFDM     ratio")
for index, length in enumerate(lengths):
    sc_ms = runtime["single carrier"][index]
    ofdm_ms = runtime["OFDM"][index]
    print(f"{length:6d} {sc_ms:13.1f} ms {ofdm_ms:7.2f} ms {sc_ms/ofdm_ms:8.0f}")

measured_runtime = {}
for name, values in runtime.items():
    measured_runtime[name] = np.array(values)

# A runtime is not an error rate. Two log axes and a grid is the whole
# figure, so it is written out rather than routed through a helper whose
# name would then be describing something else.
fig5, ax5 = plt.subplots()
for name, values in measured_runtime.items():
    ax5.loglog(lengths, values, "o-", fillstyle="none", label=name)
ax5.set_xlabel("block length $N$")
ax5.set_ylabel("receiver runtime [ms]")
ax5.set_title("what equalization costs")
ax5.grid(True, which="both")
ax5.legend()
plt.savefig(f"{img_dir}/one_shot_ofdm_fig5.png")
