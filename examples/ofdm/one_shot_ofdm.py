"""Single-carrier equalization against OFDM, on one EPA channel.

Run from this directory: it writes the tutorial's figures and diagrams
into ../../docs/tutorials/.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN, FIRChannel, TappedDelayLineChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.fading import get_delay_profile
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import get_alphabet
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.ofdm.chains import OFDMReceiver, OFDMTransmitter

img_dir = "../../docs/tutorials/img/"
mermaid_dir = "../../docs/tutorials/mermaid/"

M = 16
N = 1280
fs = 7.68e6
snr_dB = 18
alphabet = get_alphabet("QAM", M)

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
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        FIRChannel(h),
        AWGN(snr_dB=snr_dB, name="data_rx"),
        LinearEqualizer(h, method="zf", name="data_rx_eq"),
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx", "data_rx_eq"])

sc_chain.seed(1)
start = time.perf_counter()
detected = sc_chain(N)
sc_time = time.perf_counter() - start
sc_ser = compute_ser(sc_chain.tap("data_tx"), detected)
print(f"single carrier: SER {sc_ser:.4f}, {sc_time * 1e3:.0f} ms")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
for index, (block, title) in enumerate(
        [("data_rx", "received"), ("data_rx_eq", "after ZF equalization")]):
    plot_iq(sc_chain.tap(block), reference=alphabet, title=title,
            ax=axes[index])
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_ofdm_fig2.png")

# --- the other strategy ----------------------------------------------
N_carrier = 128
N_cp = 10
ofdm_chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        OFDMTransmitter(N_carrier, N_cp),
        FIRChannel(h),
        AWGN(snr_dB=snr_dB, name="data_rx"),
        OFDMReceiver(N_carrier, N_cp, h=h, name="data_rx_eq"),
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx_eq"])

ofdm_chain.seed(1)
start = time.perf_counter()
detected = ofdm_chain(N)
ofdm_time = time.perf_counter() - start
ofdm_ser = compute_ser(ofdm_chain.tap("data_tx"), detected)
print(f"OFDM          : SER {ofdm_ser:.4f}, {ofdm_time * 1e3:.2f} ms "
      f"({sc_time / ofdm_time:.0f} times faster)")

plot_iq(ofdm_chain.tap("data_rx_eq"), reference=alphabet,
        title="OFDM, after one-tap equalization")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig3.png")

# --- error rate ------------------------------------------------------
# One operating point is not a conclusion. Each sweep is repeated over
# independent noise seeds because 1280 symbols only resolve down to
# 8e-4; the channel is the same throughout, by construction.
snr_list = np.arange(6, 22, 2)
measured = {}
for name, chain in (("single carrier", sc_chain), ("OFDM", ofdm_chain)):
    runs = [sweep(chain, "data_rx.snr_dB", snr_list, {"ser": compute_ser}, N,
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
lengths = [128, 256, 512, 1024]
runtime = {"single carrier": [], "OFDM": []}
for length in lengths:
    for name, chain in (("single carrier", sc_chain), ("OFDM", ofdm_chain)):
        chain.seed(1)
        start = time.perf_counter()
        chain(length)
        runtime[name].append((time.perf_counter() - start) * 1e3)

print("\n     N   single carrier      OFDM     ratio")
for index, length in enumerate(lengths):
    sc_ms = runtime["single carrier"][index]
    ofdm_ms = runtime["OFDM"][index]
    print(f"{length:6d} {sc_ms:13.1f} ms {ofdm_ms:7.2f} ms {sc_ms/ofdm_ms:8.0f}")

plot_error_rate(np.array(lengths),
                {name: np.array(values) for name, values in runtime.items()},
                xlabel="block length $N$", ylabel="receiver runtime [ms]",
                xscale="log", yscale="log", title="what equalization costs")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig5.png")

# The diagram is exported from the chain itself (D33c), so the picture
# cannot say something the code does not.
with open(f"{mermaid_dir}/ofdm_chain.mmd", "w") as stream:
    stream.write(ofdm_chain.to_mermaid())

# --- where the difference comes from ---------------------------------
# OFDM gives subcarrier k the gain of one frequency, so its symbols see a
# spread of SNRs. Single-carrier zero forcing inverts a *linear*
# convolution, which spreads the enhanced noise over the whole block, so
# every symbol sees the same gain -- the harmonic mean of |H|^2.
from scipy.linalg import toeplitz              # noqa: E402  (local to this study)

gain = np.abs(np.fft.fft(h, N_carrier)) ** 2
convolution = toeplitz(np.r_[h, np.zeros(N - 1)], np.r_[h[0], np.zeros(N - 1)])
enhancement = np.real(np.diag(
    np.linalg.inv(convolution.conj().T @ convolution)))
# AWGN(snr_dB=) measures the power at its input, i.e. after the channel
rho = 10 ** (snr_dB / 10) / gain.mean()

print(f"\nper-symbol SNR at a nominal {snr_dB} dB")
print(f"  OFDM  : {10 * np.log10(rho * gain.min()):5.1f} to "
      f"{10 * np.log10(rho * gain.max()):5.1f} dB")
print(f"  SC-ZF : {10 * np.log10(rho / enhancement.max()):5.1f} to "
      f"{10 * np.log10(rho / enhancement.min()):5.1f} dB")
print(f"  arithmetic mean of |H|^2 {gain.mean():.3f}, "
      f"harmonic mean {1 / np.mean(1 / gain):.3f}")
