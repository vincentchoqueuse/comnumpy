import numpy as np
import matplotlib.pyplot as plt
import time
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import (AWGN, FIRChannel,
                                    TappedDelayLineChannel)
from comnumpy.core.fading import get_delay_profile
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser
from comnumpy.core.visualizers import plot_iq
from comnumpy.ofdm.chains import OFDMTransmitter, OFDMReceiver

img_dir = "../../docs/tutorials/img/"

M = 16
N = 1280
fs = 7.68e6
sigma2 = 0.015
alphabet = get_alphabet("QAM", M)

channel_model = TappedDelayLineChannel(get_delay_profile("EPA"), fs=fs,
                                       seed=18, name="sounder")
h = channel_model.impulse_response()
print(f"EPA at {fs/1e6:.2f} MHz: {len(h)} taps, delay spread "
      f"{get_delay_profile('EPA').rms_delay_spread_ns:.0f} ns")

simple_chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        FIRChannel(h),
        AWGN(sigma2=sigma2, name="data_rx"),
        LinearEqualizer(h, method="zf", name="data_rx_eq"),
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx", "data_rx_eq"])

simple_chain.seed(1)
start_time = time.time()
s_rx = simple_chain(N)
stop_time = time.time()

s_tx = simple_chain.tap("data_tx")
ser = compute_ser(s_tx, s_rx)
elapsed_time = stop_time - start_time
print(f"SER: {ser}")
print(f"elapsed time: {elapsed_time} s")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
for indice, processor_name in enumerate(["data_rx", "data_rx_eq"]):
    plot_iq(simple_chain.tap(processor_name), reference=alphabet,
            title=f"Received signal ({processor_name})", ax=axes[indice])
    axes[indice].set_xlim([-2, 2])
    axes[indice].set_ylim([-2, 2])

plt.savefig(f"{img_dir}/one_shot_ofdm_fig1.png")

N_carrier = 128
N_cp = 10
ofdm_chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        OFDMTransmitter(N_carrier, N_cp),   # <- add OFDM transmitter
        FIRChannel(h),
        AWGN(sigma2=sigma2),
        OFDMReceiver(N_carrier, N_cp, h=h, name="data_rx"), # <- add OFDM receiver
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx"])

ofdm_chain.seed(1)
start_time = time.time()
s_rx = ofdm_chain(N)
stop_time = time.time()

s_tx = ofdm_chain.tap("data_tx")
data_rx = ofdm_chain.tap("data_rx")
ser = compute_ser(s_tx, s_rx)
elapsed_time = stop_time - start_time
print(f"SER: {ser}")
print(f"elapsed time: {elapsed_time} s")

plot_iq(data_rx, reference=alphabet, title="OFDM Chain: received data")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig2.png")

print("sigma2     single carrier      OFDM     |H| spans "
      f"{np.max(np.abs(np.fft.fft(h, N_carrier))) / np.min(np.abs(np.fft.fft(h, N_carrier))):.0f}")
for variance in [0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]:
    row = []
    for chain, block in ((simple_chain, "data_rx"), (ofdm_chain, "awgn")):
        chain.seed(1)
        chain.set_params(**{f"{block}.sigma2": variance})
        detected = chain(N)
        row.append(compute_ser(chain.tap("data_tx"), detected))
    print(f"{variance:8.4f}   {row[0]:14.4f} {row[1]:9.4f}")

mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("ofdm_single_carrier", simple_chain),
        ("ofdm_chain", ofdm_chain),
        ("ofdm_transmitter", ofdm_chain[2].chain),
        ("ofdm_receiver", ofdm_chain[5].chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()


# --- the channel itself, which is what the subcarriers are answering to ---
H = np.fft.fftshift(np.fft.fft(h, N_carrier))
bins = np.fft.fftshift(np.fft.fftfreq(N_carrier, d=1 / fs)) / 1e6
gain_dB = 20 * np.log10(np.abs(H))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))
axes[0].stem(np.arange(len(h)) / fs * 1e9, np.abs(h), basefmt=" ")
axes[0].set_xlabel("delay [ns]")
axes[0].set_ylabel("|h|")
axes[0].set_title(f"EPA impulse response, {len(h)} taps")
axes[0].grid(True, alpha=0.4)

axes[1].plot(bins, gain_dB, lw=1.2)
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].fill_between(bins, gain_dB, -30, where=gain_dB < -10,
                     color="C3", alpha=0.20, label="more than 10 dB down")
axes[1].set_xlabel("frequency [MHz]")
axes[1].set_ylabel(r"$|H(f)|$ [dB]")
axes[1].set_title("what each subcarrier sees")
axes[1].set_ylim(-30, 15)
axes[1].legend()
axes[1].grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_ofdm_fig3.png")

deep = int(np.sum(gain_dB < -10))
print(f"\nchannel: {gain_dB.max() - gain_dB.min():.1f} dB peak-to-null across "
      f"{fs/1e6:.2f} MHz, {deep} of {N_carrier} subcarriers more than 10 dB "
      f"down; 1/tau_rms = {1e3/get_delay_profile('EPA').rms_delay_spread_ns:.1f} MHz "
      f"against {fs/N_carrier/1e3:.1f} kHz of subcarrier spacing")
