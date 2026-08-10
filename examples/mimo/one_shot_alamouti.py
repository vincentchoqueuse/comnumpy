import numpy as np
import matplotlib.pyplot as plt

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ser_rayleigh_psk
from comnumpy.core.processors import Amplifier
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

img_dir = "../../docs/tutorials/img/"

M = 4
alphabet = get_alphabet("PSK", M)
code = get_code("alamouti")
power = 1 / np.sqrt(code.n_tx)          # split the power over the antennas
sigma2 = 0.2

H = rayleigh_channel(1, 2, seed=42)
alamouti = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    Amplifier(power),
    SpaceTimeEncoder(code),
    FlatMIMOChannel(H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    SpaceTimeDecoder(code, H=H, name="detector"),
    Amplifier(1 / power),
    SymbolDemapper(alphabet),
], taps=["tx", "noise", "detector"], name="Alamouti 2x1")

alamouti.seed(7)                        # every stochastic block, reproducibly
s_rx = alamouti(500 * code.n_symbols)
print(f"one-shot SER: {compute_ser(alamouti.tap('tx'), s_rx):.4f}")

alamouti.summary(500 * code.n_symbols)

received = alamouti.tap("noise")[0]
combined = alamouti.tap("detector") / power
fig1, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
plot_iq(received, marker=".", ax=ax_left)
ax_left.set_title("tap('noise'): the single receive antenna")
plot_iq(combined, reference=alphabet, ax=ax_right)
ax_right.set_title("tap('detector'): after Alamouti combining")
for ax in (ax_left, ax_right):
    ax.set_xlabel("in phase")
    ax.set_ylabel("quadrature")
    ax.axis("equal")
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig1.png")

siso_H = rayleigh_channel(1, 1, seed=1)
siso = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    FlatMIMOChannel(siso_H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    LinearDetector(alphabet, H=siso_H, name="detector"),
], taps=["tx"], name="1 Tx, 1 Rx")

mrc_H = rayleigh_channel(2, 1, seed=2)
mrc = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    FlatMIMOChannel(mrc_H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    LinearDetector(alphabet, H=mrc_H, name="detector"),
], taps=["tx"], name="MRC 1 Tx, 2 Rx")


def average_ser(chain, n_rx, n_tx, snr_dB, stimulus, n_channels, seed=0):
    """Average one chain over independent quasi-static fading draws.

    The chain is built once. ``sweep`` takes several dotted parameters
    at once and zips them, so one sweep point sets the channel the
    signal goes through *and* the channel the detector inverts -- which
    is exactly what a fading realization is.
    """
    rng = np.random.default_rng(seed)
    channels = [rayleigh_channel(n_rx, n_tx, rng=rng)
                for _ in range(n_channels)]
    chain.set_params(**{"noise.sigma2": 10 ** (-snr_dB / 10)})
    results = sweep(chain, ("channel.H", "detector.H"),
                    [(H, H) for H in channels],
                    {"ser": compute_ser}, stimulus=stimulus,
                    reference="tx", seed=seed)
    return float(np.mean(results["ser"]))


snr_dB_list = np.arange(4, 25, 4)
draws = [1500, 2000, 2500, 3500, 5000, 6000]
n_symbols = 80
curves = {
    "1 Tx, 1 Rx (no diversity)": [average_ser(siso, 1, 1, value,
                                              (1, n_symbols), count)
                                  for value, count in zip(snr_dB_list, draws, strict=True)],
    "Alamouti, 2 Tx, 1 Rx": [average_ser(alamouti, 1, 2, value, n_symbols,
                                         count)
                             for value, count in zip(snr_dB_list, draws, strict=True)],
    "MRC, 1 Tx, 2 Rx": [average_ser(mrc, 2, 1, value, (1, n_symbols), count)
                        for value, count in zip(snr_dB_list, draws, strict=True)],
}

fine = np.linspace(snr_dB_list[0], snr_dB_list[-1], 100)
per_bit = 10 ** (fine / 10) / np.log2(M)
theory = {
    "1 Tx, 1 Rx (no diversity)": compute_ser_rayleigh_psk(M, per_bit),
    "Alamouti, 2 Tx, 1 Rx": compute_ser_rayleigh_psk(M, per_bit / code.n_tx,
                                                     diversity=2),
    "MRC, 1 Tx, 2 Rx": compute_ser_rayleigh_psk(M, per_bit, diversity=2),
}
ax = plot_error_rate(snr_dB_list, curves, theory=theory, x_theory=fine,
                     ylabel="symbol error rate",
                     title=f"{M}-PSK over Rayleigh, equal transmit power")
ax.set_ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig2.png")

exact = {name: compute_ser_rayleigh_psk(
    M, 10 ** (snr_dB_list / 10) / np.log2(M) / (code.n_tx if "Alamouti" in name
                                                else 1),
    diversity=1 if "no diversity" in name else 2)
    for name in curves}
for name, values in curves.items():
    ratio = np.array(values) / exact[name]
    print(f"{name:28s} measured / closed form  "
          + " ".join(f"{value:.2f}" for value in ratio))

high = np.array([30.0, 40.0])
for name in curves:
    reference = compute_ser_rayleigh_psk(
        M, 10 ** (high / 10) / np.log2(M) / (code.n_tx if "Alamouti" in name
                                             else 1),
        diversity=1 if "no diversity" in name else 2)
    slope = np.polyfit(high / 10, np.log10(reference), 1)[0]
    print(f"{name:28s} diversity order {-slope:.2f}")

alamouti_snr = 10 ** (np.linspace(0, 30, 601) / 10) / np.log2(M)
target = 1e-3
of = {"MRC": compute_ser_rayleigh_psk(M, alamouti_snr, diversity=2),
      "Alamouti": compute_ser_rayleigh_psk(M, alamouti_snr / code.n_tx,
                                           diversity=2)}
needed = {name: np.interp(np.log10(target), np.log10(values)[::-1],
                          np.linspace(0, 30, 601)[::-1])
          for name, values in of.items()}
print(f"SNR for SER = {target:g}: MRC {needed['MRC']:.1f} dB, Alamouti "
      f"{needed['Alamouti']:.1f} dB, gap {needed['Alamouti'] - needed['MRC']:.2f} dB "
      f"(10log10(N_t) = {10 * np.log10(code.n_tx):.2f} dB)")

mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("alamouti", alamouti), ("mrc", mrc)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())
