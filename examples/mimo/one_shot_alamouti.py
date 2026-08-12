"""Rayleigh fading, receive diversity, and the Alamouti space-time code.

Run from this directory: it writes the tutorial's figures and diagrams
into ../../docs/tutorials/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ser_rayleigh_psk
from comnumpy.core.processors import Amplifier
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

img_dir = "../../docs/tutorials/img/"
mermaid_dir = "../../docs/tutorials/mermaid/"

constellation = Constellation("PSK", 4)
M = constellation.order
code = get_code("alamouti")
power = 1 / np.sqrt(code.n_tx)          # split the power over the antennas
sigma2 = 0.2

# --- the fading channel ----------------------------------------------
# One Rayleigh coefficient is a complex Gaussian, so its squared modulus
# is exponential with unit mean: the instantaneous SNR is the average one
# multiplied by that draw.
rng = np.random.default_rng(0)
gain = np.abs(rayleigh_channel(20000, 1, rng=rng).ravel()) ** 2
grid = np.linspace(0, 6, 200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gain, bins=80, range=(0, 6), density=True, alpha=0.5,
        label="20 000 draws of $|h|^2$")
ax.plot(grid, np.exp(-grid), lw=2, label=r"$e^{-\gamma}$, exponential(1)")
ax.set_xlabel(r"channel power gain $|h|^2$")
ax.set_ylabel("probability density")
ax.set_title("a Rayleigh channel is a random SNR")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig1.png")

for threshold in (0.1, 0.01):
    print(f"P[|h|^2 < {threshold:g}] = {np.mean(gain < threshold):.4f}  "
          f"(1 - exp(-{threshold:g}) = {1 - np.exp(-threshold):.4f})")

# --- the three links --------------------------------------------------


def get_link(kind, H, sigma2=sigma2):
    """Build the chain of one scheme, on a given channel realization.

    The three schemes differ by two blocks -- what is put on the
    antennas, and what is done with what comes back -- so they are one
    function and not three scripts.
    """
    source = [SymbolGenerator(constellation.order, name="tx"),
              SymbolMapper(constellation)]
    channel = [FlatMIMOChannel(H, name="channel"),
               AWGN(sigma2=sigma2, name="noise")]
    match kind:
        case "alamouti":
            return Sequential([
                *source, Amplifier(power), SpaceTimeEncoder(code), *channel,
                SpaceTimeDecoder(code, H=H, name="detector"),
                Amplifier(1 / power), SymbolDemapper(constellation),
            ], taps=["tx", "noise", "detector"], name="Alamouti 2x1")
        case "linear":
            # zero forcing on an (N_r, 1) channel *is* maximum ratio
            # combining: the pseudo-inverse of a column is h^H / ||h||^2
            return Sequential([
                *source, *channel,
                LinearDetector(constellation, H=H, name="detector"),
            ], taps=["tx"], name=f"{H.shape[0]} Rx, {H.shape[1]} Tx")
        case _:
            raise ValueError(f"unknown link {kind!r}")


alamouti = get_link("alamouti", rayleigh_channel(1, 2, seed=42))
siso = get_link("linear", rayleigh_channel(1, 1, seed=1))
mrc = get_link("linear", rayleigh_channel(2, 1, seed=2))

# --- one shot ---------------------------------------------------------
alamouti.seed(7)                        # every stochastic block, reproducibly
s_rx = alamouti(500 * code.n_symbols)
print(f"\none-shot SER: {compute_ser(alamouti.tap('tx'), s_rx):.4f}")
alamouti.summary(500 * code.n_symbols)

received = alamouti.tap("noise")[0]
combined = alamouti.tap("detector") / power
fig2, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
plot_iq(received, marker=".", ax=ax_left)
ax_left.set_title("tap('noise'): the single receive antenna")
plot_iq(combined, reference=constellation, ax=ax_right)
ax_right.set_title("tap('detector'): after Alamouti combining")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig2.png")

# --- averaging over fading --------------------------------------------


def average_ser(chain, n_rx, n_tx, snr_dB, stimulus, n_channels, seed=0):
    """Average one chain over independent quasi-static fading draws.

    The chain is built once. ``sweep`` takes several dotted parameters
    at once and zips them, so one sweep point sets the channel the
    signal goes through *and* the channel the detector inverts -- which
    is exactly what a fading realization is.
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_channels):
        H = rayleigh_channel(n_rx, n_tx, rng=rng)
        draws.append((H, H))
    chain.set_params(noise__sigma2=10 ** (-snr_dB / 10))
    results = sweep(chain, ("channel.H", "detector.H"), draws,
                    {"ser": compute_ser}, stimulus=stimulus,
                    reference="tx", seed=seed)
    return float(np.mean(results["ser"]))


# The accuracy of an average over fading is set by the number of channel
# draws, not by the symbol count, so the draw count grows with the SNR.
snr_dB_list = np.arange(4, 25, 4)
draws = [1500, 2000, 2500, 3500, 5000, 6000]
n_symbols = 80
links = {
    "1 Tx, 1 Rx (no diversity)": (siso, 1, 1, (1, n_symbols)),
    "Alamouti, 2 Tx, 1 Rx": (alamouti, 1, 2, n_symbols),
    "MRC, 1 Tx, 2 Rx": (mrc, 2, 1, (1, n_symbols)),
}
curves = {}
for name, (chain, n_rx, n_tx, stimulus) in links.items():
    values = []
    for snr_dB, count in zip(snr_dB_list, draws, strict=True):
        values.append(average_ser(chain, n_rx, n_tx, snr_dB, stimulus, count))
    curves[name] = values

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
plt.savefig(f"{img_dir}/one_shot_alamouti_fig3.png")

# --- the two claims, read off the closed form -------------------------
def closed_form(name, snr_dB):
    """The Rayleigh SER this curve should follow, at equal transmit power."""
    power_split = code.n_tx if "Alamouti" in name else 1
    diversity = 1 if "no diversity" in name else 2
    return compute_ser_rayleigh_psk(
        M, 10 ** (snr_dB / 10) / np.log2(M) / power_split,
        diversity=diversity)


for name, values in curves.items():
    ratio = np.array(values) / closed_form(name, snr_dB_list)
    line = f"{name:28s} measured / closed form  "
    for value in ratio:
        line += f"{value:.2f} "
    print(line)

high = np.array([30.0, 40.0])
for name in curves:
    reference = closed_form(name, high)
    slope = np.polyfit(high / 10, np.log10(reference), 1)[0]
    print(f"{name:28s} diversity order {-slope:.2f}")

grid_dB = np.linspace(0, 30, 601)
alamouti_snr = 10 ** (grid_dB / 10) / np.log2(M)
target = 1e-3
of = {"MRC": compute_ser_rayleigh_psk(M, alamouti_snr, diversity=2),
      "Alamouti": compute_ser_rayleigh_psk(M, alamouti_snr / code.n_tx,
                                           diversity=2)}
needed = {}
for name, values in of.items():
    needed[name] = np.interp(np.log10(target), np.log10(values)[::-1],
                             grid_dB[::-1])
print(f"SNR for SER = {target:g}: MRC {needed['MRC']:.1f} dB, Alamouti "
      f"{needed['Alamouti']:.1f} dB, gap {needed['Alamouti'] - needed['MRC']:.2f} dB "
      f"(10log10(N_t) = {10 * np.log10(code.n_tx):.2f} dB)")

for diagram_name, diagram_chain in [("alamouti", alamouti), ("mrc", mrc)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())
