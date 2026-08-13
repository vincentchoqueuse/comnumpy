import time

import numpy as np
import matplotlib.pyplot as plt

from comnumpy import monte_carlo, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector, SphereDecoder)
from comnumpy.mimo.utils import rayleigh_channel

img_dir = "../../docs/tutorials/img/"

N = 1000
N_r, N_t = 3, 2
constellation = Constellation("PSK", 4)
sigma2 = 0.1
H = rayleigh_channel(N_r, N_t, seed=0)


def get_chain(detector):
    """The same link, closed by one detector or another.

    Only the last block changes, which is the point: a detector is a
    chain block like any other, and the comparison further down is four
    chains that differ by that block alone.
    """
    return Sequential([
        SymbolGenerator(constellation.order, name="tx"),
        SymbolMapper(constellation),
        FlatMIMOChannel(H, name="channel"),
        AWGN(sigma2=sigma2, name="noise"),
        detector,
    ], taps=["tx", "noise"], name=f"{N_r}x{N_t} MIMO, {detector.name}")


detectors = {
    "ZF": LinearDetector(constellation, H=H, method="zf", name="detector"),
    "MMSE": LinearDetector(constellation, H=H, sigma2=sigma2, method="mmse",
                           name="detector"),
    "OSIC": OrderedSuccessiveInterferenceCancellationDetector(
        constellation, osic_type="sinr", H=H, sigma2=sigma2,
        name="detector"),
    "ML": MaximumLikelihoodDetector(constellation, H=H, name="detector"),
    "SD": SphereDecoder(constellation, H=H, name="detector"),
}
chains = {}
for name, detector in detectors.items():
    chains[name] = get_chain(detector)

for name, chain in chains.items():
    chain.seed(0)
    detected = chain((N_t, N))
    print(f"* detector {name:5s}: ser={compute_ser(chain.tap('tx'), detected):.4f}")

Y = chains["ZF"].tap("noise")
fig1, axes1 = plt.subplots(nrows=1, ncols=N_r, figsize=(4 * N_r, 4))
for index in range(N_r):
    axes1[index].plot(np.real(Y[index, :]), np.imag(Y[index, :]), ".")
    axes1[index].set_title(f"Received signal (antenna {index + 1})")
    axes1[index].set_xlim([-2, 2])
    axes1[index].set_ylim([-2, 2])
    style.apply(axes1[index], "iq")
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig1.png")

Z = detectors["ZF"].linear_estimator(Y)
fig2, axes2 = plt.subplots(nrows=1, ncols=N_t, figsize=(4 * N_t, 4))
for index in range(N_t):
    axes2[index].plot(np.real(Z[index, :]), np.imag(Z[index, :]), ".",
                      label="estimated")
    axes2[index].plot(np.real(constellation.alphabet),
                      np.imag(constellation.alphabet), "kx",
                      markersize=9, label="transmitted alphabet")
    axes2[index].set_title(f"Estimated signal (stream {index + 1})")
    axes2[index].set_xlim([-2, 2])
    axes2[index].set_ylim([-2, 2])
    style.apply(axes2[index], "iq")
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig2.png")

snr_dB_list = np.arange(0, 20, 3)
n_channels = 200
n_symbols = 200


NOISE_AWARE = {"MMSE", "OSIC"}   # the two detectors that weight by sigma2


def average_ser(name, chain, snr_dB, seed=0):
    """Average one chain over independent Rayleigh draws at one SNR."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_channels):
        matrix = rayleigh_channel(N_r, N_t, rng=rng)
        draws.append((matrix, matrix))
    noise_variance = N_t * 10 ** (-snr_dB / 10)
    params = {"noise.sigma2": noise_variance}
    if name in NOISE_AWARE:
        params["detector.sigma2"] = noise_variance
    chain.set_params(**params)
    results = monte_carlo(chain, ("channel.H", "detector.H"), draws,
                          {"ser": compute_ser}, stimulus=(N_t, n_symbols),
                          reference="tx", seed=seed)
    return float(np.mean(results["ser"]))


curves = {}
for name, chain in chains.items():
    values = []
    for snr_dB in snr_dB_list:
        values.append(average_ser(name, chain, snr_dB))
    curves[name] = values
for name, values in curves.items():
    line = f"{name:5s} "
    for value in values:
        line += f"{value:.4f} "
    print(line)

ax = plot_error_rate(snr_dB_list, curves, ylabel="SER",
                     title=f"{N_r}x{N_t} MIMO, {constellation.order}-"
                           f"{constellation.family}, "
                           f"{n_channels} channel draws per point")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig3.png")

# --- what the pruning is worth, in seconds ---------------------------
# 16-QAM on four streams: 65 536 candidates per vector, which is where
# the two detectors stop costing the same.
print("\n16-QAM, 4x4: same decision, what it costs")
big_constellation = Constellation("QAM", 16)
big_H = rayleigh_channel(4, 4, seed=2)
big_decoder = SphereDecoder(big_constellation, H=big_H, name="detector")
big_detectors = {
    "ML": MaximumLikelihoodDetector(big_constellation, H=big_H, name="detector"),
    "SD": big_decoder,
}
elapsed = {}
for name in big_detectors:
    elapsed[name] = []
for name, detector in big_detectors.items():
    big_chain = Sequential([
        SymbolGenerator(big_constellation.order, name="tx"),
        SymbolMapper(big_constellation),
        FlatMIMOChannel(big_H, name="channel"),
        AWGN(sigma2=1.0, name="noise"),
        detector,
    ], taps=["tx"], name=f"4x4 MIMO, {name}")
    for snr_dB in snr_dB_list:
        big_chain.seed(4)
        big_chain.set_params(noise__sigma2=4 * 10 ** (-snr_dB / 10))
        start = time.perf_counter()
        detected = big_chain((4, 400))
        elapsed[name].append((time.perf_counter() - start) * 1e3)
        errors = compute_ser(big_chain.tap("tx"), detected)
        nodes = (f"{big_decoder.nodes_:7.1f} nodes" if name == "SD"
                 else f"{16 ** 4:7d} nodes")
        print(f"  {name:2s} {snr_dB:2d} dB {elapsed[name][-1]:8.1f} ms  "
              f"{nodes}   SER {errors:.4f}")

# A runtime is not an error rate, so it does not go through
# plot_error_rate and there is no style kind for it either: what this
# figure needs is a log ordinate and a grid, which is what it says.
fig4, ax = plt.subplots()
for name, values in elapsed.items():
    ax.semilogy(snr_dB_list, values, "o-", fillstyle="none", label=name)
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("detection time for 400 vectors [ms]")
ax.set_title("16-QAM, 4x4: same decision, what it costs")
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig4.png")

mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("mimo_zf", chains["ZF"])]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
