"""MIMO detection: one spatial mixture, five ways to undo it.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/.
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from comnumpy import monte_carlo, print_data, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector, SphereDecoder)
from comnumpy.mimo.utils import rayleigh_channel

style.use()

img_dir = "../../docs/tutorials/img/"

N = 1000
N_r, N_t = 3, 2
constellation = Constellation("PSK", 4)
M = constellation.order
sigma2 = 0.1

# --- the problem: one channel draw, one chain, zero forcing -----------
H = rayleigh_channel(N_r, N_t, seed=0)
chain = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    LinearDetector(constellation, H=H, method="zf", name="detector"),
], taps=["tx", "noise"], name=f"{N_r}x{N_t} MIMO, ZF")

chain.seed(0)
detected = chain((N_t, N))
print(f"ZF, one channel draw: SER = "
      f"{compute_ser(chain.tap('tx'), detected):.4f}")

Y = chain.tap("noise")
fig1, axes1 = plt.subplots(nrows=1, ncols=N_r, figsize=(4 * N_r, 4))
for index in range(N_r):
    plot_iq(Y[index, :], ax=axes1[index])
    axes1[index].set_title(f"Received signal (antenna {index + 1})")
    axes1[index].set_xlim([-2, 2])
    axes1[index].set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig1.png")

Z = chain["detector"].linear_estimator(Y)
fig2, axes2 = plt.subplots(nrows=1, ncols=N_t, figsize=(4 * N_t, 4))
for index in range(N_t):
    plot_iq(Z[index, :], reference=constellation, ax=axes2[index])
    axes2[index].set_title(f"Estimated signal (stream {index + 1})")
    axes2[index].set_xlim([-2, 2])
    axes2[index].set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig2.png")

# --- the detectors, averaged over fading ------------------------------
# The SAME K channel draws serve every SNR point: one draw is one row
# of the batch (D51). Each technique below is one chain built with the
# stack -- on the channel block and on the detector -- followed by its
# own Monte-Carlo sweep: monte_carlo moves the noise variance, and the
# detectors that weight by sigma2 receive it too, through the zip of
# two parameter paths. No simulation loop.
K = 200
n_symbols = 200
snr_dB_list = np.arange(0, 20, 3)
sigma2_list = N_t * 10.0 ** (-snr_dB_list / 10)
both = list(zip(sigma2_list, sigma2_list, strict=True))
stimulus = (K, N_t, n_symbols)
H_stack = rayleigh_channel(N_r, N_t, seed=1, size=K)
curves = {}

# --- zero forcing ----------------------------------------------------
zf = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_stack, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    LinearDetector(constellation, H=H_stack, method="zf",
                   name="detector"),
], taps=["tx"], name="ZF")
curves["ZF"] = monte_carlo(
    zf, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    stimulus, reference="tx", seed=1)["ser"]

# --- MMSE ------------------------------------------------------------
mmse = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_stack, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    LinearDetector(constellation, H=H_stack, method="mmse", sigma2=1.0,
                   name="detector"),
], taps=["tx"], name="MMSE")
curves["MMSE"] = monte_carlo(
    mmse, ("noise.sigma2", "detector.sigma2"), both, {"ser": compute_ser},
    stimulus, reference="tx", seed=2)["ser"]

# --- OSIC ------------------------------------------------------------
osic = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_stack, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    OrderedSuccessiveInterferenceCancellationDetector(
        constellation, osic_type="sinr", H=H_stack, sigma2=1.0,
        name="detector"),
], taps=["tx"], name="OSIC")
curves["OSIC"] = monte_carlo(
    osic, ("noise.sigma2", "detector.sigma2"), both, {"ser": compute_ser},
    stimulus, reference="tx", seed=3)["ser"]

# --- maximum likelihood ----------------------------------------------
ml = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_stack, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    MaximumLikelihoodDetector(constellation, H=H_stack, name="detector"),
], taps=["tx"], name="ML")
curves["ML"] = monte_carlo(
    ml, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    stimulus, reference="tx", seed=4)["ser"]

# --- sphere decoder --------------------------------------------------
sd = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_stack, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    SphereDecoder(constellation, H=H_stack, name="detector"),
], taps=["tx"], name="SD")
curves["SD"] = monte_carlo(
    sd, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    stimulus, reference="tx", seed=5)["ser"]

# --- results: table and figure ---------------------------------------
print()
print_data({"x": snr_dB_list, "curves": curves},
           xlabel="snr_dB", ylabel="SER")

ax = plot_error_rate(snr_dB_list, curves, ylabel="SER",
                     title=f"{N_r}x{N_t} MIMO, {M}-PSK, "
                           f"{K} channel draws per point")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig3.png")

# --- what the pruning is worth, in seconds ---------------------------
# 16-QAM on four streams: 65 536 candidates per vector, which is where
# the exhaustive search and the sphere decoder stop costing the same.
print("\n16-QAM, 4x4: same decision, what it costs")
big_constellation = Constellation("QAM", 16)
big_H = rayleigh_channel(4, 4, seed=2)

ml_16 = Sequential([
    SymbolGenerator(big_constellation.order, name="tx"),
    SymbolMapper(big_constellation),
    FlatMIMOChannel(big_H, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    MaximumLikelihoodDetector(big_constellation, H=big_H,
                              name="detector"),
], taps=["tx"], name="4x4 MIMO, ML")

sd_16 = Sequential([
    SymbolGenerator(big_constellation.order, name="tx"),
    SymbolMapper(big_constellation),
    FlatMIMOChannel(big_H, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    SphereDecoder(big_constellation, H=big_H, name="detector"),
], taps=["tx"], name="4x4 MIMO, SD")

# --- metrics, pre-allocated ------------------------------------------
elapsed = {}
for name in ("ML", "SD"):
    elapsed[name] = np.zeros(len(snr_dB_list))

# --- simulation loop -------------------------------------------------
# A wall time is measured per point, so this one stays a loop.
for name, big_chain in (("ML", ml_16), ("SD", sd_16)):
    for index, snr_dB in enumerate(snr_dB_list):
        big_chain.seed(4)
        big_chain.set_params(noise__sigma2=4 * 10 ** (-snr_dB / 10))
        start = time.perf_counter()
        detected = big_chain((4, 400))
        elapsed[name][index] = (time.perf_counter() - start) * 1e3
        errors = compute_ser(big_chain.tap("tx"), detected)
        nodes = (f"{big_chain['detector'].nodes_:7.1f} nodes"
                 if name == "SD" else f"{16 ** 4:7d} nodes")
        print(f"  {name:2s} {snr_dB:2d} dB {elapsed[name][index]:8.1f} ms  "
              f"{nodes}   SER {errors:.4f}")

# --- results: figure -------------------------------------------------
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
