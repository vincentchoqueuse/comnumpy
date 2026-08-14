"""What an OFDM waveform looks like in amplitude, and how often it peaks.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ccdf
from comnumpy.core.processors import Serial2Parallel
from comnumpy.core.utils import Constellation
from comnumpy.ofdm.metrics import compute_papr, compute_papr_ccdf_theo
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

N_sc = 1024
os = 4                                  # oversampling, to see the peaks
constellation = Constellation("PSK", 4)


# --- what the transmitter puts out -----------------------------------
# One transmitter, up to the IFFT. Oversampling is zero padding: the
# data occupy N_sc of the os * N_sc bins, so the IFFT interpolates
# between the Nyquist samples and the peaks stop hiding between them.
carrier_1024 = np.zeros(os * N_sc)
carrier_1024[:N_sc] = 1
ofdm_1024 = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation),
    Serial2Parallel(N_sc, name="s2p"),
    CarrierAllocator(carrier_type=carrier_1024, name="carrier_allocator"),
    IFFTProcessor(),
], name="ofdm 1024")

ofdm_1024.seed(0)
blocks = ofdm_1024(4 * N_sc)            # four OFDM symbols
signal = np.ravel(blocks)               # C-order flatten of the (T, F) blocks
power = np.abs(signal) ** 2 / np.mean(np.abs(signal) ** 2)

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(power, lw=0.6)
ax.axhline(1.0, color="k", lw=1.0, label="average power")
ax.set_xlabel("$n$ [sample]")
ax.set_ylabel(r"$|x[n]|^2 / \mathbb{E}[|x[n]|^2]$")
ax.set_title(f"four OFDM symbols, {N_sc} subcarriers, oversampling {os}")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig1.png")

# --- why it looks like that ------------------------------------------
# Each sample is a sum of N_sc independent terms, so the central limit
# theorem applies: the samples are complex Gaussian, and their power is
# exponential. Nothing about OFDM is needed beyond that.
long_run = np.ravel(ofdm_1024(200 * N_sc))
normalized = long_run / np.std(long_run)
gain = np.abs(long_run) ** 2 / np.mean(np.abs(long_run) ** 2)

fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
grid = np.linspace(-4, 4, 200)
ax_left.hist(np.real(normalized), bins=100, range=(-4, 4), density=True,
             alpha=0.5, label=r"$\Re\{x[n]\}$")
ax_left.plot(grid, np.exp(-grid ** 2) / np.sqrt(np.pi), lw=2,
             label=r"$\mathcal{N}(0, 1/2)$")
ax_left.set_xlabel("normalized amplitude")
ax_left.set_ylabel("probability density")
ax_left.set_title("the samples are Gaussian")

grid = np.linspace(0, 8, 200)
ax_right.hist(gain, bins=100, range=(0, 8), density=True, alpha=0.5,
              label=r"$|x[n]|^2$")
ax_right.plot(grid, np.exp(-grid), lw=2, label=r"$e^{-\gamma}$")
ax_right.set_yscale("log")
ax_right.set_ylim(1e-5, 2)
ax_right.set_xlabel(r"normalized power $\gamma$")
ax_right.set_title("so their power is exponential")
for ax in (ax_left, ax_right):
    ax.legend()
    ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig2.png")

# --- the metric -------------------------------------------------------
papr_dB = compute_papr(blocks, unit="dB", axis=-1)
line = "PAPR of the four symbols above: "
for value in papr_dB:
    line += f"{value:.2f} "
print(line + "dB")
print(f"their average                 : "
      f"{compute_papr(blocks, unit='dB', axis=-1, reduction='mean'):.2f} dB")

# --- the distribution -------------------------------------------------
# Two subcarrier counts, so two chains -- the 1024 one already exists.
# The batch needs no loop: Serial2Parallel stacks the OFDM symbols on
# the leading axis, so 20 000 symbols are one call and compute_papr
# reads one value per row.
n_symbols = 20000
threshold_dB = np.arange(4, 14, 0.1)

carrier_256 = np.zeros(os * 256)
carrier_256[:256] = 1
ofdm_256 = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation),
    Serial2Parallel(256, name="s2p"),
    CarrierAllocator(carrier_type=carrier_256, name="carrier_allocator"),
    IFFTProcessor(),
], name="ofdm 256")

ofdm_256.seed(1)
papr_256 = compute_papr(ofdm_256(n_symbols * 256), unit="dB", axis=-1)
ofdm_1024.seed(1)
papr_1024 = compute_papr(ofdm_1024(n_symbols * N_sc), unit="dB", axis=-1)

measured = {"$N_{sc}$ = 256": compute_ccdf(papr_256),
            "$N_{sc}$ = 1024": compute_ccdf(papr_1024)}
reference = {
    "$N_{sc}$ = 256": compute_papr_ccdf_theo(threshold_dB, 256,
                                             oversampling=os, unit="dB"),
    "$N_{sc}$ = 1024": compute_papr_ccdf_theo(threshold_dB, 1024,
                                              oversampling=os, unit="dB")}

# Markers are placed on a logarithmic grid of the ordinate: spacing them
# evenly in index would crowd the top of the curve and leave the tail --
# the part the amplifier is sized on -- with no marker at all.
levels = np.logspace(0, -4, 25)

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, (sorted_dB, ccdf) in measured.items():
    shown = np.clip((ccdf.size * (1 - levels)).astype(int), 0, ccdf.size - 1)
    line, = ax.plot(sorted_dB[shown], ccdf[shown], "o", fillstyle="none",
                    label=f"{name}, simulated")
    ax.plot(threshold_dB, reference[name], "-", color=line.get_color(),
            label=f"{name}, theory")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlim(6, 13)
ax.set_xlabel("PAPR threshold [dB]")
ax.set_ylabel(r"$\Pr\{\mathrm{PAPR} > \gamma\}$")
ax.set_title("CCDF of the PAPR of an OFDM signal")
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_ofdm_papr_fig3.png")

for n_sub in (256, 1024):
    solved = brentq(lambda t, n=n_sub: compute_papr_ccdf_theo(
        t, n, oversampling=os, unit="dB") - 1e-3, 0, 20)
    print(f"N_sc = {n_sub:4d}: PAPR exceeded once in a thousand symbols above "
          f"{solved:.2f} dB")

