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
from comnumpy.core.utils import get_alphabet
from comnumpy.ofdm.metrics import compute_papr, compute_papr_ccdf_theo
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor

img_dir = "../../docs/tutorials/img/"

N_sc = 1024
M = 4
os = 4                                  # oversampling, to see the peaks
alphabet = get_alphabet("PSK", M)


def get_transmitter(n_sub, oversampling=os, name="ofdm"):
    """The OFDM transmitter, up to the IFFT.

    Oversampling is zero padding: the data occupy ``n_sub`` of the
    ``oversampling * n_sub`` bins, so the IFFT interpolates between the
    Nyquist samples and the peaks stop hiding between them.
    """
    carrier_type = np.zeros(oversampling * n_sub)
    carrier_type[:n_sub] = 1
    return Sequential([
        SymbolGenerator(M, name="tx"),
        SymbolMapper(alphabet),
        Serial2Parallel(n_sub, name="s2p"),
        CarrierAllocator(carrier_type=carrier_type, name="carrier_allocator"),
        IFFTProcessor(),
    ], name=name)


# --- what the transmitter puts out -----------------------------------
chain = get_transmitter(N_sc)
chain.seed(0)
blocks = chain(4 * N_sc)                # four OFDM symbols
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
long_run = np.ravel(chain(200 * N_sc))
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
print("PAPR of the four symbols above: "
      + " ".join(f"{value:.2f}" for value in papr_dB) + " dB")
print(f"their average                 : "
      f"{compute_papr(blocks, unit='dB', axis=-1, reduction='mean'):.2f} dB")

# --- the distribution -------------------------------------------------
# The CCDF is estimated over many symbols, in batches, because 20 000
# OFDM symbols at 4096 samples do not have to exist at the same time.
threshold_dB = np.arange(4, 14, 0.1)
n_batches, batch = 20, 1000
measured, reference = {}, {}
for n_sub in (256, 1024):
    transmitter = get_transmitter(n_sub)
    transmitter.seed(1)
    values = np.concatenate([
        compute_papr(transmitter(batch * n_sub), unit="dB", axis=-1)
        for _ in range(n_batches)])
    sorted_dB, ccdf = compute_ccdf(values)
    measured[f"$N_{{sc}}$ = {n_sub}"] = (sorted_dB, ccdf)
    reference[f"$N_{{sc}}$ = {n_sub}"] = compute_papr_ccdf_theo(
        threshold_dB, n_sub, oversampling=os, unit="dB")

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
