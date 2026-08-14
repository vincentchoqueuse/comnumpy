"""Why probabilistic shaping, from the capacity down to a matcher.

Run from this directory: it writes the tutorial's figures and diagrams
into ../../docs/tutorials/.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from comnumpy.core import Sequential
from comnumpy.core.capacity import (awgn_capacity, bicm_capacity,
                                    constellation_capacity)
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.shaping import (AmplitudeMapper, ConstantCompositionMatcher,
                                   DistributionMatcher, distribution_entropy,
                                   maxwell_boltzmann)
from comnumpy.core.utils import Constellation
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

snr_dB = np.arange(0, 27, 2)
snr = 10 ** (snr_dB / 10)

# --- 1. the limit ----------------------------------------------------
# Under a power constraint the Gaussian input maximizes the mutual
# information, and the value it reaches is the capacity.
capacity = awgn_capacity(snr)
print(f"AWGN capacity at 20 dB: {awgn_capacity(100.0):.2f} bit/symbol")

# --- 2. what a uniform QAM reaches -----------------------------------
orders = (4, 16, 64, 256)
uniform = {}
for order in orders:
    uniform[order] = Constellation("QAM", order).metrics(
        snr_dB, per="symbol", metrics=("mi",))["mi"]

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(snr_dB, capacity, "k", lw=2, label="AWGN capacity")
for order in orders:
    ax.plot(snr_dB, uniform[order], label=f"{order}-QAM, uniform")
    ax.axhline(np.log2(order), color="0.85", lw=0.8, zorder=0)
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("bits per symbol")
ax.set_title("a finite constellation saturates, the capacity does not")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig1.png")

header = "\nSNR   capacity"
for order in orders:
    header += f"  {order:>3d}-QAM"
print(header)
for index in range(0, len(snr_dB), 4):
    line = f"{snr_dB[index]:3d} dB {capacity[index]:8.2f}"
    for order in orders:
        line += f" {uniform[order][index]:9.2f}"
    print(line)

# --- 3. shaping the 64-QAM -------------------------------------------
# The Maxwell-Boltzmann law is the maximum-entropy law at a given energy.
# Shaping lowers the average energy, so the constellation is rescaled
# back to unit power: the comparison below is at equal transmit power,
# which is the only comparison that means anything.
qam64 = np.asarray(Constellation("QAM", 64))


def shaped_rate(lam, value):
    """Mutual information of a Maxwell-Boltzmann 64-QAM at one SNR."""
    law = maxwell_boltzmann(qam64, lam=lam)
    energy = float(law @ np.abs(qam64) ** 2)
    return float(constellation_capacity(qam64 / np.sqrt(energy), value, px=law))


# lambda is unbounded above -- at low SNR the best law collapses onto the
# innermost points, i.e. onto a smaller constellation -- so the entropy of
# the winning law is reported beside it, which is the readable quantity.
best_lam = np.zeros(snr.size)
best_entropy = np.zeros(snr.size)
shaped = np.zeros(snr.size)
for index, value in enumerate(snr):
    search = minimize_scalar(lambda lam, v=value: -shaped_rate(lam, v),
                             bounds=(0.0, 12.0), method="bounded",
                             options={"xatol": 1e-4})
    best_lam[index] = search.x
    best_entropy[index] = distribution_entropy(
        maxwell_boltzmann(qam64, lam=search.x))
    shaped[index] = shaped_rate(search.x, value)

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(snr_dB, capacity, "k", lw=2, label="AWGN capacity")
ax.plot(snr_dB, uniform[64], "o-", fillstyle="none", label="64-QAM, uniform")
ax.plot(snr_dB, shaped, "s-", fillstyle="none", label="64-QAM, shaped")
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("bits per symbol")
ax.set_title("Maxwell-Boltzmann, with the best $\\lambda$ at each SNR")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig2.png")

print("\nSNR    lambda    H(P_X)   uniform    shaped   capacity   gap closed")
for index in range(0, len(snr_dB), 2):
    gap = capacity[index] - uniform[64][index]
    closed = (shaped[index] - uniform[64][index]) / gap if gap > 1e-6 else 0.0
    print(f"{snr_dB[index]:3d} dB {best_lam[index]:8.3f} {best_entropy[index]:8.3f} "
          f"{uniform[64][index]:9.3f} {shaped[index]:9.3f} "
          f"{capacity[index]:10.3f} {100 * closed:9.0f} %")

# What the shaping is worth, read as the SNR the two need for one rate.
target = 4.0
snr_uniform = np.interp(target, uniform[64], snr_dB)
snr_shaped = np.interp(target, shaped, snr_dB)
print(f"\nto carry {target:.0f} bit/symbol: uniform {snr_uniform:.2f} dB, "
      f"shaped {snr_shaped:.2f} dB, saving {snr_uniform - snr_shaped:.2f} dB")

# The law the matcher will have to produce, at the SNR the chain below
# operates at. Probabilistic amplitude shaping shapes the *amplitudes*
# and leaves the signs uniform, so the object the matcher works on is the
# positive half of the PAM constellation -- four magnitudes for an 8-PAM,
# the sign being the bit a systematic code supplies.
middle = int(np.argmin(np.abs(snr_dB - 18)))
lam = float(best_lam[middle])
law = maxwell_boltzmann(qam64, lam=lam)
pam8 = np.sort(np.real(np.asarray(Constellation("PAM", 8))))
amplitudes = np.unique(np.abs(pam8))
amplitude_law = maxwell_boltzmann(amplitudes, lam=lam)
print(f"\nat {snr_dB[middle]} dB: lambda = {lam:.3f}, "
      f"H(amplitudes) = {distribution_entropy(amplitude_law):.3f} bits")

# --- 4. generating that law from uniform bits ------------------------
# The law is one thing, producing it from an equiprobable bit stream is
# another. A constant-composition matcher fixes the number of times each
# symbol appears in a block and enumerates the arrangements.
print("\n   n   composition                            k/n     H(X)  rate loss")
for n in (16, 64, 256, 1024):
    matcher = ConstantCompositionMatcher(amplitudes, distribution=amplitude_law,
                                         length=n)
    entropy = distribution_entropy(amplitude_law)
    counts = []
    for count in matcher.composition:
        counts.append(int(count))
    print(f"{n:4d}   {str(tuple(counts)):36s} "
          f"{matcher.rate:6.3f} {entropy:8.3f} {matcher.rate_loss:10.3f}")

# --- 5. where it goes in a chain -------------------------------------
# Probabilistic amplitude shaping: the matcher shapes the amplitudes, the
# signs stay uniform, and a systematic code's parity bits become signs.
matcher = ConstantCompositionMatcher(amplitudes, distribution=amplitude_law,
                                     length=256)
chain = Sequential([
    SymbolGenerator(2, name="bits"),
    DistributionMatcher(matcher, name="dm"),
    AmplitudeMapper(amplitudes, name="signs"),
    ], name="probabilistic amplitude shaping")
chain.seed(42)

# Run it and count. The signs are equiprobable, so the law carried by the
# signed 8-PAM is P(+/- a_i) = P_A(a_i) / 2, which is Maxwell-Boltzmann
# on the full constellation at the same lambda -- the theoretical curve
# the measured frequencies are compared against.
symbols = chain(200 * matcher.n_bits)
measured = np.zeros(pam8.size)
for index, level in enumerate(pam8):
    measured[index] = np.mean(np.isclose(symbols, level))
signed_law = maxwell_boltzmann(pam8, lam=lam)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.bar(pam8, measured, width=0.22, color="C0", alpha=0.7,
       label=f"measured, {symbols.size} symbols")
ax.plot(pam8, signed_law, "C1s--", label="Maxwell-Boltzmann, "
        f"$\\lambda$ = {lam:.2f}")
ax.axhline(1 / pam8.size, color="0.5", lw=1, ls=":", label="uniform")
ax.set_xticks(pam8)
labels = []
for value in pam8:
    labels.append(f"{value:.2f}")
ax.set_xticklabels(labels)
ax.set_xlabel("8-PAM symbol")
ax.set_ylabel("probability")
ax.set_title("what comes out of the matcher, against the law it targets")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig3.png")

print(f"\nlargest deviation from the target law: "
      f"{float(np.max(np.abs(measured - signed_law))):.4f}")

# MI is what the symbol-wise channel offers; GMI is what a bit-interleaved
# receiver with a soft demapper and a binary code can actually use.
scaled = qam64 / np.sqrt(float(law @ np.abs(qam64) ** 2))
print("\nSNR    MI uniform  GMI uniform   MI shaped  GMI shaped")
for value, label in zip(snr[::4], snr_dB[::4], strict=True):
    print(f"{label:3d} dB "
          f"{float(constellation_capacity(qam64, value)):11.3f} "
          f"{float(bicm_capacity(qam64, value)):12.3f} "
          f"{float(constellation_capacity(scaled, value, px=law)):11.3f} "
          f"{float(bicm_capacity(scaled, value, px=law)):11.3f}")
