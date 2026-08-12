r"""CCDF of the OFDM PAPR against the Rayleigh/exponential reference.

An OFDM time sample is the sum of :math:`N` independently modulated
subcarriers, so the central limit theorem makes it circularly-symmetric
Gaussian, its instantaneous power exponential, and -- if the :math:`N`
samples of a symbol are taken as independent -- the peak-to-average
power ratio follows

.. math::

    \Pr\left\{\mathrm{PAPR} > \gamma\right\}
        = 1 - \left(1 - e^{-\gamma}\right)^{N}

(van Nee & Prasad, *OFDM for Wireless Multimedia Communications*, ch. 6).

The formula rests on two approximations that this script measures rather
than hides:

1. **Independence of the samples.** It only holds for the :math:`N`
   Nyquist-rate samples; the continuous-time signal peaks *between* them,
   so a Nyquist-sampled measurement under-estimates the true PAPR. The
   script measures both the Nyquist-rate PAPR and the 4x oversampled one
   (the usual proxy for continuous time) and reports which one the
   formula describes.
2. **Asymptotics in N.** The Gaussian approximation is exact only as
   :math:`N \to \infty`. The script sweeps N = 64, 256, 1024 and reports
   how the deviation shrinks.

Both deviations are quantified as a *horizontal* gap in dB at a fixed
CCDF level -- the way a PAPR curve is read in practice -- and as the
effective exponent :math:`\alpha N` that best fits the measurement.
"""
import pathlib

import numpy as np

from comnumpy import Sequential, SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ccdf
from comnumpy.core.processors import Serial2Parallel
from comnumpy.core.utils import Constellation
from comnumpy.ofdm.metrics import compute_papr
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor

FIG_DIR = pathlib.Path(__file__).parent / "figures"

M = 4                       # QPSK: every symbol has |s| = 1, so the block
                            # mean power is exactly the ensemble mean power
                            # (Parseval), and the PAPR needs no ensemble
                            # normalization.
N_LIST = (64, 256, 1024)
OS = 4                      # oversampling factor, the common proxy for the
                            # continuous-time PAPR
N_SYMBOLS = 50_000          # 50 OFDM symbols above the 1e-3 CCDF level
SEED = 1
LEVELS = (1e-1, 1e-2, 1e-3)
COLORS = ("#0072B2", "#D55E00", "#009E73")   # Okabe-Ito, as in ofdm.allocation


def papr_dB(N, os, n_symbols, seed, chunk=500):
    """PAPR of ``n_symbols`` OFDM symbols, in dB, one value per symbol.

    ``os = 1`` samples at the Nyquist rate; ``os > 1`` zero-pads the
    spectrum, which interpolates the same symbol on a denser time grid.
    """
    carrier_type = np.zeros(os * N, dtype=int)
    carrier_type[:N] = 1        # a frequency shift leaves |x[n]| unchanged
    chain = Sequential([
        SymbolGenerator(M),
        SymbolMapper(Constellation("PSK", M)),
        Serial2Parallel(N),
        CarrierAllocator(carrier_type=carrier_type),
        IFFTProcessor(),
    ])
    chain.seed(seed)
    out, done = [], 0
    while done < n_symbols:    # chunked: N=1024 at os=4 is 2 GB in one go
        count = min(chunk, n_symbols - done)
        out.append(compute_papr(chain(count * N), unit="dB", axis=-1))
        done += count
    return np.concatenate(out)


def theory_ccdf(gamma_dB, N):
    """1 - (1 - exp(-gamma))^N, the reference of the module docstring."""
    return 1 - (1 - np.exp(-10 ** (np.asarray(gamma_dB) / 10))) ** N


def theory_quantile_dB(N, level):
    """PAPR threshold at which the reference CCDF equals ``level``."""
    return 10 * np.log10(-np.log(1 - (1 - level) ** (1.0 / N)))


def measured_quantile_dB(papr, level):
    """Empirical PAPR threshold at which the measured CCDF equals ``level``."""
    sorted_papr, ccdf = compute_ccdf(papr)
    return sorted_papr[np.searchsorted(-ccdf, -level)]   # ccdf is decreasing


def effective_exponent(papr, N, lo=1e-3, hi=3e-1):
    """Exponent alpha*N that reproduces the measurement, as alpha.

    Inverting 1 - (1 - e^-gamma)^(alpha N) = ccdf at every measured point
    of the CCDF window and taking the median: alpha = 1 says the reference
    formula holds, alpha > 1 says the signal peaks more often than N
    independent samples would.
    """
    sorted_papr, ccdf = compute_ccdf(papr)
    window = (ccdf >= lo) & (ccdf <= hi)
    gamma = 10 ** (sorted_papr[window] / 10)
    alpha = np.log(1 - ccdf[window]) / np.log(1 - np.exp(-gamma)) / N
    return float(np.median(alpha))


def main():
    papr = {(N, os): papr_dB(N, os, N_SYMBOLS, SEED)
            for N in N_LIST for os in (1, OS)}
    gap = {key: {lvl: measured_quantile_dB(p, lvl) - theory_quantile_dB(key[0], lvl)
                 for lvl in LEVELS}
           for key, p in papr.items()}
    alpha = {key: effective_exponent(p, key[0]) for key, p in papr.items()}

    for N in N_LIST:
        for os in (1, OS):
            gaps = " ".join(f"{lvl:.0e}:{gap[N, os][lvl]:+.3f}" for lvl in LEVELS)
            print(f"  N={N:5d} os={os}  alpha={alpha[N, os]:.2f}  gap[dB] {gaps}")

    # --- assertions -------------------------------------------------------
    # Thresholds come from the run above repeated over seeds 1, 2, 3 at
    # N_SYMBOLS = 50,000; the seed-to-seed spread is quoted next to each
    # measurement, so no threshold sits closer than 3 sigma to the data.

    # (1) At the Nyquist rate the reference formula holds. Worst measured
    #     |gap| = 0.126 dB (N=64, CCDF=1e-2, spread 0.002); the 1e-3 level
    #     is excluded because only 50 symbols land above it (spread up to
    #     0.06 dB).
    nyquist = [abs(gap[N, 1][lvl]) for N in N_LIST for lvl in (1e-1, 1e-2)]
    assert max(nyquist) < 0.20, f"Nyquist-rate PAPR off the reference: {nyquist}"

    # (2) The formula is asymptotic in N: the gap at CCDF=1e-2 collapses
    #     from 0.126 dB (N=64, spread 0.002) to 0.003 dB (N=1024, spread
    #     0.015). Thresholds 0.08 / 0.06 dB sit between the two, more than
    #     3 sigma away from each.
    assert abs(gap[64, 1][1e-2]) > 0.08, (
        f"N=64 should still show a finite-N gap, got {gap[64, 1][1e-2]:+.3f} dB")
    assert abs(gap[1024, 1][1e-2]) < 0.06, (
        f"N=1024 should have converged, got {gap[1024, 1][1e-2]:+.3f} dB")

    # (3) Oversampling reveals the peaks the Nyquist grid misses, so the
    #     true PAPR is *above* the formula. Measured gap at CCDF=1e-2:
    #     +0.362 / +0.400 / +0.401 dB for N = 64 / 256 / 1024 (spread
    #     <= 0.03 dB). The [0.20, 0.80] window is >= 5 sigma from the data.
    over = [gap[N, OS][1e-2] for N in N_LIST]
    assert all(0.20 < g < 0.80 for g in over), (
        f"{OS}x oversampled PAPR should exceed the Nyquist reference: {over}")

    # (4) Effective exponent. Nyquist: 0.95 / 0.98 / 0.99 (spread <= 0.01),
    #     i.e. the exponent really is N. Oversampled: 2.23 / 2.49 / 2.73,
    #     climbing towards the alpha = 2.8 quoted by van Nee & Prasad for
    #     4x oversampling; the steps (0.26, 0.24) are >= 6 sigma.
    assert all(0.90 < alpha[N, 1] < 1.05 for N in N_LIST), \
        f"Nyquist exponent is not N: {[alpha[N, 1] for N in N_LIST]}"
    assert all(2.0 < alpha[N, OS] < 3.0 for N in N_LIST), \
        f"oversampled exponent outside the published range: {[alpha[N, OS] for N in N_LIST]}"
    assert alpha[64, OS] < alpha[256, OS] < alpha[1024, OS], \
        f"oversampled exponent should grow with N: {[alpha[N, OS] for N in N_LIST]}"

    # --- figure -----------------------------------------------------------
    import matplotlib.pyplot as plt  # local import (D36)
    _, ax = plt.subplots(figsize=(7, 5))
    grid_dB = np.linspace(4, 14, 400)
    # markers spaced logarithmically in CCDF, so the decades that matter
    # (1e-2, 1e-3) get as many points as the head of the distribution
    marks = np.geomspace(0.9, 2e-4, 22)
    for N, color in zip(N_LIST, COLORS, strict=True):
        ax.semilogy(grid_dB, theory_ccdf(grid_dB, N), "-", color=color,
                    lw=1.2, label=f"theory, N={N}")
    for os, marker in ((1, "o"), (OS, "s")):
        for N, color in zip(N_LIST, COLORS, strict=True):
            sorted_papr, ccdf = compute_ccdf(papr[N, os])
            idx = np.clip((len(ccdf) * (1 - marks)).astype(int), 0, len(ccdf) - 1)
            ax.semilogy(sorted_papr[idx], ccdf[idx], marker, color=color,
                        fillstyle="none", ms=5,
                        label=f"{'Nyquist' if os == 1 else f'{OS}x oversampled'}, N={N}")
    ax.set_xlim(5, 13)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"PAPR threshold $\gamma$ [dB]")
    ax.set_ylabel(r"$\Pr\{\mathrm{PAPR} > \gamma\}$")
    ax.set_title(f"OFDM PAPR CCDF, QPSK, {N_SYMBOLS:,} symbols per curve")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "ofdm_papr_ccdf.png", dpi=150, bbox_inches="tight")

    print(f"PASS PAPR CCDF (Nyquist): max |gap| {max(nyquist):.3f} dB over "
          f"N={N_LIST} at CCDF 1e-1 and 1e-2, effective exponent alpha = "
          f"{', '.join(f'{alpha[N, 1]:.2f}' for N in N_LIST)}")
    print(f"PASS PAPR CCDF (asymptotics): gap at CCDF=1e-2 shrinks "
          f"{gap[64, 1][1e-2]:+.3f} -> {gap[256, 1][1e-2]:+.3f} -> "
          f"{gap[1024, 1][1e-2]:+.3f} dB for N = 64 -> 256 -> 1024")
    print(f"PASS PAPR CCDF ({OS}x oversampled): the formula under-estimates the "
          f"PAPR by {min(over):.3f}-{max(over):.3f} dB at CCDF=1e-2; effective "
          f"exponent alpha = {', '.join(f'{alpha[N, OS]:.2f}' for N in N_LIST)} "
          f"(published value 2.8)")


if __name__ == "__main__":
    main()
