"""Soft-decision Viterbi BER over AWGN vs the union bound of the code's own spectrum.

The distance spectrum is enumerated from the trellis
(:func:`comnumpy.fec.analysis.distance_spectrum`), never copied from a
table; the enumerator returning ``d_free = 10`` for the NASA (133, 171)
K=7 code and ``d_free = 5`` for (5, 7) is the cross-check against the
literature (Proakis & Salehi, section 8.2). The bound it feeds,

    P_b <= (1/k) sum_d beta_d Q(sqrt(2 d R Eb/N0)),

is then compared to a Monte-Carlo simulation of the soft Viterbi decoder
on the K=3 (5, 7) code -- short enough that its BER stays measurable
where the bound is informative.

**What is and is not asserted.** The union bound is a *truncated* sum,
and the full series only converges above the code's cutoff rate. Below
about 3 dB the truncated sum keeps growing with ``d_max`` and exceeds 1:
asserting "simulation < bound" there would be a tautology (the BER of
anything is below 1), so the script instead measures the truncation
sensitivity and shows it is large. From 4 dB up, the sum is converged
(two truncations agree to better than 1 %) and the comparison has
content: the bound must majorize, and must tighten as Eb/N0 grows.

Runtime is about a minute: the 5 dB point needs ~24 million bits before
its BER estimate is worth comparing to a bound that sits only ~20 %
above it.
"""
import pathlib

import numpy as np

from comnumpy.fec import ConvolutionalEncoder, ViterbiDecoder
from comnumpy.fec.analysis import distance_spectrum, union_bound_ber

FIG_DIR = pathlib.Path(__file__).parent / "figures"

G_SIM = (0o5, 0o7)            # K=3, rate 1/2: BER measurable up to 5 dB
G_NASA = (0o133, 0o171)       # K=7, rate 1/2: the d_free = 10 cross-check
CODE_RATE = 0.5

EBN0_DB_RANGE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
FRAME_BITS = 2000
# more frames at high SNR: the bound is tight there, so the estimate must
# be sharp (same reasoning as validation/mimo_zf_ml_ber.py)
N_FRAMES_PER_POINT = [250, 250, 1000, 4000, 12000]
FRAMES_PER_CHUNK = 2000       # bounds peak memory, not the statistics

D_MAX = 26                    # converged for Eb/N0 >= 4 dB (checked below)
D_MAX_SHORT = 16              # deliberately too short, to expose truncation
INFORMATIVE_FROM_DB = 4.0     # where the truncated sum is a bound worth the name


def simulate_ber(ebn0_dB, n_frames, seed):
    """Soft-decision Viterbi BER, real BPSK over AWGN, terminated frames."""
    encoder = ConvolutionalEncoder(G_SIM)
    decoder = ViterbiDecoder(G_SIM, soft=True)
    rng = np.random.default_rng(seed)
    # real BPSK, Es = 1: Eb = Es / R, and the real noise variance is N0/2
    sigma2 = 1.0 / (2 * CODE_RATE * 10 ** (ebn0_dB / 10))

    errors = 0
    total = 0
    remaining = n_frames
    while remaining > 0:
        batch = min(remaining, FRAMES_PER_CHUNK)
        bits = rng.integers(0, 2, (batch, FRAME_BITS))
        tx = 1.0 - 2.0 * encoder(bits)              # bit 0 -> +1, bit 1 -> -1
        rx = tx + rng.normal(scale=np.sqrt(sigma2), size=tx.shape)
        llr = 2 * rx / sigma2                       # log P(0)/P(1)
        errors += int(np.count_nonzero(decoder(llr) != bits))
        total += bits.size
        remaining -= batch
    return errors / total, errors


def main():
    # --- 1. spectra derived from the trellis, checked against the literature
    nasa = distance_spectrum(G_NASA, n_terms=5)
    assert nasa.d_free == 10, f"(133,171) K=7: d_free={nasa.d_free}, expected 10"
    print(f"(133,171) K=7  d_free={nasa.d_free}  "
          f"a_d={nasa.a_d[:9].tolist()}  beta_d={nasa.beta_d[:9].tolist()}")

    spectrum = distance_spectrum(G_SIM, d_max=D_MAX)
    assert spectrum.d_free == 5, f"(5,7) K=3: d_free={spectrum.d_free}, expected 5"
    print(f"(5,7)     K=3  d_free={spectrum.d_free}  "
          f"a_d={spectrum.a_d[:6].tolist()}  beta_d={spectrum.beta_d[:6].tolist()}")

    # --- 2. is the truncated sum a bound at all? measure, do not assume
    bound = union_bound_ber(spectrum, EBN0_DB_RANGE)
    bound_short = union_bound_ber(distance_spectrum(G_SIM, d_max=D_MAX_SHORT),
                                  EBN0_DB_RANGE)
    truncation = np.abs(bound - bound_short) / bound
    informative = EBN0_DB_RANGE >= INFORMATIVE_FROM_DB
    assert np.all(truncation[informative] < 0.01), \
        f"truncation still moves the bound above {INFORMATIVE_FROM_DB} dB: {truncation}"
    assert np.all(truncation[EBN0_DB_RANGE <= 2.0] > 0.2), \
        "the low-SNR sum was expected to depend heavily on d_max"
    assert bound[0] > 1.0, \
        "at 1 dB the truncated union bound was expected to exceed 1 (vacuous)"

    # --- 3. Monte-Carlo BER of the soft Viterbi decoder
    ber = np.empty_like(EBN0_DB_RANGE)
    points = zip(EBN0_DB_RANGE, N_FRAMES_PER_POINT, strict=True)
    for i, (ebn0_dB, n_frames) in enumerate(points):
        ber[i], count = simulate_ber(ebn0_dB, n_frames, seed=1000 + i)
        print(f"Eb/N0 = {ebn0_dB:.0f} dB  bits = {n_frames * FRAME_BITS:>9,}  "
              f"errors = {count:>6d}  BER = {ber[i]:.3e}  "
              f"bound = {bound[i]:.3e}  bound/BER = {bound[i] / ber[i]:6.2f}")

    # --- 4. the two things the bound actually claims
    ratio = bound / ber
    assert np.all(ratio[informative] > 1.0), \
        f"the union bound must majorize above {INFORMATIVE_FROM_DB} dB: {ratio}"
    tightening = ratio[EBN0_DB_RANGE >= 3.0]
    assert np.all(np.diff(tightening) < 0), \
        f"the bound must tighten as Eb/N0 grows, got ratios {tightening}"

    # --- 5. figure
    import matplotlib.pyplot as plt
    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax.semilogy(EBN0_DB_RANGE, bound, "k-", label=f"union bound ($d\\leq{D_MAX}$)")
    ax.semilogy(EBN0_DB_RANGE, bound_short, "k:",
                label=f"union bound ($d\\leq{D_MAX_SHORT}$)")
    ax.semilogy(EBN0_DB_RANGE, ber, "o-", fillstyle="none",
                label="simulated soft Viterbi")
    ax.axhline(1.0, color="0.6", lw=1)
    ax.axvspan(EBN0_DB_RANGE[0], INFORMATIVE_FROM_DB, color="0.92", zorder=0)
    ax.text(INFORMATIVE_FROM_DB - 0.1, 0.25, "truncated sum not converged\n"
            "(and above 1: no information)",
            fontsize=8, color="0.35", ha="right", va="top")
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("BER")
    ax.set_title(f"(5,7) K=3 rate-1/2, $d_{{free}}={spectrum.d_free}$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both")

    ax_ratio.plot(EBN0_DB_RANGE, ratio, "s-", fillstyle="none")
    ax_ratio.axhline(1.0, color="k", lw=1)
    ax_ratio.axvspan(EBN0_DB_RANGE[0], INFORMATIVE_FROM_DB, color="0.92", zorder=0)
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel(r"$E_b/N_0$ [dB]")
    ax_ratio.set_ylabel("bound / simulation")
    ax_ratio.set_title("the union bound tightens with SNR")
    ax_ratio.grid(True, which="both")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fec_union_bound.png", dpi=150)

    print(f"PASS union bound: majorizes from {INFORMATIVE_FROM_DB:.0f} dB "
          f"(ratio {ratio[informative][0]:.2f} down to {ratio[-1]:.2f}); "
          f"d_free = {spectrum.d_free} for (5,7) and {nasa.d_free} for (133,171), "
          f"both enumerated from the trellis.")


if __name__ == "__main__":
    main()
