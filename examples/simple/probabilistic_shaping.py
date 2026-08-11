import numpy as np
import matplotlib.pyplot as plt

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.capacity import (bicm_capacity, constellation_capacity)
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.information import compute_gmi, compute_mi
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.shaping import (AmplitudeDemapper, AmplitudeMapper,
                                   ConstantCompositionMatcher,
                                   DistributionDematcher, DistributionMatcher,
                                   SphereShaper, blahut_arimoto,
                                   distribution_entropy, maxwell_boltzmann,
                                   shaping_gain_dB)
from comnumpy.core.utils import get_alphabet

img_dir = "../../docs/tutorials/img/"

# 16-PAM on the odd-integer grid, in the library's own label order: the
# most significant bit carries the sign and the three others the
# amplitude, which is exactly the labelling PAS is built on.
PAM16 = np.real(get_alphabet("PAM", 16)) * np.sqrt(85.0)
UNIFORM = np.full(16, 1 / 16)
AMPLITUDES = np.arange(1, 16, 2).astype(float)      # the eight |a| values
ULTIMATE_GAIN_dB = 10 * np.log10(np.pi * np.e / 6)


# ===========================================================================
#  Part 1 -- a constellation is a set of points and a law on it
# ===========================================================================

def energy_of(law):
    """Average energy the law spends on this constellation."""
    return float(np.asarray(law) @ PAM16 ** 2)


def mutual_information(law, snr_dB):
    """Exact MI, at an SNR defined on the power the law actually spends.

    ``constellation_capacity`` splits its noise over two dimensions, so a
    *real* channel at SNR ``s`` on a constellation of energy ``E`` is the
    complex convention's ``rho = s / (2E)``. Normalizing by ``E`` is not
    bookkeeping either: it is what makes the comparison fair. A shaped
    law spends less, so at equal power its constellation is *wider* --
    and that extra width is where the whole shaping gain comes from.
    """
    rho = 10 ** (np.asarray(snr_dB, dtype=float) / 10) / (2 * energy_of(law))
    return constellation_capacity(PAM16, rho, px=law)


def bitwise_rate(snr_dB):
    """The same, for a bit-wise decoder. Uniform input only."""
    rho = 10 ** (np.asarray(snr_dB, dtype=float) / 10) / (2 * energy_of(UNIFORM))
    return bicm_capacity(PAM16, rho)


snr_axis = np.linspace(0.0, 30.0, 121)
print(f"16-PAM, uniform law: H = {distribution_entropy(UNIFORM):.3f} "
      f"bit/symbol, energy {energy_of(UNIFORM):.0f}, "
      f"shaping gain {shaping_gain_dB(PAM16, UNIFORM):.3f} dB")

_, (ax_pmf, ax_rate) = plt.subplots(ncols=2, figsize=(11, 4.2),
                                    layout="constrained")
order = np.argsort(PAM16)
ax_pmf.stem(PAM16[order], UNIFORM[order], basefmt=" ")
ax_pmf.set_xlabel("16-PAM constellation point")
ax_pmf.set_ylabel("$P_X(a)$")
ax_pmf.set_ylim(0, 0.20)
ax_pmf.set_title(f"The law we never chose: $H$ = "
                 f"{distribution_entropy(UNIFORM):.0f} bit/symbol")
ax_pmf.grid(True, alpha=0.4)

noise = AWGN(snr_dB=0.0, name="noise")
study = Sequential([SymbolGenerator(16, name="tx"), SymbolMapper(PAM16), noise],
                   taps=["tx"], name="16-PAM over AWGN")


def measure(law, snr_dB_list, n_symbols=120000, seed=7):
    """MI and GMI read off samples rather than integrated."""
    study.set_params(**{"tx.distribution": law})

    def symbolwise(symbols, received):
        return compute_mi(received, symbols, PAM16,
                          snr=1 / (2 * noise.sigma2_), px=law)

    def bitwise(symbols, received):
        return compute_gmi(received, symbols, PAM16,
                           snr=1 / (2 * noise.sigma2_))

    return sweep(study, "noise.snr_dB", snr_dB_list,
                 {"mi": symbolwise, "gmi": bitwise}, n_symbols,
                 reference="tx", seed=seed)


checkpoints = np.array([4.0, 10.0, 16.0, 22.0, 28.0])
measured = measure(UNIFORM, checkpoints)
ax_rate.plot(snr_axis, mutual_information(UNIFORM, snr_axis), "-",
             label="MI, symbol-wise decoder")
ax_rate.plot(snr_axis, bitwise_rate(snr_axis), "--",
             label="GMI, bit-wise decoder")
ax_rate.plot(snr_axis, 0.5 * np.log2(1 + 10 ** (snr_axis / 10)), ":",
             color="0.4", label="Shannon, $\\frac{1}{2}\\log_2(1+\\rho)$")
ax_rate.plot(checkpoints, measured["mi"], "o", color="C0")
ax_rate.plot(checkpoints, measured["gmi"], "s", color="C1")
ax_rate.axhline(4.0, color="0.7", linewidth=1)
ax_rate.set_xlabel("SNR [dB]")
ax_rate.set_ylabel("rate [bit/symbol]")
ax_rate.set_ylim(0, 5.2)
ax_rate.set_title("What it carries: markers measured, lines integrated")
ax_rate.legend(loc="upper left", fontsize=9)
ax_rate.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig1.png")

print("\n SNR      MI       GMI    MI-GMI    measured MI   measured GMI")
for point, mi_hat, gmi_hat in zip(checkpoints, measured["mi"],
                                  measured["gmi"], strict=True):
    exact_mi = float(mutual_information(UNIFORM, point))
    exact_gmi = float(bitwise_rate(point))
    print(f"{point:5.0f} {exact_mi:8.4f} {exact_gmi:8.4f} "
          f"{exact_mi - exact_gmi:8.4f} {mi_hat:13.4f} {gmi_hat:14.4f}")


# ===========================================================================
#  Part 2 -- which law maximizes the rate, and the one we use instead
# ===========================================================================

def matched_maxwell_boltzmann(energy):
    """The Maxwell-Boltzmann law spending exactly ``energy``.

    The family is parameterized by lambda or by entropy, and neither is
    the energy; but the energy decreases monotonically in lambda, so a
    bisection on it is exact and costs nothing -- the law itself is a
    closed form.

    The refusal matters. The family spends at most what the uniform law
    spends, at lambda = 0, so a bisection asked for more would run to
    lambda = 0 and hand back the uniform law as if it had succeeded.
    """
    if energy > energy_of(UNIFORM):
        raise ValueError(
            f"no Maxwell-Boltzmann law on this constellation spends "
            f"{energy:.2f}: the family tops out at the uniform law's "
            f"{energy_of(UNIFORM):.2f}, at lambda = 0.")
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if energy_of(maxwell_boltzmann(PAM16, lam=middle)) > energy:
            low = middle
        else:
            high = middle
    return maxwell_boltzmann(PAM16, lam=0.5 * (low + high))


OPERATING_SNR_dB = 18.0
sigma2 = energy_of(UNIFORM) / 10 ** (OPERATING_SNR_dB / 10)
print(f"\nAt {OPERATING_SNR_dB:.0f} dB on the uniform law, sigma^2 = "
      f"{sigma2:.3f}\n")
print("  lambda   energy   H(P)   MI(best)   MI(Maxwell-Boltzmann)     gap")
for lam in (0.0, 0.002, 0.006, 0.02, 0.06):
    best = blahut_arimoto(PAM16, sigma2=sigma2, lam=lam)
    spent = energy_of(best)
    rho = 1 / (2 * sigma2)
    best_rate = float(constellation_capacity(PAM16, rho, px=best))
    if spent > energy_of(UNIFORM):
        # No Maxwell-Boltzmann law spends more than the uniform one, so
        # there is nothing to compare against -- see the warning below.
        print(f"  {lam:6.3f} {spent:8.2f} {distribution_entropy(best):6.3f} "
              f"{best_rate:10.4f}       (no MB law spends that much)")
        continue
    closed = matched_maxwell_boltzmann(spent)
    closed_rate = float(constellation_capacity(PAM16, rho, px=closed))
    print(f"  {lam:6.3f} {spent:8.2f} {distribution_entropy(best):6.3f} "
          f"{best_rate:10.4f} {closed_rate:20.4f} {best_rate - closed_rate:8.4f}")

SHOWN_LAMBDA = 0.006
best = blahut_arimoto(PAM16, sigma2=sigma2, lam=SHOWN_LAMBDA)
closed = matched_maxwell_boltzmann(energy_of(best))
starved = blahut_arimoto(PAM16, sigma2=energy_of(UNIFORM) / 10 ** 0.6,
                         lam=0.004)

_, (ax_pair, ax_low) = plt.subplots(ncols=2, figsize=(11, 4.2),
                                    layout="constrained")
ax_pair.stem(PAM16[order], best[order], basefmt=" ", linefmt="C0-",
             markerfmt="C0o", label="Blahut-Arimoto, the true maximizer")
ax_pair.plot(PAM16[order], closed[order], "C1^--",
             label="Maxwell-Boltzmann, same energy")
ax_pair.set_xlabel("16-PAM constellation point")
ax_pair.set_ylabel("$P_X(a)$")
ax_pair.set_title(f"At {OPERATING_SNR_dB:.0f} dB, energy "
                  f"{energy_of(best):.0f}: the same curve")
ax_pair.legend(fontsize=9)
ax_pair.grid(True, alpha=0.4)

ax_low.stem(PAM16[order], starved[order], basefmt=" ", linefmt="C3-",
            markerfmt="C3o")
ax_low.set_xlabel("16-PAM constellation point")
ax_low.set_ylabel("$P_X(a)$")
ax_low.set_title("At 6 dB: points dropped outright")
ax_low.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig2.png")

print(f"\nAt 6 dB the maximizer sets {int(np.sum(starved < 1e-9))} of the 16 "
      f"probabilities to exactly zero, and spends {energy_of(starved):.1f} "
      f"doing it -- more than the uniform {energy_of(UNIFORM):.0f}.")
print(f"No Maxwell-Boltzmann law does either. At lambda = 0.5 the outermost "
      f"point still keeps {np.min(maxwell_boltzmann(PAM16, lam=0.5)):.1e}, "
      f"and asking the family for that energy is refused outright:")
try:
    matched_maxwell_boltzmann(energy_of(starved))
except ValueError as error:
    print(f"  {error}")


# ===========================================================================
#  Part 3 -- drawing from the law, and what it buys
# ===========================================================================

TARGET = maxwell_boltzmann(PAM16, entropy=3.5)
print(f"\ntarget law: H = {distribution_entropy(TARGET):.3f} bit/symbol, "
      f"energy {energy_of(TARGET):.2f} against {energy_of(UNIFORM):.0f}, "
      f"shaping gain {shaping_gain_dB(PAM16, TARGET):.3f} dB")

_, axes = plt.subplots(ncols=3, figsize=(13, 4), layout="constrained",
                       sharey=True)
print("\n  symbols   empirical H   empirical energy   total variation")
for panel, n_draws in zip(axes, (200, 20000, 2000000), strict=True):
    drawn = SymbolGenerator(16, distribution=TARGET, seed=1)(n_draws)
    seen = np.bincount(drawn, minlength=16) / n_draws
    panel.bar(PAM16[order], seen[order], width=1.4, alpha=0.55,
              label="drawn")
    panel.plot(PAM16[order], TARGET[order], "ko-", markersize=4,
               label="$P_X$")
    panel.set_xlabel("16-PAM constellation point")
    panel.set_title(f"{n_draws} symbols")
    panel.grid(True, alpha=0.4)
    panel.legend(fontsize=9)
    print(f"{n_draws:9d} {distribution_entropy(seen / seen.sum()):13.4f} "
          f"{energy_of(seen):18.3f} {0.5 * np.sum(np.abs(seen - TARGET)):17.4f}")
axes[0].set_ylabel("frequency")
plt.savefig(f"{img_dir}/probabilistic_shaping_fig3.png")

shaped_rate = mutual_information(TARGET, snr_axis)
uniform_rate = mutual_information(UNIFORM, snr_axis)
_, (ax_gain, ax_dB) = plt.subplots(ncols=2, figsize=(11, 4.2),
                                   layout="constrained")
ax_gain.plot(snr_axis, uniform_rate, "-", label="uniform 16-PAM")
ax_gain.plot(snr_axis, shaped_rate, "-",
             label=f"shaped, $H$ = {distribution_entropy(TARGET):.2f}")
ax_gain.plot(snr_axis, 0.5 * np.log2(1 + 10 ** (snr_axis / 10)), ":",
             color="0.4", label="Shannon")
ax_gain.set_xlabel("SNR [dB]")
ax_gain.set_ylabel("mutual information [bit/symbol]")
ax_gain.set_title("Same power, more bits")
ax_gain.legend(fontsize=9)
ax_gain.grid(True, alpha=0.4)

rate_grid = np.linspace(0.5, 3.4, 60)
saved = (np.interp(rate_grid, uniform_rate, snr_axis)
         - np.interp(rate_grid, shaped_rate, snr_axis))
ax_dB.plot(rate_grid, saved, "-")
ax_dB.axhline(ULTIMATE_GAIN_dB, color="k", linestyle="--",
              label=f"{ULTIMATE_GAIN_dB:.2f} dB, the ultimate gain")
ax_dB.set_xlabel("rate [bit/symbol]")
ax_dB.set_ylabel("SNR saved [dB]")
ax_dB.set_title("The same gap, read horizontally")
ax_dB.legend(fontsize=9)
ax_dB.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig4.png")

for rate in (1.5, 2.0, 2.5, 3.0):
    plain = float(np.interp(rate, uniform_rate, snr_axis))
    shaped = float(np.interp(rate, shaped_rate, snr_axis))
    print(f"rate {rate:.1f} bit/symbol: uniform needs {plain:5.2f} dB, "
          f"shaped {shaped:5.2f} dB -- {plain - shaped:+.2f} dB saved")


# ===========================================================================
#  Part 4 -- data is uniform, so the law has to be built
# ===========================================================================

amplitude_law = 2 * TARGET[PAM16 > 0][np.argsort(PAM16[PAM16 > 0])]
print(f"\nthe eight amplitudes carry {distribution_entropy(amplitude_law):.3f} "
      f"bit each, the sign the remaining one")

# The sphere's table is O(n * E_max) and the outer amplitude of 16-PAM
# costs 225 on its own, so doubling the block roughly multiplies the
# construction by ten: n = 128 already takes seconds, n = 512 minutes.
lengths = [16, 32, 64, 128]
rates = {"CCDM (constant composition)": [], "ESS (sphere, same energy)": []}
for n in lengths:
    ccdm = ConstantCompositionMatcher(AMPLITUDES, distribution=amplitude_law,
                                      length=n)
    same_rate = SphereShaper(AMPLITUDES, length=n, n_bits=ccdm.n_bits)
    budget = int(np.dot(ccdm.composition, same_rate.energies))
    same_energy = SphereShaper(AMPLITUDES, length=n, max_energy=budget)
    rates["CCDM (constant composition)"].append(ccdm.rate)
    rates["ESS (sphere, same energy)"].append(same_energy.rate)
    print(f"n = {n:4d}   composition {ccdm.composition}   "
          f"rate {ccdm.rate:.4f} vs {same_energy.rate:.4f} bit/amplitude")

_, ax_loss = plt.subplots(figsize=(7, 4.2), layout="constrained")
for label, values in rates.items():
    ax_loss.semilogx(lengths, values, "o-", base=2, label=label)
ax_loss.axhline(distribution_entropy(amplitude_law), color="k",
                linestyle="--",
                label=f"$H(P)$ = {distribution_entropy(amplitude_law):.3f}, "
                      f"the ceiling")
ax_loss.set_xlabel("blocklength n [amplitudes]")
ax_loss.set_ylabel("rate [bit/amplitude]")
ax_loss.set_title("Rate loss is what a finite block costs")
ax_loss.legend(fontsize=9)
ax_loss.grid(True, which="both", alpha=0.4)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig5.png")

n_block = 64
shaper = ConstantCompositionMatcher(AMPLITUDES, distribution=amplitude_law,
                                    length=n_block)
link = Sequential([
    SymbolGenerator(2, name="bits"),
    DistributionMatcher(shaper, name="matcher"),
    AmplitudeMapper(AMPLITUDES, name="mapper"),
    AWGN(snr_dB=32.0, name="channel"),
    AmplitudeDemapper(AMPLITUDES),
    DistributionDematcher(shaper),
], taps=["bits", "mapper"], name="PAS transmitter")

n_blocks = 200
link.seed(12)
recovered = link(n_blocks * shaper.n_bits)
print(f"\n{shaper.n_bits} bits -> {n_block} amplitudes per block, recovered "
      f"exactly: {np.array_equal(recovered, link.tap('bits'))}")
link.summary(n_blocks * shaper.n_bits)

sent = link.tap("mapper").ravel()
emitted = np.array([np.mean(sent == point) for point in PAM16])
_, ax_emit = plt.subplots(figsize=(7, 4.2), layout="constrained")
ax_emit.bar(PAM16[order], emitted[order], width=1.4, alpha=0.55,
            label=f"emitted by the matcher, {n_blocks * n_block} symbols")
ax_emit.plot(PAM16[order], TARGET[order], "ko-", markersize=4,
             label="$P_X$, the target")
ax_emit.set_xlabel("16-PAM constellation point")
ax_emit.set_ylabel("frequency")
ax_emit.set_title("What a matcher emits, against what it was asked for")
ax_emit.legend(fontsize=9)
ax_emit.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig6.png")

link.set_params(**{"channel.snr_dB": 20.0})
try:
    link(n_blocks * shaper.n_bits)
except ValueError as error:
    print(f"\nat 20 dB -- {error}")


# ===========================================================================
#  Part 5 -- the same law, on a complex constellation
# ===========================================================================

qam256 = get_alphabet("QAM", 256)
axis = np.unique(np.real(qam256))
axis_law = maxwell_boltzmann(axis, entropy=3.5)
square_law = np.array([axis_law[np.argmin(np.abs(axis - np.real(point)))]
                       * axis_law[np.argmin(np.abs(axis - np.imag(point)))]
                       for point in qam256])
print(f"\n256-QAM as a product of two shaped 16-PAM axes: "
      f"H = {distribution_entropy(square_law):.3f} bit/symbol, "
      f"{2 * distribution_entropy(axis_law):.3f} expected")

_, ax_qam = plt.subplots(figsize=(6, 5.6), layout="constrained")
ax_qam.scatter(np.real(qam256), np.imag(qam256),
               s=square_law / square_law.max() * 260, alpha=0.75)
ax_qam.set_xlabel("in phase")
ax_qam.set_ylabel("quadrature")
ax_qam.set_title("Shaped 256-QAM: area is probability")
ax_qam.set_aspect("equal")
ax_qam.grid(True, alpha=0.3)
plt.savefig(f"{img_dir}/probabilistic_shaping_fig7.png")

wide_axis = np.linspace(0.0, 34.0, 69)
print("\n   M   best SNR saved   at rate   still short of 1.53 dB")
for order_pam in (4, 8, 16, 32, 64):
    grid = np.arange(-(order_pam - 1), order_pam, 2).astype(float)
    flat = np.full(order_pam, 1 / order_pam)
    law = maxwell_boltzmann(grid, entropy=0.75 * np.log2(order_pam))

    def rate_curve(p, points=grid):
        rho = 10 ** (wide_axis / 10) / (2 * float(p @ points ** 2))
        return constellation_capacity(points, rho, px=p)

    plain, shaped = rate_curve(flat), rate_curve(law)
    grid_rates = np.linspace(0.5, 0.95 * np.log2(order_pam), 40)
    saved_dB = (np.interp(grid_rates, plain, wide_axis)
                - np.interp(grid_rates, shaped, wide_axis))
    peak = int(np.argmax(saved_dB))
    print(f"{order_pam:4d} {saved_dB[peak]:14.3f} dB {grid_rates[peak]:9.2f} "
          f"{ULTIMATE_GAIN_dB - saved_dB[peak]:21.3f} dB")

mermaid_dir = "../../docs/tutorials/mermaid/"
for name, chain in (("shaping_pas", link), ("shaping_study", study)):
    with open(f"{mermaid_dir}/{name}.mmd", "w") as stream:
        stream.write(chain.to_mermaid())

plt.show()
