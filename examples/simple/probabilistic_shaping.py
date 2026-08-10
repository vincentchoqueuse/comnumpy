import numpy as np
import matplotlib.pyplot as plt

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.information import compute_mi
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.shaping import (AmplitudeDemapper, AmplitudeMapper,
                                   ConstantCompositionMatcher,
                                   DistributionDematcher, DistributionMatcher,
                                   SphereShaper, distribution_entropy,
                                   maxwell_boltzmann, shaping_gain_dB)

img_dir = "../../docs/examples/img/"

pam8 = np.arange(-7, 8, 2).astype(float)   # 8-PAM on the odd-integer grid
amplitudes = pam8[pam8 > 0]                # the half a matcher shapes
entropies = [3.0, 2.5, 2.0, 1.5]

laws = {H: maxwell_boltzmann(pam8, entropy=H) for H in entropies}
for H, law in laws.items():
    print(f"H = {H:.2f} bit/symbol    energy {np.sum(law * pam8 ** 2):7.3f}"
          f"    shaping gain {shaping_gain_dB(pam8, law):.3f} dB")
print(f"ultimate gain 10log10(pi e / 6) = "
      f"{10 * np.log10(np.pi * np.e / 6):.3f} dB")

fig1, ax1 = plt.subplots(figsize=(7, 4))
for H, law in laws.items():
    ax1.plot(pam8, law, "o-", label=f"H = {H:.1f} bit/symbol")
ax1.set_xlabel("8-PAM constellation point")
ax1.set_ylabel("probability")
ax1.set_title("Maxwell-Boltzmann: the same shape, one parameter")
ax1.set_xticks(pam8)
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig1.png")

target = maxwell_boltzmann(amplitudes, entropy=1.25)
lengths = [16, 32, 64, 128, 256, 512]
rates = {"CCDM (constant composition)": [], "ESS (sphere, same energy)": []}
for n in lengths:
    ccdm = ConstantCompositionMatcher(amplitudes, distribution=target,
                                      length=n)
    same_rate = SphereShaper(amplitudes, length=n, n_bits=ccdm.n_bits)
    budget = int(np.dot(ccdm.composition, same_rate.energies))
    same_energy = SphereShaper(amplitudes, length=n, max_energy=budget)
    rates["CCDM (constant composition)"].append(ccdm.rate)
    rates["ESS (sphere, same energy)"].append(same_energy.rate)
    print(f"n = {n:4d}   composition {ccdm.composition}   "
          f"rate {ccdm.rate:.4f} vs {same_energy.rate:.4f} bit/amplitude   "
          f"energy {budget:4d} vs {same_rate.max_energy:4d} at equal rate")

fig2, ax2 = plt.subplots(figsize=(7, 4))
for label, values in rates.items():
    ax2.semilogx(lengths, values, "o-", base=2, label=label)
ax2.axhline(distribution_entropy(target), color="k", linestyle="--",
            label=f"H(P) = {distribution_entropy(target):.3f}, the ceiling")
ax2.set_xlabel("blocklength n [amplitudes]")
ax2.set_ylabel("rate [bit/amplitude]")
ax2.set_title("Rate loss is what a finite block costs")
ax2.legend()
ax2.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig2.png")

n_block = 64
shaper = ConstantCompositionMatcher(amplitudes, distribution=target,
                                    length=n_block)
link = Sequential([
    SymbolGenerator(2, name="bits"),
    DistributionMatcher(shaper, name="matcher"),
    AmplitudeMapper(amplitudes, name="mapper"),
    AWGN(snr_dB=25.0, name="noise"),
    AmplitudeDemapper(amplitudes),
    DistributionDematcher(shaper),
], taps=["bits", "matcher", "mapper"], name="PAS transmitter")

n_blocks = 200
link.seed(12)
recovered = link(n_blocks * shaper.n_bits)
print(f"{shaper.n_bits} bits -> {n_block} amplitudes per block, "
      f"recovered exactly: {np.array_equal(recovered, link.tap('bits'))}")
link.summary(n_blocks * shaper.n_bits)

fig3, ax3 = plt.subplots(figsize=(7, 4))
sent = link.tap("mapper").ravel()
counts = np.array([np.mean(sent == point) for point in pam8])
ax3.bar(pam8, counts, width=1.2, alpha=0.6, label="measured at tap('mapper')")
ax3.plot(pam8, maxwell_boltzmann(pam8, entropy=2.25), "ko-",
         label="Maxwell-Boltzmann, H = 2.25")
ax3.set_xlabel("8-PAM constellation point")
ax3.set_ylabel("frequency")
ax3.set_title(f"What the matcher emits, {n_blocks * n_block} symbols")
ax3.set_xticks(pam8)
ax3.set_ylim(0, 0.40)
ax3.legend()
ax3.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig3.png")

link.set_params(**{"noise.snr_dB": 12.0})
try:
    link(n_blocks * shaper.n_bits)
except ValueError as error:
    print(f"at 12 dB -- {error}")

n_symbols = 60000
snr_dB_list = np.arange(0, 25, 2.0)
noise = AWGN(snr_dB=0.0, name="noise")
study = Sequential([
    SymbolGenerator(8, name="tx"),
    SymbolMapper(pam8),
    noise,
], name="8-PAM over AWGN")


def rate_curve(law):
    """Mutual information over the SNR range, for one input law."""
    def mutual_information(symbols, received):
        return compute_mi(received, symbols, pam8, px=law,
                          snr=1 / (2 * noise.sigma2_))

    study.set_params(**{"tx.distribution": law})
    return sweep(study, "noise.snr_dB", snr_dB_list, {"mi": mutual_information},
                 stimulus=n_symbols, reference="tx", seed=3)["mi"]


uniform = rate_curve(np.full(8, 1 / 8))
fixed = rate_curve(maxwell_boltzmann(pam8, entropy=2.25))

best = np.full(snr_dB_list.size, -np.inf)
best_H = np.zeros(snr_dB_list.size)
for H in np.arange(1.2, 3.01, 0.15):
    values = rate_curve(maxwell_boltzmann(pam8, entropy=float(H)))
    best_H[values > best] = H
    best = np.maximum(best, values)

fig4, (ax4, ax5) = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.2))
ax4.plot(snr_dB_list, uniform, "s-", label="uniform 8-PAM")
ax4.plot(snr_dB_list, fixed, "^-", label="shaped, fixed H = 2.25")
ax4.plot(snr_dB_list, best, "o-", label="shaped, H optimized per SNR")
ax4.set_xlabel("SNR [dB]")
ax4.set_ylabel("mutual information [bit/symbol]")
ax4.set_title("Same power, more bits")
ax4.legend()
ax4.grid(True)

rate_grid = np.linspace(0.5, 2.9, 49)
gap = (np.interp(rate_grid, uniform, snr_dB_list)
       - np.interp(rate_grid, best, snr_dB_list))
ax5.plot(rate_grid, gap, "-")
ax5.axhline(10 * np.log10(np.pi * np.e / 6), color="k", linestyle="--",
            label="1.53 dB, the ultimate shaping gain")
ax5.set_xlabel("rate [bit/symbol]")
ax5.set_ylabel("SNR saved [dB]")
ax5.set_title("The same gap, read horizontally")
ax5.legend()
ax5.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/probabilistic_shaping_fig4.png")

print("best entropy per SNR point:", np.round(best_H, 2))
for rate in (1.5, 2.0, 2.5):
    plain = np.interp(rate, uniform, snr_dB_list)
    shaped = np.interp(rate, best, snr_dB_list)
    print(f"rate {rate} bit/symbol: uniform needs {plain:.2f} dB, shaped "
          f"{shaped:.2f} dB -- {plain - shaped:.2f} dB saved")

mermaid_dir = "../../docs/examples/mermaid/"
for diagram_name, diagram_chain in [("shaping_pas", link), ("shaping_study", study)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
