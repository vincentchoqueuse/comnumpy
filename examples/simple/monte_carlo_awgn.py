import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.special import erfc
from comnumpy import monte_carlo
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import Constellation, ebn0_to_snr_dB
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.core.visualizers import plot_error_rate

img_dir = "../../docs/tutorials/img/"

# parameters
N = 1000000
constellation = Constellation("QAM", 16)
snr_dB_list = np.arange(0, 22)


def get_chain(constellation):
    """Symbols in, decided symbols out, over an AWGN channel.

    The last section of this page runs one of these per constellation
    order, so the constellation is an argument rather than a global. The
    transmitter is named because a sweep has to compare against what it
    produced, and the channel is named because a sweep has to reconfigure
    it.
    """
    return Sequential([
        SymbolGenerator(constellation.order, name="tx"),
        SymbolMapper(constellation),
        AWGN(snr_dB=0, name="awgn_channel"),
        SymbolDemapper(constellation),
        ], taps=["tx"])


chain = get_chain(constellation)

# A Monte Carlo simulation is a stateless loop: reconfigure the chain,
# run it, measure, collect. Written out, at three SNR values:
by_hand = []
for snr_dB in snr_dB_list[::8]:
    chain.seed(1)
    chain.set_params(awgn_channel__snr_dB=snr_dB)
    detected = chain(N)                  # run first: the tap is filled by it
    by_hand.append(compute_ser(chain.tap("tx"), detected))

# monte_carlo (decision D35) is that loop, and nothing more: the same
# reconfigure-reseed-run-collect, over every point, in one call.
results = monte_carlo(chain, "awgn_channel.snr_dB", snr_dB_list,
                      {"ser": compute_ser}, N, reference="tx", seed=1)
ser_array = results["ser"]
loop_line = "loop        :"
for value in by_hand:
    loop_line += f" {value:.3e}"
print(loop_line)
sweep_line = "monte_carlo :"
for value in ser_array[::8]:
    sweep_line += f" {value:.3e}"
print(sweep_line)


def qfunc(x):
    """Gaussian tail function, the one the textbooks write as Q."""
    return 0.5 * erfc(x / np.sqrt(2))


# The closed form of Proakis, written out. A square M-QAM is two
# independent sqrt(M)-PAM constellations, so a symbol is correct only
# when both components are. AWGN(snr_dB=) is a symbol SNR and the
# expression is parameterized by the SNR per bit, hence the division by k.
M = constellation.order
k = int(np.log2(M))
gamma_b = 10 ** (snr_dB_list / 10) / k
P_pam = 2 * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * k * gamma_b / (M - 1)))
ser_theo_array = 1 - (1 - P_pam) ** 2

# The same expression is already in the library, and the constellation
# knows which one applies to it. per="symbol" says that the swept SNR is
# the symbol SNR, and the conversion to Eb/N0 above happens inside.
from_object = constellation.metrics(snr_dB_list, per="symbol")["ser"]
gap = np.max(np.abs(ser_theo_array - from_object))
print(f"largest gap between the two closed forms: {gap:.3e}")

# plot_error_rate draws the measurements as markers and the closed form
# as a line of the same colour: the figure every sweep ends with
label = f"{constellation.order}-{constellation.family}"
plot_error_rate(snr_dB_list, {label: ser_array},
                theory={label: ser_theo_array},
                ylabel="SER", title=f"SER performance for {label}")
plt.savefig(f"{img_dir}/monte_carlo_awgn.png")

# Comparing orders is only fair at equal energy per *bit*: at equal
# energy per symbol a dense constellation is favoured, because it spends
# the same energy on more bits. ebn0_to_snr_dB does that conversion, and
# it needs k, which is chain-level knowledge the channel does not have.
ebn0_dB_list = np.arange(0, 25, 2)
N_compare = 100000
fine_dB_list = np.arange(0, 25, 0.05)
measured_ber = {}
theory_ber = {}
required_rows = []
for order in [4, 16, 64, 256]:
    other = Constellation("QAM", order)
    bits = int(np.log2(order))
    snr_dB_values = ebn0_to_snr_dB(ebn0_dB_list, bits_per_symbol=bits)
    collected = monte_carlo(get_chain(other), "awgn_channel.snr_dB",
                            snr_dB_values,
                            {"ber": partial(compute_ber, width=bits)},
                            N_compare, reference="tx", seed=1)
    order_label = f"{order}-QAM"
    measured_ber[order_label] = collected["ber"]
    theory_ber[order_label] = other.metrics(ebn0_dB_list, per="bit")["ber"]
    # What the density costs, read off the closed form rather than the
    # measurement: the sweep grid is 2 dB coarse and noisy at 1e-3.
    fine_ber = other.metrics(fine_dB_list, per="bit")["ber"]
    above_underflow = fine_ber > 0            # the tail reaches 0 in float64
    required_dB = float(
        np.interp(-3.0, np.log10(fine_ber[above_underflow])[::-1],
                  fine_dB_list[above_underflow][::-1]))
    required_rows.append((order_label, bits, required_dB))

print(f"{'order':>8s} {'bits/symbol':>12s} {'Eb/N0 at BER=1e-3':>19s}")
for order_label, bits, ebn0_dB in required_rows:
    print(f"{order_label:>8s} {bits:>12d} {ebn0_dB:>16.1f} dB")

ax = plot_error_rate(ebn0_dB_list, measured_ber, theory=theory_ber,
                     xlabel="$E_b/N_0$ [dB]", ylabel="BER",
                     title="Square QAM at equal energy per bit")
# The closed forms run down to 1e-12; the measurements cannot follow them
# past a few 1e-6, so the axis stops where the simulation stops speaking.
ax.set_ylim(1e-6, 1)
plt.savefig(f"{img_dir}/monte_carlo_awgn_orders.png")

# The chain diagrams this tutorial shows are exported from the chains
# themselves (D33c), so the picture cannot drift from the code -- the
# smoke test compares what a run writes with what the page displays.
mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("awgn_chain", chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())
