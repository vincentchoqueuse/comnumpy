import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.special import erfc
from comnumpy import monte_carlo, print_data
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import Constellation, ebn0_to_snr_dB
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.core.visualizers import plot_error_rate
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

# parameters
N = 1000000
constellation = Constellation("QAM", 16)
snr_dB_list = np.arange(0, 22)


# One chain, transmitter to decision. Every block that a sweep must
# reconfigure is named: the channel for the SNR, the generator, mapper
# and demapper for the constellation the last section changes.
chain = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation, name="mapper"),
    AWGN(snr_dB=0, name="awgn_channel"),
    SymbolDemapper(constellation, name="demapper"),
], observations=["tx"])

# A Monte Carlo simulation is three parts, always in this order: the
# metric pre-allocated, the loop that fills it, the display at the end.
# Written out, at three SNR values:
probed_snr_dB = snr_dB_list[::8]
ser_by_hand = np.zeros(len(probed_snr_dB))
for index, snr_dB in enumerate(probed_snr_dB):
    chain.seed(1)
    chain.set_params(awgn_channel__snr_dB=snr_dB)
    detected = chain(N)                  # run first: the observation is filled by it
    ser_by_hand[index] = compute_ser(chain.observation("tx"), detected)

# monte_carlo (decision D35) is that loop, and nothing more: the same
# reconfigure-reseed-run-collect, over every point, in one call.
results = monte_carlo(chain, "awgn_channel.snr_dB", snr_dB_list,
                      {"ser": compute_ser}, N, reference="tx", seed=1)
ser_array = results["ser"]

# An abscissa and one series per curve is what a sweep produces, and
# print_data is what shows it: the same dictionary would draw the figure.
print_data({"x": probed_snr_dB,
            "curves": {"loop": ser_by_hand,
                       "monte_carlo": ser_array[::8]}},
           xlabel="SNR [dB]", ylabel="SER")


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
orders = np.array([4, 16, 64, 256])
bits_per_symbol = np.log2(orders).astype(int)     # vectorized, not a loop
curves = {}
theory = {}
for index, order in enumerate(orders):
    other = Constellation("QAM", int(order))
    bits = int(bits_per_symbol[index])
    snr_dB_values = ebn0_to_snr_dB(ebn0_dB_list, bits_per_symbol=bits)
    # the same chain, reconfigured (D50): set_params re-runs the
    # coercions a bare assignment would skip
    chain.set_params(tx__M=other.order, mapper__alphabet=other,
                     demapper__alphabet=other)
    collected = monte_carlo(chain, "awgn_channel.snr_dB",
                            snr_dB_values,
                            {"ber": partial(compute_ber, width=bits)},
                            N_compare, reference="tx", seed=1)
    order_label = f"{order}-QAM"
    curves[order_label] = collected["ber"]
    theory[order_label] = other.metrics(ebn0_dB_list, per="bit")["ber"]

ax = plot_error_rate(ebn0_dB_list, curves, theory=theory,
                     xlabel="$E_b/N_0$ [dB]", ylabel="BER",
                     title="Square QAM at equal energy per bit")
# The closed forms run down to 1e-12; the measurements cannot follow them
# past a few 1e-6, so the axis stops where the simulation stops speaking.
ax.set_ylim(1e-6, 1)
plt.savefig(f"{img_dir}/monte_carlo_awgn_orders.png")

