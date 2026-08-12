import numpy as np
import matplotlib.pyplot as plt
from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo
from comnumpy.core.visualizers import plot_error_rate


# parameters
M = 16
N = 1000000
modulation = "QAM"
alphabet = get_alphabet(modulation, M)
snr_dB_list = np.arange(0, 22)

# create chain
chain = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    AWGN(snr_dB=0, name="awgn_channel"),
    SymbolDemapper(alphabet),
    ])

# A Monte Carlo simulation is a stateless loop: reconfigure the chain,
# run it, measure, collect. Written out, at three SNR values:
chain.taps = ["tx"]
by_hand = []
for snr_dB in snr_dB_list[::8]:
    chain.seed(1)
    chain.set_params(awgn_channel__snr_dB=snr_dB)
    detected = chain(N)                  # run first: the tap is filled by it
    by_hand.append(compute_ser(chain.tap("tx"), detected))

# sweep (decision D35) is that loop, and nothing more: the same
# reconfigure-reseed-run-collect, over every point, in one call.
results = sweep(chain, "awgn_channel.snr_dB", snr_dB_list,
                {"ser": compute_ser}, N, reference="tx", seed=1)
ser_array = results["ser"]
print("loop  :", " ".join(f"{value:.3e}" for value in by_hand))
print("sweep :", " ".join(f"{value:.3e}" for value in ser_array[::8]))

# compute theoretical SER metric
snr_per_bit = (10**(snr_dB_list/10))/np.log2(M)
ser_theo_array = compute_metric_awgn_theo(modulation, M, snr_per_bit)["ser"]

# plot_error_rate draws the measurements as markers and the closed form
# as a line of the same colour: the figure every sweep ends with
plot_error_rate(snr_dB_list, {f"{M}-{modulation}": ser_array},
                theory={f"{M}-{modulation}": ser_theo_array},
                ylabel="SER", title=f"SER performance for {M}-{modulation}")

# The chain diagrams this tutorial shows are exported from the chains
# themselves (D33c), so the picture cannot drift from the code -- the
# smoke test compares what a run writes with what the page displays.
mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("awgn_chain", chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
