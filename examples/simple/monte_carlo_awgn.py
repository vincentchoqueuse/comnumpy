import numpy as np
import matplotlib.pyplot as plt
from comnumpy import monte_carlo
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import Constellation
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser
from comnumpy.core.visualizers import plot_error_rate


# parameters
N = 1000000
constellation = Constellation("QAM", 16)
snr_dB_list = np.arange(0, 22)

# create chain
chain = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation),
    AWGN(snr_dB=0, name="awgn_channel"),
    SymbolDemapper(constellation),
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
results = monte_carlo(chain, "awgn_channel.snr_dB", snr_dB_list,
                      {"ser": compute_ser}, N, reference="tx", seed=1)
ser_array = results["ser"]
loop_line = "loop  :"
for value in by_hand:
    loop_line += f" {value:.3e}"
print(loop_line)
sweep_line = "sweep :"
for value in ser_array[::8]:
    sweep_line += f" {value:.3e}"
print(sweep_line)

# The closed form comes from the constellation itself: it knows its
# family and its order, so the theory cannot describe a modulation other
# than the one the chain transmits. AWGN(snr_dB=) is a symbol SNR, hence
# per="symbol".
ser_theo_array = constellation.metrics(snr_dB_list, per="symbol")["ser"]

# plot_error_rate draws the measurements as markers and the closed form
# as a line of the same colour: the figure every sweep ends with
label = f"{constellation.order}-{constellation.family}"
plot_error_rate(snr_dB_list, {label: ser_array},
                theory={label: ser_theo_array},
                ylabel="SER", title=f"SER performance for {label}")

# The chain diagrams this tutorial shows are exported from the chains
# themselves (D33c), so the picture cannot drift from the code -- the
# smoke test compares what a run writes with what the page displays.
mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("awgn_chain", chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
