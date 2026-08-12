import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential, plot_iq
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo

img_dir = "../../docs/getting_started/img/"

modulation = "QAM"
M = 4       # Modulation order (4-QAM)
N = 10000   # Number of symbols
snr_dB = 10 # Signal-to-Noise Ratio in dB

alphabet = get_alphabet(modulation, M)

chain = Sequential([
            SymbolGenerator(M, name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=snr_dB, name="awgn"),
            SymbolDemapper(alphabet)
        ], taps=["tx", "awgn"])

y = chain(N)

data_tx = chain.tap("tx")
ser = compute_ser(data_tx, y)

snr_per_bit = (10**(snr_dB/10))/np.log2(M)
ser_theo = compute_metric_awgn_theo(modulation, M, snr_per_bit)["ser"]

print(f"SER (simu)= {ser}")
print(f"SER (theo)= {ser_theo}")

ax = plot_iq(chain.tap("awgn"), title="Received Constellation Diagram")
ax.grid(True)

plt.savefig(f"{img_dir}/first_simulation_fig1.png")

mermaid_dir = "../../docs/getting_started/mermaid/"
for diagram_name, diagram_chain in [("first_simulation", chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
