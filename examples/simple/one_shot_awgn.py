import matplotlib.pyplot as plt
from comnumpy.core import Sequential, plot_iq
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import Constellation
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser

img_dir = "../../docs/getting_started/img/"

N = 10000   # Number of symbols
snr_dB = 10 # Signal-to-Noise Ratio in dB

# The constellation carries its own order and its own theory, so the
# modulation is named once and the two cannot disagree.
constellation = Constellation("QAM", 4)

chain = Sequential([
            SymbolGenerator(constellation.order, name="tx"),
            SymbolMapper(constellation),
            AWGN(snr_dB=snr_dB, name="awgn"),
            SymbolDemapper(constellation)
        ], taps=["tx", "awgn"])

# seeded, so the number this page quotes is the number you get (D6)
chain.seed(0)
y = chain(N)

data_tx = chain.tap("tx")
ser = compute_ser(data_tx, y)

# AWGN(snr_dB=) is a symbol SNR, and the closed form is quoted against
# Eb/N0: per="symbol" makes the conversion, which is 10log10(k) dB.
ser_theo = constellation.metrics(snr_dB, per="symbol")["ser"]

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
