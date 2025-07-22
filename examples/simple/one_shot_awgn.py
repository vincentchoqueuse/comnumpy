import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.channels import AWGN
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser

img_dir = "../../docs/getting_started/img/"

# parameters
M = 4           # Modulation order (4-QAM)
N = 1000        # Number of symbols
alphabet = get_alphabet("QAM", M)
snr_dB = 10     # Signal-to-Noise Ratio in dB

# define a communication chain
chain = Sequential([
            SymbolGenerator(M),
            Recorder(name="data_tx"),
            SymbolMapper(alphabet),
            AWGN(value=snr_dB, unit="snr_dB", name="awgn_channel"),
        ])

# test chain
y = chain(N)

# estimate performance
data_tx = chain["data_tx"].get_data()
ser = compute_ser(data_tx, y)
print(f"SER = {ser}")

# plot signals
plt.scatter(np.real(y), np.imag(y))
plt.title("Received Constellation Diagram")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.savefig(f"{img_dir}/first_simulation_fig1.png")
plt.show()
