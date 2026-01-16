import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo

img_dir = "docs/getting_started/img"

# parameters
modulation = "QAM"
M = 4       # Modulation order (4-QAM)
N = 10000   # Number of symbols
snr_dB = 10 # Signal-to-Noise Ratio in dB

alphabet = get_alphabet(modulation, M)

# define a communication chain
chain = Sequential([
            SymbolGenerator(M),
            Recorder(name="recorder_tx"),
            SymbolMapper(alphabet),
            AWGN(value=snr_dB, unit="snr_dB"),
            Recorder(name="recorder_rx"),
            SymbolDemapper(alphabet)
        ])

# test chain
y = chain(N)

# estimate simulation performance
data_tx = chain["recorder_tx"].get_data()
ser = compute_ser(data_tx, y)

# extract theoretical performance
snr_per_bit = (10**(snr_dB/10))/np.log2(M)
ser_theo = compute_metric_awgn_theo(modulation, M, snr_per_bit, "ser")

# display results
print(f"SER (simu)= {ser}")
print(f"SER (theo)= {ser_theo}")

# plot signals
data_rx = chain["recorder_rx"].get_data()
plt.scatter(np.real(data_rx), np.imag(data_rx))
plt.title("Received Constellation Diagram")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.grid(True)

plt.savefig(f"{img_dir}/first_simulation_fig1.png")
plt.show()
