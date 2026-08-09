import numpy as np
import matplotlib.pyplot as plt
from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo


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

# perform the Monte Carlo simulation (decision D35): reconfigure the
# SNR at every point and compare the output with the tapped tx symbols
results = sweep(chain, "awgn_channel.snr_dB", snr_dB_list,
                {"ser": compute_ser}, N, reference="tx", seed=1)
ser_array = results["ser"]

# compute theoretical SER metric
snr_per_bit = (10**(snr_dB_list/10))/np.log2(M)
ser_theo_array = compute_metric_awgn_theo(modulation, M, snr_per_bit, "ser")

plt.semilogy(snr_dB_list, ser_array, label="exp")
plt.semilogy(snr_dB_list, ser_theo_array, "--", label="theo")
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.title(f"SER performance for {M}-{modulation}")
plt.legend()
plt.grid()
plt.show()
