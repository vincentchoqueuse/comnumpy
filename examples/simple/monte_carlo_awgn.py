import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from comnumpy.core import Sequential, Recorder
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
    SymbolGenerator(M),
    Recorder(name="recorder_tx"),
    SymbolMapper(alphabet),
    AWGN(unit="snr_dB", name="awgn_channel"),
    SymbolDemapper(alphabet),
    ])

# perform monte Carlo simulation
ser_array = np.zeros(len(snr_dB_list))

for index, snr_dB in enumerate(tqdm(snr_dB_list)):

    # change simulation parameters
    chain["awgn_channel"].value = snr_dB

    # run chain
    y = chain(N)

    # evaluate metrics
    data_tx = chain["recorder_tx"].get_data()
    ser = compute_ser(data_tx, y)

    # save and display metrics
    ser_array[index] = ser


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
