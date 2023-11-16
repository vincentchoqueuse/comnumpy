import numpy as np
import matplotlib.pyplot as plt
from comnumpy import Sequential
from comnumpy.generators.data import Symbol_Generator
from comnumpy.dsp.modem import get_alphabet, Modulator
from comnumpy.dsp.ofdm import Carrier_Allocator, IFFT_Processor
from comnumpy.dsp.utils import Serial_2_Parallel
from comnumpy.metrics.ofdm import compute_PAPR, compute_ccdf
from comnumpy.metrics.recorder import Metric_Recorder


def get_carrier_allocation(N_sc, os):
    # specify carrier structure with zero padding for oversampling
    N = os*N_sc
    middle = N//2
    width = N_sc//2

    carrier_type = np.zeros(N)
    carrier_type[middle-width:middle+width] = 1
    return carrier_type


# see paper: Jiang, Tao, and Yiyan Wu. "An overview: Peak-to-average power ratio reduction techniques for OFDM signals." IEEE Transactions on broadcasting 54.2 (2008): 257-268.

# parameters
N_cp = 0
N_sc = 128
L = 100000
type, M = "QAM", 16
alphabet = get_alphabet(type, M)
alphabet_generator = np.arange(M)
papr_dB_threshold = np.arange(6, 13, 0.1)
gamma = 10**(papr_dB_threshold/10)

# for this simulation, I prefer to store the metric in the sequential object.
# the metric class allows to compute the metric for each frame.

metric = Metric_Recorder(compute_PAPR, params={"unit": "dB"})

for os in range(1, 3):
    carrier_type = get_carrier_allocation(N_sc, os)
    N = N_sc*os*L

    # perform simulation
    chain = Sequential(
        [
            Symbol_Generator(alphabet=alphabet_generator),
            Modulator(alphabet),
            Serial_2_Parallel(N_sc),
            Carrier_Allocator(carrier_type),
            IFFT_Processor(),
            metric
        ]
    )
    y = chain(N)

    # extract metric and convert to 1D
    papr_dB_array = metric.get_data()

    # display experimental curves
    ccdf = compute_ccdf(papr_dB_array, papr_dB_threshold=papr_dB_threshold)
    label_exp = "exp: os={}".format(os)
    plt.semilogy(papr_dB_threshold, ccdf, label=label_exp)

    # display theoretical curves
    label_theo = "theo: os={}".format(os)
    ccdf_theo = 1 - (1 - np.exp(-gamma))**(N_sc*os)
    plt.semilogy(papr_dB_threshold, ccdf_theo, label=label_theo)


plt.ylim([0.0001, 1])
plt.xlabel("PAPR (dB)")
plt.ylabel("CCDF")
plt.title("PAPR for different Oversampling rate")
plt.legend()
plt.show()
