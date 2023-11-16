import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from comnumpy import Sequential
from comnumpy.generators.data import Symbol_Generator
from comnumpy.dsp.modem import get_alphabet, Modulator
from comnumpy.dsp.ofdm import Carrier_Allocator, IFFT_Processor, ICT_PAPR_Reductor
from comnumpy.dsp.utils import Serial_2_Parallel, Parallel_2_Serial
from comnumpy.metrics.ofdm import compute_PAPR, compute_ccdf
from comnumpy.metrics.recorder import Metric_Recorder
from comnumpy.analysers.scope import Spectrum_Scope


def get_carrier_allocation(N_sc, os):
    # specify carrier structure with zero padding for oversampling
    N = os*N_sc
    middle = N//2
    width = N_sc//2

    carrier_type = np.zeros(N)
    carrier_type[middle-width:middle+width] = 1
    return carrier_type


# see paper: Wang, Y-C., and Z-Q. Luo. "Optimized iterative clipping and filtering for PAPR reduction of OFDM signals." IEEE Transactions on communications 59.1 (2010): 33-37.
# figure 2

# parameters
N_cp = 0
N_sc = 128
L = 100
os = 4
type, M = "PSK", 4
alphabet = get_alphabet(type, M)

alphabet_generator = np.arange(M)
papr_dB_threshold = np.arange(5, 11.01, 0.1)
carrier_type = get_carrier_allocation(N_sc, os)
N = N_sc*os*L

technique_list = [
    { "name": "no reduction", "processor": IFFT_Processor, "params": {}},
    { "name": "ICT 1 iteration", "processor": ICT_PAPR_Reductor, "params": {"PAPR_max_dB": 5, "filter_weight": carrier_type, "N_it": 1}},
    { "name": "ICT 8 iterations", "processor": ICT_PAPR_Reductor, "params": {"PAPR_max_dB": 5, "filter_weight": carrier_type, "N_it": 8}},
    { "name": "ICT 16 iterations", "processor": ICT_PAPR_Reductor, "params": {"PAPR_max_dB": 5, "filter_weight": carrier_type, "N_it": 16}},
    ]


for technique in tqdm(technique_list):

    name = technique["name"]
    processor = technique["processor"]
    params = technique["params"]

    ifft_processor = processor(**params)
    metric = Metric_Recorder(compute_PAPR, params={"unit": "dB"})

    # perform simulation
    chain = Sequential(
        [
            Symbol_Generator(alphabet=alphabet_generator),
            Modulator(alphabet),
            Serial_2_Parallel(N_sc),
            Carrier_Allocator(carrier_type),
            ifft_processor,
            metric,
            Parallel_2_Serial(),
        ]
    )
    y = chain(N)

    # extract metric
    papr_dB_array = metric.get_data()
    ccdf = compute_ccdf(papr_dB_array, papr_dB_threshold=papr_dB_threshold)
    plt.semilogy(papr_dB_threshold, ccdf, label=name)


# display theoretical curves
label_theo = "theo: os={}".format(os)
gamma = 10**(papr_dB_threshold/10)
ccdf_theo = 1 - (1 - np.exp(-gamma))**(N_sc*os)
plt.semilogy(papr_dB_threshold, ccdf_theo, label=label_theo)

plt.ylim([0.001, 1])
plt.xlabel("PAPR (dB)")
plt.ylabel("CCDF")
plt.title("PAPR statistics for an OFDM systems (128 subcarriers, QPSK, OS=4)")
plt.grid()
plt.legend()

scope = Spectrum_Scope(apply_fftshift=False, ylim=[-50, 0], label=name)
scope(y)

plt.show()
