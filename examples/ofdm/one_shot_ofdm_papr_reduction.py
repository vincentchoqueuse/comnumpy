import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.processors import Serial2Parallel
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ccdf
from comnumpy.ofdm.processors import CarrierAllocator, IFFTProcessor
from comnumpy.ofdm.predistorders import ICT_PAPR_Reductor
from comnumpy.ofdm.metrics import compute_PAPR


# This script reproduces the second figure of the following article :
# - "Optimized iterative clipping and filtering for PAPR reduction of OFDM signals." Wang, Y-C., and Z-Q. Luo., IEEE Transactions on communications 59.1 (2010): 33-37.

# parameters
N_cp = 0
N_sc = 128
L = 1000
os = 4
type, M = "PSK", 4
alphabet = get_alphabet(type, M)
alphabet_generator = np.arange(M)
papr_dB_threshold = np.arange(5, 11.01, 0.1)
carrier_type = np.zeros(N_sc*os)
carrier_type[:N_sc] = 1
N = N_sc*os*L

# create chain
chain = Sequential([
            SymbolGenerator(M),
            SymbolMapper(alphabet),
            Serial2Parallel(N_sc),
            CarrierAllocator(carrier_type),
        ])
x = chain(N)

# create list of processors
processor_list = [
    IFFTProcessor(name="no reduction"),
    ICT_PAPR_Reductor(PAPR_max_dB=5, filter_weight=carrier_type, N_it=1, name="ICT 1 iteration"),
    ICT_PAPR_Reductor(PAPR_max_dB=5, filter_weight=carrier_type, N_it=8, name="ICT 8 iterations"),
    ICT_PAPR_Reductor(PAPR_max_dB=5, filter_weight=carrier_type, N_it=16, name="ICT 16 iterations")
]

# test the different processors
for processor in processor_list:
    # run chain
    y = processor(x)

    # evaluate metric
    papr_dB_array = compute_PAPR(y, unit="dB", axis=0)
    papr_dB, ccdf = compute_ccdf(papr_dB_array)
    plt.semilogy(papr_dB, ccdf, label=processor.name)


# display theoretical curves
gamma = 10**(papr_dB_threshold/10)
ccdf_theo = 1 - (1 - np.exp(-gamma))**(N_sc*os)
plt.semilogy(papr_dB_threshold, ccdf_theo, label=f"theo: os={os}")
plt.ylim([0.001, 1])
plt.xlabel("PAPR (dB)")
plt.ylabel("CCDF")
plt.title("PAPR statistics for an OFDM systems (128 subcarriers, QPSK, OS=4)")
plt.grid()
plt.legend()
plt.show()
