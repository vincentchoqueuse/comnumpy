import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.utils import get_alphabet
from comnumpy.core.visualizers import plot_chain_profiling
from comnumpy.ofdm.processors import CarrierAllocator, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover, CarrierExtractor
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from comnumpy.ofdm.allocation import get_allocation


# parameters
M = 16
N_h = 5
N_cp = 10
N = 100000
sigma2 = 0.01
alphabet = get_alphabet("QAM", M)
allocation = get_allocation("802.11ac-40")   # 128 subcarriers

# extract carrier information
N_carriers = allocation.N_fft
N_carrier_data = allocation.N_data
N_carrier_pilots = allocation.N_pilots

# generate channel
h = 0.1*(np.random.randn(N_h) + 1j*np.random.randn(N_h))
h[0] = 1
pilots = 10*np.ones(N_carrier_pilots)

# create sequential
chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet, name="mapper_tx"),
        Serial2Parallel(N_carrier_data),
        CarrierAllocator(carrier_type=allocation, pilots=pilots, name="carrier_allocator_tx"),
        IFFTProcessor(),
        CyclicPrefixer(N_cp),
        Parallel2Serial(),
        FIRChannel(h),
        AWGN(sigma2=sigma2),
        Serial2Parallel(N_carriers+N_cp),
        CyclicPrefixRemover(N_cp),
        FFTProcessor(),
        FrequencyDomainEqualizer(h=h),
        CarrierExtractor(allocation, name="data_rx"),
        Parallel2Serial(),
        SymbolDemapper(alphabet)
    ])

# run chain
plot_chain_profiling(chain, input=N)
plt.show()
