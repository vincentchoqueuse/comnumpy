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
modulation = "QAM"
M = 16          # Modulation order
N_h = 5         # Number of channel taps
N_cp = 10       # Cyclic prefix length
N = 100000      # Number of symbols
sigma2 = 0.01   # Noise variance

alphabet = get_alphabet(modulation, M)  # Get alphabet for QAM modulation
allocation = get_allocation("802.11ac-40")  # 128 subcarriers, from the catalog

# the allocation carries its own counts, checked against the standard's table
N_carriers = allocation.N_fft
N_carrier_data = allocation.N_data       # Number of data carriers
N_carrier_pilots = allocation.N_pilots   # Number of pilot carriers

# channel parameters
h = 0.1 * (np.random.randn(N_h) + 1j * np.random.randn(N_h))
h[0] = 1
pilots = 10 * np.ones(N_carrier_pilots)  # Pilot values

# communication chain
chain = Sequential([
    SymbolGenerator(M),
    SymbolMapper(alphabet),
    Serial2Parallel(N_carrier_data),
    CarrierAllocator(carrier_type=allocation, pilots=pilots),
    IFFTProcessor(),
    CyclicPrefixer(N_cp),
    Parallel2Serial(),
    FIRChannel(h),
    AWGN(sigma2=sigma2),
    Serial2Parallel(N_carriers + N_cp),
    CyclicPrefixRemover(N_cp),
    FFTProcessor(),
    FrequencyDomainEqualizer(h=h),
    CarrierExtractor(allocation),
    Parallel2Serial(),
    SymbolDemapper(alphabet)
])

# profiling chain
plot_chain_profiling(chain, input=N)
plt.savefig("../../docs/getting_started/img/profiling_chain_fig1.png")
plt.show()
