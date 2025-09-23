import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser
from comnumpy.core.visualizers import plot_chain_profiling
from comnumpy.ofdm.processors import CarrierAllocator, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover, CarrierExtractor
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from comnumpy.ofdm.utils import get_standard_carrier_allocation


# parameters
modulation = "QAM"
M = 16          # Modulation order
N_h = 5         # Number of channel taps
N_cp = 10       # Cyclic prefix length
N = 100000      # Number of symbols
sigma2 = 0.01   # Noise variance

alphabet = get_alphabet(modulation, M)  # Get alphabet for QAM modulation
carrier_type = get_standard_carrier_allocation("802.11ah_128")  # Standard carrier allocation

# extract carrier information
N_carriers = len(carrier_type)
N_carrier_data = np.sum(carrier_type == 1)  # Number of data carriers
N_carrier_pilots = np.sum(carrier_type == 2)  # Number of pilot carriers

# channel parameters
h = 0.1 * (np.random.randn(N_h) + 1j * np.random.randn(N_h))
h[0] = 1
pilots = 10 * np.ones(N_carrier_pilots)  # Pilot values

# communication chain
chain = Sequential([
    SymbolGenerator(M),
    SymbolMapper(alphabet),
    Serial2Parallel(N_carrier_data),
    CarrierAllocator(carrier_type=carrier_type, pilots=pilots),
    IFFTProcessor(),
    CyclicPrefixer(N_cp),
    Parallel2Serial(),
    FIRChannel(h),
    AWGN(sigma2),
    Serial2Parallel(N_carriers + N_cp),
    CyclicPrefixRemover(N_cp),
    FFTProcessor(),
    FrequencyDomainEqualizer(h=h),
    CarrierExtractor(carrier_type),
    Parallel2Serial(),
    SymbolDemapper(alphabet)
])

# profiling chain
plot_chain_profiling(chain, input=N)
plt.savefig(f"../../docs/getting_started/img/profiling_chain_fig1.png")
plt.show()
