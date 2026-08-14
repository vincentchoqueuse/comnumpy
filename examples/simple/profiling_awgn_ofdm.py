import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import (AWGN, FIRChannel,
                                    TappedDelayLineChannel)
from comnumpy.core.fading import get_delay_profile
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_chain_profiling
from comnumpy.ofdm.processors import CarrierAllocator, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover, CarrierExtractor
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from comnumpy.ofdm.allocation import get_allocation
from comnumpy import style

style.use()


fs = 5e6        # Sample rate the channel is resolved at
N_cp = 10       # Cyclic prefix length
N = 100000      # Number of symbols
sigma2 = 0.01   # Noise variance

constellation = Constellation("QAM", 16)
allocation = get_allocation("802.11ac-40")  # 128 subcarriers, from the catalog

N_carriers = allocation.N_fft
N_carrier_data = allocation.N_data       # Number of data carriers
N_carrier_pilots = allocation.N_pilots   # Number of pilot carriers

channel_model = TappedDelayLineChannel(get_delay_profile("TDL-D"), fs=fs,
                                       seed=5)
h = channel_model.impulse_response()
pilots = 10 * np.ones(N_carrier_pilots)  # Pilot values

chain = Sequential([
    SymbolGenerator(constellation.order),
    SymbolMapper(constellation),
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
    SymbolDemapper(constellation)
])

plot_chain_profiling(chain, input=N)
plt.savefig("../../docs/getting_started/img/profiling_chain_fig1.png")

mermaid_dir = "../../docs/getting_started/mermaid/"
for diagram_name, diagram_chain in [("profiling_chain", chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())
