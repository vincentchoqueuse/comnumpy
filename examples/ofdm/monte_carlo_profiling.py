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


fs = 5e6
N_cp = 10
N = 100000
sigma2 = 0.01
constellation = Constellation("QAM", 16)
allocation = get_allocation("802.11ac-40")   # 128 subcarriers

N_carriers = allocation.N_fft
N_carrier_data = allocation.N_data
N_carrier_pilots = allocation.N_pilots

channel_model = TappedDelayLineChannel(get_delay_profile("TDL-D"), fs=fs,
                                       seed=5)
h = channel_model.impulse_response()
pilots = 10*np.ones(N_carrier_pilots)

chain = Sequential([
        SymbolGenerator(constellation.order, name="data_tx"),
        SymbolMapper(constellation, name="mapper_tx"),
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
        SymbolDemapper(constellation)
    ])

plot_chain_profiling(chain, input=N)
plt.show()
