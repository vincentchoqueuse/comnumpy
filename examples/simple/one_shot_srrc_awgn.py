import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential, Recorder, Scope
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Upsampler, Downsampler, DataExtractor
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo


# parameters
M = 16
N = 10000
modulation = "QAM"
alphabet = get_alphabet(modulation, M)
oversampling = 8
rolloff = 0.25
N_h = 1000
sigma2 = 3e-2

# create chain
chain = Sequential([
    SymbolGenerator(M),
    Recorder(name="recorder_tx"),
    SymbolMapper(alphabet),
    Upsampler(oversampling),
    SRRCFilter(rolloff, oversampling, N_h=N_h),
    AWGN(value=sigma2, name="awgn_channel"),
    Scope(num=1, scope_type="spectrum", title="received signal"),
    SRRCFilter(rolloff, oversampling, N_h=N_h),
    Downsampler(oversampling, phase=2*oversampling*N_h),
    DataExtractor(selector=(0, N)),
    Scope(num=2, scope_type="iq", title="after SRRC+downsampling+extractor"),
    SymbolDemapper(alphabet),
    ])

# run chain
y = chain(N)

# evaluate metrics
data_tx = chain["recorder_tx"].get_data()
ser_exp = compute_ser(data_tx, y)

# plot error distribution
N_min = np.min([len(data_tx), len(y)])
plt.figure()
plt.stem(np.abs(data_tx[:N_min]-y[:N_min]) > 0.01)
plt.xlabel("n [samples]")
plt.ylabel("error")
plt.title("error distribution")

# theoretical metrics
snr_per_bit = (1/sigma2) / np.log2(M)
ser_theo = compute_metric_awgn_theo(modulation, M, snr_per_bit, "ser")

# print metric and plot
print(f"exp: SER={ser_exp}")
print(f"theo: SER={ser_theo}")
plt.show()
