import numpy as np
import matplotlib.pyplot as plt

from comnumpy import style
from comnumpy.core import Sequential, plot_spectrum, plot_iq
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Upsampler, Downsampler, DataExtractor
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.utils import Constellation
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ser

style.use()


# parameters
N = 10000
constellation = Constellation("QAM", 16)
oversampling = 8
rolloff = 0.25
N_h = 1000
sigma2 = 3e-2

# create chain; taps extract the signals to observe after the run
chain = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation),
    Upsampler(oversampling),
    SRRCFilter(rolloff, oversampling, N_h=N_h),
    AWGN(sigma2=sigma2, name="awgn_channel"),
    SRRCFilter(rolloff, oversampling, N_h=N_h),
    Downsampler(oversampling, phase=2*oversampling*N_h),
    DataExtractor(selector=(0, N), name="extractor"),
    SymbolDemapper(constellation),
    ], taps=["tx", "awgn_channel", "extractor"])

# run chain
y = chain(N)

# evaluate metrics
data_tx = chain.tap("tx")
ser_exp = compute_ser(data_tx, y)

# plot the extracted signals
plot_spectrum(chain.tap("awgn_channel"), title="received signal")
plot_iq(chain.tap("extractor"), title="after SRRC+downsampling+extractor")

# plot error distribution
N_min = np.min([len(data_tx), len(y)])
plt.figure()
plt.stem(np.abs(data_tx[:N_min]-y[:N_min]) > 0.01)
plt.xlabel("n [samples]")
plt.ylabel("error")
plt.title("error distribution")

# theoretical metrics
ser_theo = constellation.metrics(-10 * np.log10(sigma2),
                                 per="symbol")["ser"]

# print metric and plot
print(f"exp: SER={ser_exp}")
print(f"theo: SER={ser_theo}")
plt.show()
