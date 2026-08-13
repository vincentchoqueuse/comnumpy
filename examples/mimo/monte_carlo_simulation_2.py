import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import ErrorCounter
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import OrderedSuccessiveInterferenceCancellationDetector

# This script shows the performance of several OSIC MIMO detectors. It reproduces the figure 11.3 of the book [1]
# [1] Cho, Yong Soo, et al. MIMO-OFDM wireless communications with MATLAB. John Wiley & Sons, 2010.

# Parameters
N_test = 5000  # increase the value for smoothing the ber
N = 100
N_r, N_t = 4, 4
constellation = Constellation("QAM", 16)
M = constellation.order

H = rayleigh_channel(N_r=N_r, N_t=N_t)

# construct chain
chain = Sequential([SymbolGenerator(constellation.order, name="data_tx"),
                    SymbolMapper(constellation),
                    FlatMIMOChannel(H, name="channel"),
                    AWGN(sigma2=0, name="noise"),
                    ], taps=["data_tx"])

# prepare MC trial
detector_names = ["colnorm", "snr", "sinr"]

# compute snr list from snr per bit in dB
snr_dB_list = np.arange(0, 45, 5)

# perform simulation. One counter per (SNR, detector): it accumulates
# errors and bits over the trials, so the rate is a ratio of totals and
# the count that makes it credible stays available.
counters = {}
for index_snr in range(len(snr_dB_list)):
    for index in range(len(detector_names)):
        counters[index_snr, index] = ErrorCounter(width=int(np.log2(M)))

for index_snr, snr_dB in enumerate(snr_dB_list):
    sigma2 = N_t*(10**(-snr_dB/10))
    chain.set_params(noise__sigma2=sigma2)

    for _ in range(N_test):

        # new channel realization
        H = rayleigh_channel(N_r=N_r, N_t=N_t)
        chain.set_params(channel__H=H)

        # generate data
        Y = chain((N_t, N))
        S_ref = chain.tap("data_tx")

        # test detector
        for index, detector_name in enumerate(detector_names):
            # create detector
            detector = OrderedSuccessiveInterferenceCancellationDetector(alphabet=constellation, osic_type=detector_name, H=H, sigma2=sigma2)
            # perform detection
            S_est = detector(Y)
            # evaluate metrics
            counters[index_snr, index].update(S_ref, S_est)

ber_data = np.zeros((len(snr_dB_list), len(detector_names)))
for (index_snr, index), counter in counters.items():
    ber_data[index_snr, index] = counter.rate

worst = min(counters.values(), key=lambda counter: counter.n_errors)
print(f"fewest errors at any point: {worst.n_errors} over "
      f"{worst.n_symbols * worst.width} bits")

# plot figures
for index, detector_name in enumerate(detector_names):
    plt.semilogy(snr_dB_list, ber_data[:, index], label=detector_name)
plt.ylabel("BER")
plt.xlabel("SNR (dB)")
plt.xlim([0, 40])
plt.ylim([10**-4, 1])
plt.legend()
plt.grid(True)
plt.title("Performance of OSIC methods (NT=4, Nr=4, 16QAM)")
plt.show()
