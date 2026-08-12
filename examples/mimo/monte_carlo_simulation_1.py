import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ber
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import MaximumLikelihoodDetector, LinearDetector, OrderedSuccessiveInterferenceCancellationDetector

# This script reproduces the figure 2 of the article [1]
#
# X. Li, H. C. Huang, A. Lozano and G. J. Foschini, "Reduced-complexity detection algorithms for systems using multi-element arrays,"
# Globecom '00 - IEEE. Global Telecommunications Conference. Conference Record (Cat. No.00CH37137), San Francisco, CA, USA, 2000, pp. 1072-1076 vol.2,


# Parameters
N_test = 500  # increase the value for smoothing the ber
N = 400
N_r, N_t = 2, 2
constellation = Constellation("PSK", 4)
M = constellation.order

H = rayleigh_channel(N_r=N_r, N_t=N_t)

# construct chain
chain = Sequential([SymbolGenerator(constellation.order, name="data_tx"),
                    SymbolMapper(constellation),
                    FlatMIMOChannel(H, name="channel"),
                    AWGN(sigma2=0, name="noise")
                    ], taps=["data_tx"])

# prepare MC trial
detector_names = ["ML", "ZF", "OSIC"]

# compute snr list from snr per bit in dB
snr_dB_list = np.arange(0, 45, 5)

# perform simulation
bler_data = np.zeros((len(snr_dB_list), len(detector_names)))
sig_power = N_t

for index_snr, snr_dB in enumerate(snr_dB_list):
    sigma2 = sig_power*(10**(-snr_dB/10))
    chain["noise"].sigma2 = sigma2

    for _ in range(N_test):

        # new channel realization
        H = rayleigh_channel(N_r=N_r, N_t=N_t)
        chain["channel"].H = H

        # generate data
        Y = chain((N_t, N))
        S_ref = chain.tap("data_tx")

        # test detector
        for index, detector_name in enumerate(detector_names):

            match detector_name:
                case "ML":
                    detector = MaximumLikelihoodDetector(alphabet=constellation, H=H)
                case "ZF":
                    detector = LinearDetector(alphabet=constellation, H=H, method="zf")
                case "OSIC":
                    detector = OrderedSuccessiveInterferenceCancellationDetector(constellation, osic_type="sinr", H=H, sigma2=sigma2, name="OSIC")

            # perform detection
            S_est = detector(Y)
            # evaluate metrics
            bler_data[index_snr, index] += (compute_ber(S_ref, S_est, width=int(np.log2(M))) > 0)

    bler_data[index_snr, :] /= N_test

# plot figures
for index, detector_name in enumerate(detector_names):
    plt.semilogy(snr_dB_list, bler_data[:, index], label=detector_name)
plt.ylabel("BLER")
plt.xlabel("SNR (dB)")
plt.xlim([0, 40])
plt.ylim([10**-3, 1])
plt.legend()
plt.grid(True)
plt.title("Performance Comparison of ZF and ML detector, 2*2 QPSK")
plt.show()
