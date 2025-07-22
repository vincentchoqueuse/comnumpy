import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import FlatFadingRayleighChannel
from comnumpy.mimo.detectors import MaximumLikelihoodDetector, ZeroForcingDetector, MinimumMeanSquaredErrorDetector, OrderedSuccessiveInterferenceCancellationDetector

# This script reproduces the figure 2 of the article [1]
#
# X. Li, H. C. Huang, A. Lozano and G. J. Foschini, "Reduced-complexity detection algorithms for systems using multi-element arrays," 
# Globecom '00 - IEEE. Global Telecommunications Conference. Conference Record (Cat. No.00CH37137), San Francisco, CA, USA, 2000, pp. 1072-1076 vol.2, 


# Parameters
N_test = 10000  # increase the value for smoothing the ber
N = 400
N_r, N_t = 2, 2
M = 4
alphabet = get_alphabet("PSK", M)
M = len(alphabet)

# construct chain
chain = Sequential([SymbolGenerator(M),
                    Recorder(name="data_tx"),
                    SymbolMapper(alphabet),
                    FlatFadingRayleighChannel(N_r=N_r, N_t=N_t, noise_unit="sigma2", name="channel"),
                    ])

# prepare MC trial
detector_names = ["ML", "ZF"]

# compute snr list from snr per bit in dB
snr_dB_list = np.arange(0, 45, 5)

# perform simulation
bler_data = np.zeros((len(snr_dB_list), len(detector_names)))
sig_power = N_t

for index_snr, snr_dB in enumerate(tqdm(snr_dB_list)):
    sigma2 = sig_power*(10**(-snr_dB/10))
    chain["channel"].noise_value = sigma2

    for trial in range(N_test):
   
        # new channel realization
        chain["channel"].channel_matrix_rvs()
        H = chain["channel"].H

        # generate data
        Y = chain((N_t, N))
        S_ref = chain["data_tx"].get_data()

        # test detector
        for index, detector_name in enumerate(detector_names):

            match detector_name:
                case "ML":
                    detector = MaximumLikelihoodDetector(alphabet=alphabet, H=H)
                case "ZF":
                    detector = ZeroForcingDetector(alphabet=alphabet, H=H)
            
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
