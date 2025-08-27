import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import OrderedSuccessiveInterferenceCancellationDetector

# This script shows the performance of several OSIC MIMO detectors. It reproduces the figure 11.3 of the book [1]
# [1] Cho, Yong Soo, et al. MIMO-OFDM wireless communications with MATLAB. John Wiley & Sons, 2010.

# Parameters
N_test = 5000  # increase the value for smoothing the ber
N = 100
N_r, N_t = 4, 4
M = 16
alphabet = get_alphabet("QAM", M)
M = len(alphabet)

H = rayleigh_channel(N_r=N_r, N_t=N_t)

# construct chain
chain = Sequential([SymbolGenerator(M),
                    Recorder(name="data_tx"),
                    SymbolMapper(alphabet),
                    FlatMIMOChannel(H, name="channel"),
                    AWGN(0, unit="sigma2", name="noise"),
                    ])

# prepare MC trial
detector_names = ["colnorm", "snr", "sinr"]

# compute snr list from snr per bit in dB
snr_dB_list = np.arange(0, 45, 5)

# perform simulation
ber_data = np.zeros((len(snr_dB_list), len(detector_names)))

for index_snr, snr_dB in enumerate(tqdm(snr_dB_list)):
    sigma2 = N_t*(10**(-snr_dB/10))
    chain["noise"].value = sigma2

    for trial in range(N_test):
   
        # new channel realization
        H = rayleigh_channel(N_r=N_r, N_t=N_t)
        chain["channel"].H = H

        # generate data
        Y = chain((N_t, N))
        S_ref = chain["data_tx"].get_data()

        # test detector
        for index, detector_name in enumerate(detector_names):
            # create detector
            detector = OrderedSuccessiveInterferenceCancellationDetector(alphabet=alphabet, osic_type=detector_name, H=H, sigma2=sigma2)
            # perform detection
            S_est = detector(Y)
            # evaluate metrics
            ber_data[index_snr, index] += compute_ber(S_ref, S_est, width=int(np.log2(M)))

    ber_data[index_snr, :] /= N_test

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
