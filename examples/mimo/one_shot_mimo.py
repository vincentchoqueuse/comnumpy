import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from tqdm import tqdm
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import FlatMIMOChannel, AWGN
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import MaximumLikelihoodDetector, LinearDetector, OrderedSuccessiveInterferenceCancellationDetector

img_dir = "../../docs/examples/img/"

# Parameters
N = 1000
N_r, N_t = 3, 2
M = 4
alphabet = get_alphabet("PSK", M)
M = len(alphabet)
sigma2 = 0.1
H = rayleigh_channel(N_r, N_t)

# construct chain
chain = Sequential([SymbolGenerator(M),
                    Recorder(name="data_tx"),
                    SymbolMapper(alphabet),
                    FlatMIMOChannel(H, name="channel"),
                    AWGN(sigma2, name="noise")
                    ])
Y = chain((N_t, N))

# extract signals
S_tx = chain["data_tx"].get_data()

# Figure 1: received signal
fig1, axes1 = plt.subplots(nrows=1, ncols=N_r, figsize=(4 * N_r, 4))
for num_channel in range(N_r):
    y = Y[num_channel, :]
    ax = axes1[num_channel]
    ax.plot(np.real(y), np.imag(y), ".")
    ax.set_title(f"Received signal (antenna {num_channel+1})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig1.png")

# ZF equalization
H_inv = linalg.pinv(H)
X_est = np.matmul(H_inv, Y)

# Figure 2: estimated signal
fig2, axes2 = plt.subplots(nrows=1, ncols=N_t, figsize=(4 * N_t, 4))
for num_channel in range(N_t):
    x_est = X_est[num_channel, :]
    ax = axes2[num_channel]
    ax.plot(np.real(x_est), np.imag(x_est), ".")
    ax.set_title(f"Estimated signal (antenna {num_channel+1})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig2.png")

# evaluate the BER for several detectors
detector_list = [
    LinearDetector(alphabet, H, method="zf", name="ZF"),
    LinearDetector(alphabet, H, sigma2=sigma2, method="mmse", name="MMSE"),
    OrderedSuccessiveInterferenceCancellationDetector(alphabet, "sinr", H, sigma2=sigma2, name="OSIC"),
    MaximumLikelihoodDetector(alphabet, H, name="ML")
    ]
for detector in detector_list:
    S_est = detector(Y)
    ser = compute_ser(S_tx, S_est)
    name = detector.name
    print(f"* detector {name}: ser={ser}")


# perform monte carlo simulation
snr_dB_list = np.arange(0, 20, 2)
N_test = 1000
ser_data = np.zeros((len(snr_dB_list), len(detector_list)))

for index_snr, snr_dB in enumerate(tqdm(snr_dB_list)):
    sigma2 = N_t * (10**(-snr_dB/10))
    chain["noise"].value = sigma2

    # update sigma2 for the MMSE and OSIC detector
    detector_list[1].sigma2 = sigma2
    detector_list[2].sigma2 = sigma2

    for trial in range(N_test):
   
        # new channel realization
        H = rayleigh_channel(N_r, N_t)
        chain["channel"].H = H

        # generate data
        Y = chain((N_t, N))
        S_tx = chain["data_tx"].get_data()

        # test detector
        for index, detector in enumerate(detector_list):

            # update channel information
            detector.H = H
            # perform detection
            S_est = detector(Y)
            # evaluate metrics
            ser_data[index_snr, index] += compute_ser(S_tx, S_est)

    ser_data[index_snr, :] /= N_test

# plot figures
plt.figure()
for index, detector in enumerate(detector_list):
    plt.semilogy(snr_dB_list, ser_data[:, index], label=detector.name)
plt.ylabel("SER")

plt.xlabel("SNR (dB)")
plt.xlim([0, 20])
plt.ylim([10**-3, 1])
plt.legend()
plt.grid(True)
plt.title("Performance Comparison of several MIMO detectors")
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig3.png")
plt.show()
