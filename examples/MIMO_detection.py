from numpy.random import randn, randint
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, '../src')

from dsp.modem import get_alphabet, Modulator, Demodulator
from core import Sequential
from channels.mimo import Random_Channel, AWGN
from dsp.mimo import ML, ZF, AMP, OAMP
from metrics.functional import compute_ser, compute_ber

# This script shows the performance of several MIMO detectors (ML, ZF, AMP and OAMP)

# Parameters
N_test = 1000 # increase the value for smoothing the ber
N = 1
N_r, N_t = 5, 4 
k = 2
M = 2**k
alphabet = get_alphabet("PSK", M)
M = len(alphabet)

# construct chain
modulator = Modulator(alphabet)
channel = Random_Channel(N_r, N_t, norm=True)
noise = AWGN(0)
model = Sequential([modulator, 
                    channel, 
                    noise])

# prepare MC trial
detector_names = ["ML", "ZF", "AMP", "OAMP"]
Nb_techniques = len(detector_names)
SNR_vect = np.arange(0,30,5)
ber_data = np.zeros((len(SNR_vect),Nb_techniques))
ser_data = np.zeros((len(SNR_vect),Nb_techniques))

for index_snr, SNR_dB in enumerate(SNR_vect):

    ser_temp = np.zeros((N_test, Nb_techniques))
    ber_temp = np.zeros((N_test, Nb_techniques))
    SNR = 10**(SNR_dB/20)
    
    for trial in range(N_test):
        
        channel.rvs()  # new channel realization
        H = channel.H 
        H_H = np.transpose(np.conjugate(H))
        sigma2 = np.real(np.trace(np.matmul(H, H_H)) / (SNR * N_r))
        noise.sigma2 = sigma2

        # generate data
        S = randint(0, high=M, size=(N_t,N))
        Y = model(S)

        detector_list = [
                ML(H, alphabet),
                ZF(H, alphabet),
                AMP(H, alphabet, sigma2=sigma2),
                OAMP(H, alphabet, sigma2=sigma2)
        ]

        for index, detector in enumerate(detector_list):
            S_est = detector(Y)
            ser = compute_ser(S, S_est)
            ber = compute_ber(S, S_est, k)
            ser_temp[trial,index] = ser
            ber_temp[trial,index] = ber

    ser = np.mean(ser_temp, axis=0)
    ber = np.mean(ber_temp, axis=0)
    ser_data[index_snr,:] = ser
    ber_data[index_snr,:] = ber
    print("ser={} - ber={}".format(ser, ber))

# plot figures
for index in range(Nb_techniques):
    plt.semilogy(SNR_vect, ser_data[:,index],label = detector_names[index])
plt.ylabel("ser")
plt.legend()

plt.figure()
for index in range(Nb_techniques):
    plt.semilogy(SNR_vect, ser_data[:,index],label = detector_names[index])
plt.ylabel("ber")
plt.xlabel("SNR (dB)")
plt.legend()
plt.show()