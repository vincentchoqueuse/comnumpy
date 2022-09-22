import numpy as np
import sys

sys.path.insert(0, '../src')

from core import Sequential
from dsp.modem import get_alphabet, Modulator, Demodulator
from dsp.optical import FIR_CD_Compensator, LS_FIR_CD_Compensator
from dsp.frontend import Upsampler, Downsampler, SRRC_filter
from channels.optical import CD
from channels.noise import awgn_theo, AWGN
from metrics.functional import compute_ber
import matplotlib.pyplot as plt


# This script reproduces the figures 3 and 4 in the paper Optimal Least-Squares FIR Digital Filters
#for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers
# see: https://www.researchgate.net/profile/Amir-Eghbali-2/publication/260642085_Optimal_Least-Squares_FIR_Digital_Filters_for_Compensation_of_Chromatic_Dispersion_in_Digital_Coherent_Optical_Receivers/links/02e7e534adbd8b0d77000000/Optimal-Least-Squares-FIR-Digital-Filters-for-Compensation-of-Chromatic-Dispersion-in-Digital-Coherent-Optical-Receivers.pdf

system = 2
type = "QAM"
k_vect = [4, 6, 8]
N = 200000  # increase number for smoothing MC curves
N_h = 1000  # number of delay samples for the SRRC filter 
oversampling = 2  # oversampling factor
rolloff = 0.25  # rolloff factor
D = 17*(10**-6)
lamb=1.553*(10**-6)
SNR_vect = range(10, 31, 1) 


# system selection (see Table 1)
if system == 1:
    z = 4000*(10**3)
    F_s = 21.4 * (10**9)
    
if system == 2:
    z = 1000*(10**3)
    F_s = 80 * (10**9)


comp_ref = FIR_CD_Compensator(z, D=D, lamb=lamb, F_s=F_s)
N_filter = comp_ref.get_N()  # compute length of filter
print("filter length: {}".format(N_filter))
total_delay = int(2*N_h*oversampling + np.floor(N_filter/2)) # total delay of the filters (need to be compensated before sampling)

plt.figure()

noise = AWGN(0)

for k in k_vect:

    M = 2**k
    
    print("Modulation: {}{}".format(type,M))
    alphabet = get_alphabet(type, M)
    epsilon_s = np.sum(np.abs(alphabet)**2)/len(alphabet)
    epsilon_b = epsilon_s/k

    # create your chain
    channel = Sequential([
                Modulator(alphabet),
                Upsampler(oversampling),
                SRRC_filter(rolloff, oversampling, N_h=N_h),
                CD(z, D=D, lamb=lamb, F_s=F_s),
                noise
                ])

    full_compensator1 = Sequential([
                FIR_CD_Compensator(z, D=D, lamb=lamb, F_s=F_s),
                SRRC_filter(rolloff, oversampling, N_h=N_h),
                Downsampler(oversampling, pre_delay=total_delay),
                Demodulator(alphabet)
                ])

    full_compensator2 = Sequential([
                LS_FIR_CD_Compensator(z, N_filter, D=D, lamb=lamb, F_s=F_s, w_vect=[-np.pi, np.pi]),
                SRRC_filter(rolloff, oversampling, N_h=N_h),
                Downsampler(oversampling, pre_delay=total_delay),
                Demodulator(alphabet)
                ])

    compensator_list = [full_compensator1, full_compensator2]

    # monte carlo simulations
    ber_theo = []
    ber_exp = []
    ber_list_names = ["theoretical", "Savory", "LS"]
    ber_list = np.zeros((len(SNR_vect),3))

    for index_SNR, SNR_bitdB in enumerate(SNR_vect):
        snr_per_bit = 10 ** (SNR_bitdB/10)
        
        # compute theoretical ber
        ber_theo = awgn_theo(type, M, snr_per_bit, "bin")
        ber_list[index_SNR, 0] = ber_theo

        # perform MC simulations
        N0 = epsilon_b/snr_per_bit  #snr_bit = epsilon_b/N0 = (epsilon_s/log2(order))/N0 
        noise.sigma2 = N0  # change noise variance
        s = np.random.randint(0, high=M, size=N)
        y = channel(s)

        # get compensated symbols
        for index_comp, full_compensator in enumerate(compensator_list):

            s_est = full_compensator(y)
            s_trunc = s_est[:N]
            ber_exp_temp = compute_ber(s_trunc, s, width=k)
            ber_list[index_SNR, index_comp+1] = ber_exp_temp

        print("SNR={}dB/bit".format(SNR_bitdB))
        print("BER={}".format(ber_list[index_SNR, :]))

    # plot curves
    color_list = ["k-","r-","b-"]
    for index in range(3):
        name = "{} ({}{})".format(ber_list_names[index],type,M)
        plt.semilogy(SNR_vect, ber_list[:, index], color_list[index], label=name)


plt.ylabel("BER")
plt.xlabel("SNR per bit (dB)")
plt.ylim([10**-6,0.2])
plt.xlim([SNR_vect[0], SNR_vect[-1]])
plt.legend()
plt.show()


