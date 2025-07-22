import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.core.utils import get_alphabet
from comnumpy.core.channels import AWGN
from comnumpy.core.metrics import compute_ber, compute_metric_awgn_theo
from comnumpy.core.filters import SRRCFilter
from comnumpy.optical.compensators import ChromaticDispersionFIRCompensator, ChromaticDispersionLSFIRCompensator
from comnumpy.optical.channels import ChromaticDispersion

# This script reproduces the figures 3 and 4 in the paper Optimal Least-Squares FIR Digital Filters
# for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers
# see: https://www.researchgate.net/profile/Amir-Eghbali-2/publication/260642085_Optimal_Least-Squares_FIR_Digital_Filters_for_Compensation_of_ChromaticDispersion_in_Digital_Coherent_Optical_Receivers/links/02e7e534adbd8b0d77000000/Optimal-Least-Squares-FIR-Digital-Filters-for-Compensation-of-Chromatic-Dispersion-in-Digital-Coherent-Optical-Receivers.pdf

system = 1
type = "QAM"
N = 200000  # increase number for smoothing MC curves
N_h = 1000  # number of delay samples for the SRRC filter
oversampling = 2  # oversampling factor
rolloff = 0.25  # rolloff factor

# system selection (see Table 1)
if system == 1:
    z = 4000  # km
    fs = 21.4*(10**9)

if system == 2:
    z = 1000  # km
    fs = 80*(10**9)

comp_ref = ChromaticDispersionFIRCompensator(z, fs=fs)
K = comp_ref.K  # compute length of filter
N_filter = len(comp_ref.h)  # compute length of filter
print("K: {}".format(K))
print("filter length: {}".format(N_filter))
total_delay = int(2*N_h*oversampling + np.floor(N_filter/2))  # total delay of the filters (need to be compensated before sampling)

plt.figure()

# create your chain and compensator list
chain = Sequential([
            SymbolGenerator(M=4, name="generator"),
            Recorder(name="data_tx"),
            SymbolMapper([], name="mapper"),
            Upsampler(oversampling),
            SRRCFilter(rolloff, oversampling, N_h=N_h),
            ChromaticDispersion(z, fs=fs),
            AWGN(value=0, unit="sigma2", name="noise")
            ])

full_compensator1 = Sequential([
            ChromaticDispersionFIRCompensator(z, fs=fs),
            SRRCFilter(rolloff, oversampling, N_h=N_h),
            Downsampler(oversampling, phase=total_delay),
            SymbolDemapper([], name="demapper")
            ])

full_compensator2 = Sequential([
            ChromaticDispersionLSFIRCompensator(z, N_filter, fs=fs, w_vect=[-np.pi, np.pi]),
            SRRCFilter(rolloff, oversampling, N_h=N_h),
            Downsampler(oversampling, phase=total_delay),
            SymbolDemapper([], name="demapper")
            ])

compensator_list = [full_compensator1, full_compensator2]

# Monte carlo simulation
ber_list_names = ["theoretical", "Savory", "LS"]
SNR_vect = range(10, 31, 1)
k_vect = [4, 6, 8]

for k in k_vect:
    M = 2**k
    print("Modulation: {}{}".format(type, M))
    alphabet = get_alphabet(type, M)
    epsilon_s = np.sum(np.abs(alphabet)**2)/len(alphabet)
    epsilon_b = epsilon_s/k

    # update chain and compensator parameters
    chain["generator"].M = M
    chain["mapper"].alphabet = alphabet
    full_compensator1["demapper"].alphabet = alphabet
    full_compensator2["demapper"].alphabet = alphabet

    # perform simulation
    ber_theo = []
    ber_exp = []
    ber_list = np.zeros((len(SNR_vect), len(ber_list_names)))

    for index_SNR, SNR_bitdB in enumerate(tqdm(SNR_vect)):
        snr_per_bit = 10 ** (SNR_bitdB/10)

        # compute theoretical ber
        ber_list[index_SNR, 0] = compute_metric_awgn_theo(type, M, snr_per_bit, "bin")

        # perform MC simulations
        N0 = epsilon_b/snr_per_bit  # snr_bit = epsilon_b/N0 = (epsilon_s/log2(order))/N0
        chain["noise"].value = N0  # change noise variance
        y = chain(N)

        # evaluate metric
        s = chain["data_tx"].get_data()

        for index_comp, full_compensator in enumerate(compensator_list):
            s_est = full_compensator(y)
            ber_exp_temp = compute_ber(s_est, s[:len(s_est)], width=k)
            ber_list[index_SNR, index_comp+1] = ber_exp_temp

    # plot curves
    color_list = ["k-", "r-", "b-"]
    for index in range(3):
        name = "{} ({}{})".format(ber_list_names[index], type, M)
        plt.semilogy(SNR_vect, ber_list[:, index], color_list[index], label=name)


plt.ylabel("BER")
plt.xlabel("SNR per bit (dB)")
plt.ylim([10**-6, 0.2])
plt.xlim([SNR_vect[0], SNR_vect[-1]])
plt.legend()
plt.grid()
plt.show()
