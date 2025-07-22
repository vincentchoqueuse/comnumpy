import numpy as np
from scipy import signal
from comnumpy.optical.compensators import ChromaticDispersionFIRCompensator, ChromaticDispersionLSFIRCompensator
import matplotlib.pyplot as plt


# This script reproduces the figures 6 and 7 in the paper Optimal Least-Squares FIR Digital Filters
# for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers
# see: https://www.researchgate.net/profile/Amir-Eghbali-2/publication/260642085_Optimal_Least-Squares_FIR_Digital_Filters_for_Compensation_of_ChromaticDispersion_in_Digital_Coherent_Optical_Receivers/links/02e7e534adbd8b0d77000000/Optimal-Least-Squares-FIR-Digital-Filters-for-Compensation-of-Chromatic-Dispersion-in-Digital-Coherent-Optical-Receivers.pdf
z = 500.            # in km
fs = 21.4 * (10**9) # in Hz
D = 17              # in ps/nm/km
lamb = 1553         # in nm

comp_ref = ChromaticDispersionFIRCompensator(z, fs=fs)
N_filter = len(comp_ref.h)

compensator_list = [
    ChromaticDispersionFIRCompensator(z, fs=fs),
    ChromaticDispersionLSFIRCompensator(z, N_filter, fs=fs, w_vect=[-np.pi, np.pi])
]

fig, axs = plt.subplots(3, 1)
fig.suptitle("Fig6. Impulse responses")
markerfmt_list = ["b.", "rx", "k."]

for index, compensator in enumerate(compensator_list):
    h = compensator.h
    name = compensator.name
    bound = len(h)//2
    markerfmt = markerfmt_list[index]
    bound = np.floor(N_filter/2)
    n = np.arange(-bound, bound+1)

    axs[0].stem(n, np.real(h), markerfmt=markerfmt, label=name)
    axs[1].stem(n, np.imag(h), markerfmt=markerfmt, label=name)
    axs[2].stem(n, np.abs(h), markerfmt=markerfmt, label=name)
    axs[2].set_xlabel("n")

fig, axs = plt.subplots(2, 1)
fig.suptitle("Fig7. Magnitude response and group delay")
color_list = ["b-", "r--", "k:"]
for index, compensator in enumerate(compensator_list):
    h = compensator.h
    name = compensator.name
    color = color_list[index]
    w, H = signal.freqz(h)
    w2, gd = signal.group_delay((h, 1))

    axs[0].plot(w, np.abs(H), color, label=name)
    axs[1].plot(w2, gd, color, label=name)

axs[1].set_xlabel("w.T")
axs[0].grid()
axs[1].grid()
plt.show()
