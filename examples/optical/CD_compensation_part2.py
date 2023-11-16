import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from comnumpy.dsp.optical import FIR_CD_Compensator, LS_FIR_CD_Compensator



# This script reproduces the figures 6 and 7 in the paper Optimal Least-Squares FIR Digital Filters
#for Compensation of Chromatic Dispersion in Digital Coherent Optical Receivers
# see: https://www.researchgate.net/profile/Amir-Eghbali-2/publication/260642085_Optimal_Least-Squares_FIR_Digital_Filters_for_Compensation_of_Chromatic_Dispersion_in_Digital_Coherent_Optical_Receivers/links/02e7e534adbd8b0d77000000/Optimal-Least-Squares-FIR-Digital-Filters-for-Compensation-of-Chromatic-Dispersion-in-Digital-Coherent-Optical-Receivers.pdf


z = 500*10**3
F_s = 21.4 * (10**9)
D = 17 * 10**-6 # s/m/m
lamb = 1553*(10**-9) # m

comp_ref = FIR_CD_Compensator(z, D=D, lamb=lamb, F_s=F_s)
N_filter = comp_ref.get_N()

compensator_list = [
    FIR_CD_Compensator(z, D=D, lamb=lamb, F_s=F_s),
    LS_FIR_CD_Compensator(z, N_filter, D=D, lamb=lamb, F_s=F_s, w_vect=[-np.pi, np.pi])
]

fig, axs = plt.subplots(3, 1)
fig.suptitle("Fig6. Impulse responses")
markerfmt_list= ["b.","rx","k."]
for index, compensator in enumerate(compensator_list):
    h = compensator.h()
    print(h)
    name = compensator.name
    bound = compensator.get_delay()
    markerfmt = markerfmt_list[index]
    bound = np.floor(N_filter/2)
    n = np.arange(-bound,bound+1)

    axs[0].stem(n, np.real(h), markerfmt=markerfmt, label = name)
    axs[1].stem(n, np.imag(h), markerfmt=markerfmt, label = name)
    axs[2].stem(n, np.abs(h), markerfmt=markerfmt, label = name)
    axs[2].set_xlabel("n")

fig, axs = plt.subplots(2, 1)
fig.suptitle("Fig7. Magnitude response and group delay")
color_list= ["b-","r--","k:"]
for index, compensator in enumerate(compensator_list):
    h = compensator.h()
    name = compensator.name
    color = color_list[index]
    w, H = signal.freqz(h)
    w2, gd = signal.group_delay((h, 1))

    axs[0].plot(w, np.abs(H), color, label = name)
    axs[1].plot(w2, gd, color, label = name)

axs[1].set_xlabel("w.T")
plt.show()