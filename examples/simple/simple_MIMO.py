from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
from comnumpy import Sequential
from comnumpy.dsp.modem import get_alphabet, Modulator
from comnumpy.channels.mimo import Random_Channel, AWGN


# This script simulate a simple MIMO Communication

# Parameters
N = 1000
N_r, N_t = 5, 4 
k = 2
M = 2**k
alphabet = get_alphabet("PSK", M)
M = len(alphabet)

# construct chain
model = Sequential([Modulator(alphabet), 
                    Random_Channel(N_r, N_t, norm=True, name="channel"), 
                    AWGN(0.01)
                    ])

S = randint(0, high=M, size=(N_t, N))
Y = model(S)

for n_r in range(N_r):
    plt.figure()
    plt.plot(np.real(Y[n_r,:]), np.imag(Y[n_r,:]), "*")
    plt.title("Receiver {}".format(n_r+1))

plt.show()