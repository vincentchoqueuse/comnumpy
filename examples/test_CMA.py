import numpy as np
from scipy import signal
import sys

sys.path.insert(0, '../src')

from core import Sequential, MIMO_Wrapper
from dsp.modem import get_alphabet, Modulator
from dsp.frontend import Upsampler, SRRC_filter
from dsp.optical import CMA
from channels.mimo import Selective_Channel, AWGN
from analysers.scope import IQ_Scope
import matplotlib.pyplot as plt

def generate_H_list(N_tap, N_t, coef=0.3):
    """
    Generate a list of MIMO channel (one matrix by tap)
    """

    I_mat = np.eye(N_t)
    H_list = []
    for indice in range(N_tap):
        B = np.random.randn(N_t, N_t) + 1j*np.random.randn(N_t, N_t)
        H_temp = (coef**indice)*(I_mat + B)
        H_list.append(H_temp)
    
    return H_list


# parameters
type, M = "PSK", 4
N_t = 2
N_tap = 8
N = 100000
oversampling = 2
rolloff = 0.2
sigma2 = 0.001
nlim = [N-2000, N]

alphabet = get_alphabet(type, M)

# generate siso stream at transmitter side
tx_siso = Sequential([Modulator(alphabet), Upsampler(oversampling), SRRC_filter(rolloff, oversampling)])

# generate random channel
H_list = generate_H_list(N_tap, N_t)

# create chain
channel = Sequential([
            MIMO_Wrapper(tx_siso, N_t),
            Selective_Channel(H_list),
            AWGN(0.001),
            MIMO_Wrapper(IQ_Scope(num="scope1", nlim=nlim), N_t),
            CMA(10, alphabet, mu=0.0001, oversampling=oversampling),
            MIMO_Wrapper(IQ_Scope(num="scope2", nlim=nlim), N_t),
            ])

# simulate communication
S = np.random.randint(0, high=M, size=(2,N))
Y = channel(S)
plt.show()