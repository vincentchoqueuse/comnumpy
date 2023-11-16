import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import Sequential
from comnumpy.dsp.core import Processor 
from comnumpy.dsp.modem import get_alphabet, Modulator, Demodulator
from comnumpy.channels.noise import AWGN
from comnumpy.channels.linear import FIR_Channel
from comnumpy.analysers.recorder import Recorder
from comnumpy.dsp.utils import Serial_2_Parallel, Parallel_2_Serial
from comnumpy.dsp.ofdm import FFT_Processor, IFFT_Processor, Cyclic_Prefix_Adder, Cyclic_Prefix_Remover, Carrier_Extractor, Frequency_Domain_Equalizer


class Equalizer(Processor):

    def __init__(self, h, NFFT, name="equalizer"):
        self.h = h
        self.NFFT = NFFT
        self.name = name

    def forward(self, X):
        H = np.fft.fft(h, self.NFFT)
        H_inv = 1/ np.fft.fftshift(H)
        Y = H_inv[:, np.newaxis] * X  [:, np.newaxis]
        return Y


M = 16
nb_carrier_data = 64
N = nb_carrier_data*1000
modulation = "QAM"
order = 16
h = [0.9, 0.3+0.1j, -0.05+0.01j] # FIR channel
N_cp = len(h) # size of the cyclic prefix
sigma2 = 0.01
alphabet = get_alphabet(modulation, order)

chain = Sequential(
    [
        Modulator(alphabet),
        Recorder(name="tx_recorder"),
        Serial_2_Parallel(nb_carrier_data),
        IFFT_Processor(),
        Cyclic_Prefix_Adder(N_cp),
        Parallel_2_Serial(),
        FIR_Channel(h, name="channel"),
        AWGN(sigma2),
        Recorder(name="rx_recorder"),
        Serial_2_Parallel(nb_carrier_data+N_cp),
        Cyclic_Prefix_Remover(N_cp),
        FFT_Processor(),
        Recorder(name="rx_recorder2"),
        Equalizer(h, nb_carrier_data),
        Parallel_2_Serial(),
        Recorder(name="rx_recorder3"),
        Demodulator(alphabet)
    ]
)

x = np.random.randint(0, M, size=N)
y = chain(x)

for name in ["tx_recorder", "rx_recorder"]:
    data = chain[name].get_data()
    plt.figure()
    plt.plot(np.real(data), np.imag(data),'*')
    plt.title(name)


# plot channel transfer function
w, Hjw = chain["channel"].freqresp(512)
plt.figure()
plt.plot(w, np.abs(Hjw))
plt.xlabel("w [rad/sample]")
plt.ylabel("$|H(j\omega)|$")

# plot data after fft
data = chain["rx_recorder2"].get_data()
for channel in [0, 8, 19, 26]:
    name = "channel_{}".format(channel)
    data_channel = data[channel,:]
    plt.figure()
    plt.plot(np.real(data_channel), np.imag(data_channel),'*')
    plt.title(name)

# plot data after equalization 
data = chain["rx_recorder3"].get_data()
plt.figure()
plt.plot(np.real(data), np.imag(data),'*')
plt.title(name)

plt.show()