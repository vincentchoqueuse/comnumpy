import numpy as np
import matplotlib.pyplot as plt
from comnumpy import Sequential
from comnumpy.generators.data import Symbol_Generator
from comnumpy.analysers.recorder import Recorder
from comnumpy.channels.noise import AWGN, awgn_theo
from comnumpy.dsp.modem import get_alphabet, Modulator, Demodulator
from comnumpy.metrics.detection import compute_ser


# parameters
M = 16
N = 100000
modulation = "QAM"
alphabet = get_alphabet(modulation, M)
snr_dB_list = np.arange(0, 22)

# keep objects to easily extract symbols and change noise variance
recorder_data_tx = Recorder()
channel = AWGN() 

# create chain
chain = Sequential(
    [
    Symbol_Generator(alphabet=np.arange(M)),
    recorder_data_tx,
    Modulator(alphabet),
    channel,
    Demodulator(alphabet),
    ]
)


# compute SER metric

sigma2_list = 1/(10**(snr_dB_list/10))
snr_per_bit = (10**(snr_dB_list/10))/np.log2(M)
ser_theo_array = awgn_theo(modulation, M, snr_per_bit, "ser")

# perform monte Carlo simulation
ser_array = np.zeros(len(sigma2_list))

for index, sigma2 in enumerate(sigma2_list):

    # calibrate channel
    channel.sigma2 = sigma2

    # run chain
    y = chain(N)

    # evaluate symbol error rate
    data_tx = recorder_data_tx.get_data()
    ser = compute_ser(data_tx, y)
    ser_array[index] = ser

    # save ser
    print("SNR={}: ser={} (theo)".format(snr_dB_list[index], ser_theo_array[index]))
    print("SNR={}: ser={} (exp)".format(snr_dB_list[index], ser))

plt.semilogy(snr_dB_list , ser_array, label="exp")
plt.semilogy(snr_dB_list , ser_theo_array, "--", label="theo") 
plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.title("SER performance (modulation={}-{})".format(M, modulation))
plt.legend()
plt.grid()
plt.show()