import numpy as np

from comnumpy import Sequential
from comnumpy.generators.data import Symbol_Generator
from comnumpy.analysers.recorder import Recorder
from comnumpy.channels.noise import AWGN
from comnumpy.channels.linear import FIR_Channel
from comnumpy.dsp.modem import get_alphabet, Modulator, Demodulator
from comnumpy.dsp.utils import Serial_2_Parallel, Parallel_2_Serial
from comnumpy.dsp.ofdm import get_standard_carrier_allocation, Carrier_Allocator, FFT_Processor, IFFT_Processor, Cyclic_Prefix_Adder, Cyclic_Prefix_Remover, Carrier_Extractor, Frequency_Domain_Equalizer
from comnumpy.metrics.recorder import Metric_Recorder
from comnumpy.metrics.detection import compute_ser
from comnumpy.metrics.ofdm import compute_PAPR

# parameters
M = 16
N_h = 5
N_cp = 10
N = 1000
sigma2 = 0.01

carrier_type = get_standard_carrier_allocation("802.11ah_128")
nb_carriers = len(carrier_type)
nb_carrier_data = len(np.where(carrier_type==1)[0])
nb_carrier_pilots = len(np.where(carrier_type==2)[0])
nb_pilots = nb_carrier_pilots

h = 0.2*(np.random.randn(N_h) + 1j*np.random.randn(N_h))
h[0] = 1
pilots = 10*np.ones(nb_pilots)
alphabet = get_alphabet("QAM", M)

# create recorder
recorder_data_tx = Recorder()
recorder_data_rx = Recorder()
recorder_metric = Metric_Recorder(compute_PAPR, params={"unit": "dB"})

# create sequential
transmitter = Sequential(
    [
        Symbol_Generator(alphabet=np.arange(M)),
        recorder_data_tx,
        Modulator(alphabet),
        Serial_2_Parallel(nb_carrier_data),
        Carrier_Allocator(carrier_type, pilots=pilots),
        IFFT_Processor(),
        recorder_metric,
        Cyclic_Prefix_Adder(N_cp),
        Parallel_2_Serial()
    ]
)

channel = Sequential(
    [
        FIR_Channel(h),
        AWGN(sigma2=sigma2)
    ]
)

receiver = Sequential(
    [
        Serial_2_Parallel(nb_carriers+N_cp),
        Cyclic_Prefix_Remover(N_cp),
        FFT_Processor(),
        Frequency_Domain_Equalizer(h),
        Carrier_Extractor(carrier_type),
        Parallel_2_Serial(),
        Demodulator(alphabet),
        recorder_data_rx
    ]
)

chain = Sequential(
    [
    transmitter,
    channel,
    receiver
    ],
    name="chain"
)

# run chain
y = chain(N)

# evaluate symbol error rate
data_tx = recorder_data_tx.get_data()
data_rx = recorder_data_rx.get_data()
ser = compute_ser(data_tx, data_rx)

chain.to_json("chain.json")

print("ser={}".format(ser))