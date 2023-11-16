import numpy as np
import matplotlib.pyplot as plt

from comnumpy import Sequential
from channels.noise import AWGN
from channels.impairment import IQ_imbalance
from dsp.optical import Blind_Phase_Compensation
from dsp.modem import get_alphabet, Modulator
from analysers.scope import IQ_Scope



# parameters
type, M = "QAM", 16
N = 5000
alphabet = get_alphabet(type, M)
theta = np.random.randn()

channel = Sequential([
            Modulator(alphabet),
            IQ_imbalance(np.exp(1j*theta), 0),
            AWGN(0.01),
            IQ_Scope(),
            Blind_Phase_Compensation(alphabet),
            IQ_Scope()
            ])

# simulate communication
s = np.random.randint(0, high=M, size=N)
y = channel(s)
plt.show()

