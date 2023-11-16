import numpy as np
import matplotlib.pyplot as plt

from comnumpy import Sequential
from comnumpy.channels.noise import AWGN
from comnumpy.channels.impairment import IQ_imbalance
from comnumpy.dsp.optical import GSOP
from comnumpy.dsp.modem import get_alphabet, Modulator
from comnumpy.analysers.scope import IQ_Scope
from comnumpy.analysers.logger import IQ_Reporter


# parameters
type, M = "QAM", 16
N = 5000
alphabet = get_alphabet(type, M)

channel = Sequential([
            Modulator(alphabet),
            IQ_imbalance(0.2+0.8j, -0.3+0.1j),
            AWGN(0.001),
            IQ_Scope(num=1, axis="equal"),
            IQ_Reporter(name="before GSOP"),
            GSOP(),
            IQ_Scope(num=2, axis="equal"),
            IQ_Reporter(name="after GSOP"),
            ])

# simulate communication
s = np.random.randint(0, high=M, size=N)
y = channel(s)

plt.show()

