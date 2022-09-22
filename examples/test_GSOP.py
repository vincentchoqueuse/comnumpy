import numpy as np
from scipy import signal
import sys

sys.path.insert(0, '../src')

from channels.noise import AWGN
from channels.impairment import IQ_imbalance
from dsp.optical import GSOP
from dsp.modem import get_alphabet, Modulator
from analysers.scope import IQ_Scope
from analysers.logger import IQ_Reporter
from core import Sequential
import matplotlib.pyplot as plt


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

