import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential, plot_iq
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN
from comnumpy.core.processors import Amplifier
from comnumpy.core.compensators import BlindPhaseCompensation
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser


# parameters
type, M = "QAM", 16
N = 5000
alphabet = get_alphabet(type, M)
sigma2 = 0.01

# generate random IQ imbalance
true_phase = 0.1
amplifier_param = np.exp(1j*0.23)

chain = Sequential([
            SymbolGenerator(M, name="data_tx"),
            SymbolMapper(alphabet),
            Amplifier(amplifier_param),
            AWGN(sigma2=sigma2, name="awgn"),
            BlindPhaseCompensation(alphabet, name="phase_compensation"),
            SymbolDemapper(alphabet)
            ], taps=["data_tx", "awgn", "phase_compensation"])

# simulate communication
y = chain(N)

# print phase estimation
estimated_phase = chain["phase_compensation"].theta_
print(f"true phase: {true_phase}")
print(f"compensation phase: {estimated_phase}")

# compute metric
data_tx = chain.tap("data_tx")
ser_after = compute_ser(data_tx, y)

# print metric and plot
print(f"after: SER={ser_after}")

plot_iq(chain.tap("awgn"), title="received data")
plot_iq(chain.tap("phase_compensation"), title="after phase correction")
plt.show()
