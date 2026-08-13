import numpy as np
import matplotlib.pyplot as plt

from comnumpy import style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN
from comnumpy.core.processors import Amplifier
from comnumpy.core.compensators import BlindPhaseCompensation
from comnumpy.core.utils import Constellation
from comnumpy.core.metrics import compute_ser

style.use()


# parameters
constellation = Constellation("QAM", 16)
N = 5000
sigma2 = 0.01

# generate random IQ imbalance
true_phase = 0.1
amplifier_param = np.exp(1j*0.23)

chain = Sequential([
            SymbolGenerator(constellation.order, name="data_tx"),
            SymbolMapper(constellation),
            Amplifier(amplifier_param),
            AWGN(sigma2=sigma2, name="awgn"),
            BlindPhaseCompensation(constellation, name="phase_compensation"),
            SymbolDemapper(constellation)
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

for tap, name in [("awgn", "received data"),
                  ("phase_compensation", "after phase correction")]:
    symbols = chain.tap(tap)
    _, ax = plt.subplots()
    ax.plot(np.real(symbols), np.imag(symbols), ".")
    ax.set_title(name)
    style.apply(ax, "iq")
plt.show()
