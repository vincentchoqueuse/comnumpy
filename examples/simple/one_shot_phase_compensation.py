import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential, Recorder, Scope
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.impairments import IQImbalance, CFO
from comnumpy.core.channels import AWGN
from comnumpy.core.processors import Amplifier
from comnumpy.core.compensators import BlindIQCompensator, BlindCFOCompensator, TrainedBasedPhaseCompensator, BlindPhaseCompensation
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

# add a recorder to use transmitted data during phase correction
signal_recorder_tx = Recorder(name="recorder tx")

chain = Sequential([
            SymbolGenerator(M),
            Recorder(name="data_tx"),
            SymbolMapper(alphabet),
            signal_recorder_tx,
            Amplifier(amplifier_param),
            AWGN(value=sigma2),
            Scope(num=1, scope_type="iq", title="received data"),
            BlindPhaseCompensation(alphabet, name="phase_compensation"),
            Scope(num=4, scope_type="iq", title="after phase correction"),
            SymbolDemapper(alphabet)
            ])

# simulate communication
y = chain(N)

# print phase estimation
estimated_phase = chain["phase_compensation"].theta
print(f"true phase: {true_phase}")
print(f"compensation phase: {estimated_phase}")

# compute metric
data_tx = chain["data_tx"].get_data()
ser_after = compute_ser(data_tx, y)

# print metric and plot
print(f"after: SER={ser_after}")

plt.show()
