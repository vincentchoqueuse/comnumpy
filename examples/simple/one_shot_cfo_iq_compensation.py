import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential, Recorder, Scope
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.impairments import IQImbalance, CFO
from comnumpy.core.channels import AWGN
from comnumpy.core.compensators import BlindIQCompensator, BlindCFOCompensator, TrainedBasedPhaseCompensator
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser


# parameters
type, M = "QAM", 16
N = 5000
alphabet = get_alphabet(type, M)

# generate random IQ imbalance
iq_params = np.array([1, 0]) + 0.2*(np.random.randn(2) + 1j*np.random.randn(2))

# add a recorder to use transmitted data during phase correction
signal_recorder_tx = Recorder(name="recorder tx")

chain = Sequential([
            SymbolGenerator(M),
            Recorder(name="data_tx"),
            SymbolMapper(alphabet),
            signal_recorder_tx,
            CFO(0.001),
            IQImbalance(iq_params[0], iq_params[1]),
            AWGN(value=0.005),
            Scope(num=1, scope_type="iq", title="received signal"),
            BlindIQCompensator(),
            Scope(num=2, scope_type="iq", title="after GSOP"),
            BlindCFOCompensator(save_history=True, name="cfo_comp"),
            Scope(num=3, scope_type="iq", title="after GSOP+CFO comp"),
            TrainedBasedPhaseCompensator(target_data=signal_recorder_tx),
            Scope(num=4, scope_type="iq", title="after GSOP + CFO comp + phase correction"),
            SymbolDemapper(alphabet)
            ])

# simulate communication
y = chain(N)

# compute metric
data_tx = chain["data_tx"].get_data()
ser_after = compute_ser(data_tx, y)

# print metric and plot
print(f"after: SER={ser_after}")

# show evolution of the angular frequency estimate
w0_history = chain["cfo_comp"].history
plt.figure()
plt.plot(w0_history)
plt.xlabel("number of iteration")
plt.title("w0 estimate")
plt.show()
