import numpy as np
import matplotlib.pyplot as plt

from comnumpy import style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.impairments import IQImbalance, CFO
from comnumpy.core.channels import AWGN
from comnumpy.core.compensators import BlindIQCompensator, BlindCFOCompensator, DataAidedPhaseCompensator
from comnumpy.core.utils import Constellation
from comnumpy.core.metrics import compute_ser


# parameters
constellation = Constellation("QAM", 16)
N = 5000

# generate random IQ imbalance
iq_params = np.array([1, 0]) + 0.2*(np.random.randn(2) + 1j*np.random.randn(2))

# One chain describing the whole system. The phase compensator is data
# aided: `wiring` feeds it the transmitted symbols of the current run,
# produced upstream by the mapper -- so the reference is never stale.
# `taps` extracts the signals we want to look at afterwards.
chain = Sequential([
            SymbolGenerator(constellation.order, name="data_tx"),
            SymbolMapper(constellation, name="signal_tx"),
            CFO(0.001),
            IQImbalance(iq_params[0], iq_params[1]),
            AWGN(sigma2=0.005, name="awgn"),
            BlindIQCompensator(name="gsop"),
            BlindCFOCompensator(save_history=True, name="cfo_comp"),
            DataAidedPhaseCompensator(reference=np.zeros(1), name="phase_comp"),
            SymbolDemapper(constellation)
            ],
            taps=["data_tx", "awgn", "gsop", "cfo_comp", "phase_comp"],
            wiring={"phase_comp.reference": "signal_tx"})

# simulate communication
y = chain(N)

# compute metric
ser_after = compute_ser(chain.tap("data_tx"), y)

# print metric and plot
print(f"after: SER={ser_after}")

# Four IQ planes, one per stage: a scatter of real against imaginary,
# and style.apply for the labels, the grid and the equal aspect ratio.
for tap, name in [("awgn", "received signal"),
                  ("gsop", "after GSOP"),
                  ("cfo_comp", "after GSOP+CFO comp"),
                  ("phase_comp", "after GSOP + CFO comp + phase correction")]:
    symbols = chain.tap(tap)
    _, ax = plt.subplots()
    ax.plot(np.real(symbols), np.imag(symbols), ".")
    ax.set_title(name)
    style.apply(ax, "iq")

# show evolution of the angular frequency estimate
w0_history = chain["cfo_comp"].history
plt.figure()
plt.plot(w0_history)
plt.xlabel("number of iteration")
plt.title("w0 estimate")
plt.show()
