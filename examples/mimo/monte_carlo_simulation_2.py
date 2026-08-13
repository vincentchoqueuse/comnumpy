"""BER of three OSIC orderings over Rayleigh fading, 4x4 16-QAM.

Reproduces figure 11.3 of [1]: ordered successive interference
cancellation, the ordering criterion being the column norm, the
post-detection SNR or the post-detection SINR.

[1] Cho, Yong Soo, et al. MIMO-OFDM wireless communications with
MATLAB. John Wiley & Sons, 2010.
"""
import numpy as np
import matplotlib.pyplot as plt

from comnumpy import plot_data, print_data, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import ErrorCounter
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import (
    OrderedSuccessiveInterferenceCancellationDetector)

style.use()

# parameters
N = 100          # symbols per frame
N_test = 5000    # channel draws per point; more smooths the curve
N_r, N_t = 4, 4
constellation = Constellation("QAM", 16)
snr_dB_list = np.arange(0, 45, 5)
seed = 42        # one master seed reproduces the whole figure
orderings = ("colnorm", "snr", "sinr")

chain = Sequential([
    SymbolGenerator(constellation.order, name="data_tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(rayleigh_channel(N_r=N_r, N_t=N_t), name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    ], taps=["data_tx"])

# Storage, declared before the loop: one array per ordering, full of
# zeros, indexed by name. The error counts get one too -- they say
# where the estimate runs out of samples, so they are part of the
# record even though they stay out of the figure.
ber = {}
errors = {}
for ordering in orderings:
    ber[ordering] = np.zeros(len(snr_dB_list))
    errors[ordering] = np.zeros(len(snr_dB_list))

# simulation loop: one child seed per SNR point (D6/D35), so the whole
# figure is reproduced by the master seed alone
point_seeds = np.random.SeedSequence(seed).spawn(len(snr_dB_list))
for index, snr_dB in enumerate(snr_dB_list):
    rng = np.random.default_rng(int(point_seeds[index].generate_state(1)[0]))
    sigma2 = N_t * 10 ** (-snr_dB / 10)
    chain.set_params(noise__sigma2=sigma2)

    # one ErrorCounter per ordering: each BER is a ratio of error and
    # bit totals over the whole point, not a mean of per-frame rates
    counters = {}
    for ordering in orderings:
        counters[ordering] = ErrorCounter(
            width=constellation.bits_per_symbol)

    for _ in range(N_test):
        H = rayleigh_channel(N_r=N_r, N_t=N_t, rng=rng)
        chain.seed(int(rng.integers(2 ** 31)))
        chain.set_params(channel__H=H)
        Y = chain((N_t, N))
        S_ref = chain.tap("data_tx")

        for ordering, counter in counters.items():
            detector = OrderedSuccessiveInterferenceCancellationDetector(
                alphabet=constellation, osic_type=ordering, H=H,
                sigma2=sigma2)
            counter.update(S_ref, detector(Y))

    for ordering, counter in counters.items():
        ber[ordering][index] = counter.rate
        errors[ordering][index] = counter.n_errors

# display, from the same dictionaries the loop filled
print_data({"x": snr_dB_list, "curves": ber}, xlabel="snr_dB", ylabel="BER")
print()
print_data({"x": snr_dB_list, "curves": errors},
           xlabel="snr_dB", ylabel="errors behind each point")

ax = plot_data({"x": snr_dB_list, "curves": ber}, xlabel="SNR [dB]",
               ylabel="BER", yscale="log", marker="o", fillstyle="none")
ax.set_xlim(0, 40)
ax.set_ylim(1e-4, 1)
ax.set_title("OSIC orderings, 4x4 16-QAM over Rayleigh fading")
plt.show()
