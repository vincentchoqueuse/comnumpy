"""Block error rate of three MIMO detectors over Rayleigh fading.

Reproduces figure 2 of [1]: a 2x2 QPSK link, maximum-likelihood
detection against zero-forcing and ordered successive interference
cancellation, one new channel draw per frame.

[1] X. Li, H. C. Huang, A. Lozano and G. J. Foschini,
"Reduced-complexity detection algorithms for systems using
multi-element arrays", Globecom '00, San Francisco, 2000.
"""
import numpy as np
import matplotlib.pyplot as plt

from comnumpy import plot_data, print_data, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector)

style.use()

# parameters
N = 400          # symbols per frame
N_test = 500     # channel draws per point; more smooths the curve
N_r, N_t = 2, 2
constellation = Constellation("PSK", 4)
snr_dB_list = np.arange(0, 45, 5)
seed = 42        # one master seed reproduces the whole figure
detector_names = ("ML", "ZF", "OSIC")

chain = Sequential([
    SymbolGenerator(constellation.order, name="data_tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(rayleigh_channel(N_r=N_r, N_t=N_t), name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    ], observations=["data_tx"])

# --- metrics, pre-allocated ------------------------------------------
# One array per detector, full of zeros, indexed by name -- a column
# never has to be counted to be found, and the dictionary is what the
# table and the figure render.
bler = {}
for name in detector_names:
    bler[name] = np.zeros(len(snr_dB_list))

# --- simulation loop -------------------------------------------------
# One child seed per SNR point (D6/D35), and one *stack* of channel
# draws per point (D51): channel k propagates frame k, and each
# detector, built with the same stack, decides frame k against channel
# k -- the comparison is over identical frames, and the chain runs once
# per point instead of once per draw.
point_seeds = np.random.SeedSequence(seed).spawn(len(snr_dB_list))
for index, snr_dB in enumerate(snr_dB_list):
    rng = np.random.default_rng(int(point_seeds[index].generate_state(1)[0]))
    sigma2 = N_t * 10 ** (-snr_dB / 10)
    H = rayleigh_channel(N_r=N_r, N_t=N_t, rng=rng, size=N_test)
    chain.set_params(noise__sigma2=sigma2, channel__H=H)
    chain.seed(int(rng.integers(2 ** 31)))
    Y = chain((N_test, N_t, N))
    S_ref = chain.observation("data_tx")

    detectors = {
        "ML": MaximumLikelihoodDetector(alphabet=constellation, H=H),
        "ZF": LinearDetector(alphabet=constellation, H=H, method="zf"),
        "OSIC": OrderedSuccessiveInterferenceCancellationDetector(
            constellation, osic_type="sinr", H=H, sigma2=sigma2),
    }
    for name, detector in detectors.items():
        S_est = detector(Y)
        # a frame is in error when at least one of its symbols is --
        # with Gray labelling, exactly "at least one of its bits"
        bler[name][index] = float(np.mean(
            np.any(S_est != S_ref, axis=(-2, -1))))

# --- results: table and figure ---------------------------------------
data = {"x": snr_dB_list, "curves": bler}
print_data(data, xlabel="snr_dB", ylabel="BLER")

ax = plot_data(data, xlabel="SNR [dB]", ylabel="BLER", yscale="log",
               marker="o", fillstyle="none")
ax.set_xlim(0, 40)
ax.set_ylim(1e-3, 1)
ax.set_title("2x2 QPSK over Rayleigh fading, one draw per frame")
plt.show()
