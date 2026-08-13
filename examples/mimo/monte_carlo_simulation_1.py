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

from comnumpy import Experiment, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ber
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.utils import rayleigh_channel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector)

style.use()

# The experimental conditions, in one place. The studied parameter --
# the SNR -- is deliberately not here: the experiment sets it.
config = {
    "N": 400,        # symbols per frame
    "N_test": 500,   # channel draws per point; more smooths the curve
    "N_r": 2,
    "N_t": 2,
}
constellation = Constellation("PSK", 4)

chain = Sequential([
    SymbolGenerator(constellation.order, name="data_tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(rayleigh_channel(N_r=2, N_t=2), name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    ], taps=["data_tx"])


def simulate(config, seed):
    """One SNR point: N_test channel draws, the three detectors on each.

    A frame is in error when at least one of its bits is, and every
    detector decides the *same* received frame, so the comparison is
    over identical realizations. The point's seed drives both the
    channel draws and the chain, so the whole curve is reproduced by
    the experiment's seed alone.
    """
    rng = np.random.default_rng(seed)
    sigma2 = config["N_t"] * 10 ** (-config["snr_dB"] / 10)
    chain.set_params(noise__sigma2=sigma2)
    bler = {"ML": 0.0, "ZF": 0.0, "OSIC": 0.0}
    for _ in range(config["N_test"]):
        H = rayleigh_channel(N_r=config["N_r"], N_t=config["N_t"], rng=rng)
        chain.seed(int(rng.integers(2 ** 31)))
        chain.set_params(channel__H=H)
        Y = chain((config["N_t"], config["N"]))
        S_ref = chain.tap("data_tx")

        detectors = {
            "ML": MaximumLikelihoodDetector(alphabet=constellation, H=H),
            "ZF": LinearDetector(alphabet=constellation, H=H, method="zf"),
            "OSIC": OrderedSuccessiveInterferenceCancellationDetector(
                constellation, osic_type="sinr", H=H, sigma2=sigma2),
        }
        for name, detector in detectors.items():
            ber = compute_ber(S_ref, detector(Y),
                              width=constellation.bits_per_symbol)
            bler[name] += float(ber > 0) / config["N_test"]
    return {"BLER": bler}


experiment = Experiment(config, parameter="snr_dB",
                        values=np.arange(0, 45, 5), seed=42)
result = experiment.run(simulate)
result.print()

ax = result.plot("BLER", yscale="log", marker="o", fillstyle="none")
ax.set_xlim(0, 40)
ax.set_ylim(1e-3, 1)
ax.set_title("2x2 QPSK over Rayleigh fading, one draw per frame")
plt.show()
