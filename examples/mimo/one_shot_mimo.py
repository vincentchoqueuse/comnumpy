import numpy as np
import matplotlib.pyplot as plt

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import get_alphabet
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.detectors import (
    LinearDetector, MaximumLikelihoodDetector,
    OrderedSuccessiveInterferenceCancellationDetector)
from comnumpy.mimo.utils import rayleigh_channel

img_dir = "../../docs/examples/img/"

# Parameters
N = 1000
N_r, N_t = 3, 2
M = 4
alphabet = get_alphabet("PSK", M)
sigma2 = 0.1
H = rayleigh_channel(N_r, N_t, seed=0)


def link(detector):
    """The same link, closed by one detector or another.

    Only the last block changes, which is the point: a detector is a
    chain block like any other, and the comparison further down is four
    chains that differ by that block alone.
    """
    return Sequential([
        SymbolGenerator(M, name="tx"),
        SymbolMapper(alphabet),
        FlatMIMOChannel(H, name="channel"),
        AWGN(sigma2=sigma2, name="noise"),
        detector,
    ], taps=["tx", "noise"], name=f"{N_r}x{N_t} MIMO, {detector.name}")


detectors = {
    "ZF": LinearDetector(alphabet, H=H, method="zf", name="detector"),
    "MMSE": LinearDetector(alphabet, H=H, sigma2=sigma2, method="mmse",
                           name="detector"),
    "OSIC": OrderedSuccessiveInterferenceCancellationDetector(
        alphabet, osic_type="sinr", H=H, sigma2=sigma2, name="detector"),
    "ML": MaximumLikelihoodDetector(alphabet, H=H, name="detector"),
}
chains = {name: link(detector) for name, detector in detectors.items()}

# One shot, on one channel realization. Every chain is given the same
# seed, so they see the same symbols and the same noise: the only
# difference between the four numbers is the detector.
for name, chain in chains.items():
    chain.seed(0)
    detected = chain((N_t, N))
    print(f"* detector {name:5s}: ser={compute_ser(chain.tap('tx'), detected):.4f}")

# Figure 1: what each receive antenna sees -- the streams are summed by
# the channel, so no constellation is visible on any of them
Y = chains["ZF"].tap("noise")
fig1, axes1 = plt.subplots(nrows=1, ncols=N_r, figsize=(4 * N_r, 4))
for index in range(N_r):
    axes1[index].plot(np.real(Y[index, :]), np.imag(Y[index, :]), ".")
    axes1[index].set_title(f"Received signal (antenna {index + 1})")
    axes1[index].set_aspect("equal", adjustable="box")
    axes1[index].set_xlim([-2, 2])
    axes1[index].set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig1.png")

# Figure 2: the same run after zero forcing. The detector applies the
# pseudo-inverse and then decides; `linear_estimator` is that first step
# alone, which is what a constellation plot needs.
Z = detectors["ZF"].linear_estimator(Y)
fig2, axes2 = plt.subplots(nrows=1, ncols=N_t, figsize=(4 * N_t, 4))
for index in range(N_t):
    axes2[index].plot(np.real(Z[index, :]), np.imag(Z[index, :]), ".")
    axes2[index].plot(np.real(alphabet), np.imag(alphabet), "kx", markersize=9)
    axes2[index].set_title(f"Estimated signal (stream {index + 1})")
    axes2[index].set_aspect("equal", adjustable="box")
    axes2[index].set_xlim([-2, 2])
    axes2[index].set_ylim([-2, 2])
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig2.png")

# Monte Carlo evaluation. Averaging over fading means running the chain
# once per channel realization, and that is a sweep whose parameter is
# the channel: one point sets the matrix the signal goes through *and*
# the one the detector inverts, which is what a realization is.
snr_dB_list = np.arange(0, 20, 3)
n_channels = 200
n_symbols = 200


NOISE_AWARE = {"MMSE", "OSIC"}   # the two detectors that weight by sigma2


def average_ser(name, chain, snr_dB, seed=0):
    """Average one chain over independent Rayleigh draws at one SNR."""
    rng = np.random.default_rng(seed)
    channels = [rayleigh_channel(N_r, N_t, rng=rng) for _ in range(n_channels)]
    noise_variance = N_t * 10 ** (-snr_dB / 10)
    params = {"noise.sigma2": noise_variance}
    if name in NOISE_AWARE:
        # ZF and ML have no sigma2 at all: one ignores the noise by
        # construction, the other only compares distances
        params["detector.sigma2"] = noise_variance
    chain.set_params(**params)
    results = sweep(chain, ("channel.H", "detector.H"),
                    [(matrix, matrix) for matrix in channels],
                    {"ser": compute_ser}, stimulus=(N_t, n_symbols),
                    reference="tx", seed=seed)
    return float(np.mean(results["ser"]))


curves = {name: [average_ser(name, chain, snr_dB) for snr_dB in snr_dB_list]
          for name, chain in chains.items()}
for name, values in curves.items():
    print(f"{name:5s} " + " ".join(f"{value:.4f}" for value in values))

# Figure 3: the four detectors on one figure. There is no closed form
# here -- ZF over Rayleigh has one (see the Alamouti tutorial), the
# three others do not -- so the plot carries measurements only.
ax = plot_error_rate(snr_dB_list, curves, ylabel="SER",
                     title=f"{N_r}x{N_t} MIMO, {M}-PSK, "
                           f"{n_channels} channel draws per point")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/monte_carlo_mimo_fig3.png")
plt.show()
