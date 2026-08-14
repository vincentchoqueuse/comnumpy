"""Rayleigh fading, receive diversity, and the Alamouti space-time code.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import monte_carlo, print_data, style
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser, compute_ser_rayleigh_psk
from comnumpy.core.processors import Amplifier
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate, plot_iq
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

style.use()

img_dir = "../../docs/tutorials/img/"

constellation = Constellation("PSK", 4)
M = constellation.order
code = get_code("alamouti")
power = 1 / np.sqrt(code.n_tx)          # split the power over the antennas

# --- the fading channel ----------------------------------------------
# One Rayleigh coefficient is a complex Gaussian, so its squared modulus
# is exponential with unit mean: the instantaneous SNR is the average one
# multiplied by that draw.
rng = np.random.default_rng(0)
gain = np.abs(rayleigh_channel(20000, 1, rng=rng).ravel()) ** 2
grid = np.linspace(0, 6, 200)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(gain, bins=80, range=(0, 6), density=True, alpha=0.5,
        label="20 000 draws of $|h|^2$")
ax.plot(grid, np.exp(-grid), lw=2, label=r"$e^{-\gamma}$, exponential(1)")
ax.set_xlabel(r"channel power gain $|h|^2$")
ax.set_ylabel("probability density")
ax.set_title("a Rayleigh channel is a random SNR")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig1.png")

for threshold in (0.1, 0.01):
    print(f"P[|h|^2 < {threshold:g}] = {np.mean(gain < threshold):.4f}  "
          f"(1 - exp(-{threshold:g}) = {1 - np.exp(-threshold):.4f})")

# --- one shot ---------------------------------------------------------
# One channel draw, one chain, transmitter to decision: what Alamouti
# combining does to a single receive antenna.
H_one = rayleigh_channel(1, 2, seed=42)
one_shot = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    Amplifier(power),
    SpaceTimeEncoder(code),
    FlatMIMOChannel(H_one, name="channel"),
    AWGN(sigma2=0.2, name="noise"),
    SpaceTimeDecoder(code, H=H_one, name="detector"),
    Amplifier(1 / power),
    SymbolDemapper(constellation),
], observations=["tx", "noise", "detector"], name="Alamouti 2x1")

one_shot.seed(7)                        # every stochastic block, reproducibly
s_rx = one_shot(500 * code.n_symbols)
print(f"\none-shot SER: {compute_ser(one_shot.observation('tx'), s_rx):.4f}")
one_shot.summary(500 * code.n_symbols)

received = one_shot.observation("noise")[0]
combined = one_shot.observation("detector") / power
fig2, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
plot_iq(received, marker=".", ax=ax_left)
ax_left.set_title("observation('noise'): the single receive antenna")
plot_iq(combined, reference=constellation, ax=ax_right)
ax_right.set_title("observation('detector'): after Alamouti combining")
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig2.png")

# --- averaging over fading --------------------------------------------
# The SAME K channel draws serve every SNR point: one draw is one row
# of the batch (D51), each chain is built once with its stack -- the
# channel block propagates draw k on frame k, the detector holds the
# same stack -- and only the noise moves along the sweep.
K = 5000
n_symbols = 80
rng = np.random.default_rng(3)
H_siso = rayleigh_channel(1, 1, rng=rng, size=K)
H_alamouti = rayleigh_channel(1, 2, rng=rng, size=K)
H_mrc = rayleigh_channel(2, 1, rng=rng, size=K)

siso = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_siso, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    # zero forcing on an (N_r, 1) channel *is* maximum ratio combining:
    # the pseudo-inverse of a column is h^H / ||h||^2
    LinearDetector(constellation, H=H_siso, name="detector"),
], observations=["tx"], name="1 Tx, 1 Rx")

alamouti = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    Amplifier(power),
    SpaceTimeEncoder(code),
    FlatMIMOChannel(H_alamouti, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    SpaceTimeDecoder(code, H=H_alamouti, name="detector"),
    Amplifier(1 / power),
    SymbolDemapper(constellation),
], observations=["tx"], name="Alamouti 2x1")

mrc = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(constellation),
    FlatMIMOChannel(H_mrc, name="channel"),
    AWGN(sigma2=1.0, name="noise"),
    LinearDetector(constellation, H=H_mrc, name="detector"),
], observations=["tx"], name="MRC 1x2")

# --- the sweep: no simulation loop -----------------------------------
# monte_carlo moves the noise variance; everything else is frozen in
# the chains above.
snr_dB_list = np.arange(4, 25, 4)
sigma2_list = 10.0 ** (-snr_dB_list / 10)

curves = {}
curves["1 Tx, 1 Rx (no diversity)"] = monte_carlo(
    siso, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    (K, 1, n_symbols), reference="tx", seed=1)["ser"]
curves["Alamouti, 2 Tx, 1 Rx"] = monte_carlo(
    alamouti, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    (K, n_symbols), reference="tx", seed=2)["ser"]
curves["MRC, 1 Tx, 2 Rx"] = monte_carlo(
    mrc, "noise.sigma2", sigma2_list, {"ser": compute_ser},
    (K, 1, n_symbols), reference="tx", seed=3)["ser"]

# --- results: table and figure against the closed forms --------------
print()
print_data({"x": snr_dB_list, "curves": curves},
           xlabel="snr_dB", ylabel="SER")

fine = np.linspace(snr_dB_list[0], snr_dB_list[-1], 100)
per_bit = 10 ** (fine / 10) / np.log2(M)
theory = {
    "1 Tx, 1 Rx (no diversity)": compute_ser_rayleigh_psk(M, per_bit),
    "Alamouti, 2 Tx, 1 Rx": compute_ser_rayleigh_psk(M, per_bit / code.n_tx,
                                                     diversity=2),
    "MRC, 1 Tx, 2 Rx": compute_ser_rayleigh_psk(M, per_bit, diversity=2),
}
ax = plot_error_rate(snr_dB_list, curves, theory=theory, x_theory=fine,
                     ylabel="symbol error rate",
                     title=f"{M}-PSK over Rayleigh, {K} channel draws, "
                           f"equal transmit power")
ax.set_ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig3.png")
