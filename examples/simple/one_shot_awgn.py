"""One AWGN simulation, written twice: by hand, then as a chain.

Run from this directory: it writes the figures and diagrams of the
getting-started page into ../../docs/getting_started/.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from comnumpy import style
from comnumpy.core import Sequential, plot_iq
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import Constellation

style.use()

img_dir = "../../docs/getting_started/img/"
mermaid_dir = "../../docs/getting_started/mermaid/"

N = 10000     # number of symbols
snr_dB = 10   # signal-to-noise ratio per symbol, in dB

# --- 1. by hand, in plain numpy ---------------------------------------
# A 4-QAM alphabet on the odd-integer grid, rescaled to unit average
# energy. That rescaling is a *convention*, and everything below depends
# on it being the same one the noise variance assumes.
rng = np.random.default_rng(0)
axis = np.array([-1.0, +1.0])
raw = (axis[:, None] + 1j * axis[None, :]).ravel()
alphabet = raw / np.sqrt(np.mean(np.abs(raw) ** 2))

symbols = rng.integers(0, alphabet.size, N)
sent = alphabet[symbols]

# Es = 1 after the rescaling, so sigma^2 = 10^(-SNR/10); the noise power
# is split evenly between the real and the imaginary part.
sigma2 = 10 ** (-snr_dB / 10)
noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(N)
                               + 1j * rng.standard_normal(N))
received = sent + noise

# hard decision: the nearest constellation point
distance = np.abs(received[:, None] - alphabet[None, :])
detected = np.argmin(distance, axis=1)
ser_by_hand = np.mean(detected != symbols)

# and the closed form to compare it against, which is where the
# modulation has to be described a second time -- the square-QAM error
# probability, with the order written into it three times
order, bits = 4, 2
snr_per_bit = 10 ** (snr_dB / 10) / bits
component = 2 * (1 - 1 / np.sqrt(order)) * norm.sf(
    np.sqrt(3 * bits * snr_per_bit / (order - 1)))
ser_theo_by_hand = 1 - (1 - component) ** 2

print(f"by hand : SER = {ser_by_hand:.4f}, theory = {ser_theo_by_hand:.4f}")

# --- 2. the same thing, as a chain ------------------------------------
# The constellation is one object: it holds the alphabet, its order and
# its own closed form, so the modulation is named once.
constellation = Constellation("QAM", 4)

chain = Sequential([
    SymbolGenerator(constellation.order, name="tx"),
    SymbolMapper(constellation),
    AWGN(snr_dB=snr_dB, name="awgn"),
    SymbolDemapper(constellation),
    ], taps=["tx", "awgn"])

# seeded, so the number this page quotes is the number you get (D6)
chain.seed(0)
y = chain(N)

ser = compute_ser(chain.tap("tx"), y)
# AWGN(snr_dB=) is a symbol SNR and the closed form is quoted against
# Eb/N0: per="symbol" makes the conversion, which is 10log10(k) dB.
ser_theo = constellation.metrics(snr_dB, per="symbol")["ser"]

print(f"chain   : SER = {ser:.4f}, theory = {ser_theo:.4f}")

# --- 3. what the object knows about itself ----------------------------
print("\nconstellation.info()")
for key, value in constellation.info().items():
    print(f"  {key:16s} {value}")

# --- 4. the figures ---------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
constellation.plot(ax=axes[0])
plot_iq(chain.tap("awgn"), title=f"received at {snr_dB} dB", ax=axes[1])
plt.tight_layout()
plt.savefig(f"{img_dir}/first_simulation_fig1.png")

with open(f"{mermaid_dir}/first_simulation.mmd", "w") as stream:
    stream.write(chain.to_mermaid())
