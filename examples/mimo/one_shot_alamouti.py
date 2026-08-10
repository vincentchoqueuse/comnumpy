import numpy as np
import matplotlib.pyplot as plt

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.processors import Amplifier
from comnumpy.core.utils import get_alphabet
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

img_dir = "../../docs/examples/img/"

# Parameters
M = 4
alphabet = get_alphabet("PSK", M)
code = get_code("alamouti")
power = 1 / np.sqrt(code.n_tx)          # split the power over the antennas
sigma2 = 0.2

# The Alamouti link, as one chain
H = rayleigh_channel(1, 2, seed=42)
alamouti = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    Amplifier(power),
    SpaceTimeEncoder(code),
    FlatMIMOChannel(H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    SpaceTimeDecoder(code, H=H, name="detector"),
    Amplifier(1 / power),
    SymbolDemapper(alphabet),
], taps=["tx", "noise", "detector"], name="Alamouti 2x1")

alamouti.seed(7)                        # every stochastic block, reproducibly
s_rx = alamouti(500 * code.n_symbols)
print(f"one-shot SER: {compute_ser(alamouti.tap('tx'), s_rx):.4f}")

# what each block costs and what it hands to the next one (D33b)
alamouti.summary(500 * code.n_symbols)

# Figure 1: the tapped signals, before and after combining
received = alamouti.tap("noise")[0]
combined = alamouti.tap("detector") / power
fig1, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
ax_left.plot(np.real(received), np.imag(received), ".", markersize=3)
ax_left.set_title("tap('noise'): the single receive antenna")
ax_right.plot(np.real(combined), np.imag(combined), ".", markersize=3)
ax_right.plot(np.real(alphabet), np.imag(alphabet), "kx", markersize=9)
ax_right.set_title("tap('detector'): after Alamouti combining")
for ax in (ax_left, ax_right):
    ax.set_xlabel("in phase")
    ax.set_ylabel("quadrature")
    ax.axis("equal")
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig1.png")

# The two references are chains differing only in their blocks. Zero
# forcing on an (N_r, 1) channel *is* maximum ratio combining -- the
# pseudo-inverse of a column vector is h^H / ||h||^2 -- so one
# LinearDetector covers both the no-diversity and the diversity case.
siso_H = rayleigh_channel(1, 1, seed=1)
siso = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    FlatMIMOChannel(siso_H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    LinearDetector(alphabet, H=siso_H, name="detector"),
], taps=["tx"], name="1 Tx, 1 Rx")

mrc_H = rayleigh_channel(2, 1, seed=2)
mrc = Sequential([
    SymbolGenerator(M, name="tx"),
    SymbolMapper(alphabet),
    FlatMIMOChannel(mrc_H, name="channel"),
    AWGN(sigma2=sigma2, name="noise"),
    LinearDetector(alphabet, H=mrc_H, name="detector"),
], taps=["tx"], name="MRC 1 Tx, 2 Rx")


def draw_channels(n_rx, n_tx, n_channels, seed):
    """The quasi-static fading realizations one sweep point each."""
    rng = np.random.default_rng(seed)
    return [rayleigh_channel(n_rx, n_tx, rng=rng) for _ in range(n_channels)]


def ser_by_hand(chain, channels, snr_dB, stimulus, seed=0):
    """The loop, spelled out: reconfigure, run, count, average."""
    chain.seed(seed)
    chain.set_params(**{"noise.sigma2": 10 ** (-snr_dB / 10)})
    errors = total = 0
    for realization in channels:
        chain.set_params(**{"channel.H": realization,
                            "detector.H": realization})
        detected = chain(stimulus)
        errors += int(np.sum(chain.tap("tx") != detected))
        total += detected.size
    return errors / total


def ser_by_sweep(chain, channels, snr_dB, stimulus, seed=0):
    """The same average, as one call: the sweep is over the channel.

    ``sweep`` takes several dotted parameters at once and zips them, so
    one sweep point sets the channel of the propagation block *and* the
    channel the detector inverts -- which is what a realization is.
    """
    chain.set_params(**{"noise.sigma2": 10 ** (-snr_dB / 10)})
    results = sweep(chain, ("channel.H", "detector.H"),
                    [(H, H) for H in channels],
                    {"ser": compute_ser}, stimulus=stimulus,
                    reference="tx", seed=seed)
    return float(np.mean(results["ser"]))


# The two agree on the same realizations. They do not draw the same
# noise -- sweep gives every point its own child seed (D6/D35) -- so
# they agree to within the Monte-Carlo error, not to the last digit.
n_symbols = 80
probe = draw_channels(1, 2, 300, seed=0)
print(f"8 dB over 300 realizations: loop "
      f"{ser_by_hand(alamouti, probe, 8.0, n_symbols):.5f}, sweep "
      f"{ser_by_sweep(alamouti, probe, 8.0, n_symbols):.5f}")


def average_ser(chain, n_rx, n_tx, snr_dB, stimulus, n_channels=2500):
    return ser_by_sweep(chain, draw_channels(n_rx, n_tx, n_channels, seed=0),
                        snr_dB, stimulus)


# Monte-Carlo comparison, at equal total transmit power
snr_dB_list = np.arange(4, 29, 4)
curves = {
    "1 Tx, 1 Rx (no diversity)": [average_ser(siso, 1, 1, value, (1, n_symbols))
                                  for value in snr_dB_list],
    "Alamouti, 2 Tx, 1 Rx": [average_ser(alamouti, 1, 2, value, n_symbols)
                             for value in snr_dB_list],
    "MRC, 1 Tx, 2 Rx": [average_ser(mrc, 2, 1, value, (1, n_symbols))
                        for value in snr_dB_list],
}

# Figure 2: the diversity order is the slope
fig2, ax = plt.subplots(figsize=(6.5, 5))
for label, ser in curves.items():
    ax.semilogy(snr_dB_list, ser, "o-", label=label)
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("symbol error rate")
ax.set_ylim(1e-5, 1)
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig2.png")

# The diversity order is the *asymptotic* slope, so the interesting
# thing is the local slope converging to it. Intervals whose upper end
# rests on a handful of errors are dropped: their slope is noise.
for label, ser in curves.items():
    values = np.array(ser)
    steps = [(np.log10(values[index + 1]) - np.log10(values[index]))
             / ((snr_dB_list[index + 1] - snr_dB_list[index]) / 10)
             for index in range(len(values) - 1) if values[index + 1] > 3e-4]
    print(f"{label:28s} local slope "
          + " ".join(f"{value:+.2f}" for value in steps))

# and the gap Alamouti pays for transmitting blind: about 3 dB.
# Points where no error was seen carry no information, so they are
# dropped rather than read as a zero error rate.
target = 1e-3
gaps = {}
for label, ser in curves.items():
    seen = np.array(ser) > 0
    gaps[label] = np.interp(np.log10(target),
                            np.log10(np.array(ser)[seen])[::-1],
                            snr_dB_list[seen][::-1])
print(f"SNR needed for SER = {target:g}: "
      + ", ".join(f"{label} {value:.1f} dB" for label, value in gaps.items()))
