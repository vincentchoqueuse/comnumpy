"""Diversity schemes against the closed form they are supposed to reach.

Three links that spend the same total transmit power over an i.i.d.
Rayleigh channel, each one an instance of the *same* expression -- the
MGF average of Simon and Alouini, exposed as
``comnumpy.core.metrics.compute_ser_rayleigh_psk``:

    one antenna each side          L = 1,        SNR per branch = gamma
    maximum ratio combining 1x2    L = N_r = 2,  SNR per branch = gamma
    Alamouti 2x1                   L = N_t N_r = 2, per branch = gamma / N_t

The last line is what the script is really about. An orthogonal design
reaches the same diversity *order* as receive combining and pays 3 dB
for it, because the transmitter has no channel knowledge and splits its
power evenly. That factor N_t is the only difference between the second
and the third row, so a simulation that lands on the curve confirms the
convention as much as the code: get the split wrong and the Alamouti
points sit 3 dB away from their own theory.

Accuracy here is limited by the number of *channel* draws, not by the
symbol count -- the error rate is dominated by the rare deep fades, and
an under-sampled tail reads systematically **low**. At 12 dB and 4000
draws the estimate is already 16 % under the curve, which is why the
unit tests stop at 8 dB and this script draws two orders of magnitude
more.
"""
import pathlib

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser_rayleigh_psk
from comnumpy.core.processors import Amplifier
from comnumpy.core.utils import Constellation
from comnumpy.mimo.channels import AWGN, FlatMIMOChannel
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code
from comnumpy.mimo.detectors import LinearDetector
from comnumpy.mimo.utils import rayleigh_channel

FIG_DIR = pathlib.Path(__file__).parent / "figures"

ORDER = 4
ALPHABET = Constellation("PSK", ORDER)
SNR_DB_RANGE = np.array([6.0, 12.0, 18.0])
# more draws where the error rate is lower: the deep fades that dominate
# the variance become rarer exactly where the curve is being read
N_CHANNELS = [20000, 40000, 80000]
N_SYMBOLS = 20
TOLERANCE = 0.12


def simulate(chain, n_rx, n_tx, snr_dB, stimulus, n_channels, seed):
    """Average one chain over independent quasi-static fading draws."""
    rng = np.random.default_rng(seed)
    chain.seed(seed)
    chain.set_params(noise__sigma2=10 ** (-snr_dB / 10))
    errors = total = 0
    for _ in range(n_channels):
        realization = rayleigh_channel(n_rx, n_tx, rng=rng)
        chain.set_params(channel__H=realization,
                         detector__H=realization)
        detected = chain(stimulus)
        errors += int(np.sum(chain.tap("tx") != detected))
        total += detected.size
    return errors / total


def linear_chain(H):
    """One antenna or receive combining: zero forcing covers both."""
    return Sequential([
        SymbolGenerator(ORDER, name="tx"),
        SymbolMapper(ALPHABET),
        FlatMIMOChannel(H, name="channel"),
        AWGN(sigma2=0.1, name="noise"),
        LinearDetector(ALPHABET, H=H, name="detector"),
    ], taps=["tx"])


def alamouti_chain(H, code):
    power = 1 / np.sqrt(code.n_tx)
    return Sequential([
        SymbolGenerator(ORDER, name="tx"),
        SymbolMapper(ALPHABET),
        Amplifier(power),
        SpaceTimeEncoder(code),
        FlatMIMOChannel(H, name="channel"),
        AWGN(sigma2=0.1, name="noise"),
        SpaceTimeDecoder(code, H=H, name="detector"),
        Amplifier(1 / power),
        SymbolDemapper(ALPHABET),
    ], taps=["tx"])


def theory(diversity, branch_split=1):
    """The one expression, read for one scheme."""
    snr_per_bit = 10 ** (SNR_DB_RANGE / 10) / np.log2(ORDER) / branch_split
    return compute_ser_rayleigh_psk(ORDER, snr_per_bit, diversity=diversity)


def main():
    code = get_code("alamouti")
    schemes = {
        "1 Tx, 1 Rx": (linear_chain(rayleigh_channel(1, 1, seed=0)),
                       1, 1, (1, N_SYMBOLS), theory(1)),
        "MRC 1 Tx, 2 Rx": (linear_chain(rayleigh_channel(2, 1, seed=1)),
                           2, 1, (1, N_SYMBOLS), theory(2)),
        "Alamouti 2 Tx, 1 Rx": (alamouti_chain(rayleigh_channel(1, 2, seed=2),
                                               code),
                                1, 2, N_SYMBOLS, theory(2, code.n_tx)),
    }

    measured = {}
    for name, (chain, n_rx, n_tx, stimulus, expected) in schemes.items():
        values = np.array([
            simulate(chain, n_rx, n_tx, snr_dB, stimulus, count, seed=10 + index)
            for index, (snr_dB, count) in enumerate(zip(SNR_DB_RANGE,
                                                        N_CHANNELS,
                                                        strict=True))])
        measured[name] = values
        deviation = np.abs(values - expected) / expected
        assert np.all(deviation < TOLERANCE), (name, values, expected, deviation)
        print(f"PASS {name:22s} within {deviation.max():.1%} of the closed "
              f"form at {SNR_DB_RANGE.min():.0f}-{SNR_DB_RANGE.max():.0f} dB")

    # the 3 dB is the whole point: same slope, shifted curve
    ratio = measured["Alamouti 2 Tx, 1 Rx"] / measured["MRC 1 Tx, 2 Rx"]
    assert np.all(ratio > 1.5), ratio
    shift = np.interp(np.log10(measured["Alamouti 2 Tx, 1 Rx"][-1]),
                      np.log10(measured["MRC 1 Tx, 2 Rx"])[::-1],
                      SNR_DB_RANGE[::-1])
    print(f"PASS Alamouti pays {SNR_DB_RANGE[-1] - shift:.1f} dB against "
          f"receive diversity for the same diversity order "
          f"(10log10(N_t) = {10 * np.log10(code.n_tx):.1f} dB)")

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    fine = np.linspace(SNR_DB_RANGE[0], SNR_DB_RANGE[-1], 100)
    per_bit = 10 ** (fine / 10) / np.log2(ORDER)
    curves = {"1 Tx, 1 Rx": compute_ser_rayleigh_psk(ORDER, per_bit),
              "MRC 1 Tx, 2 Rx": compute_ser_rayleigh_psk(ORDER, per_bit,
                                                         diversity=2),
              "Alamouti 2 Tx, 1 Rx": compute_ser_rayleigh_psk(
                  ORDER, per_bit / code.n_tx, diversity=2)}
    for (name, curve), marker in zip(curves.items(), "osd", strict=True):
        line, = ax.semilogy(fine, curve, "-", label=f"{name}, closed form")
        ax.semilogy(SNR_DB_RANGE, measured[name], marker, fillstyle="none",
                    color=line.get_color(), label=f"{name}, simulation")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.set_title(f"QPSK over i.i.d. Rayleigh, {min(N_CHANNELS)}+ "
                 f"channel draws per point")
    ax.legend(fontsize=8)
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "mimo_diversity_ber.png", dpi=150)


if __name__ == "__main__":
    main()
