"""Convolutional coding gain over AWGN vs the uncoded baseline.

BPSK over AWGN with the standard (133, 171) K=7 rate-1/2 code: the
soft-decision Viterbi curve must show its coding gain over hard
decisions, which in turn beat the uncoded baseline at moderate Eb/N0
(Proakis & Salehi, Digital Communications, section 8.2). The sweep uses
the shared build/sweep/collect skeleton (decision D35).
"""
import pathlib

import numpy as np

from comnumpy import AWGN, Sequential, SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.fec import ConvolutionalEncoder, ViterbiDecoder
from comnumpy.monte_carlo import monte_carlo

FIG_DIR = pathlib.Path(__file__).parent / "figures"

N_BITS = 50_000
EBN0_DB_RANGE = np.arange(0, 7, 1)
CODE_RATE = 0.5


BPSK = np.array([1.0 + 0.0j, -1.0 + 0.0j])  # bit 0 -> +1, bit 1 -> -1


def get_coded_chain(soft):
    bpsk = BPSK
    demapper = (SymbolDemapper(bpsk, soft=True, name="demap") if soft
                else SymbolDemapper(bpsk, name="demap"))
    return Sequential([
        SymbolGenerator(2, name="tx"),
        ConvolutionalEncoder(),
        SymbolMapper(bpsk),
        AWGN(snr_dB=0, name="noise"),
        demapper,
        ViterbiDecoder(soft=soft),
    ])


def get_uncoded_chain():
    bpsk = BPSK
    return Sequential([
        SymbolGenerator(2, name="tx"),
        SymbolMapper(bpsk),
        AWGN(snr_dB=0, name="noise"),
        SymbolDemapper(bpsk, name="demap"),
    ])


def main():
    # per-coded-bit SNR: Eb/N0 + 10 log10(R) for BPSK (1 bit/symbol)
    snr_coded = EBN0_DB_RANGE + 10 * np.log10(CODE_RATE)

    ber = {}
    ber["uncoded"] = monte_carlo(get_uncoded_chain(), "noise.snr_dB", EBN0_DB_RANGE,
                                 {"ber": compute_ber_bits}, N_BITS,
                                 reference="tx", seed=10)["ber"]
    ber["hard"] = monte_carlo(get_coded_chain(soft=False), "noise.snr_dB", snr_coded,
                              {"ber": compute_ber_bits}, N_BITS,
                              reference="tx", seed=20)["ber"]
    ber["soft"] = monte_carlo(get_coded_chain(soft=True), "noise.snr_dB", snr_coded,
                              {"ber": compute_ber_bits}, N_BITS,
                              reference="tx", seed=30)["ber"]

    # Hard decisions operate 10*log10(1/R) = 3 dB below the uncoded channel
    # SNR, so the hard curve crosses the uncoded one around 3-4 dB (Proakis
    # 8.2): assert the crossover region, not a naive global ordering.
    high = EBN0_DB_RANGE >= 5
    assert np.all(ber["hard"][high] < ber["uncoded"][high]), \
        (ber["hard"], ber["uncoded"])
    low = EBN0_DB_RANGE <= 1
    assert np.all(ber["hard"][low] > ber["uncoded"][low]), \
        "hard decisions should still lose below the crossover"
    # soft never does worse than hard, and gains ~2 dB where measurable
    measurable = ber["hard"] > 100 / N_BITS
    assert np.all(ber["soft"][measurable] <= ber["hard"][measurable]), \
        (ber["soft"], ber["hard"])

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    for label, marker in (("uncoded", "o"), ("hard", "s"), ("soft", "^")):
        ax.semilogy(EBN0_DB_RANGE, np.where(ber[label] > 0, ber[label], np.nan),
                    marker + "-", fillstyle="none", label=label)
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("BER")
    ax.set_title(f"(133,171) K=7 rate-1/2 convolutional code, {N_BITS:,} bits")
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "fec_coding_gain.png", dpi=150)

    print(f"PASS coding gain: at {EBN0_DB_RANGE[-1]} dB, uncoded "
          f"{ber['uncoded'][-1]:.2e}, hard {ber['hard'][-1]:.2e}, "
          f"soft {ber['soft'][-1]:.2e}")


def compute_ber_bits(bits_tx, bits_rx):
    """Bit error rate between two bit streams (decoded length = tx length)."""
    return float(np.mean(bits_tx != bits_rx))


if __name__ == "__main__":
    main()
