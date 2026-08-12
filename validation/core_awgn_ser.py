"""Monte-Carlo symbol error rate over AWGN vs theory.

Simulated SER for QPSK and 16-QAM must match the closed-form expressions
(Proakis, Digital Communications, section 4.3) within Monte-Carlo
accuracy across the SNR range.
"""
import pathlib

import numpy as np

from comnumpy import (AWGN, Sequential, SymbolDemapper,
                      SymbolGenerator, SymbolMapper, compute_ser,
                      Constellation)
from comnumpy.sweep import sweep

FIG_DIR = pathlib.Path(__file__).parent / "figures"

N_SYMBOLS = 200_000
SNR_DB_RANGE = np.arange(0, 16, 1)


def simulate(order, seed):
    constellation = Constellation("QAM", order)
    chain = Sequential([
        SymbolGenerator(order, name="tx"),
        SymbolMapper(constellation),
        AWGN(snr_dB=0, name="noise"),
        SymbolDemapper(constellation),
    ])
    results = sweep(chain, "noise.snr_dB", SNR_DB_RANGE,
                    {"ser": compute_ser}, N_SYMBOLS,
                    reference="tx", seed=seed)
    return results["ser"]


def main():
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()

    for order, seed, marker in ((4, 10, "o"), (16, 20, "s")):
        constellation = Constellation("QAM", order)
        ser_theo = constellation.metrics(SNR_DB_RANGE, per="symbol")["ser"]
        ser_sim = simulate(order, seed)

        # compare where the theoretical SER is measurable with N_SYMBOLS
        mask = ser_theo > 50 / N_SYMBOLS
        rel_err = np.abs(ser_sim[mask] - ser_theo[mask]) / ser_theo[mask]
        assert np.all(rel_err < 0.15), \
            f"QAM-{order}: SER deviates from theory: {rel_err}"

        ax.semilogy(SNR_DB_RANGE, ser_theo, "-",
                    label=f"QAM-{order} theory")
        ax.semilogy(SNR_DB_RANGE, np.where(ser_sim > 0, ser_sim, np.nan),
                    marker, fillstyle="none", label=f"QAM-{order} simulation")
        print(f"PASS QAM-{order}: max relative SER error "
              f"{rel_err.max():.3f} over {mask.sum()} SNR points")

    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.set_title(f"SER over AWGN, {N_SYMBOLS:,} symbols per point")
    ax.set_ylim(1e-5, 1)
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "core_awgn_ser.png", dpi=150)


if __name__ == "__main__":
    main()
