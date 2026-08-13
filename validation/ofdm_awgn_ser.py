"""SER of a full OFDM chain over AWGN vs the plain QAM theory.

With an orthonormal (I)FFT and an ideal channel, OFDM over AWGN
preserves the per-subcarrier SNR (Parseval), so the measured SER of the
complete chain -- serial/parallel, carrier allocation, IFFT, cyclic
prefix, AWGN, and the receiver mirror -- must match the closed-form QAM
expression. This validates the whole OFDM stack, not one block. The
sweep uses the shared build/sweep/collect skeleton (decision D35).
"""
import pathlib

import numpy as np

from comnumpy import AWGN, Sequential, SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.processors import Parallel2Serial, Serial2Parallel
from comnumpy.core.utils import Constellation
from comnumpy.ofdm.allocation import get_allocation
from comnumpy.ofdm.processors import (CarrierAllocator, CarrierExtractor,
                                      CyclicPrefixer, CyclicPrefixRemover,
                                      FFTProcessor, IFFTProcessor)
from comnumpy.monte_carlo import monte_carlo
from comnumpy import style

style.use()

FIG_DIR = pathlib.Path(__file__).parent / "figures"

M = 16
N_SYMBOLS = 200_000
SNR_DB_RANGE = np.arange(4, 20, 1)
N_CP = 16


def ofdm_chain():
    alloc = get_allocation("802.11a")
    constellation = Constellation("QAM", M)
    return Sequential([
        SymbolGenerator(M, name="tx"),
        SymbolMapper(constellation),
        Serial2Parallel(alloc.N_data),
        CarrierAllocator(alloc, pilots=1.0),
        IFFTProcessor(),
        CyclicPrefixer(N_CP),
        Parallel2Serial(),
        AWGN(snr_dB=0, name="noise"),
        Serial2Parallel(alloc.N_fft + N_CP),
        CyclicPrefixRemover(N_CP),
        FFTProcessor(),
        CarrierExtractor(alloc),
        Parallel2Serial(),
        SymbolDemapper(constellation),
    ])


def main():
    results = monte_carlo(ofdm_chain(), "noise.snr_dB", SNR_DB_RANGE,
                          {"ser": compute_ser}, N_SYMBOLS,
                          reference="tx", seed=7)
    ser_sim = results["ser"]

    # The AWGN block measures the *time-domain* power. With unit-power
    # symbols on the 52 occupied subcarriers of a 64-FFT (orthonormal
    # IFFT), the time-domain power is 52/64, so sigma^2 = (52/64) *
    # 10^(-SNR/10) per sample -- and by Parseval the per-subcarrier SNR
    # is *higher* than the time-domain SNR by 64/52. The CP changes
    # nothing: it copies samples of identical average power.
    alloc = get_allocation("802.11a")
    occupied = alloc.N_data + alloc.N_pilots
    snr_data_dB = SNR_DB_RANGE + 10 * np.log10(alloc.N_fft / occupied)
    ser_theo = Constellation("QAM", M).metrics(snr_data_dB,
                                               per="symbol")["ser"]

    mask = ser_theo > 50 / N_SYMBOLS
    rel_err = np.abs(ser_sim[mask] - ser_theo[mask]) / ser_theo[mask]
    assert np.all(rel_err < 0.25), (ser_sim[mask], ser_theo[mask], rel_err)

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.semilogy(SNR_DB_RANGE, ser_theo, "-", label="QAM-16 theory (occupancy-corrected)")
    ax.semilogy(SNR_DB_RANGE, np.where(ser_sim > 0, ser_sim, np.nan), "o",
                fillstyle="none", label="OFDM chain simulation")
    ax.set_xlabel("time-domain SNR [dB]")
    ax.set_ylabel("SER")
    ax.set_title(f"802.11a OFDM chain over AWGN, {N_SYMBOLS:,} symbols per point")
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "ofdm_awgn_ser.png", dpi=150)

    print(f"PASS OFDM chain SER: max relative error {rel_err.max():.3f} "
          f"over {mask.sum()} SNR points")


if __name__ == "__main__":
    main()
