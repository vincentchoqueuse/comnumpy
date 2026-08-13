"""MIMO detectors over 2x2 i.i.d. Rayleigh fading vs closed-form references.

Zero forcing on an i.i.d. Rayleigh :math:`2 \\times 2` channel reduces
each stream to SISO Rayleigh fading (post-processing SNR
:math:`\\gamma = \\bar{\\gamma} Z` with :math:`Z \\sim \\mathrm{Exp}(1)`,
diversity order :math:`N_r - N_t + 1 = 1`), so the average BPSK BER has
the closed form

    BER_ZF = (1/2) (1 - sqrt(g / (1 + g))),      g = 1 / sigma^2

(Tse & Viswanath, Fundamentals of Wireless Communication, chapter 8;
Proakis & Salehi, section 13.2). Maximum likelihood achieves full
receive diversity (order 2), which the script checks through the slope
of the BER curve. Monte-Carlo accuracy here is limited by the number of
*channel realizations* (deep fades dominate the variance), not by the
bit count.
"""
import pathlib

import numpy as np

from comnumpy.core.metrics import compute_ser_rayleigh_psk
from comnumpy.mimo.detectors import LinearDetector, MaximumLikelihoodDetector
from comnumpy import style

style.use()

FIG_DIR = pathlib.Path(__file__).parent / "figures"

SNR_DB_RANGE = np.array([4.0, 8.0, 12.0])
# more channel draws at high SNR: deep fades dominate the MC variance there
N_REALIZATIONS_PER_POINT = [4000, 4000, 16000]
N_SYMBOLS = 40
BPSK = np.array([1.0 + 0.0j, -1.0 + 0.0j])


def simulate(snr_dB, n_realizations, seed):
    rng = np.random.default_rng(seed)
    sigma2 = 10 ** (-snr_dB / 10)
    errors = {"zf": 0, "ml": 0}
    total = 0
    for _ in range(n_realizations):
        H = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
        bits = rng.integers(0, 2, (2, N_SYMBOLS))
        x = BPSK[bits]
        noise = np.sqrt(sigma2 / 2) * (rng.normal(size=x.shape)
                                       + 1j * rng.normal(size=x.shape))
        y = H @ x + noise
        errors["zf"] += np.sum(LinearDetector(BPSK, H=H)(y) != bits)
        errors["ml"] += np.sum(MaximumLikelihoodDetector(BPSK, H=H)(y) != bits)
        total += bits.size
    return {k: v / total for k, v in errors.items()}


def zf_closed_form(snr_dB):
    """Diversity 1, from the library rather than transcribed here.

    Zero forcing on an i.i.d. Rayleigh N_r x N_t channel leaves each
    stream with diversity N_r - N_t + 1, which is 1 for the 2x2 case.
    """
    return compute_ser_rayleigh_psk(2, 10 ** (snr_dB / 10), diversity=1)


def main():
    ber_zf, ber_ml = [], []
    for i, snr_dB in enumerate(SNR_DB_RANGE):
        point = simulate(snr_dB, N_REALIZATIONS_PER_POINT[i], seed=100 + i)
        ber_zf.append(point["zf"])
        ber_ml.append(point["ml"])
    ber_zf, ber_ml = np.array(ber_zf), np.array(ber_ml)
    ber_theo = zf_closed_form(SNR_DB_RANGE)

    # ZF matches the diversity-1 closed form
    rel_err = np.abs(ber_zf - ber_theo) / ber_theo
    assert np.all(rel_err < 0.10), (ber_zf, ber_theo, rel_err)

    # ML beats ZF and shows a steeper (diversity-2) slope
    assert np.all(ber_ml < ber_zf / 2), (ber_ml, ber_zf)
    slope_zf = np.log10(ber_zf[0]) - np.log10(ber_zf[-1])
    slope_ml = np.log10(ber_ml[0]) - np.log10(ber_ml[-1])
    assert slope_ml / slope_zf > 1.4, (slope_ml, slope_zf)

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    snr_fine = np.linspace(SNR_DB_RANGE[0], SNR_DB_RANGE[-1], 100)
    ax.semilogy(snr_fine, zf_closed_form(snr_fine), "-",
                label="ZF closed form (diversity 1)")
    ax.semilogy(SNR_DB_RANGE, ber_zf, "o", fillstyle="none", label="ZF simulation")
    ax.semilogy(SNR_DB_RANGE, ber_ml, "s", fillstyle="none",
                label="ML simulation (diversity 2)")
    ax.set_xlabel("SNR per receive antenna [dB]")
    ax.set_ylabel("BER")
    ax.set_title("2x2 i.i.d. Rayleigh, BPSK, "
                 f"{min(N_REALIZATIONS_PER_POINT)}+ channel draws/point")
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "mimo_zf_ml_ber.png", dpi=150)

    print(f"PASS MIMO detectors: ZF within {rel_err.max():.1%} of the closed "
          f"form; ML/ZF slope ratio {slope_ml / slope_zf:.2f} (diversity 2)")


if __name__ == "__main__":
    main()
