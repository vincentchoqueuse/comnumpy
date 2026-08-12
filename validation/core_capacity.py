"""Capacity against references that do not share a line of code with it.

``comnumpy.core.capacity`` is meant to be the reference axis the rest of
the library is read against, so checking it against its own properties
is not enough. Four confrontations (decision D7):

1. **Constellation capacity vs Monte-Carlo mutual information.** The
   module integrates over the noise with Gauss-Hermite quadrature. Here
   the same mutual information is estimated by *simulating the channel*
   -- draw symbols, draw noise, average the log-likelihood ratio. The
   two methods share nothing but the definition.

2. **Shaping loss.** A uniform constellation cannot reach Shannon: the
   asymptotic gap is exactly :math:`10 \\log_{10}(\\pi e / 6) =
   1.533` dB, the ultimate shaping gain. The measured gap must be
   positive, below that bound, and *increase* with the constellation
   size at matched spectral efficiency -- which is the statement that
   the bound is approached from below.

3. **Rayleigh ergodic capacity vs Monte-Carlo.** The closed form uses
   the exponential integral; the reference averages
   :math:`\\log_2(1 + \\rho |h|^2)` over drawn channels.

4. **Water-filling vs uniform power.** Water-filling is optimal, so it
   never loses; and its advantage must vanish at high SNR, where the
   water level dwarfs every noise level and the allocation flattens.

References
----------
G. Ungerboeck, "Channel coding with multilevel/phase signals", IEEE
Trans. Inform. Theory, vol. 28, no. 1, pp. 55-67, 1982.

G. D. Forney, L.-F. Wei, "Multidimensional constellations -- Part I",
IEEE J. Sel. Areas Commun., vol. 7, no. 6, pp. 877-892, 1989 (the
1.53 dB ultimate shaping gain).

I. E. Telatar, "Capacity of multi-antenna Gaussian channels", European
Trans. Telecomm., vol. 10, no. 6, pp. 585-595, 1999.
"""
import pathlib

import numpy as np
from scipy.optimize import brentq

from comnumpy.core.capacity import (awgn_capacity, constellation_capacity,
                                    rayleigh_ergodic_capacity, waterfilling)
from comnumpy.core.utils import Constellation

FIG_DIR = pathlib.Path(__file__).parent / "figures"

SNR_DB = np.arange(-4, 31, 1.0)
N_MONTE_CARLO = 200_000
ULTIMATE_SHAPING_GAIN_DB = 10 * np.log10(np.pi * np.e / 6)      # 1.5329 dB


def monte_carlo_mutual_information(constellation, snr, n=N_MONTE_CARLO, seed=0):
    r"""Estimate I(X; Y) by simulating the channel, not by quadrature.

    For equiprobable symbols on a complex AWGN channel,

    .. math::

        I = \log_2 M - \mathbb{E}\left[\log_2 \sum_m
            e^{-(|y - a_m|^2 - |y - a_i|^2)/\sigma^2}\right]

    with :math:`i` the transmitted index. The inner difference is taken
    against the transmitted symbol so the exponentials stay bounded by 1
    -- writing it as a ratio of raw exponentials underflows well before
    30 dB.
    """
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(constellation)
    order = len(alphabet)
    sigma2 = 1.0 / snr
    index = rng.integers(0, order, n)
    x = alphabet[index]
    y = x + np.sqrt(sigma2 / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    distance2 = np.abs(y[:, None] - alphabet[None, :]) ** 2
    transmitted = distance2[np.arange(n), index]
    inner = np.log2(np.sum(np.exp(-(distance2 - transmitted[:, None]) / sigma2),
                           axis=1))
    return float(np.log2(order) - np.mean(inner))


def snr_for_rate(capacity_of, rate):
    """SNR at which a capacity curve reaches ``rate``, in linear scale."""
    return brentq(lambda snr: capacity_of(snr) - rate, 1e-3, 1e7,
                  xtol=1e-10, rtol=1e-12)


def check_against_monte_carlo():
    """1. Quadrature vs simulation, on three constellations."""
    worst = 0.0
    for family, order, seed in (("PSK", 4, 1), ("QAM", 16, 2), ("QAM", 64, 3)):
        constellation = Constellation(family, order)
        for snr_dB in (0.0, 6.0, 12.0, 18.0):
            snr = 10 ** (snr_dB / 10)
            quadrature = float(constellation_capacity(constellation, snr))
            simulated = monte_carlo_mutual_information(constellation, snr, seed=seed)
            error = abs(quadrature - simulated)
            worst = max(worst, error)
            # 0.01 bit/symbol: measured worst case 4.8e-3 over these twelve
            # points, and the Monte-Carlo standard error alone is ~3e-3 at
            # 200 000 samples, so a tighter bound would be testing the RNG
            assert error < 0.01, (
                f"{family}-{order} at {snr_dB} dB: quadrature {quadrature:.5f} "
                f"vs simulation {simulated:.5f}")
    print(f"PASS quadrature vs Monte-Carlo mutual information: worst "
          f"deviation {worst:.2e} bit/symbol over 12 points")
    return worst


def check_shaping_loss():
    """2. The gap to Shannon is positive, bounded by 1.53 dB, and grows."""
    gaps = []
    for order in (16, 64, 256):
        constellation = Constellation("QAM", order)
        rate = 0.6 * np.log2(order)          # well below saturation
        shannon = snr_for_rate(lambda snr: float(awgn_capacity(snr)), rate)
        uniform = snr_for_rate(
            lambda snr, a=constellation: float(constellation_capacity(a, snr)), rate)
        gap_dB = 10 * np.log10(uniform / shannon)
        gaps.append(gap_dB)
        assert 0 < gap_dB < ULTIMATE_SHAPING_GAIN_DB, (
            f"QAM-{order}: shaping loss {gap_dB:.4f} dB is outside "
            f"(0, {ULTIMATE_SHAPING_GAIN_DB:.4f}) dB")
    assert np.all(np.diff(gaps) > 0), \
        f"the shaping loss must grow towards 1.53 dB, got {gaps}"
    print("PASS shaping loss at 0.6 log2(M): "
          + ", ".join(f"QAM-{o} {g:.4f} dB" for o, g in zip((16, 64, 256), gaps,
                                                            strict=True))
          + f" -- increasing, all below the {ULTIMATE_SHAPING_GAIN_DB:.4f} dB limit")
    return gaps


def check_rayleigh_closed_form():
    """3. The exponential-integral form vs averaging over drawn channels."""
    rng = np.random.default_rng(4)
    gains = rng.exponential(size=1_000_000)      # |h|^2 for unit-power Rayleigh
    worst = 0.0
    for snr_dB in (0.0, 10.0, 20.0, 30.0):
        snr = 10 ** (snr_dB / 10)
        closed_form = float(rayleigh_ergodic_capacity(snr))
        simulated = float(np.mean(np.log2(1 + snr * gains)))
        error = abs(closed_form - simulated)
        worst = max(worst, error)
        assert error < 0.01, (
            f"{snr_dB} dB: closed form {closed_form:.5f} vs simulation "
            f"{simulated:.5f}")
    print(f"PASS Rayleigh ergodic capacity vs Monte-Carlo: worst deviation "
          f"{worst:.2e} bit/s/Hz")
    return worst


def check_waterfilling():
    """4. Never worse than uniform, and the advantage vanishes at high SNR."""
    rng = np.random.default_rng(5)
    gains = rng.exponential(size=16)
    advantages = []
    for snr in (0.1, 1.0, 10.0, 100.0, 1000.0):
        _, optimal = waterfilling(gains, snr)
        uniform = float(np.mean(np.log2(1 + snr * gains)))
        advantages.append(optimal - uniform)
        assert optimal >= uniform - 1e-12, \
            f"snr={snr}: water-filling {optimal} below uniform {uniform}"
    assert advantages[0] > advantages[-1], \
        f"the water-filling advantage must decay with SNR, got {advantages}"
    assert advantages[-1] < 0.01, \
        f"at 30 dB the advantage should be negligible, got {advantages[-1]}"
    print(f"PASS water-filling: advantage {advantages[0]:.4f} bit/s/Hz at "
          f"-10 dB, {advantages[-1]:.2e} at 30 dB -- optimal everywhere")
    return advantages


def main():
    import matplotlib.pyplot as plt

    check_against_monte_carlo()
    check_shaping_loss()
    check_rayleigh_closed_form()
    check_waterfilling()

    snr = 10 ** (SNR_DB / 10)
    _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SNR_DB, awgn_capacity(snr), "k-", lw=2, label="Shannon")
    for family, order, marker in (("PSK", 4, "o"), ("QAM", 16, "s"),
                                  ("QAM", 64, "^"), ("QAM", 256, "d")):
        constellation = Constellation(family, order)
        ax.plot(SNR_DB, constellation_capacity(constellation, snr), "-",
                label=f"{family}-{order}")
        # the independent estimate, on a few points only: it is expensive
        sparse = SNR_DB[::8]
        ax.plot(sparse, [monte_carlo_mutual_information(constellation, 10 ** (s / 10))
                         for s in sparse], marker, fillstyle="none", color="0.3")
    ax.plot([], [], "ko", fillstyle="none", label="Monte-Carlo estimate")
    ax.plot(SNR_DB, rayleigh_ergodic_capacity(snr), "--", color="0.5",
            label="Rayleigh ergodic")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("capacity [bit/symbol]")
    ax.set_title("Constellation-constrained capacity vs Shannon")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", fontsize="small")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "core_capacity.png", dpi=150)


if __name__ == "__main__":
    main()
