"""Digital back-propagation round trip.

In the noiseless case, DBP with steps matched to the forward SSFM must
invert the fiber link exactly (every split-step operation is invertible
and applied in reverse order), so the residual NMSE is at machine
precision regardless of launch power. Linear-only (CD) equalization, by
contrast, leaves the nonlinear distortion in place, and that residual
grows with launch power (Ip & Kahn, J. Lightwave Technol. 26(20), 2008).
"""
import pathlib

import numpy as np

from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.links import FiberLink

FIG_DIR = pathlib.Path(__file__).parent / "figures"

FS = 50e9
N = 4096
N_SPANS = 4
FIBER = dict(L_span=80.0, StPS=50, fs=FS,
             fiber=FiberSpec(0.2, gamma=1.3, cd_coefficient=17))


def band_limited_signal(rng):
    X = rng.normal(size=N) + 1j * rng.normal(size=N)
    Xf = np.fft.fft(X)
    Xf[np.abs(np.fft.fftfreq(N)) > 0.2] = 0
    x = np.fft.ifft(Xf)
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def main():
    rng = np.random.default_rng(42)
    x_unit = band_limited_signal(rng)

    powers_dBm = np.arange(-2, 9, 2)
    nmse_dbp, nmse_cd = [], []
    for p_dBm in powers_dBm:
        P0 = 10 ** (p_dBm / 10) * 1e-3
        x = x_unit * np.sqrt(P0)
        y = FiberLink(N_SPANS, noise_scaling=0, **FIBER)(x)

        x_dbp = DBP(N_SPANS, **FIBER)(y)
        x_cd = DBP(N_SPANS, use_only_linear=True, **FIBER)(y)

        nmse_dbp.append(np.sum(np.abs(x_dbp - x) ** 2) / np.sum(np.abs(x) ** 2))
        nmse_cd.append(np.sum(np.abs(x_cd - x) ** 2) / np.sum(np.abs(x) ** 2))

    nmse_dbp, nmse_cd = np.array(nmse_dbp), np.array(nmse_cd)
    assert np.all(nmse_dbp < 1e-20), f"DBP is not an exact inverse: {nmse_dbp}"
    assert np.all(np.diff(nmse_cd) > 0), \
        f"CD-only residual should grow with power: {nmse_cd}"
    assert nmse_cd[-1] > 1e3 * nmse_dbp[-1], "DBP shows no gain over CD-only"

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.semilogy(powers_dBm, nmse_cd, "s-", label="CD-only equalization")
    ax.semilogy(powers_dBm, np.maximum(nmse_dbp, 1e-30), "o-",
                label="DBP (matched steps)")
    ax.set_xlabel("launch power [dBm]")
    ax.set_ylabel("residual NMSE")
    ax.set_title(f"DBP round trip, {N_SPANS}x{FIBER['L_span']:.0f} km, noiseless")
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "optical_dbp_roundtrip.png", dpi=150)

    print(f"PASS DBP round trip: max NMSE {nmse_dbp.max():.2e} (machine precision); "
          f"CD-only residual {nmse_cd[0]:.2e} -> {nmse_cd[-1]:.2e} "
          f"from {powers_dBm[0]} to {powers_dBm[-1]} dBm")


if __name__ == "__main__":
    main()
