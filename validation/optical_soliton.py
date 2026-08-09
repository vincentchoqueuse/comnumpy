"""Fundamental soliton propagation through the full split-step method.

In the anomalous dispersion regime, a first-order soliton

    A(0, t) = sqrt(P0) * sech(t / T0),      P0 = |beta2| / (gamma * T0^2)

propagates without changing shape: dispersion and Kerr nonlinearity
cancel exactly (Agrawal, section 5.2). This is the strongest single test
of the SSFM implementation, because it fails if either the CD/Kerr sign
convention, the step normalization, or the beta2 unit conversion is
wrong. As a counter-check, the same pulse must disperse when the Kerr
effect is switched off.
"""
import pathlib

import numpy as np

from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import compute_beta2

FIG_DIR = pathlib.Path(__file__).parent / "figures"

FS = 1e12
N = 4096
T0 = 25e-12
GAMMA = 1.3  # rad/W/km


def main():
    t = (np.arange(N) - N // 2) / FS
    beta2 = compute_beta2(1550, 17, 299792458) * 1e-24  # s^2/km, anomalous (<0)
    P0 = abs(beta2) / (GAMMA * T0**2)
    L_D = T0**2 / abs(beta2)
    x = (np.sqrt(P0) / np.cosh(t / T0)).astype(complex)

    L = 2 * L_D
    link = FiberLink(1, L_span=L, StPS=int(4 * L), fs=FS, noise_scaling=0,
                     fiber=FiberSpec(0, gamma=GAMMA, cd_coefficient=17))
    y = link(x)
    nmse = np.sum(np.abs(np.abs(y) - np.abs(x)) ** 2) / np.sum(np.abs(x) ** 2)
    assert nmse < 1e-9, f"soliton not preserved: NMSE={nmse}"

    # counter-check: without Kerr, the pulse must broaden substantially
    link_lin = FiberLink(1, L_span=L, StPS=1, fs=FS, noise_scaling=0,
                         fiber=FiberSpec(0, gamma=0, cd_coefficient=17))
    y_lin = link_lin(x)
    nmse_lin = np.sum(np.abs(np.abs(y_lin) - np.abs(x)) ** 2) / np.sum(np.abs(x) ** 2)
    assert nmse_lin > 1e-2, f"linear pulse did not disperse: NMSE={nmse_lin}"

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ps = t * 1e12
    ax.plot(ps, np.abs(x) ** 2 * 1e3, "-", label="input $|A(0,t)|^2$")
    ax.plot(ps, np.abs(y) ** 2 * 1e3, "--",
            label=f"soliton after {L:.0f} km (2 $L_D$)")
    ax.plot(ps, np.abs(y_lin) ** 2 * 1e3, ":",
            label="same distance, Kerr off (disperses)")
    ax.set_xlim(-250, 250)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("power [mW]")
    ax.set_title(f"Fundamental soliton (T0={T0*1e12:.0f} ps, "
                 f"P0={P0*1e3:.2f} mW, L_D={L_D:.1f} km)")
    ax.legend()
    ax.grid(True)
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "optical_soliton.png", dpi=150)

    print(f"PASS soliton: shape NMSE {nmse:.2e} after 2 L_D "
          f"(vs {nmse_lin:.2e} with Kerr off)")


if __name__ == "__main__":
    main()
