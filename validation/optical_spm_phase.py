"""Self-phase modulation vs the analytic Kerr phase rotation.

For a dispersion-free fiber, the Kerr effect rotates the phase of a
constant-power signal by

    phi_NL = gamma * P * L                      (lossless)
    phi_NL = gamma * P * L_eff                  (with loss)

with L_eff = (1 - exp(-alpha L)) / alpha (Agrawal, section 4.1). The
lossless case is exact for any step count; with loss, the split-step
approximation must converge to the effective-length formula as the step
count grows (O(StPS^-2) for the symmetric scheme).
"""
import pathlib

import numpy as np

from comnumpy.optical.links import FiberLink

FIG_DIR = pathlib.Path(__file__).parent / "figures"

FS = 1e12
N = 1024
P0 = 5e-3      # 5 mW launch power
GAMMA = 1.3    # rad/W/km
L = 80.0       # km
ALPHA_DB = 0.2


def spm_phase(alpha_dB, StPS):
    x = np.sqrt(P0) * np.ones(N, dtype=complex)
    link = FiberLink(1, L_span=L, StPS=StPS, fs=FS, gamma=GAMMA,
                     alpha_dB=alpha_dB, cd_coefficient=0, noise_scaling=0)
    y = link(x)
    # the EDFA restores the launch power exactly in the noiseless case
    power_ratio = np.abs(y[0]) ** 2 / P0
    assert abs(power_ratio - 1) < 1e-12, f"power not restored: {power_ratio}"
    return float(np.angle(y[0] * np.conj(x[0])))


def main():
    # lossless: exact for any step count
    phi = spm_phase(alpha_dB=0, StPS=1)
    phi_theo = GAMMA * P0 * L
    assert abs(phi - phi_theo) < 1e-12, (phi, phi_theo)
    print(f"PASS SPM lossless: phi = {phi:.6f} rad = gamma*P*L (exact, single step)")

    # with loss: converges to gamma * P * L_eff
    alpha = np.log(10) / 10 * ALPHA_DB
    L_eff = (1 - np.exp(-alpha * L)) / alpha
    phi_theo = GAMMA * P0 * L_eff
    steps = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    rel_err = np.array([abs(spm_phase(ALPHA_DB, s) - phi_theo) / phi_theo
                        for s in steps])
    assert rel_err[-1] < 1e-6, f"no convergence: {rel_err[-1]}"
    # observed order: halving the step size divides the error by ~4
    order = np.polyfit(np.log(steps), np.log(rel_err), 1)[0]
    assert order < -1.8, f"convergence order too low: {order}"

    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.loglog(steps, rel_err, "o-", label="simulation vs $\\gamma P L_{eff}$")
    ax.loglog(steps, rel_err[0] * (steps / steps[0]) ** -2.0, "--",
              label=r"$O(\mathrm{StPS}^{-2})$ slope")
    ax.set_xlabel("steps per span")
    ax.set_ylabel("relative phase error")
    ax.set_title(f"SPM phase convergence ({L:.0f} km, {ALPHA_DB} dB/km, "
                 f"L_eff = {L_eff:.1f} km)")
    ax.legend()
    ax.grid(True, which="both")
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "optical_spm_phase.png", dpi=150)

    print(f"PASS SPM with loss: relative error {rel_err[-1]:.2e} at StPS=1000, "
          f"convergence order {order:.2f}")


if __name__ == "__main__":
    main()
