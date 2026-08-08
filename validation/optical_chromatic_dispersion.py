"""Chromatic dispersion vs analytic Gaussian pulse broadening.

A Gaussian pulse of half-width T0 propagating in a purely dispersive
fiber broadens as

    T1(z) = T0 * sqrt(1 + (z / L_D)^2),      L_D = T0^2 / |beta2|

(Agrawal, Nonlinear Fiber Optics, section 3.2). The script measures the
RMS width of the simulated intensity profile and compares it to the
closed form; without loss, energy must also be conserved.
"""
import pathlib

import numpy as np

from comnumpy.optical.channels import ChromaticDispersion
from comnumpy.optical.utils import compute_beta2

FIG_DIR = pathlib.Path(__file__).parent / "figures"

# time grid: 1 ps resolution, 8192 samples
FS = 1e12
N = 8192
T0 = 20e-12  # 20 ps Gaussian half-width


def measured_half_width(t, y):
    """Gaussian half-width from the RMS width of the intensity profile."""
    intensity = np.abs(y) ** 2
    center = np.sum(t * intensity) / np.sum(intensity)
    rms = np.sqrt(np.sum((t - center) ** 2 * intensity) / np.sum(intensity))
    return rms * np.sqrt(2)  # I(t) ~ exp(-t^2/T1^2) has RMS width T1/sqrt(2)


def main():
    t = (np.arange(N) - N // 2) / FS
    x = np.exp(-t**2 / (2 * T0**2)).astype(complex)

    beta2 = compute_beta2(1550, 17, 299792458)  # ps^2/km
    L_D = T0**2 / abs(beta2 * 1e-24)            # km
    distances = np.array([0.5, 1, 2, 4]) * L_D

    T1_meas, T1_theo = [], []
    for z in distances:
        y = ChromaticDispersion(z, alpha_dB=0, fs=FS)(x)
        T1_meas.append(measured_half_width(t, y))
        T1_theo.append(T0 * np.sqrt(1 + (z / L_D) ** 2))
        energy_ratio = np.sum(np.abs(y) ** 2) / np.sum(np.abs(x) ** 2)
        assert abs(energy_ratio - 1) < 1e-12, f"energy not conserved: {energy_ratio}"

    T1_meas, T1_theo = np.array(T1_meas), np.array(T1_theo)
    rel_err = np.abs(T1_meas - T1_theo) / T1_theo
    assert np.all(rel_err < 1e-10), f"width mismatch: {rel_err}"

    # figure
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    z_fine = np.linspace(0, distances[-1], 200)
    ax.plot(z_fine, T0 * np.sqrt(1 + (z_fine / L_D) ** 2) * 1e12, "-",
            label=r"theory: $T_0\sqrt{1+(z/L_D)^2}$")
    ax.plot(distances, T1_meas * 1e12, "o", label="comnumpy simulation")
    ax.set_xlabel("distance z [km]")
    ax.set_ylabel(r"pulse half-width $T_1$ [ps]")
    ax.set_title(f"Gaussian pulse broadening under CD (T0={T0*1e12:.0f} ps, "
                 f"L_D={L_D:.1f} km)")
    ax.legend()
    ax.grid(True)
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "optical_chromatic_dispersion.png", dpi=150)

    print(f"PASS chromatic dispersion: max relative width error {rel_err.max():.2e} "
          f"over {len(distances)} distances up to {distances[-1]:.1f} km")


if __name__ == "__main__":
    main()
