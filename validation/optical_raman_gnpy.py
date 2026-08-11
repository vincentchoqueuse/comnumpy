"""Counter-pumped Raman against a second implementation (decision D7).

``validation/optical_raman.py`` confronts the solver with an exact
solution under arbitrary depletion, with photon-number conservation and
with the undepleted limit. Those are real checks of the **integrator**,
and they share a blind spot: photon conservation is imposed by the
coupling matrix by construction (``C_ji = -(nu_j/nu_i) C_ij``), the
closed form is derived from the same two-wave system, and the
undepleted limit fixes a magnitude at one point. A wrong gain *shape*
passes all three.

This script closes that gap with a case whose answer is dominated by
the shape: 96 channels across 4.8 THz, amplified by two
counter-propagating pumps, where the gain each channel receives depends
on where it sits under the Raman spectrum of the two pumps.

The reference is GNPy (Telecom Infra Project), an independently
developed planning tool used by network operators, together with the
measured SSMF gain profile it carries. Provenance and licence are in
``data/gnpy/README.md``; nothing from GNPy is in ``src/``.

**This comparison does not agree to the last decibel, and the residual
is reported rather than tuned away.** Two conventions were established
by reading GNPy's source rather than by fitting:

- its gain coefficient scales as ``nu_pump / nu_reference`` and as
  ``1 / A_eff``, both of which this library now applies through
  ``RamanGainSpectrum(quoted_at=...)``;
- its depletion, after the ``vibrational_loss`` factor, conserves
  **energy** where this library conserves **photons**. That difference
  has the wrong sign to explain the residual: conserving photons
  depletes the pump *more*, which would lower our gain further.

The open candidate is the solver itself. GNPy defaults to a
*perturbative* expansion where this library solves the boundary value
problem exactly, and a truncated series under-estimates gain the harder
it is driven -- which is the shape of what is left, largest where the
gain peaks. If that is the explanation then the remaining gap is
GNPy's, and closing it would be the error. GNPy can be asked for a
``numerical`` solution instead; regenerating its expected results that
way is the way to settle it, and is not done here.
"""
import pathlib

import numpy as np

from comnumpy.optical.raman import RamanGainSpectrum, solve_raman

SPEED_OF_LIGHT = 2.99792458e8
DATA = pathlib.Path(__file__).parent / "data" / "gnpy"
FIG_DIR = pathlib.Path(__file__).parent / "figures"

# The span of tests/data/test_science_utils_fiber_config.json
LENGTH_KM = 80.0
LOSS_dB_KM = 0.2
CONNECTOR_dB = 0.5
PASSIVE_dB = LENGTH_KM * LOSS_dB_KM + 2 * CONNECTOR_dB
PUMP_Hz = np.array([205e12, 201e12])
PUMP_W = np.array([0.224403, 0.231135])
TEMPERATURE_K = 283.0
N_CHANNELS = 96
CHANNEL_Hz = 191.3e12 + 50e9 * np.arange(N_CHANNELS)
CHANNEL_W = 1e-3

# The profile is quoted at 1454 nm with these waveguide numbers.
REFERENCE_Hz = 206184634112792.0
REFERENCE_AREA_um2 = 75.74659443542413
CORE_RADIUS_um = 4.2


def measured_spectrum():
    """The GNPy SSMF profile, in this library's convention."""
    rows = [line.split(",") for line in (DATA / "raman_gain_ssmf.csv").read_text().splitlines()
            if line.strip() and not line.startswith("#") and not line[0].isalpha()]
    shift = np.array([float(row[0]) for row in rows])
    gamma = np.array([float(row[1]) for row in rows])
    peak_W_km = float(np.max(gamma)) / (REFERENCE_AREA_um2 * 1e-12) * 1e3
    spectrum = RamanGainSpectrum(
        tabulated=(shift, gamma),
        quoted_at=(SPEED_OF_LIGHT / REFERENCE_Hz * 1e9,
                   REFERENCE_AREA_um2, CORE_RADIUS_um),
        standard="SSMF",
        reference="D'Amico et al., J. Lightwave Technol. 40, 3499 (2022)")
    return spectrum, peak_W_km


def gnpy_on_off_gain_dB():
    """On-off gain implied by GNPy's expected per-channel output."""
    lines = (DATA / "raman_reference_expected.csv").read_text().splitlines()[1:]
    signal_W = np.array([float(line.split(",")[0]) for line in lines])
    return 10 * np.log10(signal_W / CHANNEL_W) + PASSIVE_dB


def ours_on_off_gain_dB(spectrum, peak_W_km):
    connector = 10 ** (-CONNECTOR_dB / 10)
    solution = solve_raman(
        length_km=LENGTH_KM, gain_peak_W_km=peak_W_km,
        signal_W=np.full(N_CHANNELS, CHANNEL_W * connector),
        pump_backward_W=PUMP_W,
        alpha_signal_dB_km=LOSS_dB_KM, alpha_pump_dB_km=LOSS_dB_KM,
        wavelength_signal_nm=SPEED_OF_LIGHT / CHANNEL_Hz * 1e9,
        wavelength_pump_nm=SPEED_OF_LIGHT / PUMP_Hz * 1e9,
        spectrum=spectrum, temperature_K=TEMPERATURE_K, n_nodes=301)
    return np.asarray(solution.on_off_gain_dB)


def check_the_peak_gain(peak_W_km):
    """The table, in our units, must be the textbook SSMF figure."""
    assert 0.35 < peak_W_km < 0.50, peak_W_km
    print(f"PASS the measured profile normalizes to {peak_W_km:.4f} /W/km, "
          f"inside the 0.35-0.50 /W/km")
    print("       the literature quotes for SSMF -- which is what says the "
          "conversion from GNPy's")
    print("       units (per-metre, over an effective area) into this "
          "library's is right")


def check_the_gain_profile(ours, theirs):
    error = ours - theirs
    assert np.all(ours > 0), "no gain at all"
    # The two disagree; the bound is the measured residual, recorded so
    # that it cannot quietly grow.
    assert np.abs(error).max() < 1.5, np.abs(error).max()
    print("\nPASS 96-channel counter-pumped profile, 2 pumps, 80 km:")
    print("       channel      ours    GNPy    error")
    for index in (0, 24, 48, 72, 95):
        print(f"       {CHANNEL_Hz[index]/1e12:7.2f} THz  {ours[index]:6.2f}  "
              f"{theirs[index]:6.2f}  {error[index]:+6.2f} dB")
    print(f"       mean error {error.mean():+.2f} dB, worst "
          f"{np.abs(error).max():.2f} dB -- a real disagreement, see the "
          f"module docstring")


def check_the_tilt(ours, theirs):
    """The part that tests the *shape*, which nothing else here does."""
    tilt_ours = float(ours.max() - ours.min())
    tilt_theirs = float(theirs.max() - theirs.min())
    # The gain peaks ~12.75 THz below the pump, so the 205 THz pump peaks
    # near the bottom of the band: more gain at low frequency.
    assert ours[0] > ours[-1], "the tilt has the wrong sign"
    assert theirs[0] > theirs[-1], "the reference tilt has the wrong sign"
    assert abs(tilt_ours - tilt_theirs) < 1.0, (tilt_ours, tilt_theirs)
    print(f"\nPASS gain tilt across the band: ours {tilt_ours:.2f} dB, "
          f"GNPy {tilt_theirs:.2f} dB")
    print("       both fall from low to high frequency, which is a "
          "prediction of the spectrum's")
    print("       shape rather than its scale, and is the statement no "
          "closed form here can make")


def check_the_waveguide_correction(spectrum, peak_W_km, theirs):
    """What `quoted_at=` buys, measured rather than asserted."""
    plain = RamanGainSpectrum(tabulated=spectrum.tabulated, standard="SSMF")
    without = ours_on_off_gain_dB(plain, peak_W_km)
    with_it = ours_on_off_gain_dB(spectrum, peak_W_km)
    tilt_error = [abs(float(curve.max() - curve.min())
                      - float(theirs.max() - theirs.min()))
                  for curve in (without, with_it)]
    assert tilt_error[1] < tilt_error[0], tilt_error
    print("\nPASS the effective-area correction earns its place:")
    print(f"       tilt error {tilt_error[0]:.2f} dB without "
          f"quoted_at=, {tilt_error[1]:.2f} dB with it")
    print(f"       worst-channel error {np.abs(without-theirs).max():.2f} dB "
          f"-> {np.abs(with_it-theirs).max():.2f} dB")
    return without, with_it


def main():
    import matplotlib.pyplot as plt

    spectrum, peak_W_km = measured_spectrum()
    check_the_peak_gain(peak_W_km)
    theirs = gnpy_on_off_gain_dB()
    ours = ours_on_off_gain_dB(spectrum, peak_W_km)
    check_the_gain_profile(ours, theirs)
    check_the_tilt(ours, theirs)
    without, _ = check_the_waveguide_correction(spectrum, peak_W_km, theirs)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    frequency = CHANNEL_Hz / 1e12
    axes[0].plot(frequency, theirs, "-", label="GNPy (reference)")
    axes[0].plot(frequency, ours, "--", label="comnumpy")
    axes[0].plot(frequency, without, ":", color="0.6",
                 label="comnumpy, no $A_{eff}$ scaling")
    axes[0].set_xlabel("channel frequency [THz]")
    axes[0].set_ylabel("on-off Raman gain [dB]")
    axes[0].set_title("96 channels, two counter-pumps, 80 km SMF")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(frequency, ours - theirs, "-", label="with $A_{eff}$ scaling")
    axes[1].plot(frequency, without - theirs, ":", color="0.6",
                 label="without")
    axes[1].axhline(0, color="k", linewidth=1)
    axes[1].set_xlabel("channel frequency [THz]")
    axes[1].set_ylabel("error vs GNPy [dB]")
    axes[1].set_title("What the waveguide correction removes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / "optical_raman_gnpy.png", dpi=150)

    print("\nAll checks passed. The agreement is not exact and the module "
          "docstring says why; what")
    print("this pins is the gain *shape*, which the analytic confrontations "
          "in optical_raman.py cannot.")


if __name__ == "__main__":
    main()
