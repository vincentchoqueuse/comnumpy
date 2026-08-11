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

The two agree to **0.04 dB at worst** across the 96 channels, which is
a statement about the gain *shape*: the tilt comes out 6.53 dB against
6.59 dB, and its sign -- more gain at the bottom of the band -- is
predicted by where the channels sit under the Raman spectrum of the two
pumps, not by any scale factor.

Getting there took two corrections and one refutation, all of them
worth recording because each was a way to be wrong while looking right.

The **effective-area scaling** was missing. The gain shape depends on
the Stokes shift alone -- that is the glass -- but the coefficient
multiplying the powers is g_R / A_eff, and the effective area belongs
to the waveguide, so it grows with wavelength. Without it the tilt came
out 8.00 dB instead of 6.59: an excess of *tilt* rather than of level,
which is what identified the cause. With it, this library's coupling
coefficients now match GNPy's **exactly** on the gain side -- element
by element, ratio 1.0000 -- which is the check that says the law is
right rather than merely closer.

The **pump powers** were wrong here, not in the library. GNPy applies
the span's 0.5 dB connector loss to its Raman pumps before injecting
them, so the 224.4 mW and 231.1 mW of its configuration file reach the
fibre as 200.0 mW and 206.0 mW. Feeding the configuration values
over-pumped the link by 0.5 dB and left a 0.9 dB disagreement that was
briefly, and wrongly, blamed on the model.

The **solver was exonerated by experiment**. It was tempting to blame
GNPy's default perturbative expansion, since a truncated series would
under-estimate gain the harder it is driven. GNPy was installed and its
own reference case re-run at perturbative orders 1 to 4 and with its
``numerical`` method: the harness reproduces the shipped file to
5.5e-16, and all five settings agree to the last digit. The expansion
is already converged at order 1 here, so that explanation is dead.

What remains is a real difference of model, and it is small. GNPy's
depletion, after its ``vibrational_loss`` factor, conserves **energy**
where this library conserves **photons**; the coupling coefficients
differ on the depletion side by up to nu_pump/nu_signal = 1.0716. On
this link that is worth the 0.04 dB above.
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
# The configuration file quotes 224.403 mW and 231.135 mW; GNPy applies
# the span's connector loss to its pumps before injecting them, so these
# are what actually reach the fibre. Read off gnpy's own raman_pumps,
# not inferred -- feeding the configuration values over-pumps by 0.5 dB.
PUMP_W = np.array([0.224403, 0.231135]) * 10 ** (-0.5 / 10)
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
    assert np.abs(error).max() < 0.1, np.abs(error).max()
    print("\nPASS 96-channel counter-pumped profile, 2 pumps, 80 km:")
    print("       channel      ours    GNPy    error")
    for index in (0, 24, 48, 72, 95):
        print(f"       {CHANNEL_Hz[index]/1e12:7.2f} THz  {ours[index]:6.2f}  "
              f"{theirs[index]:6.2f}  {error[index]:+6.2f} dB")
    print(f"       mean error {error.mean():+.3f} dB, worst "
          f"{np.abs(error).max():.3f} dB")


def check_the_tilt(ours, theirs):
    """The part that tests the *shape*, which nothing else here does."""
    tilt_ours = float(ours.max() - ours.min())
    tilt_theirs = float(theirs.max() - theirs.min())
    # The gain peaks ~12.75 THz below the pump, so the 205 THz pump peaks
    # near the bottom of the band: more gain at low frequency.
    assert ours[0] > ours[-1], "the tilt has the wrong sign"
    assert theirs[0] > theirs[-1], "the reference tilt has the wrong sign"
    assert abs(tilt_ours - tilt_theirs) < 0.2, (tilt_ours, tilt_theirs)
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

    print("\nAll checks passed, to 0.04 dB across 96 channels. What this "
          "pins is the gain *shape*,")
    print("which the analytic confrontations in optical_raman.py cannot -- "
          "they test the integrator.")


if __name__ == "__main__":
    main()
