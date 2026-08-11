"""Four Raman regimes, and the spontaneous emission, against GNPy (D7).

``optical_raman_gnpy.py`` confronts one counter-pumped span with GNPy and
lands at 0.04 dB.  One case, however well it agrees, leaves two questions
open: whether the agreement survives regimes the first one never visits,
and what the residual is *made of*.  This script answers both, and the
answer to the second turns out to be a single term.

Four regimes, each able to break a different assumption:

=================  =========================================================
``counter``        the reference case, two pumps below the band
``co``             same pumps, launched *with* the signals: the pump falls
                   25 dB while feeding them, so depletion shapes the whole
                   profile instead of only its end
``counter_strong`` 600 mW per pump, driven to saturation
``wideband``       202 channels over 10 THz, pumps up to 215 THz, which
                   asks the effective-area law to extrapolate far outside
                   the C band where it was calibrated
=================  =========================================================

Under GNPy's own conservation convention the four agree to **0.011 dB at
worst**, against a reference whose own convergence is 0.030 dB.  That is a
statement about the *implementation*: two independently written solvers,
one a boundary-value formulation and the other a forward integration, put
the same numbers on 96 and then 202 channels.

The residual under *this* library's convention is not noise, and not a
defect.  It is one term.  Standard stimulated Raman scattering conserves
photon number -- one pump photon annihilated gives one Stokes photon plus
one optical phonon -- so the pump must lose :math:`\\nu_p/\\nu_s` times what
the signal gains, and the difference leaves the optical field.  GNPy's
coupling matrix, after its ``vibrational_loss`` factor, comes out exactly
antisymmetric, so its total power is conserved and the phonon is never
paid for.  Switching this library to that convention closes the gap on
all four cases, which is what identifies it as the whole of the residual
rather than a plausible part of it.

Which of the two is right is not settled here, and not by GNPy: it is
settled in ``tests/optical/test_raman.py``, where the closed-form
logistic valid under arbitrary depletion is matched to 1e-11 by the
photon convention and missed by 6.5% by the power-conserving one.  The
6.5% is not a fitted number -- it is the ratio of the two saturation
limits, :math:`(P_{s0}+P_{p0}) / (P_{s0}+\\frac{\\nu_s}{\\nu_p}P_{p0})`.

The **spontaneous emission** is checked separately, because none of the
above touches it.  Restricted to the source GNPy models -- the Raman
pumps -- the two agree to 0.002 dB, which tests the coefficient, the Bose
occupancy, the dual-polarization factor and the integrating factor at
once.  This library additionally seeds each channel from every channel
above it, which GNPy omits; on this comb that is worth +0.17 dB of ASE,
and the third line of the table below shows it accounts for the whole
difference.
"""
import pathlib

import numpy as np

from comnumpy.optical.raman import (PLANCK, _coupling_matrix,
                                    _photon_occupancy, solve_raman)

from optical_raman_gnpy import (CHANNEL_W, LENGTH_KM, LOSS_dB_KM,
                                SPEED_OF_LIGHT, TEMPERATURE_K,
                                measured_spectrum)

DATA = pathlib.Path(__file__).parent / "data" / "gnpy"
FIG_DIR = pathlib.Path(__file__).parent / "figures"
CONNECTOR = 10 ** (-0.5 / 10)
BAUD_RATE = 32e9

# Calling GNPy's RamanSolver directly bypasses Fiber.propagate: the signals
# keep their launch power and only the pumps see the input connector, so the
# reference profile is separated from ours by the fibre loss alone.
PASSIVE_dB = LENGTH_KM * LOSS_dB_KM

CASES = {
    "counter": ([(205e12, 0.199999), (201e12, 0.205999)], "back", 401, 1e-8),
    "co": ([(205e12, 0.199999), (201e12, 0.205999)], "fwd", 401, 1e-8),
    "counter_strong": ([(205e12, 0.60), (201e12, 0.60)], "back", 401, 1e-8),
    "wideband": ([(215e12, 0.30), (209e12, 0.25), (203e12, 0.20)],
                 "back", 81, 1e-5),
}


def power_conserving(frequency_Hz, gain_peak_W_km, spectrum):
    """GNPy's convention: same gain side, but ``C_ji = -C_ij``.

    Kept here rather than in ``src`` on purpose. It is not an option this
    library offers -- it is the hypothesis under test, and the only way to
    show that the residual against GNPy is this one term and nothing else.
    """
    full = _coupling_matrix(frequency_Hz, gain_peak_W_km, spectrum)
    gain = np.where(full > 0, full, 0.0)
    return gain - gain.T


def reference_cases():
    """Per-channel output power of the four regimes, keyed by case."""
    rows = (DATA / "raman_regimes_expected.csv").read_text().splitlines()[1:]
    cases = {}
    for row in rows:
        name, frequency, power = row.split(",")
        cases.setdefault(name, ([], []))
        cases[name][0].append(float(frequency))
        cases[name][1].append(float(power))
    return {name: (np.array(f), np.array(p)) for name, (f, p) in cases.items()}


def reference_ase():
    rows = (DATA / "raman_ase_expected.csv").read_text().splitlines()[1:]
    values = np.array([[float(cell) for cell in row.split(",")] for row in rows])
    return values[:, 0], values[:, 1]


def ours(name, frequency_Hz, spectrum, peak_W_km, coupling=None):
    pumps, direction, n_nodes, tol = CASES[name]
    key = "pump_backward_W" if direction == "back" else "pump_forward_W"
    original = None
    if coupling is not None:
        import comnumpy.optical.raman as module
        original, module._coupling_matrix = module._coupling_matrix, coupling
    try:
        return solve_raman(
            length_km=LENGTH_KM, gain_peak_W_km=peak_W_km,
            signal_W=np.full(frequency_Hz.size, CHANNEL_W),
            alpha_signal_dB_km=LOSS_dB_KM, alpha_pump_dB_km=LOSS_dB_KM,
            wavelength_signal_nm=SPEED_OF_LIGHT / frequency_Hz * 1e9,
            wavelength_pump_nm=SPEED_OF_LIGHT
            / np.array([f for f, _ in pumps]) * 1e9,
            spectrum=spectrum, temperature_K=TEMPERATURE_K,
            bandwidth_Hz=BAUD_RATE, n_nodes=n_nodes, tol=tol,
            **{key: np.array([p for _, p in pumps]) * CONNECTOR})
    finally:
        if original is not None:
            import comnumpy.optical.raman as module
            module._coupling_matrix = original


def check_the_four_regimes(spectrum, peak_W_km):
    """Same physics, four regimes, and one term of difference."""
    print("PASS four regimes against GNPy's numerical solver:")
    print("       case             convention      tilt    GNPy     mean    "
          "worst")
    residuals, curves = {}, {}
    for name, (frequency, power) in reference_cases().items():
        target = 10 * np.log10(power / CHANNEL_W) + PASSIVE_dB
        for label, coupling in (("photon (ours)", None),
                                ("power (GNPy)", power_conserving)):
            gain = np.asarray(
                ours(name, frequency, spectrum, peak_W_km, coupling).on_off_gain_dB)
            error = gain - target
            if coupling is None:
                residuals[name] = float(np.abs(error).max())
                curves[name] = (frequency, error)
            else:
                assert np.abs(error).max() < 0.05, (name, np.abs(error).max())
            print(f"       {name:16s} {label:14s} {gain.max()-gain.min():6.2f} "
                  f" {target.max()-target.min():6.2f}  {error.mean():+7.3f}  "
                  f"{np.abs(error).max():6.3f} dB")
    print("       under GNPy's own convention the four agree to the solvers' "
          "tolerance;")
    print("       the residual under ours grows with depletion, which is the "
          "phonon term")
    return residuals, curves


def check_the_residual_is_ordered_by_depletion(residuals):
    """A defect would not sort itself by how hard the pumps are driven."""
    order = ["counter", "co", "counter_strong", "wideband"]
    values = [residuals[name] for name in order]
    assert values == sorted(values), values
    print("\nPASS the residual is ordered by depletion, not scattered:")
    print("       " + "  <  ".join(f"{name} {value:.3f}"
                                   for name, value in zip(order, values, strict=True)))
    print("       a wrong coefficient would not sort itself this way; a term "
          "that only exists")
    print("       when the pump is depleted is the one thing that would")


def check_the_spontaneous_emission(spectrum, peak_W_km):
    """The one piece of physics none of the stimulated checks touches."""
    frequency, theirs = reference_ase()
    solution = ours("counter", frequency, spectrum, peak_W_km, power_conserving)
    full = np.asarray(solution.ase_W)[:, -1]

    # the same integrating factor GNPy uses, restricted to its source term
    pumps = np.array([f for f, _ in CASES["counter"][0]])
    waves = np.concatenate([frequency, pumps])
    coupling = power_conserving(waves, peak_W_km, spectrum)
    signal = np.asarray(solution.signal_W)
    loss = signal / signal[:, :1]
    pump_profile = np.asarray(solution.pump_backward_W)
    pump_only = np.zeros(frequency.size)
    for index in range(pumps.size):
        shift = waves[frequency.size + index] - frequency
        source = (2 * PLANCK * frequency * BAUD_RATE
                  * coupling[:frequency.size, frequency.size + index]
                  * _photon_occupancy(shift, TEMPERATURE_K)) * (shift > 0)
        pump_only += source * np.trapz(
            pump_profile[index][None, :] / loss, np.asarray(solution.z_km), axis=1)
    pump_only *= loss[:, -1]

    shared = 10 * np.log10(pump_only / theirs)
    extra = 10 * np.log10(full / pump_only)
    total = 10 * np.log10(full / theirs)
    assert np.abs(shared).max() < 0.01, np.abs(shared).max()
    assert np.abs(total - extra).max() < 0.01, np.abs(total - extra).max()
    print(f"\nPASS spontaneous Raman ASE, {10*np.log10(theirs.min()*1e3):.2f} "
          f"to {10*np.log10(theirs.max()*1e3):.2f} dBm over "
          f"{frequency.size} channels:")
    print(f"       pump-sourced only, vs GNPy   worst {np.abs(shared).max():.3f} dB")
    print(f"       channel-to-channel, extra    mean  {extra.mean():+.3f} dB")
    print(f"       everything, vs GNPy          mean  {total.mean():+.3f} dB")
    print("       the shared model agrees to the third decimal; the whole "
          "difference is the")
    print("       inter-channel spontaneous emission GNPy does not model, "
          "not a disagreement")
    return frequency, theirs, full, pump_only


def main():
    import matplotlib.pyplot as plt

    spectrum, peak_W_km = measured_spectrum()
    residuals, curves = check_the_four_regimes(spectrum, peak_W_km)
    check_the_residual_is_ordered_by_depletion(residuals)
    frequency, theirs, full, pump_only = check_the_spontaneous_emission(
        spectrum, peak_W_km)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name in ("counter", "co", "counter_strong", "wideband"):
        reference_frequency, error = curves[name]
        axes[0].plot(reference_frequency / 1e12, error, label=name)
    axes[0].axhline(0, color="k", linewidth=1)
    axes[0].set_xlabel("channel frequency [THz]")
    axes[0].set_ylabel("error vs GNPy [dB]")
    axes[0].set_title("What the phonon term costs, by regime")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(frequency / 1e12, 1e3 * theirs * 1e3, "-",
                 label="GNPy (pump-sourced)")
    axes[1].plot(frequency / 1e12, 1e3 * pump_only * 1e3, "--",
                 label="comnumpy, same source")
    axes[1].plot(frequency / 1e12, 1e3 * full * 1e3, ":",
                 label="comnumpy, all sources")
    axes[1].set_xlabel("channel frequency [THz]")
    axes[1].set_ylabel(r"ASE power [$\mu$W]")
    axes[1].set_title("Spontaneous Raman emission")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "optical_raman_gnpy_regimes.png"
    plt.savefig(path, dpi=150)
    print(f"\nfigure written to {path}")


if __name__ == "__main__":
    main()
