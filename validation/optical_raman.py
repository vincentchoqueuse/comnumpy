r"""Distributed Raman amplification vs closed forms and conservation laws.

Four confrontations of ``comnumpy.optical.raman`` (decision D7), chosen
so that the hard regime is covered too -- the undepleted formula is
exact but says nothing once the signal starts eating the pump, which is
precisely where a numerical solver could be wrong and look right.

0. **Exact solution of the simple case.** Co-propagating pumping
   without loss has a closed form valid under *arbitrary* depletion,
   not just a weak-pump approximation. Photon-number conservation
   eliminates the pump, and what is left is a logistic equation

   .. math::

       \frac{\mathrm{d} P_s}{\mathrm{d}z} = r P_s
           \left(1 - \frac{P_s}{P_s^{\infty}}\right),
       \qquad
       P_s(z) = \frac{P_s^{\infty}}
                {1 + \left(\frac{P_s^{\infty}}{P_s(0)} - 1\right) e^{-r z}}

   with :math:`r = g\left(P_p(0) + \frac{\nu_p}{\nu_s} P_s(0)\right)`
   and :math:`P_s^{\infty} = P_s(0) + \frac{\nu_s}{\nu_p} P_p(0)` the
   power the signal reaches once every pump photon has been converted.
   With **equal** losses the same solution holds in the effective
   length :math:`\zeta(z) = (1 - e^{-\alpha z})/\alpha`, because the
   substitution :math:`Q = P e^{\alpha z}` turns the pair back into the
   lossless one. This is the strongest reference here: it pins the
   whole profile, not one number, in the regime the approximation
   below cannot describe.

1. **Undepleted limit.** With a weak pump the on-off gain has the
   closed form

   .. math::

       G_\mathrm{on\text{-}off} = \exp\left(g P_p L_\mathrm{eff}\right),
       \qquad L_\mathrm{eff} = \frac{1 - e^{-\alpha_p L}}{\alpha_p}

   which all three pumping schemes must reproduce, and which they must
   then fall *below* as the pump rises -- the formula ignores
   depletion, so it can only overstate.

2. **Photon-number conservation.** Switch the losses off and one pump
   photon becomes exactly one signal photon, so

   .. math::

       \frac{P_s(z)}{\nu_s} + \frac{P_p(z)}{\nu_p} = \text{const}

   This holds *under arbitrary depletion*, so it is the reference for
   the regime where no closed form exists. It is also what catches a
   wrong :math:`\nu_p / \nu_s` factor, which the undepleted check
   cannot see at all.

3. **Co- versus counter-pumping.** The two agree in the undepleted
   limit -- the pump integral along the span is the same by symmetry --
   and separate as depletion sets in, with the counter-propagating
   pump always ahead: it is strongest where the signal is weakest, so
   it depletes less. A sign error on the direction flip breaks this.

4. **The boundary value problem against the initial value problem.**
   Co-propagating pumping is an IVP and is solved as one. Feeding the
   same configuration to the BVP path with a vanishing counter pump
   must give the same answer, which tests the boundary conditions
   themselves rather than the physics.

References
----------
G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013,
Chapter 8.

M. N. Islam, "Raman amplifiers for telecommunications", IEEE J. Sel.
Topics Quantum Electron., vol. 8, no. 3, pp. 548-559, 2002.
"""
import pathlib

import numpy as np

from comnumpy.optical.raman import get_gain_spectrum, solve_raman

FIG_DIR = pathlib.Path(__file__).parent / "figures"

LENGTH_KM = 80.0
GAIN_W_KM = 0.4                    # g_R/A_eff, typical SMF at a 13 THz shift
ALPHA_S_DB, ALPHA_P_DB = 0.2, 0.25
PUMPS_W = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
NU_S = 2.99792458e8 / 1550e-9
NU_P = 2.99792458e8 / 1455e-9


def effective_length_km(alpha_dB_km, length_km):
    alpha = alpha_dB_km / (10 / np.log(10))
    return (1 - np.exp(-alpha * length_km)) / alpha


def undepleted_gain_dB(pump_W):
    return 10 * np.log10(np.exp(GAIN_W_KM * pump_W
                                * effective_length_km(ALPHA_P_DB, LENGTH_KM)))


def solve(**pumping):
    return solve_raman(length_km=LENGTH_KM, gain_peak_W_km=GAIN_W_KM,
                       alpha_signal_dB_km=ALPHA_S_DB,
                       alpha_pump_dB_km=ALPHA_P_DB, **pumping)


def exact_copumped_signal(z_km, signal_W, pump_W, alpha_dB_km=0.0):
    r"""Exact co-pumped signal profile, valid under full depletion.

    Photon-number conservation removes the pump from the pair, leaving
    a logistic equation in :math:`z` -- or in the effective length when
    the two losses are equal. See the module docstring for the
    derivation.
    """
    alpha = alpha_dB_km / (10 / np.log(10))
    zeta = z_km if alpha == 0 else (1 - np.exp(-alpha * z_km)) / alpha
    limit = signal_W + (NU_S / NU_P) * pump_W
    rate = GAIN_W_KM * (pump_W + (NU_P / NU_S) * signal_W)
    profile = limit / (1 + (limit / signal_W - 1) * np.exp(-rate * zeta))
    return profile * np.exp(-alpha * z_km)


def check_exact_solution():
    """0. The closed form of the simple case, over the whole profile."""
    worst = {"lossless": 0.0, "equal losses": 0.0}
    for pump in (0.2, 0.5, 1.0, 2.0):
        for label, alpha_dB in (("lossless", 0.0), ("equal losses", 0.22)):
            solution = solve_raman(
                length_km=LENGTH_KM, gain_peak_W_km=GAIN_W_KM,
                pump_forward_W=pump, alpha_signal_dB_km=alpha_dB,
                alpha_pump_dB_km=alpha_dB, bandwidth_Hz=0.0, tol=1e-12)
            exact = exact_copumped_signal(solution.z_km, 1e-3, pump, alpha_dB)
            error = float(np.max(np.abs(solution.signal_W - exact) / exact))
            worst[label] = max(worst[label], error)
            assert error < 1e-9, (
                f"{label} at {pump} W: the profile deviates from the exact "
                f"logistic by {error:.3e}")
        depletion = 1 - float(solve_raman(
            length_km=LENGTH_KM, gain_peak_W_km=GAIN_W_KM, pump_forward_W=pump,
            alpha_signal_dB_km=0.0, alpha_pump_dB_km=0.0,
            bandwidth_Hz=0.0).pump_forward_W[-1] / pump)
        assert depletion > 0.5, \
            f"the check is weak at {pump} W: only {depletion:.1%} depleted"
    print(f"PASS exact logistic solution over the whole profile: "
          f"{worst['lossless']:.2e} lossless, {worst['equal losses']:.2e} "
          f"with equal losses, with the pump depleted by more than 50 %")
    return worst


def check_undepleted_limit():
    """1. Every scheme lands on the closed form, then falls below it."""
    weak = 0.05
    reference = undepleted_gain_dB(weak)
    errors = {}
    for name, pumping in (("co", dict(pump_forward_W=weak)),
                          ("counter", dict(pump_backward_W=weak)),
                          ("bidirectional", dict(pump_forward_W=weak / 2,
                                                 pump_backward_W=weak / 2))):
        errors[name] = abs(solve(**pumping).on_off_gain_dB - reference)
        assert errors[name] < 0.01, \
            f"{name} pumping deviates by {errors[name]:.4f} dB at 50 mW"
    # and the formula must become optimistic, never pessimistic
    for pump in PUMPS_W:
        measured = solve(pump_forward_W=pump).on_off_gain_dB
        assert measured <= undepleted_gain_dB(pump) + 1e-9, \
            f"at {pump} W the solver exceeds the undepleted bound"
    print("PASS undepleted limit: "
          + ", ".join(f"{k} {v:.4f} dB" for k, v in errors.items())
          + f" from the closed form {reference:.4f} dB; "
          f"the bound is never exceeded over {PUMPS_W[0]}-{PUMPS_W[-1]} W")
    return errors


def check_photon_conservation():
    """2. The invariant that survives depletion."""
    nu_s = 2.99792458e8 / 1550e-9
    nu_p = 2.99792458e8 / 1455e-9
    worst = 0.0
    for pump in (0.2, 0.5, 1.0):
        solution = solve_raman(length_km=LENGTH_KM, gain_peak_W_km=GAIN_W_KM,
                               pump_forward_W=pump, alpha_signal_dB_km=0.0,
                               alpha_pump_dB_km=0.0, bandwidth_Hz=0.0,
                               tol=1e-12)
        flux = solution.signal_W / nu_s + solution.pump_forward_W / nu_p
        drift = float(np.ptp(flux) / flux[0])
        depletion = 1 - float(solution.pump_forward_W[-1] / pump)
        worst = max(worst, drift)
        assert drift < 1e-8, f"photon flux drifts by {drift:.2e} at {pump} W"
        assert depletion > 0.2, \
            f"the check is vacuous at {pump} W: the pump only depleted " \
            f"by {depletion:.1%}"
    print(f"PASS photon-number conservation: worst drift {worst:.2e} with "
          f"the pump depleted by more than 20 %")
    return worst


def check_pumping_schemes():
    """3. Convergence at low pump, ordered divergence under depletion."""
    gaps = []
    for pump in PUMPS_W:
        co = solve(pump_forward_W=pump).on_off_gain_dB
        counter = solve(pump_backward_W=pump).on_off_gain_dB
        both = solve(pump_forward_W=pump / 2,
                     pump_backward_W=pump / 2).on_off_gain_dB
        assert co <= both <= counter + 1e-9, (
            f"at {pump} W the schemes are out of order: co {co:.4f}, "
            f"bidirectional {both:.4f}, counter {counter:.4f}")
        gaps.append(counter - co)
    assert gaps[0] < 0.01, \
        f"the schemes must agree in the undepleted limit, got {gaps[0]:.4f} dB"
    assert np.all(np.diff(gaps) > 0), \
        f"the gap must widen monotonically with the pump, got {gaps}"
    print(f"PASS pumping schemes: counter-co gap {gaps[0]:.4f} dB at "
          f"{PUMPS_W[0]} W growing monotonically to {gaps[-1]:.4f} dB at "
          f"{PUMPS_W[-1]} W, bidirectional between the two throughout")
    return gaps


def check_bvp_against_ivp():
    """4. The two solver paths must agree where both are valid."""
    worst = 0.0
    for pump in (0.2, 0.5, 1.0):
        ivp = solve(pump_forward_W=pump).on_off_gain_dB
        # a vanishing counter pump forces the boundary value path without
        # changing the physics
        bvp = solve(pump_forward_W=pump, pump_backward_W=1e-12).on_off_gain_dB
        worst = max(worst, abs(ivp - bvp))
        assert abs(ivp - bvp) < 1e-3, \
            f"at {pump} W the IVP gives {ivp:.6f} dB and the BVP {bvp:.6f} dB"
    print(f"PASS boundary conditions: the BVP path reproduces the IVP path "
          f"to {worst:.2e} dB")
    return worst


def main():
    import matplotlib.pyplot as plt

    check_exact_solution()
    check_undepleted_limit()
    check_photon_conservation()
    check_pumping_schemes()
    check_bvp_against_ivp()

    fig, (ax_gain, ax_profile, ax_spectrum) = plt.subplots(
        1, 3, figsize=(14, 4.2))

    ax_gain.plot(PUMPS_W, [undepleted_gain_dB(p) for p in PUMPS_W], "k--",
                 label="undepleted closed form")
    for name, key, marker in (("co-propagating", "pump_forward_W", "o"),
                              ("counter-propagating", "pump_backward_W", "s")):
        ax_gain.plot(PUMPS_W, [solve(**{key: p}).on_off_gain_dB for p in PUMPS_W],
                     marker + "-", fillstyle="none", label=name)
    ax_gain.set(xlabel="pump power [W]", ylabel="on-off gain [dB]",
                title=f"{LENGTH_KM:.0f} km span")
    ax_gain.grid(True)
    ax_gain.legend(fontsize="small")

    for name, key in (("co-propagating", "pump_forward_W"),
                      ("counter-propagating", "pump_backward_W")):
        solution = solve(**{key: 0.5})
        ax_profile.plot(solution.z_km, solution.gain_profile_dB, label=name)
    ax_profile.set(xlabel="distance [km]", ylabel="accumulated on-off gain [dB]",
                   title="where the gain is delivered, 500 mW")
    ax_profile.grid(True)
    ax_profile.legend(fontsize="small")

    shift = np.linspace(0, 25e12, 2000)
    for name in ("blow-wood", "triangular"):
        spectrum = get_gain_spectrum(name)
        ax_spectrum.plot(shift / 1e12, spectrum.shape(shift),
                         label=f"{name} (peak {spectrum.peak_shift_THz:.2f} THz)")
    ax_spectrum.axvline(13.2, color="0.6", ls=":", label="published 13.2 THz")
    ax_spectrum.set(xlabel="Stokes shift [THz]", ylabel="normalized gain",
                    title="gain shapes")
    ax_spectrum.grid(True)
    ax_spectrum.legend(fontsize="small")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "optical_raman.png", dpi=150)


if __name__ == "__main__":
    main()
