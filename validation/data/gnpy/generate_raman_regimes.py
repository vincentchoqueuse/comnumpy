"""Regenerate ``raman_regimes_expected.csv`` and ``raman_ase_expected.csv``.

This script is committed for **provenance**, not for continuous
integration: it needs GNPy installed (``pip install gnpy``, BSD-3-Clause)
and is not imported by the test suite.  It is here so that anyone can
audit, or reproduce, exactly how the reference numbers next to it were
obtained -- including the two settings that turn out to matter.

Two things must be pinned explicitly, and neither is the default.

**The solver method.**  ``RamanParams`` defaults to
``method='perturbative', order=2``.  On a counter-pumped span that is
already converged, and every setting agrees to the last digit.  On a
*co-propagating* span, where the pump falls by 25 dB while travelling
with the signals it feeds, the expansion is used far outside its domain:
the default disagrees with GNPy's own ``numerical`` integration by about
0.4 dB, and with the perturbative solution at order 4 by a comparable
amount.  Everything here is therefore generated with ``numerical``.

**The spatial step.**  ``solver_spatial_resolution`` is generated at both
20 m and 5 m so that the reference can state its own convergence rather
than assume it; the shipped files are the 5 m ones, and the two differ by
at most 0.030 dB on the worst case.

Note also that calling ``RamanSolver`` directly, as here, bypasses
``Fiber.propagate``: the connector loss is **not** applied to the signals
(``power_profile[:, 0]`` is exactly the launch power) but **is** applied
to the pumps inside the solver.  ``raman_reference_expected.csv``, which
comes from GNPy's own test data, went through the full element instead
and has both.  Mixing the two accountings silently cancels a 0.5 dB pump
error against a 1 dB connector error, which is a way to look right.
"""
import json

import numpy as np
from gnpy.core.elements import RamanFiber
from gnpy.core.info import create_input_spectral_information
from gnpy.core.parameters import SimParams
from gnpy.core.science_utils import RamanSolver

BAUD_RATE = 32e9
TEMPERATURE_K = 283

# The span of GNPy's tests/data/test_science_utils_fiber_config.json.
FIBRE = {
    "uid": "Span1",
    "params": {"length": 80, "loss_coef": 0.2, "length_units": "km",
               "att_in": 0, "con_in": 0.5, "con_out": 0.5,
               "type_variety": "SSMF", "dispersion": 1.67e-5,
               "effective_area": 83e-12, "pmd_coef": 1.265e-15},
    "operational": {"temperature": TEMPERATURE_K, "raman_pumps": []},
    "metadata": {"location": {"latitude": 1, "longitude": 0,
                              "city": None, "region": ""}},
}

# Four regimes, chosen so that a single explanation cannot cover all four:
# the first is the easy one, the second reverses the pump direction, the
# third drives the pumps to saturation, the fourth spans 10 THz so that the
# effective-area scaling is asked to extrapolate well outside the C band.
CASES = {
    "counter": ([(205e12, 0.199999, "counterprop"),
                 (201e12, 0.205999, "counterprop")], 191.3e12, 196.1e12),
    "co": ([(205e12, 0.199999, "coprop"),
            (201e12, 0.205999, "coprop")], 191.3e12, 196.1e12),
    "counter_strong": ([(205e12, 0.60, "counterprop"),
                        (201e12, 0.60, "counterprop")], 191.3e12, 196.1e12),
    "wideband": ([(215e12, 0.30, "counterprop"),
                  (209e12, 0.25, "counterprop"),
                  (203e12, 0.20, "counterprop")], 186.0e12, 196.1e12),
}


def solve(pumps, f_min, f_max, resolution):
    SimParams.set_params({
        "raman_params": {"flag": True, "result_spatial_resolution": 1e3,
                         "solver_spatial_resolution": resolution,
                         "method": "numerical", "order": 1},
        "nli_params": {"method": "ggn_spectrally_separated",
                       "dispersion_tolerance": 1, "phase_shift_tolerance": 0.1,
                       "computed_channels": [1]}})
    spectral_info = create_input_spectral_information(
        f_min=f_min, f_max=f_max, roll_off=0.15, baud_rate=BAUD_RATE,
        spacing=50e9, tx_osnr=40.0, tx_power=1e-3)
    fibre = json.loads(json.dumps(FIBRE))
    fibre["operational"]["raman_pumps"] = [
        {"power": p, "frequency": f, "propagation_direction": d}
        for f, p, d in pumps]
    fibre = RamanFiber(**fibre)
    return spectral_info, fibre, RamanSolver.calculate_stimulated_raman_scattering(
        spectral_info, fibre)


def main():
    rows = ["case,frequency_Hz,output_W"]
    for name, (pumps, f_min, f_max) in CASES.items():
        coarse = solve(pumps, f_min, f_max, 20)[2]
        spectral_info, _, fine = solve(pumps, f_min, f_max, 5)
        n = spectral_info.frequency.size
        drift = np.abs(10 * np.log10(fine.power_profile[:n, -1]
                                     / coarse.power_profile[:n, -1])).max()
        print(f"{name:15s} {n:3d} channels, 20 m vs 5 m: {drift:.3f} dB")
        for frequency, power in zip(fine.frequency[:n], fine.power_profile[:n, -1], strict=True):
            rows.append(f"{name},{frequency:.6e},{power:.12e}")
    with open("raman_regimes_expected.csv", "w") as handle:
        handle.write("\n".join(rows) + "\n")

    # ASE of the counter-pumped case.  GNPy returns it referred to the fibre
    # input, so the physical output is the product with the loss profile --
    # that is what Fiber.propagate does with it.
    pumps, f_min, f_max = CASES["counter"]
    spectral_info, fibre, srs = solve(pumps, f_min, f_max, 5)
    n = spectral_info.frequency.size
    ase = (RamanSolver.calculate_spontaneous_raman_scattering(
        spectral_info, srs, fibre) * srs.loss_profile[:n, -1])
    rows = ["frequency_Hz,ase_W"]
    for frequency, power in zip(srs.frequency[:n], ase, strict=True):
        rows.append(f"{frequency:.6e},{power:.12e}")
    with open("raman_ase_expected.csv", "w") as handle:
        handle.write("\n".join(rows) + "\n")
    print(f"ASE {10 * np.log10(ase.min() * 1e3):.2f} .. "
          f"{10 * np.log10(ase.max() * 1e3):.2f} dBm")


if __name__ == "__main__":
    main()
