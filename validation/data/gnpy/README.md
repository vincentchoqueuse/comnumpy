# Third-party reference data: GNPy

Files in this directory come from **GNPy** (the Telecom Infra Project's
open-source optical network planning tool) and are redistributed under
its own licence, **BSD-3-Clause**, reproduced in `LICENSE.BSD-3-Clause`.
comnumpy itself is MIT; BSD-3-Clause is compatible with it, and the
obligation the licence attaches to *these files* is the one satisfied
here — the copyright notice, the conditions and the disclaimer travel
with them.

Two things this directory deliberately does **not** do. It does not put
any of this material in `src/`, so the installed library stays free of
third-party data and anyone vendoring comnumpy inherits no extra
licence obligation. And it does not claim, or imply, that the Telecom
Infra Project or the GNPy contributors endorse comnumpy — the third
clause of the licence forbids exactly that. GNPy is used here as an
independent implementation to disagree with, which is the only role a
reference implementation should have.

| File | What it is | Origin |
|---|---|---|
| `raman_gain_ssmf.csv` | Measured SSMF Raman gain profile, 90 points from 0 to 42 THz of Stokes shift, referenced at 206.185 THz (1454 nm) and an effective area of 75.75 um^2 | `gnpy/core/parameters.py`, `DEFAULT_RAMAN_COEFFICIENT`. GNPy cites D'Amico *et al.*, *J. Lightwave Technol.* **40**, 3499-3511 (2022), Section III.D |
| `raman_reference_expected.csv` | Per-channel signal, ASE and NLI power at the output of the reference span below, computed by GNPy's own solver | `tests/data/test_raman_fiber_expected_results.csv` |

## Why this data and not a closed form

`validation/optical_raman.py` already confronts the Raman solver with an
exact solution valid under arbitrary depletion, with photon-number
conservation, and with the undepleted limit. Those are real checks of
the **integrator**, and they are silent about the **gain spectrum**:
photon conservation is imposed by the coupling matrix by construction
(`C_ji = -(nu_j/nu_i) C_ij`), and the closed form is derived from the
same two-wave system. A wrong `g(delta nu)` passes all of them.

The reference case below fixes that, because its answer is dominated by
the shape of the gain spectrum rather than by its magnitude.

## The reference case

From `tests/data/test_science_utils_fiber_config.json` and
`tests/test_science_utils.py::test_raman_fiber`:

- 80 km of SSMF, 0.2 dB/km, 0.5 dB of connector loss in and out
- two **counter-propagating** pumps: 224.403 mW at 205 THz and
  231.135 mW at 201 THz
- 96 channels from 191.3 to 196.1 THz, 32 GBd on a 50 GHz grid,
  0 dBm each, roll-off 0.15
- temperature 283 K

Decoded from the expected results, the case asks for:

| Quantity | Value |
|---|---|
| Passive span loss | 17.0 dB |
| On-off Raman gain | 12.15 dB at 191.3 THz down to 5.56 dB at 196.05 THz |
| Gain tilt across the band | 6.59 dB |
| OSNR | 36.7 to 39.2 dB |

The **sign** of that tilt is the part worth having. The Raman gain peaks
around 12.75 THz below the pump, so the 205 THz pump peaks near
192 THz -- the *bottom* of the band -- while the 196 THz channels sit
only 9 THz from it, on the rising flank. More gain at the low end is
therefore a prediction of the spectrum's shape, not of its scale, and
reproducing 6.59 dB of tilt is a statement about the shape.
