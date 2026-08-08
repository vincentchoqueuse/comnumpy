# Validation scripts

Standalone scripts comparing comnumpy simulations against analytical or
published references (architecture decision D7). Each script produces a
figure under `validation/figures/` **and** asserts the comparison, so it
doubles as an executable proof: if a script exits 0, the curve it plots
is the curve the library computes.

Run them from anywhere:

```bash
python validation/optical_chromatic_dispersion.py
python validation/optical_spm_phase.py
python validation/optical_soliton.py
python validation/optical_dbp_roundtrip.py
python validation/core_awgn_ser.py
```

| Script | What it validates | Reference |
|---|---|---|
| `optical_chromatic_dispersion.py` | Gaussian pulse broadening under CD matches the closed-form width to machine precision; energy conserved | Agrawal, *Nonlinear Fiber Optics*, §3.2 |
| `optical_spm_phase.py` | Kerr-only propagation applies exactly the analytic self-phase-modulation rotation; with loss it converges to the effective-length formula in O(StPS^-2) | Agrawal, §4.1 |
| `optical_soliton.py` | A fundamental soliton propagates shape-invariant through the full SSFM (validates the sign consistency between CD and Kerr steps and the step normalization); the same pulse disperses when Kerr is off | Agrawal, §5.2 |
| `optical_dbp_roundtrip.py` | Noiseless DBP with matched steps inverts the fiber to machine precision; CD-only equalization leaves a nonlinear residual growing with launch power | Ip & Kahn, JLT 26(20), 2008 |
| `core_awgn_ser.py` | Monte-Carlo SER over AWGN matches theory for QPSK/16-QAM | Proakis, *Digital Communications*, §4.3 |
