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
| `fading_doppler.py` | Tap autocorrelation of the classical spectrum matches the Bessel form `J0(2 pi f_D tau)` on four sampling grids; delay profiles reproduce their published RMS spread | Clarke, BSTJ 47(6), 1968; 3GPP TS 36.101 annex B.2 |
| `mimo_zf_ml_ber.py` | Zero forcing on a 2x2 i.i.d. Rayleigh channel matches the closed-form SISO Rayleigh BER (diversity 1); ML detection shows diversity 2 | Proakis §14.4; Tse & Viswanath §3.3 |
| `ofdm_awgn_ser.py` | SER of the complete OFDM chain over AWGN matches plain QAM theory -- validates the whole stack, not one block | Proakis §4.3 (Parseval through the orthonormal FFT) |
| `ofdm_papr_ccdf.py` | PAPR CCDF against the Rayleigh/exponential reference, at Nyquist rate and 4x oversampled; the fitted exponent converges to the published 2.8N | van Nee & Prasad, *OFDM for Wireless Multimedia Communications*, ch. 2 |
| `fec_coding_gain.py` | Soft-decision Viterbi beats hard decisions, which beat the uncoded baseline, for the (133, 171) K=7 code | Proakis & Salehi §8.2 |
| `fec_union_bound.py` | Simulated Viterbi BER stays under the union bound built from the code's own enumerated distance spectrum, and the bound tightens with Eb/N0 | Proakis & Salehi §8.2; Viterbi, IEEE Trans. Comm. 1971 |
