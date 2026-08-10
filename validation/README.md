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
| `optical_raman.py` | Distributed Raman: on-off gain against the undepleted closed form `exp(g P_p L_eff)` for co, counter and bidirectional pumping; photon-number conservation `P_s/nu_s + P_p/nu_p` in the lossless limit, which holds *under depletion* where no closed form does; the counter/co gap opening monotonically with the pump; and the BVP path reproducing the IVP path | Agrawal, *Nonlinear Fiber Optics*, ch. 8; Islam, IEEE JSTQE 8(3), 2002 |
| `optical_wdm_opticommpy.py` | A published 11 x 32 GBd PDM-16QAM WDM transmission over 700 km, reproduced against a **second implementation** rather than a closed form: the comb power matches the printed 8.41 dBm exactly, the ASE-limited SNR matches its own closed form to 0.02 dB, a fixed 0.5 km split step is shown to be 0.8 dB short of converged, the MI, GMI and normalized GMI land on the published 4.00 / 4.00 / 1.00, and the full link lands 1.0 dB above the published 20.63 dB -- the sign every impairment the notebook models and this one does not would give | OptiCommPy, `test_WDM_transmission.ipynb`; Essiambre et al., JLT 28(4), 2010; Marcuse et al., JLT 15(9), 1997 |
| `core_awgn_ser.py` | Monte-Carlo SER over AWGN matches theory for QPSK/16-QAM | Proakis, *Digital Communications*, §4.3 |
| `core_capacity.py` | Constellation-constrained capacity against a Monte-Carlo mutual-information estimate (independent method); shaping loss bounded by the 1.53 dB ultimate shaping gain and increasing with M; Rayleigh closed form vs simulation; water-filling optimal and its gain vanishing at high SNR | Ungerboeck, IEEE-IT 28(1), 1982; Forney & Wei, JSAC 7(6), 1989; Telatar, ETT 10(6), 1999 |
| `fading_doppler.py` | Tap autocorrelation of the classical spectrum matches the Bessel form `J0(2 pi f_D tau)` on four sampling grids; delay profiles reproduce their published RMS spread | Clarke, BSTJ 47(6), 1968; 3GPP TS 36.101 annex B.2 |
| `mimo_diversity_ber.py` | Three diversity schemes at equal transmit power -- one antenna, receive combining, Alamouti -- each within 4.4 % of `compute_ser_rayleigh_psk`, and the 3 dB an orthogonal design pays for transmitting blind measured at 3.0 dB against 10log10(N_t) | Simon & Alouini §9.2; Alamouti 1998 |
| `mimo_zf_ml_ber.py` | Zero forcing on a 2x2 i.i.d. Rayleigh channel matches the closed-form SISO Rayleigh BER (diversity 1); ML detection shows diversity 2 | Proakis §14.4; Tse & Viswanath §3.3 |
| `ofdm_awgn_ser.py` | SER of the complete OFDM chain over AWGN matches plain QAM theory -- validates the whole stack, not one block | Proakis §4.3 (Parseval through the orthonormal FFT) |
| `ofdm_papr_ccdf.py` | PAPR CCDF against the Rayleigh/exponential reference, at Nyquist rate and 4x oversampled; the fitted exponent converges to the published 2.8N | van Nee & Prasad, *OFDM for Wireless Multimedia Communications*, ch. 2 |
| `fec_coding_gain.py` | Soft-decision Viterbi beats hard decisions, which beat the uncoded baseline, for the (133, 171) K=7 code | Proakis & Salehi §8.2 |
| `fec_union_bound.py` | Simulated Viterbi BER stays under the union bound built from the code's own enumerated distance spectrum, and the bound tightens with Eb/N0 | Proakis & Salehi §8.2; Viterbi, IEEE Trans. Comm. 1971 |
