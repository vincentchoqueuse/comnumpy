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
python validation/optical_gn_model.py
python validation/optical_raman_gnpy.py
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
| `optical_gn_model.py` | The Gaussian Noise model against **a published measurement and an independent split-step solver**: the closed form reproduces the 15-channel, 5 x 100 km `a_NL = -23.5` dB of Serena & Bononi to 0.20 dB; this library's own SSFM agrees to 0.01 dB on a five-channel WDM link and to better than 1 dB single-channel across 1-20 spans (the drift being the coherence exponent, measured at 0.21); the cubic law comes out at n = 3.006; the 27/8 Manakov-vs-scalar polarization factor of Table I is measured at 5.29 dB against 5.28; and the modulation-format gap the GN model is blind to is measured (16QAM 1.4 dB, QPSK 2.2 dB) against the published 15-channel 1.6 / 2.8 dB. `--full` reproduces the whole published link by SSFM (-23.15 dB, ~15 min) | Serena & Bononi, JLT 33(7), 2015; Poggiolini, JLT 30(24), 2012; GNPy `NliSolver` |
| `optical_raman_gnpy.py` | Counter-pumped Raman against **a second implementation**, on the one thing the analytic checks cannot see: the gain *shape*. 96 channels over 4.8 THz amplified by two counter-propagating pumps, against GNPy's own expected output and its measured SSMF profile. The table normalizes to 0.4396 /W/km, the textbook SSMF figure; the tilt comes out 7.17 dB against 6.59 dB and falls the right way; the effective-area correction cuts the tilt error from 1.42 to 0.58 dB and the worst channel from 2.24 to 1.21 dB. **The agreement is not exact and the residual is reported, not tuned** -- GNPy defaults to a perturbative solver where this one is exact, which is the open candidate | GNPy `RamanSolver`, BSD-3-Clause; D'Amico et al., JLT 40, 3499-3511 (2022) §III.D |
| `core_awgn_ser.py` | Monte-Carlo SER over AWGN matches theory for QPSK/16-QAM | Proakis, *Digital Communications*, §4.3 |
| `core_capacity.py` | Constellation-constrained capacity against a Monte-Carlo mutual-information estimate (independent method); shaping loss bounded by the 1.53 dB ultimate shaping gain and increasing with M; Rayleigh closed form vs simulation; water-filling optimal and its gain vanishing at high SNR | Ungerboeck, IEEE-IT 28(1), 1982; Forney & Wei, JSAC 7(6), 1989; Telatar, ETT 10(6), 1999 |
| `fading_doppler.py` | Tap autocorrelation of the classical spectrum matches the Bessel form `J0(2 pi f_D tau)` on four sampling grids; delay profiles reproduce their published RMS spread | Clarke, BSTJ 47(6), 1968; 3GPP TS 36.101 annex B.2 |
| `fading_tdl_3gpp.py` | The five TR 38.901 tapped delay line models against **two independent transcriptions** rather than a closed form: TDL-D and TDL-E match OpenAirInterface's C tables tap by tap and reproduce its Rice factors (13.3 dB, 22.0 dB); all five satisfy the invariant the normalization guarantees (RMS spread = the requested scale, within 3e-4 for four of them and 6e-3 for TDL-D, whose published powers do not round to it); and TDL-A scaled to 30 ns reproduces the 12-tap TS 38.104 conformance profile's spread to 4e-5 and its longest path to 1e-3 | 3GPP TR 38.901 §7.7.2; TS 38.104 Table G.2.1.1-2; Sionna TDL model files; OpenAirInterface5G `random_channel.c` |
| `mimo_diversity_ber.py` | Three diversity schemes at equal transmit power -- one antenna, receive combining, Alamouti -- each within 4.4 % of `compute_ser_rayleigh_psk`, and the 3 dB an orthogonal design pays for transmitting blind measured at 3.0 dB against 10log10(N_t) | Simon & Alouini §9.2; Alamouti 1998 |
| `mimo_zf_ml_ber.py` | Zero forcing on a 2x2 i.i.d. Rayleigh channel matches the closed-form SISO Rayleigh BER (diversity 1); ML detection shows diversity 2 | Proakis §14.4; Tse & Viswanath §3.3 |
| `ofdm_awgn_ser.py` | SER of the complete OFDM chain over AWGN matches plain QAM theory -- validates the whole stack, not one block | Proakis §4.3 (Parseval through the orthonormal FFT) |
| `ofdm_papr_ccdf.py` | PAPR CCDF against the Rayleigh/exponential reference, at Nyquist rate and 4x oversampled; the fitted exponent converges to the published 2.8N | van Nee & Prasad, *OFDM for Wireless Multimedia Communications*, ch. 2 |
| `fec_coding_gain.py` | Soft-decision Viterbi beats hard decisions, which beat the uncoded baseline, for the (133, 171) K=7 code | Proakis & Salehi §8.2 |
| `fec_union_bound.py` | Simulated Viterbi BER stays under the union bound built from the code's own enumerated distance spectrum, and the bound tightens with Eb/N0 | Proakis & Salehi §8.2; Viterbi, IEEE Trans. Comm. 1971 |
