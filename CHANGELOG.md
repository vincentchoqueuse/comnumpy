# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

## [Unreleased] — 1.0.0.dev0

Two batches so far: the sanitation batch ("Lot 0" of the architecture
document) and the single breaking-change window of milestone 1
(decisions D2, D36, D38, D40, D41). All breaking changes land in this
one release; there is no compatibility layer.

### Breaking changes — migration table

| Before (0.91) | After (1.0.0) |
|---|---|
| `Serial2Parallel(N, order="F")` → shape `(N_sub, M)` | `Serial2Parallel(N)` → Block layout `(..., T, F)` = `(..., M, N_sub)`, pure C-order reshape; `order` removed |
| `Parallel2Serial(order="F")` | `Parallel2Serial()` — C-order flatten of `(..., T, F)`; `order` removed |
| OFDM blocks operate on axis 0 by default | OFDM blocks operate on the block content axis -1 (`FFTProcessor`/`IFFTProcessor` hardcode it; `axis` removed there) |
| `AWGN(value=15, unit="snr_dB", sigma2s_method="measured")` | `AWGN(snr_dB=15)` — exactly one of `snr_dB=` / `sigma2=`; applied variance exposed as `sigma2_` |
| `AWGN(0.01)` / `AWGN(value=0.01)` | `AWGN(sigma2=0.01)` |
| `compute_sigma2(value, unit)` | removed — use `AWGN(snr_dB=…)` / `AWGN(sigma2=…)`, and `ebn0_to_snr_dB()` / `esn0_to_snr_dB()` for chain-level conversions |
| `comnumpy.mimo.channels.AWGN` (duplicate class) | re-export of `comnumpy.core.channels.AWGN` (element-wise, shape-agnostic) |
| `is_mimo=` parameter on many blocks | removed (it was never read); blocks declare an axis category instead (see CONVENTIONS.md) |
| `ofdm.processors.OFDMTransmitter` (legacy duplicate) | removed — use `ofdm.chains.OFDMTransmitter` |
| Optional block parameters accepted positionally | keyword-only (D40b): only the principal first argument is positional |
| Bare `ValueError` on shape mismatch | `comnumpy.ShapeError` (still a `ValueError` subclass — existing `except ValueError` keeps working) |
| `import comnumpy` loaded matplotlib (~1 s) | lazy imports: no matplotlib at import time, ~90 ms (enforced in CI) |
| `Recorder(name="tx")` inserted in `module_list`, read with `chain["tx"].get_data()` | name the block itself and declare a tap: `Sequential([SymbolGenerator(16, name="tx"), ...], taps=["tx"])`, read with `chain.tap("tx")` |
| `Logger`, `Debugger`, `PowerReporter`, `TimeSignalMonitor` blocks | removed — tap the block and call `signal_report(chain.tap("x"))`, then log the returned dict |
| `MetricRecorder(metric_fn=f)` block | removed — `f(chain.tap("x"))` |
| `Scope(scope_type="iq", ...)`, `TimeScope`, `SpectrumScope`, `IQScope`, `KDEScope`, `WelchScope` blocks | removed — functions `plot_iq`, `plot_time`, `plot_spectrum`, `plot_kde`, `plot_welch` applied to a tapped signal |
| `ofdm.visualizers.FFTMonitor` block | `ofdm.visualizers.plot_subcarrier_amplitude(X, ax=...)` |
| `CarrierExtractor(..., pilot_recorder=rec)` | pilot content exposed as the estimated attribute `extractor.pilots_` (D23) |
| `TrainedBased*(target_data=recorder)` | `DataAided*(reference=…)` — the class family is renamed after the standard pair of the field (*data-aided* vs *blind*, which the `Blind*` classes already used), and the known signal an estimator compares against is a `reference`, the same word `sweep(reference=…)` uses. It takes a plain array; when the reference is produced by the chain itself, declare `wiring={"comp.reference": "source"}` |
| `DataAidedFIRCompensator(h, reference=…)` | `DataAidedFIRCompensator(reference=…)` — `h` was only an initial value that `fit` overwrote from scratch, i.e. a purely estimated quantity; it is now `h_` and not a constructor parameter |
| `Normalizer(gain, method, …)` (inherited from `Amplifier`) | `Normalizer(method, …)` — `gain` is no longer constructible, so `Normalizer('max')` finally means what it reads; the measured gain is `gain_` (D23) |
| `Amplifier(gain, axis=…)` | `Amplifier(gain)` — `axis` implemented no defensible model (it scaled only entries at index `axis` of the *last* axis, whatever axis was asked); use `WeightAmplifier` for a per-branch gain |
| `.gain`, `.alpha`/`.beta`, `.w0`, `.h`, `.delay`/`.scale`/`.cross_corr` read off a compensator | same names with a trailing underscore (D23): these are estimated from the data, and the convention now separates them from configured parameters |
| `FiberLink(alpha_dB=…, gamma=…, cd_coefficient=…, lamb=…, nu=…, c=…, h=…)` | `FiberLink(fiber=FiberSpec(alpha_dB, gamma=…, cd_coefficient=…, wavelength_nm=…))` (D46) — same for `DBP`. The carrier frequency is derived from the wavelength instead of being a second argument that could disagree with it; `c` and `h` are no longer settable. `FiberLink` goes from 21 constructor arguments to 15 |
| `TrainedBasedPhaseCompensator`, `TrainedBasedComplexGainCompensator`, `TrainedBasedSimpleSynchronizer`, `TrainedBasedFineSynchronizer` | `DataAidedPhaseCompensator`, `DataAidedComplexGainCompensator`, `DataAidedSimpleSynchronizer`, `DataAidedFineSynchronizer` (`DataAidedFIRCompensator` already had the right name) |
| `core.metrics.calculate_acpr` | `compute_acpr` — it was the only `calculate_*` in the library, against 17 `compute_*` |
| `core.metrics.compute_effective_SNR`, `ofdm.metrics.compute_PAPR` | `compute_effective_snr`, `compute_papr` — the two capitalized outliers among functions otherwise all lowercase (`compute_ser`, `compute_ber`, `compute_evm`, `compute_ccdf`, `compute_mi`) |

### Added (milestones 2-5)

- `RamanGainSpectrum(tabulated=(shift_Hz, gain))`: a **measured**
  spectrum, alongside the analytic `lorentzian=` and `triangular=`
  shapes. Outside the measured range the gain is zero rather than held
  at the last sample -- a table stops where the measurement stopped,
  and extrapolating a Raman spectrum off the end of the data is how a
  tilt gets invented.

- `validation/optical_raman_gnpy.py`: counter-pumped Raman against a
  second implementation. The six analytic confrontations in
  `optical_raman.py` share a blind spot -- photon conservation is
  imposed by the coupling matrix by construction, and the closed form
  comes from the same two-wave system, so a wrong gain *shape* passes
  all of them. This one is dominated by the shape, and it found a
  missing effective-area scaling on its first run.

  It also leaves a residual of +0.91 dB that is **not** explained. The
  obvious suspect -- GNPy defaulting to a perturbative solver where this
  library is exact -- was tested by installing GNPy and re-running its
  own reference case at perturbative orders 1 to 4 and with its
  `numerical` method. The harness reproduces their shipped file to
  5e-16 and all five settings agree to the last digit, so the expansion
  is already converged and the gap is ours. Recorded as an open
  question with the residual bounded, rather than closed with a story.

- `comnumpy.optical.gn_model`: the **Gaussian Noise model** in closed
  form -- `gn_model_psi` (eq. 123 of arXiv:1209.0394), `gn_model_nli_power`
  (eq. 120), `gn_model_snr` and `optimal_launch_power`. It answers "how
  much nonlinear interference will this link make, and what launch power
  minimizes the total noise" in microseconds where the split-step method
  takes minutes, which is what makes design sweeps over span length,
  channel count or amplifier noise figure possible at all.

  Validated three ways, because a model nobody can contradict is not
  validated. Against **GNPy** (Telecom Infra Project), re-transcribed in
  SI units in the tests -- metres and s^2/m against this library's
  kilometres and ps^2/km, so the agreement to twelve digits tests the
  unit conversions as well as the formula. Against a **published
  measurement**: the 15-channel, 5 x 100 km `a_NL = -23.5` dB of Serena
  & Bononi (JLT 33(7), 2015), reproduced to 0.20 dB. And against **this
  library's own split-step solver**, which shares no code with it: 0.01
  dB on a five-channel WDM link, and better than a decibel single-channel
  from 1 to 20 spans, the residual drift being the coherence exponent the
  model approximates away (measured at 0.21, exposed as
  `coherence_exponent=`).

  `polarizations=` exists because of a trap worth naming: the model's
  16/27 is a *Manakov* coefficient, so it describes a dual-polarization
  field `(..., 2, N)`. Hand `FiberLink` a one-dimensional field and it
  integrates the scalar equation instead, which makes 27/8 -- 5.3 dB --
  more interference at the same total power. Table I of Serena & Bononi
  gives the scalar weights (2 and 4); the validation script measures the
  ratio at 5.29 dB against the predicted 5.28.

  The **EGN model** -- the modulation-format correction -- is *not*
  implemented. `validation/optical_gn_model.py` and the tutorial measure
  it instead of predicting it, so the GN curve is never read as an exact
  answer for a modulated signal: it is a pessimistic bound, worth about
  1 dB for PM-16QAM and 2 dB for PM-QPSK on the links tested here.

- `docs/examples/gn_model.rst`, tutorial 10 and the last of the course:
  the same link designed twice, in closed form and by propagation, with
  the two overlaid on one figure (0.25 dB apart on the optimal launch
  power, 0.17 dB on the peak SNR).

- `sweep(n_jobs=...)`: the points of a sweep are independent by
  construction -- each one reconfigures and reseeds the chain from
  scratch -- so they can run at once. The curve is **identical** value
  for value, not merely statistically equivalent, because every point
  already draws from its own child seed; that is asserted in the tests,
  and it is why `seed=` becomes mandatory when `n_jobs` is not 1.

  Measured on four cores: below ~20 ms a point the pool costs more than
  it saves (starting a worker and pickling the chain into it is ~130 ms),
  at ~200 ms a point it returns 2.7x. Threads were measured too and are
  *four times slower* on a Viterbi sweep -- the decoder holds the GIL --
  so workers are processes, spawned rather than forked so the BLAS
  thread limits actually apply. The two traps that follow (a `lambda`
  metric cannot be pickled; a script must guard its body with
  `if __name__ == "__main__":`) raise errors that say exactly that,
  rather than letting `pickle` and `BrokenProcessPool` explain
  themselves.

- A channel coding tutorial (`examples/simple/channel_coding.py`,
  `docs/examples/coding.rst`), the one module of the library that had no
  tutorial at all: the convolutional encoder and what its generator
  polynomials mean, the Viterbi recursion and the 2 dB that separate
  hard from soft decisions, the union bound built from the code's own
  distance spectrum -- which is what answers the question a simulation
  cannot, since 40 000 bits cannot resolve a BER below 2.5e-5 -- and an
  LDPC code at the same rate, with its waterfall and what its iterations
  buy.

- `mimo.detectors.SphereDecoder`: maximum likelihood by tree search
  rather than exhaustion. The thin QR factorization of the channel makes
  the metric a sum of non-negative layer terms, so a partial sum that
  already exceeds the best complete metric prunes its whole subtree; the
  Schnorr-Euchner enumeration order reaches the successive-cancellation
  solution first, which supplies a finite radius immediately and removes
  the initial-radius parameter such decoders usually carry.

  It returns the *same* decision as `MaximumLikelihoodDetector` -- that
  is the only claim worth making, and it is tested symbol by symbol on
  six alphabet/geometry pairs, at three noise levels and on an
  ill-conditioned channel. What changes is the cost: on 16-QAM over
  four streams it visits 5.9 nodes per vector at 18 dB and 78 at 0 dB,
  against 65 536 candidates for the exhaustive search, and the average
  is reported as `nodes_`. On small problems it is *slower* than the
  vectorized exhaustive search, which the tutorial says plainly.

- A multipath fading tutorial (`examples/simple/multipath_channels.py`,
  `docs/examples/multipath.rst`), which is what the TDL catalog was
  missing: the power delay profile, the exact scaling law between delay
  spread and coherence bandwidth (B_c x sigma = 0.329 at three spreads,
  to three digits), and the cyclic prefix rule it all exists to settle
  -- 48 samples of prefix against a 40-sample channel reaches
  2e-3 at 30 dB, 10 samples floors at 0.145 whatever the SNR.

- The five 5G NR delay profiles of 3GPP TR 38.901 Section 7.7.2 --
  `TDL-A` to `TDL-E` -- in the `get_delay_profile` catalog, next to the
  LTE ones. Their delays are *normalized*, so a TDL entry is a shape and
  `delay_spread_ns=` is the RMS spread it is stretched to; TDL-D and
  TDL-E carry the Rice factor of their first tap (13.3 dB, 22.0 dB), and
  paths sharing a delay are merged into one resolvable tap.

  `validation/fading_tdl_3gpp.py` confronts them with three things that
  were not used to write them: OpenAirInterface's independent
  transcription in C (tap by tap on TDL-D/E, including its Rice
  constants), the invariant the normalization guarantees (the RMS spread
  of the table *is* the scale factor: four of the five land within 3e-4
  of it), and the 12-tap TS 38.104 conformance profile, which summarizes
  to the same channel -- same spread to 4e-5, same longest path to 1e-3,
  from half the taps and a different specification.

- The two blocks that turn `core/shaping.py` into a transmitter:
  `AmplitudeMapper` signs the shaped amplitudes (the sign is a free bit
  in PAS, spent on the FEC parity), `AmplitudeDemapper` reads the
  amplitude index back off `|y|` -- which is the maximum-likelihood
  decision on a constellation symmetric about the origin, not a
  shortcut. A shaped link is now an ordinary chain: bits, matcher,
  mapper, channel, demapper, dematcher.

- `SymbolGenerator(distribution=...)`: a source drawing from a
  non-uniform law, so what a *distribution* is worth can be measured
  without building a matcher for every parameter value. The docstring
  says what it is not -- an i.i.d. draw is the idealization a matcher
  approaches, never what it produces on a finite block.

- A probabilistic shaping tutorial (`examples/simple/probabilistic_shaping.py`,
  `docs/examples/shaping.rst`): the Maxwell-Boltzmann law and its
  shaping gain, CCDM against ESS at equal energy (where the CCDM code
  is a *subset* of the sphere, so the inequality is exact), the PAS
  chain and its measured distribution, and the achievable rate against
  a uniform 8-PAM -- 0.8 dB at best, not 1.53 dB, and the figure shows
  why.

- Closed-form error rates over Rayleigh fading, in `core/metrics.py`:
  `compute_ser_rayleigh_psk`, `compute_ser_rayleigh_qam` and the
  `compute_metric_rayleigh_theo` front-end. The library knew only AWGN,
  so a MIMO or diversity simulation had nothing analytical to be read
  against -- and the one closed form that existed was hardcoded inside
  `validation/mimo_zf_ml_ber.py`, where nothing could reuse it.

  One expression covers the four cases, through the MGF average of
  Simon and Alouini on Craig's contour: L branches at a per-branch SNR,
  with a *transmit* scheme dividing that SNR by N_t because it splits
  its power. So one antenna is L=1, receive combining is L=N_r,
  Alamouti is L=N_t*N_r at gamma/N_t, and zero forcing is
  L=N_r-N_t+1 per stream. That single division is the 3 dB an
  orthogonal design pays for transmitting blind.

  Checked three ways rather than trusted: against the closed form of
  Proakis for L-branch BPSK (1e-10 relative, L = 1 to 8), against the
  defining property that the high-SNR slope *is* L, and against chains
  built from the library's own blocks -- the only check that both sides
  mean the same thing by "SNR per branch".

  `validation/mimo_diversity_ber.py` runs that last confrontation where
  it belongs: three schemes, up to 80000 channel draws per point, all
  within 4.4 % of the curves, and the Alamouti penalty measured at
  3.0 dB against a predicted 10log10(2) = 3.01 dB.

- `core.visualizers.plot_error_rate`, the figure a Monte-Carlo sweep
  ends with: measurements as hollow markers, the closed form they are
  read against as a line of the same colour, a logarithmic ordinate and
  a grid on both decades. Fourteen scripts of this repository drew it by
  hand. Sweep points where no error was seen are dropped rather than
  plotted: they mean the estimate ran out of samples, not that the error
  rate is zero, and a logarithmic axis has no place to put them.

- A tutorial on Alamouti space-time coding,
  `docs/examples/alamouti.rst`, with `examples/mimo/one_shot_alamouti.py`
  behind it. It answers the question the code module cannot: *why*. The
  fading link is limited by its deep fades, not its average, so the
  error rate falls only one decade per decade; the Alamouti codeword
  turns two transmit antennas into two independent observations with no
  channel knowledge at the transmitter, its equivalent channel is
  orthogonal, and that identity alone makes the maximum-likelihood
  receiver a matched filter.

  The measurement is the lesson. Three links at **equal total transmit
  power** -- one antenna, Alamouti 2x1, and receive diversity 1x2 --
  give local slopes converging to 1, 2 and 2, and the SNR needed for a
  symbol error rate of 1e-3 is 28.0, 18.4 and 15.6 dB. The 2.8 dB
  between Alamouti and maximum ratio combining is the price of
  transmitting blind, and the page says why the power normalization is
  what makes that number mean anything: without it the two curves would
  land on top of each other and prove nothing.

- Space-time block codes, in `mimo/coding.py`, with a `get_code` registry
  answering by name as `get_alphabet` does for constellations:
  `alamouti`, Tarokh's four orthogonal designs (`ostbc-3-1/2`,
  `ostbc-4-1/2`, `ostbc-3-3/4`, `ostbc-4-3/4`), `golden` and
  `spatial-multiplexing`, plus `register_code` for user codes.

  Every code is stored as its **linear dispersion** matrices, `G(s) =
  sum_k A_k s_k + B_k conj(s_k)`, and that choice is what makes the
  module verifiable rather than a table of matrices. Writing the symbols
  in real and imaginary parts turns a received block into a *real*
  linear system whose matrix `M(H)` -- `SpaceTimeCode.equivalent_channel`
  -- carries every property of the code: a design is orthogonal exactly
  when `M^T M = c |H|_F^2 I`, its rate is `K/T`, and its diversity order
  is the minimum rank of a difference codeword.

  The orthogonality identity is checked **at construction** for every
  code declaring itself orthogonal (D20), which is not decoration: it
  refused a first transcription of Tarokh's G3 on the spot. The constant
  `c` is measured there too rather than assumed -- 1 for Alamouti, 2 for
  the rate-1/2 designs that repeat each symbol over a conjugated half --
  and the decoder divides by the measured value, so the two cannot
  disagree.

  `SpaceTimeDecoder` implements the matched filter that is *exactly*
  maximum likelihood for an orthogonal design, and the tests check that
  claim against an exhaustive ML search, sample by sample, including
  where noise makes both wrong. A non-orthogonal code is refused rather
  than silently zero-forced: `equivalent_channel(H)` hands the problem
  to the detectors of `mimo/detectors.py` instead.

  Provenance stated where it is not a plain transcription: the fourth
  row of H4 is the completion H3's structure allows, and which signs its
  half-sums carry is *determined* -- over the 256 sign patterns of that
  structure exactly two make the design orthogonal, and they differ by a
  global sign. The orthogonality identity fixes the convention, not a
  reading of the table.

  Measured, not asserted: the four orthogonal designs are full
  diversity, the Golden code is full rate (2) *and* full diversity with
  a non-vanishing determinant, spatial multiplexing has rank 1 and
  coding gain 0, and Alamouti's error curve has twice the slope of a
  single antenna at equal total transmit power (-2.03 against -0.98 over
  13 to 25 dB).

- Probabilistic shaping, in `core/shaping.py`. A uniform QAM loses up to
  1.53 dB of the Gaussian capacity -- the shaping gap -- and closing it
  needs two things, both provided. `maxwell_boltzmann` gives the target
  distribution, which is not a modelling choice but what maximizing the
  entropy under a power constraint *forces*; it is parameterized either
  by lambda itself or by the entropy it must reach (D41), the second
  being the useful one since the entropy is the rate.
  `ConstantCompositionMatcher` (CCDM, Schulte-Böcherer 2016) and
  `SphereShaper` (enumerative sphere shaping, Willems-Wuijts 1993,
  Gültekin et al. 2020) turn uniform bits into that distribution. Both
  are enumerative codes ranking and unranking a finite set with exact
  integer arithmetic, so `decode(encode(bits)) == bits` holds by
  construction rather than by tolerance, and the tests check it on
  thousands of random inputs and exhaustively on a small code.

  What they hold fixed differs, and so do their guarantees: CCDM emits
  its composition in *every* block, ESS only keeps every block inside an
  energy sphere -- which is a larger set at equal average energy, and is
  measured here to carry a higher rate than the best constant
  composition at short blocklengths, the reason it exists.

  `shaping_gain_dB` measures what the shaping is worth, against the
  continuous-uniform reference at equal entropy. Two properties pin the
  definition: a uniform distribution gives exactly
  `10*log10(M^2/(M^2-1))`, i.e. 0.28 dB at M=4 and 0.0007 dB at M=64, and
  no distribution over any tested constellation exceeds
  `10*log10(pi*e/6) = 1.5329` dB.

  Defect found while writing it, and worth naming because it was silent:
  `maxwell_boltzmann` returned a distribution whose entropy was *not*
  the one requested when the target was unreachable. Points of equal
  energy always keep equal probability, so a symmetric constellation
  cannot be shaped below one bit per symbol whatever lambda does; the
  bisection saturated and returned the floor without saying so. It now
  refuses, and names both the floor and why it exists.

- The pyright strict ratchet of D37 now covers **every** module of
  `src/comnumpy`, and `py.typed` ships with the package. Closing the
  last four -- `core/metrics.py`, `core/processors.py`,
  `core/compensators.py`, `mimo/detectors.py` -- was not only
  annotation. What the type checker was pointing at, in every case, was
  a value that could be `None` and was dereferenced anyway:

  - `MaximumLikelihoodDetector.get_nb_candidates()` read `self.H.shape`
    with no check at all, so it raised `AttributeError: 'NoneType'` on a
    detector whose channel had not been set;
  - `LinearDetector(method="mmse")` never checked `sigma2`, and passed
    `None` into the MMSE solver;
  - `WeightAmplifier.weight` was declared optional and dereferenced in
    `__post_init__`, so the documented default construction crashed. It
    is now a required argument, which is what the class always meant;
  - `DataAidedSimpleSynchronizer.plot()` drew `None` against `None` when
    the block had not been asked to keep its cross-correlation.

  The two validators of `mimo/detectors.py` now *return* the value they
  check instead of only asserting it, so a caller cannot use the
  unchecked one by mistake, and their messages name what is missing and
  how to supply it (D38).

  Three latent unbound-variable paths were closed the same way:
  `compute_ser_awgn_psk`, `Serial2Parallel.forward` and
  `BlindCFOCompensator.fit` each chained independent `if`s over a
  parameter that could match none of them, leaving a local undefined and
  raising `UnboundLocalError` instead of saying what was wrong.

- `sphinx.ext.napoleon` is enabled. Without it the numpydoc sections of
  every docstring -- the whole section-4.10 course-material template,
  its parameter tables and its symbol-parameter bijection -- rendered as
  raw reStructuredText, and each D23 trailing-underscore attribute
  (`sigma2_`, `pilots_`, `h_`) was read as a broken hyperlink
  reference. The build carried 32 warnings; enabling napoleon took it to
  zero.
- The docs workflow now runs on pull requests, as a `build` job separate
  from `deploy`: a pull request gets the documentation checked without
  publishing anything, and only a push to `main` deploys. The build runs
  with `-W --keep-going`, so a Sphinx warning is an error. Intersphinx
  is deliberately not enabled -- it would put the network inside a gate
  that has to be reliable.
- Documentation pages for the 21 modules that had none: capacity,
  fading, frames, sequences, utils and validators in `core`, the whole
  `fec` package, `mimo.utils`, `ofdm.allocation`/`utils`/`visualizers`,
  `optical.wdm`/`raman`/`constants`, and a new *Chain services*
  section for `exceptions`, `serialization`, `sweep` and `style`. The
  documentation covered 25 of 46 modules; Sphinx reports nothing when a
  module simply has no `automodule` directive, so
  `tests/test_docs_coverage.py` now fails on a module with no page, a
  page for a module that is gone, and a page in no toctree.
- `tests/test_docs_references.py` also checks that `literalinclude`
  line ranges still cover every executable line of the example they
  quote. That found six tutorials silently showing incomplete code --
  `processor_profiling` opened its chain block on the *second* line of
  `Sequential([`, the PAPR page never showed its oversampling factor,
  and the fibre-nonlinearity page never showed the line that runs the
  chain.
- `constellation_capacity` and `bicm_capacity` take `method=`, which
  chooses the integration rule: `"gauss-hermite"` (the default, matched
  to the Gaussian weight of the integrand) or `"simpson"` (the classical
  composite rule on a grid truncated at 8 standard deviations). Both
  compute the same quantity and share nothing but the integrand, so the
  second is an independent check of the first without leaving the
  library -- and it is not a strawman: the integrand decays with all its
  derivatives, so Euler-Maclaurin's endpoint corrections vanish and the
  classical rule converges far faster than its nominal fourth order.
  Measured on 16-QAM at rho = 10 against a 200-node reference, Simpson
  is 1.9e-2 bit off at 20 nodes, 1.2e-4 at 40 and 1.1e-9 at 80, where
  Gauss-Hermite is at 1.1e-5, 7.9e-7 and 3.8e-10 -- about a factor two
  in nodes, four in cost, for the same accuracy. The rule now lives in
  one place, `_noise_quadrature`, which returns nodes and weights;
  nothing downstream of it knows which rule it was handed.
- The MI and GMI estimators of `core/information.py` are cross-checked
  against `core/capacity.py` over eight constellations of three families
  (PSK-2/4/8, PAM-4/8, QAM-4/16/64) at 0, 8 and 16 dB. The two modules
  compute the same two quantities by genuinely different means -- a
  Monte-Carlo estimator over a record on one side, a deterministic
  quadrature over the constellation on the other -- and the worst
  deviation over the 48 comparisons is 0.006 bit on 40000 symbols. A
  longer record shrinks it, which is what separates a sampling residual
  from a disagreement on the quantity, and that separation is measured
  in RMS over several draws rather than trusted to one lucky record.
  The sweep also pins what a single constellation cannot show: BPSK,
  QPSK and 4-QAM have MI = GMI exactly, the BICM gap opens beyond four
  points and is a low-SNR phenomenon, and 8-PAM stays below 8-PSK at
  equal SNR.
- `validation/optical_wdm_opticommpy.py --dsp` answers, by measurement,
  where the decibel between this reproduction and the published result
  comes from. A finer split step does *not* close it -- it opens it
  (20.11, 21.15, 21.57, 21.69, 21.69 dB as the step goes 1000 to 62 m),
  so the converged answer is 1.06 dB above theirs and the agreement at
  a coarse step was two errors of opposite sign cancelling. The
  receiver does close it: laser phase noise, a polarization rotation, a
  blind CMA/RDE butterfly and a blind phase search bring the same link
  to 20.18 dB against their 20.63, so the published number sits inside
  the bracket. One byproduct worth keeping: the phase search *gains*
  0.8 dB with no laser phase noise to remove, because it also tracks
  the slowly varying nonlinear phase.
- `validation/optical_wdm_opticommpy.py` uses the notebook's own pulse
  shaping -- one root-raised-cosine of 1024 taps at 16 samples per
  symbol, the same object shaping the transmitter and matching the
  receiver, as OptiCommPy does. It changes nothing it should not: the
  link reads 21.59 dB against 21.57 dB through a filter four times
  longer, and the script now *checks* that agreement, because the short
  filter leaves a floor only 7 dB above the figure under test and the
  subtraction has to be earned rather than assumed.
- The WDM slot-width warning moved from `WDMDemultiplexer` to
  `WDMMultiplexer`. At the receiver the measurement was dominated by
  amplified spontaneous emission, which is white and fills the guard
  band whatever the grid says: it fired on a correctly configured comb
  after 700 km (0.12 % of rejected energy against 0.0004 % of genuinely
  clipped signal). At the transmitter there is no noise and the
  question -- does this channel fit the bandwidth the grid declares --
  is exactly the configuration mistake worth catching.
- Estimator scope (D49): every estimator now says whether what it
  measures is **shared** by the paths of a multi-path signal or belongs
  to **each path**, and behaves accordingly. `BlindCFOCompensator`
  accepts `(..., P, N)` and estimates the offset *jointly* -- one laser,
  one number, and twice the data. `BlindIQCompensator`,
  `BlindPhaseCompensation` and `BlindPhaseTracker` estimate *per path*.
  The data-aided family is fitted against one reference, so a multi-path
  input is ambiguous by construction and `validate_single_path` refuses
  it with a message naming the quantity and both ways out. Estimated
  quantities keep the library's scalar-in-scalar-out rule: `theta_` is a
  float for one path and an array otherwise.
  Two defects surfaced asking the question:
  * `BlindIQCompensator` on a `(2, N)` signal stacked the real and
    imaginary parts of *both* polarizations into one 4-row matrix and
    returned a signal worse than its input -- `var(I)/var(Q) = 2499`
    instead of 1 -- **without raising**;
  * `DCCorrector` never broadcast its own documented `axis`: the mean of
    a `(P, N)` record along `axis=-1` came back as `(P,)` and the
    subtraction raised a shape error, so the block did not work above
    one dimension at all.
- `BlindDualMIMOCompensator`: `norm` is gone -- it was declared,
  documented as "normalize the filter weights", and never read.
  `sub_block_length` now does what its name says: it bounds the block of
  recent outputs handed to `process_after_iteration`, which used to be
  `Y[:, k-1::-100]`, a stride over the *whole* history -- so the hook
  cost grew with the sample index and a pass was quadratic. The
  docstring gained the two things that cost an afternoon to rediscover:
  the equalizer has a group delay of `L` input samples, and only CMA
  converges from a cold start (RDE stalls at 3.5 dB where the noise
  floor is 24). Staged CMA -> RDE -> DD reaches 23.96 dB against a
  23.98 dB floor, and `tests/mimo/test_blind_equalizer.py` pins all of
  it.
- `comnumpy.core.information` (D48): achievable information rates
  measured on data -- `compute_mi` (symbol-wise decoder), `compute_gmi`
  (bit-wise, the structure of every soft-decision system),
  `compute_ngmi` (on the scale of a code rate) and `compute_llr`. Every
  equation is cited by its number in Alvarado et al., JLT 33(20), 2015:
  (6)/(29) exact L-values, (7)/(31) max-log, (16)/(17) MI, (21)-(26)
  GMI, (30) its estimator and **(32)** the minimization over `s` that
  the paper calls mandatory for approximated L-values -- using (30) on
  max-log L-values returns a rate lower than the true one, so
  `max_log=True` solves it. The LLR sign stays comnumpy's (positive
  favours the bit 0), the opposite of the paper's, and the module says
  so. Writing it surfaced an underflow: at high SNR the weaker
  hypothesis summed to exactly zero and the exact LLR returned an
  infinity; the log-sum-exp shift is now per hypothesis.
- `WDMDemultiplexer` warns when its brick wall cuts into the signal
  instead of only removing the neighbours. The grid's `bandwidth_Hz` is
  the *occupied* bandwidth, so a pulse shaped at roll-off rho needs
  `Rs*(1+rho)`; writing `Rs` is the natural mistake and it is expensive
  -- measured on an 11-channel 32 GBd comb at rho = 0.01, an
  implementation floor of 33.5 dB instead of 54.1 dB, and
  `validation/optical_wdm_opticommpy.py` was making exactly that
  mistake. The check looks only at the guard band between the channel
  edge and the midpoint to its neighbour, since the energy further out
  is what the mask exists to remove. Measured the other way round, once
  the mask passes the channel it does nothing at all: a mask at
  `Rs*(1+rho)`, one at the full 37.5 GHz slot and no mask give 54.13,
  54.14 and 54.13 dB, which is why the block offers no filter-shape
  option -- the matched filter downstream is what selects the channel.
- Dual-polarization propagation (D47): a field shaped `(..., 2, N)` --
  the antenna axis of D2 -- is propagated by `FiberLink` and `DBP` with
  the **Manakov equation**, the two polarizations sharing the total
  intensity and the coefficient carrying the 8/9 factor. Which model is
  integrated is read off the shape, the way which pumps are on is read
  off their powers: there is no `polarizations=` argument to contradict
  the array. `fiber.gamma` stays the fibre's own coefficient in both
  modes -- the 8/9 belongs to the model, not to the glass. Any other
  size on that axis raises, since a row-by-row Kerr step would describe
  parallel fibres. Two defects surfaced writing it:
  `apply_chromatic_dispersion` took `NFFT = len(x)`, which is **2** for
  a `(2, N)` field, and the EDFA drew `len(x)` noise samples instead of
  one per polarization.
- `validation/optical_wdm_opticommpy.py`: the first confrontation with a
  *second implementation* rather than a closed form -- the published
  coherent WDM example of OptiCommPy (11 x 32 GBd PDM-16QAM, 700 km).
  The comb power matches the printed 8.41 dBm to 0.004 dB, the
  ASE-limited SNR matches its closed form to 0.02 dB, a fixed 0.5 km
  split step is shown to be 0.83 dB short of converged, and the full
  link lands 0.79 dB above the published 20.63 dB -- the sign every
  impairment the notebook models and this script does not would give.
- `comnumpy.optical.fiber` (D46): frozen `FiberSpec` carrying the
  physical coefficients and their provenance, with a registry
  (`get_fiber`, `available_fibers`, `@register_fiber`), a catalog of
  SMF/NZDSF/DCF, a D20 self-check against the published beta2, and
  unit-plausibility guards whose messages name the unit they want.
- `solve_raman` describes a *set of waves* (D45b), not a pump-signal
  pair: every signal channel and every pump, in each direction, is one
  equation, and every pair is coupled through the gain spectrum at the
  shift separating it. Pump-to-signal gain, pump-to-pump transfer
  (hence second-order pumping) and the inter-channel tilt of a WDM comb
  all follow from the same equations. Every signal or pump argument now
  takes a scalar -- shared by the group -- or one value per wave, and a
  scalar in gives a scalar out, so the single-channel case reads
  unchanged. `spectrum=` is required as soon as a group has more than
  one wavelength: its default means "the pair sits at the gain peak",
  which would silently invent a tilt across a comb. New on
  `RamanSolution`: `n_signals`, `n_pumps`, `tilt_dB`, and a compact
  `__repr__`.
- `FiberLink` turns a multi-signal solution into a *transfer function*:
  the multiplexer sums the channels into one field (D44), so the gain
  is interpolated from the solved channels onto the FFT grid and
  applied half-step by half-step. The EDFA is flat and cannot undo a
  tilt, so it makes up the *mean* on-off gain and the channels come out
  spread around transparency by `tilt_dB` -- the physical situation a
  gain-flattening filter exists to fix.
- `validation/optical_raman.py` gained the two analytical references
  that cover exactly the two new axes: Zirngibl's closed form for the
  comb tilt (reproduced to 0.7 % of the tilt, the size of the
  approximation that model itself makes) and the sum of the undepleted
  per-pump gains, whose residual falls by a factor 101 when the pump
  power falls by ten -- the exponent that identifies it as the
  pump-to-pump transfer rather than an error. Plus photon-number
  conservation over ten simultaneous waves.
- `comnumpy.optical.raman` (D45): `RamanGainSpectrum` (frozen, two
  closed-form models -- Blow-Wood single oscillator and the triangular
  tilt model -- with a registry and a construction-time self-check
  against the published peak), and `solve_raman`, which integrates the
  coupled power equations as an initial value problem for
  co-propagating pumping and as a two-point boundary value problem
  otherwise. It returns power profiles and the gain profile `G(z)`, not
  a `Processor`: applying a lumped gain at the end of a span would
  describe a discrete amplifier, which is what distributed
  amplification is not.
- `FiberLink(..., raman=solution)` consumes that profile inside the
  split-step loop, so the Kerr term sees the power the fibre carries.
  The span EDFA is reduced by the Raman on-off gain, so a span stays
  transparent pumped or not; the Raman ASE is added once per span; and
  the link now takes a `seed` that feeds both it and the per-span EDFA
  -- the EDFA was built unseeded, so a noisy `FiberLink` was not
  reproducible at all.
- `comnumpy.optical.wdm` (D44): frozen `WDMGrid` frequency plan
  (absolute hertz, `uniform` and ITU flexible-grid constructors, ASCII
  spectral map, `validate_fs`), plus `WDMMultiplexer` /
  `WDMDemultiplexer` -- the synthesis and analysis pair, on the
  `(..., C, N)` channel axis of D2. Neither block resamples: compose
  them with `Upsampler`/`Downsampler`. `FiberLink` and `DBP` now name
  the multiplexer in their multi-channel rejection instead of pointing
  at something the library did not provide.
- `get_alphabet` builds PSK, PAM and square QAM from their definitions
  instead of reading a CSV, so **any** power-of-two order now exists --
  BPSK, 8-PSK, 1024-QAM, all of which the tables simply did not have.
  Thirty-two of the thirty-six files are deleted; the two cross
  constellations (32-QAM, 128-QAM) are not square, hence not a product
  of two PAM axes, and stay tabulated. The construction reproduces
  every deleted table entry by entry to the six decimals the files
  stored -- and is exact where they were rounded, which is the argument
  for the change. The tables are kept as test fixtures under
  `tests/core/data_reference/`.
- `comnumpy.core.capacity`: `awgn_capacity`, `constellation_capacity`
  (Gauss-Hermite quadrature), `bicm_capacity`,
  `rayleigh_ergodic_capacity`, `mimo_ergodic_capacity`,
  `outage_capacity`, `waterfilling`.
- `comnumpy.fec.analysis`: `DistanceSpectrum`, `distance_spectrum`
  (breadth-first trellis search, catastrophic codes rejected rather
  than looping forever) and `union_bound_ber`.
- `validation/` scripts pinning the optical module to analytical
  references (chromatic dispersion, SPM, fundamental soliton, DBP round
  trip) plus a Monte-Carlo SER-vs-theory check; fast golden versions run
  in CI (`tests/optical/`).
- `comnumpy.ofdm.allocation`: `CarrierType`, frozen `CarrierAllocation`
  (physical-order 2D mask + metadata + standard clause, ASCII spectral
  map repr), `band_allocation`/`scattered_allocation`, registry-backed
  `get_allocation()` catalog (802.11a/n/ac, LTE, 5G-NR) with
  construction-time self-checks; `CarrierAllocator`/`CarrierExtractor`
  accept a shared allocation object and validate the Block layout.
- `itu_grid_frequency(n, m)` (ITU-T G.694.1 flexible grid); `FiberLink`
  and `DBP` now refuse multi-channel arrays (full-field guard).
- `Sequential.set_params(**{"awgn.snr_dB": 10})` with dotted addressing
  and parametric re-precompute (D34); `block_ids()`.
- Introspection (D33): structural `__repr__`, `summary(N)` table,
  `to_mermaid()`.
- Serialization (D31/D32): `to_json`/`from_json` with array sidecar
  (.npz), explicit `inputs` field, `@register_block` for user blocks,
  and a normative round-trip test.
- Frame structures (D28): `FieldRole`, `FrameField`, `FrameStructure`
  (frozen, ASCII frame map), `Framer`/`Deframer` sharing one structure.
- Synchronization sequences (D29): `zadoff_chu`, `schmidl_cox_preamble`,
  `barker`, `golay_pair`, `m_sequence`, each with property tests.
- Estimator conventions (D22/D23): unified `fit(x, y=None)` returning
  `self` (`y=None` = blind), `partial_fit` on the adaptive CMA/RDE/DD
  compensator, trailing underscore on estimated quantities (`theta_`,
  `gain_`, and `H_` on the MIMO equalizer -- formerly homonymous with
  the *configured* channel `H`), `NotFittedError` when `forward` runs
  with `should_fit=False` before any fit.

### Milestone 6 (D6, D11, D25, D27, D37)

- `Sequential.seed(s)`: deterministic per-block seeding via
  `SeedSequence.spawn` (D6).
- Logging replaces every `print` in `src/` (D11): monitors and the
  debug path report through `logging.getLogger("comnumpy.*")`;
  `BaseMIMOChannel.info()` logs and returns its description. The only
  deliberate terminal output left is `Sequential.summary()`'s table
  (a rendering path, like the ASCII maps).
- Plotting API (D25): every plot function/method takes `ax=None`,
  returns the axis, creates a figure only when needed; `plt.show()` and
  the `num` figure-number parameter are gone from `src/` (breaking for
  the Scope constructors, which no longer take `num`).
- Figure semantics and style (D27): frozen `CARRIER_STYLE` table
  (color, glyph, hatch, label -- Okabe-Ito, color never the only
  carrier of information) shared by the ASCII spectral map and
  `CarrierAllocation.plot()`; `comnumpy.mplstyle` ships with the
  package and `comnumpy.style` exposes `PATH` and `context()` -- never
  applied at import.
- Typing (D37, ratchet): `pyrightconfig.json` runs strict on a
  grow-only allowlist (exceptions, style, sequences, frames,
  allocation, fec, serialization -- currently 0 errors), enforced in
  CI. `py.typed` intentionally not shipped yet: it ships when all of
  `src/` passes strict (the v1.0.0 tag gate), because a partial
  `py.typed` is worse than none.
- `SymbolDemapper(soft=True)` bit LLRs (D12) and the fec package (D4)
  landed in the same window (see the dedicated commits).

### Milestone 8 (D43, standard channel models)

- `comnumpy.core.fading`: `PowerDelayProfile` (frozen, carrying the table
  **and** its clause), a registry-backed catalog of the 3GPP LTE
  reference profiles (EPA, EVA, ETU, TS 36.101 Annex B.2),
  `rayleigh_process` for a Clarke/Jakes Doppler tap, and
  `core.channels.TappedDelayLineChannel` applying the model. Third
  application of the D15/D17/D20 pattern: `CarrierType` types the
  frequency axis, `FieldRole` the frame axis, `PowerDelayProfile` the
  delay axis.
- The D20 self-check earned its place immediately. TS 36.101 publishes
  three figures per model -- tap count, RMS delay spread, maximum excess
  delay. EVA and ETU match all three and are pinned on all three. **EPA
  matches two**: 7 taps and 410 ns exactly, but 43.13 ns of delay spread
  against a published 45. The gap was investigated, not tolerated. The
  definition is not in question: the same power-weighted central formula
  reproduces EVA to 0.35 ns and ETU to 0.06 ns, and no variant lands on
  45 either (amplitude weighting 69.8 ns, second moment about zero 61.8,
  mean delay 44.2). No plausible typo bridges it either -- it would take
  +1.67 dB on the last tap or +76 ns on the 410 ns delay, in a table of
  round values. Since the two matching figures confirm the *delays*, a
  discrepancy would have to be in the *powers*, which nothing
  independently confirms. The entry pins what it reproduces and leaves
  the third figure unasserted, pending a read of Table B.2.1-1 itself.
- The channel is time-selective, not block fading: each tap is a Doppler
  process synthesized directly on the output FFT grid (only the bins
  under `f_D` are filled), so nothing is resampled or interpolated and
  the realization is band-limited by construction. The reference for
  verification is the Bessel autocorrelation `J0(2 pi f_D tau)`.
- `validation/fading_doppler.py` pins all four properties against their
  analytical references: the Bessel autocorrelation (0.0048), the
  cumulative Doppler power against `arcsin(f/f_D)/pi + 1/2` (0.0043),
  the realized path powers and delay spreads against the 3GPP tables
  (EVA 354.2 ns against 357, ETU 995.2 against 991), and the frequency
  correlation against the exact transform of the profile (0.006). The
  script reports one honest deviation it does not tune away: ETU's
  0.5-coherence bandwidth is 4.9x the `1/(5 sigma_tau)` rule of thumb,
  because its delay spread is inflated by the -7 dB path at 5 us while
  the correlation is set by the 0-500 ns cluster carrying 84% of the
  power. The rule of thumb fails there, not the simulation.
- A trap the implementation surfaces rather than hides: at 15.36 MHz with
  70 Hz Doppler, the channel needs 219 000 samples before it moves at
  all, so a 4096-sample simulation silently gets block fading. The
  generator now says so through `logging` (D11).

### Milestone 7 (D35, D42, MIMO validation, D5, D3)

- `ARCHITECTURE.md` enters the repository. The decision record was the
  normative document the code kept pointing at ("decision D25", "D40a")
  while living outside it -- unreadable for a contributor or a JOSS
  reviewer. It is now versioned next to the code it governs (v0.5), and
  `README`/`CONTRIBUTING` point at it. A test executes its canonical
  example and checks the D40c line budget: that example had been wrong
  since v0.3 (it read the recorded signal and ran the chain in the same
  call, so evaluation order read an empty record), which is exactly what
  principle P3 exists to catch.
- **Docstrings converted to the course-material template (D10, section
  4.10)**: 37 `Processor` classes and 25 processing functions now carry a
  LaTeX signal model, the axis category, the symbol-to-parameter
  bijection, a textbook or standard citation, and an executed doctest.
  What that buys, concretely: `Serial2Parallel` states the index map
  `y[t,f] = x[tF+f]` and names it as the C-order reshape of D2;
  `SRRCFilter` explains why the square root is split across transmitter
  and receiver, and its doctest proves the Nyquist zeros of the cascade;
  `get_alphabet` states the unit-average-energy normalisation, which
  every SER formula in the library depends on; `esn0_to_snr_dB` /
  `ebn0_to_snr_dB` settle the Es/Eb/SNR confusion in one place. The
  ratchet now checks functions as well as classes.
- Writing those models meant reading every block line by line, which
  surfaced **24 defects**, recorded in annex A.5 of the decision record.
  All of them are now fixed, each with a test that fails against the old
  code. The four that changed numbers, measured:
  `BWFilter`'s cutoff was compared against cycles/sample while every
  caller -- including `Upsampler`, which builds its own anti-imaging
  filter as `BWFilter(1/L)` -- assumed scipy's Nyquist-normalised `Wn`,
  so the library's own interpolation filter was twice too wide;
  `compute_ccdf` did not broadcast on 2D input; the data-aided
  synchronisers applied their amplitude correction by multiplication, so
  a channel gain of 0.5 came out as 0.25 (the output now restores the
  reference exactly); and the LS dispersion compensator's `w_vect` was
  unusable, its in-band error on a half band going from 1.31 -- worse
  than doing nothing -- to 2.0e-07, with the default band bit-stable to
  3e-16.
  Four blocks turned out to be entirely non-functional dead code
  (`BlindPhaseTracker` had no `@dataclass`, `DataAidedFineSynchronizer`
  and `Downsampler(use_filter=True)` read undeclared fields,
  `DataAidedFIRCompensator` called a method it did not inherit); those
  and three silent `SRRCFilter` defects are fixed. The rest are
  documented, including four that need a call because they change
  numbers: the `BWFilter` cutoff normalisation, `compute_ccdf` on 2D
  input, and the two that make the LS compensator's `w_vect` unusable.
- Decision **D42** records the observation model implemented below:
  chains contain communication blocks only, `taps` observe, `wiring`
  feeds. It amends D11 (monitors are removed, not converted to loggers)
  and closes part of the "computation graph" open point of section 8.

- LDPC coding (D5): `fec.LDPCEncoder` (systematic, GF(2) echelon
  form, rank-deficiency aware), `fec.LDPCDecoder` (min-sum with
  optional normalization `alpha`, early stopping on the syndrome,
  `(batch, n_edges)` segmented-reduction vectorization -- the only
  Python loop is over iterations), `make_gallager_parity_check`.
  Golden test: (3,6) regular n=240 at Eb/N0 = 3 dB decodes >10x below
  uncoded BPSK.
- Internal backend dispatch (D3): `comnumpy._backend` regroups the
  FFT calls of the signal path (SSFM, OFDM (I)FFT, FIR/Butterworth
  filters) and routes them to the library owning the input array --
  numpy arrays keep going to `scipy.fft` bit-exactly, CuPy arrays go
  to `cupyx.scipy.fft`. CuPy is imported only when a CuPy array is
  seen and remains a non-dependency.

- `comnumpy.sweep(chain, param, values, metrics, stimulus, ...)` (D35):
  the parameter-sweep loop shared by the validation scripts, extracted
  after the third script needed it. Dotted `set_params` addressing,
  per-point child seeds, zip semantics for multi-parameter sweeps.
- **Observation is now taps only.** `Sequential(..., taps=["block_id"])`
  records the output of the named blocks into `chain.tapped_` during
  `forward` and exposes them through `chain.tap("block_id")`; unknown ids
  raise `KeyError` at run time with the declared list. Observation is
  chain *metadata*, so `module_list` — and with it `repr`, `summary()`,
  `to_mermaid()`, the JSON export and the block indices — describes the
  communication system and nothing else. The cost is one dict store of an
  array reference per tapped block (no copy), which relies on the
  library-wide invariant that a block never mutates its input in place
  (now stated in CONVENTIONS.md).
- Every in-chain instrumentation block is **removed** (see the migration
  table): `Recorder`, `Logger`, `Debugger`, `PowerReporter`,
  `TimeSignalMonitor`, `MetricRecorder`, the `Scope` family and
  `FFTMonitor`. `comnumpy.core.monitors` is gone. Their replacements are
  plain functions on extracted arrays: the `plot_*` family in
  `comnumpy.core.visualizers` / `comnumpy.ofdm.visualizers` (each takes
  `ax=None`, returns the axis, never calls `plt.show()`, overlays a 2D
  input as one line per stream) and
  `comnumpy.core.metrics.signal_report(x)`, which returns the statistics
  as a dict for the caller to log, tabulate or assert on. A CI guard test
  keeps these names out of the public surface.
- `sweep(..., reference="tx")` now names a tapped block and declares the
  tap itself if needed; blocks no longer hold references to other blocks
  (`reference=` takes a plain array).
- `Sequential(..., wiring={"comp.reference": "ref"})`: declares the
  extra data edge a data-aided estimator needs when its reference is
  produced *inside* the chain. Before the target block runs, it receives
  the signal the source block produced in the same pass. The source is
  tapped automatically and must run earlier; a backward edge raises
  rather than silently serving the previous run's value. This closes the
  one real gap of the Recorder removal: `reference=` alone is a frozen
  array, so a Monte-Carlo loop would have kept comparing against the
  first run's symbols. It is the bounded form of the `inputs` field of
  D31 -- a second input declared by the chain, not a general DAG.
- Serialization fixes found while checking that path:
  - a chain sharing a `CarrierAllocation` object (the form D18 asks
    for) exported but failed to re-import: value dataclasses passed as
    block parameters had no decode path. The same bug hit
    `FrameStructure`/`FrameField`. Both round trip now, and the
    normative D32 test covers the object form, not just the raw mask.
  - `comnumpy.core.frames` and `comnumpy.fec.ldpc` were missing from the
    block registry, so `Framer`, `Deframer`, `LDPCEncoder` and
    `LDPCDecoder` could not be deserialized at all.
  - complex scalar parameters (a complex gain -- ordinary here) raised
    on export; they now round trip through a `__complex__` pair.
  - `taps` and `wiring` are exported with the chain: rebuilding a chain
    that no longer records or feeds what its author declared was a
    silent fidelity loss.
  - `FrameField(role=...)` normalizes to the `FieldRole` enum, so a role
    read back from JSON is an enum again (and an invalid role fails at
    construction).
- `OFDMTransmitter` and `OFDMReceiver` gained the `name=` field every
  other block already had. Without it they could be neither addressed
  (`chain["..."]`, `set_params`) nor tapped -- a gap the taps migration
  exposed.
- Ratchets tightened by the cleanup: ruff now lints `tests/`,
  `validation/` and `examples/` in addition to `src/`; the pyright strict
  allowlist gains both visualizer modules; the coverage floor moves from
  31 % to 55 % (measured 59 % — removing untested instrumentation code
  and testing the new observation surface).
- `validation/mimo_zf_ml_ber.py` + fast golden test: 2x2 i.i.d.
  Rayleigh BPSK; ZF pinned to the diversity-1 closed form
  `BER = (1 - sqrt(g/(1+g)))/2` (within 2%), ML checked for full
  receive diversity through the BER slope ratio.

### Sanitation batch (Lot 0)

No new feature; the published package becomes consistent with what the
code actually does.

### Fixed

- `SRRCFilter(method="fft")` sized its FFT with `len(x)`, which is the
  number of *rows* of a 2-D array rather than the number of samples. A
  polarization pair `(..., 2, N)` -- the shape D47 introduced for the
  Manakov equation -- therefore transformed two samples and died inside
  `H()` complaining about negative dimensions, naming neither the filter
  nor the shape. Both methods now filter along the last axis and
  broadcast over the leading ones, so pulse shaping a dual-polarization
  signal works the way the rest of the library already assumed. One-
  dimensional results are bit-identical.

- `plot_error_rate` drew every measurement as bare markers, which reads
  as a scatter when there is no reference curve to follow. A measurement
  that has a matching entry in `theory=` keeps its markers alone, so the
  pair still reads as one statement; one that has none is now joined.

- The tutorials were nine good pages in no particular order, each
  re-explaining what the previous one had already introduced. They are
  now a course: `docs/examples/index.rst` opens with the syllabus --
  which tutorial introduces which tool -- and the order follows the
  dependencies rather than the history (AWGN, OFDM, PAPR, multipath,
  MIMO, Alamouti, shaping, optical). Every page opens with a
  *Before you start* note naming what it assumes, and closes by pointing
  at the question the next one answers.

  The chain, the tap and the error rate are introduced once, in the
  first simulation; the Monte-Carlo loop and its `sweep` equivalent
  once, in the AWGN tutorial, which now states plainly that the rest of
  the series uses `sweep` without rewriting the loop. The tone follows:
  each page says what problem it is solving before showing the code
  that solves it.

- The MIMO tutorial gave ZF and MMSE a formula and no explanation, next
  to a sphere-decoding section that derived everything. The comparison
  is now five answers to one question -- what to do with the
  interference the other streams leave on the one being read -- with
  the noise enhancement of zero forcing and its
  `sigma^2 [(H^H H)^-1]_ii`, the regularization MMSE trades a bias for,
  the two limits it sits between (ZF and the matched filter), the
  ordering rule and error propagation of OSIC, and a table of the five
  against their cost and their diversity order. With the warning the
  curves themselves call for: 18 dB is not the asymptotic regime, so
  the exponents are stated, not read off the figure -- they are checked
  in `validation/mimo_zf_ml_ber.py` where a limit can be reached (D7).

- `MaximumLikelihoodDetector` built its candidate table one column at a
  time with `itertools.product` and then looped over samples in Python.
  The distance expands as `||y||^2 - 2 Re(y^H Hx) + ||Hx||^2`, whose
  first term does not depend on the candidate, so the search is a matrix
  product -- blocked on both axes so the cost matrix stays a few
  megabytes whatever the constellation. Same decisions, 50x faster on
  4-PSK 2x2 and 6x on 16-QAM 4x4.

- `BlindDualMIMOCompensator` gained an exact fast path rather than a
  rewrite: a stochastic gradient cannot be vectorized (the update from
  y[n] is needed to compute y[n+1]), but with `mu = 0` the equalizer is
  frozen and the whole pass is a matrix product -- 25 to 30 times
  faster, bit-for-bit identical, which is the "fit on a preamble, then
  apply" regime of D22. The docstring now states what the adaptive loop
  costs (18 us per output sample) and why the only way further down is
  a compiled kernel.

- The chain diagrams in the tutorials were drawn by hand in mermaid,
  next to a library that can draw them itself. Seven examples now export
  `chain.to_mermaid()` (D33c) and the pages include the exported file,
  so a diagram shows the real block names and the real taps -- and
  `tests/test_examples_run.py` compares what a sandbox run writes with
  what is committed, so the picture cannot drift from the chain. The
  four hand-drawn OFDM graphs were the ones that had already drifted:
  they named blocks (`S2P`, `CP add`) that no longer exist under those
  names.

- The OFDM tutorial's channel was not frequency selective (one tap at 1,
  four at 0.1), which is why its single-carrier and OFDM receivers had
  nothing to disagree about. It is now drawn from an exponential power
  delay profile with a 31:1 spectral notch, and the example runs both
  receivers over a range of noise variances instead of one point. The
  ranking crosses over, and the page explains why: the single-carrier
  equalizer inverts a *linear* convolution, `N + L - 1` equations for
  `N` unknowns, so its least-squares solution never divides by a null
  and spreads the enhanced noise over the whole block; the cyclic prefix
  makes the OFDM system exactly determined, one equation per subcarrier,
  so a subcarrier in the notch is lost whatever the SNR. Concentrated
  damage wins at low SNR and loses at high SNR -- and the error floor it
  leaves is why no real OFDM system is uncoded.

- A pass over the tutorials, which had drifted from the library they
  teach. `examples/mimo/one_shot_mimo.py` inverted the channel by hand
  with `linalg.pinv`, mutated block attributes instead of
  `set_params`, rolled its own triple Monte-Carlo loop and was not
  reproducible: it is now four chains differing by their last block,
  swept over channel realizations, seeded -- and it runs in 10 s
  instead of 171 s, so the smoke test covers it again. The OFDM
  tutorial claimed OFDM won on error rate, which the script it quotes
  does not show and which is not true uncoded on that channel: the
  page now quotes the measured numbers and says where the win actually
  is (three orders of magnitude of computation). The PAPR page's
  "around 12 dB" is now the two values the expression gives, 11.41 and
  11.83 dB. The optical page shows the SER it promised to compute, and
  the example no longer passes `hard_projector`'s `(indices, symbols)`
  tuple where an array is expected -- it worked only because
  `compute_ser` truncates to the shorter input.
- `OFDMTransmitter` and `OFDMReceiver` did not accept the
  `CarrierAllocation` object that the blocks they wrap accept (D18) --
  and did not refuse it either: `np.asarray` on a dataclass gives a 0-d
  object array, so the mistake surfaced several blocks later as
  `len() of unsized object`. Both now pass the allocation through and
  read its `N_fft`.
- The two profiling examples still called the deprecated
  `get_standard_carrier_allocation`, so running the tutorial printed a
  `DeprecationWarning`; they use `get_allocation` and the catalog's
  own counts.
- `plot_chain_profiling` drew a linear time axis on a quantity that
  spans four decades, so one block filled the figure and every other
  one was a line against zero. Logarithmic axis, on whichever axis
  carries the time (the label was hardcoded to `x` and was wrong for
  `orientation="vertical"`), and a constrained layout so long block
  names are not clipped.
- Coverage was measuring the wrong thing. `[tool.coverage.run] source`
  named the *package*, so coverage imported `comnumpy` to locate it and
  everything `__init__` pulls in was already loaded when the tracer
  started -- reported as 0%. Per-file numbers therefore depended on test
  collection order (`optical/wdm.py`: 98% alone, 0% in the suite) and
  the total was 16 points low: 74.76% against a true **90.59%**.
  `source` now names the `src/comnumpy` path, and the D39 ratchet moves
  from 70 to 90.
- `ofdm.processors.CyclicPrefixer` crashed at `N_cp = 0`, a length
  `__post_init__` explicitly accepts: the mask was built with
  `[-N_cp:]`, and `[-0:]` is the whole block. Both prefix blocks also
  accept a numpy integer now, and their rejection message no longer
  says "positive" while allowing zero.
- `docs/examples/optical_fiber_nonlinearity.rst` included
  `one_shot_nli.py`; the file is `one_shot_NLI.py`. Sphinx warns rather
  than fails on a missing `literalinclude`, so the page rendered empty
  code blocks on every case-sensitive filesystem.
- `README.md` quick example now runs as written (the previous example called a
  non-existent `SymbolMapper(M=16)` signature and unpacked two return values
  from `Sequential`, which returns one).
- `optical.channels.PhaseNoise` was dead code: it raised `AttributeError` at
  construction (missing `seed` field) and called a non-existent `self.rvs()`.
  It now works and is seedable.
- `core.compensators.DataAidedComplexGainCompensator` (then `TrainedBased…`) was not functional
  (invalid dataclass field declarations, `fit()` referencing undefined
  variables). Rewritten; the estimated gain is verified numerically.
- `core.compensators.DataAidedPhaseCompensator.__post__init__` typo: the
  initializer was never called. Renamed to `__post_init__`.
- `core.compensators.DataAidedFIRCompensator` was dead code in the same
  family: it called `get_target_data()` without inheriting the mixin that
  defines it, so every call raised `AttributeError`. Found while renaming
  that method. It now inherits `DataAidedMixin`, its `fit` follows the
  `fit(x, y=None)` convention of D22 like its siblings, and a doctest
  pins the deconvolution.
- `ofdm.processors.CarrierAllocator` raised on 1D input when pilots were
  present; pilots now broadcast correctly along the allocation axis for 1D
  and N-D inputs.
- `ofdm.utils.get_standard_carrier_allocation` always nulled an odd number of
  DC subcarriers (`2*(N//2)+1`), and nulled one subcarrier even when
  `N_nulled_DC=0` (the `NoPilot_Full_*` configurations were not full).
  Exactly `N_nulled_DC` subcarriers are now nulled.
- `mimo.channels.BaseMIMOChannel.info()` crashed on `H.ndims` (typo for `ndim`).
- `mimo.utils.rayleigh_channel` / `rician_channel` used an undefined `shape`
  variable in their tap loop.
- `core.monitors` referenced `compute_PAPR` without importing it.
- All failing doctests (48 failing examples at the 2026-07-28 audit): every
  documented numeric value is now the value the code actually computes, with
  explicit seeds where randomness is involved. Doctests are executed in CI
  and block the build.
- `ofdm.utils.plot_carrier_allocation`: mutable default arguments removed;
  the documented color/label defaults no longer contradict the signature.
- Importing comnumpy no longer mutates global matplotlib state
  (`rcParams['agg.path.chunksize']`).

### Changed

- **The Raman gain is scaled by the effective area.**
  `RamanGainSpectrum` takes an optional
  `quoted_at=(wavelength_nm, effective_area_um2, core_radius_um)`. The
  gain *shape* depends on the Stokes shift alone -- that is the glass --
  but the coefficient multiplying `P_i P_j` is `g_R / A_eff`, and the
  effective area belongs to the waveguide, so it does depend on the
  absolute frequency: the mode spreads with wavelength and the same
  shift buys less gain. With a Gaussian mode this reduces to
  `A_eff(nu) = A_0 k / (ln(nu/nu_0) + k)` with `k = pi a^2 / A_0`.

  A spectrum that does not say where it was quoted scales by exactly 1,
  so the catalogue and every existing call are untouched. Across one
  band the correction is 5.3 %, which is why nothing had noticed it;
  across C+L+S it is the difference between a flat prediction and a
  wrong tilt. It was found by the confrontation below, not by reading
  the code.

- **The documentation section is called Tutorials, not Examples**, and
  lives at ``docs/tutorials/``. Ten pages meant to be read in order are
  a course, not a grab bag of samples; the scripts under ``examples/``
  keep their name, because that is what they are.

- **Three things the tutorials kept writing by hand became library
  features.** Each was extracted because several pages had the same
  lines, not because the API looked incomplete:

  - ``plot_iq(..., reference=..., label=...)``. Ten examples scattered
    a constellation, overlaid the alphabet as black crosses and
    legended the pair, in three lines each. A received cloud means
    little without the points it is supposed to be near, so the overlay
    is what makes the picture a measurement rather than a decoration --
    it belongs in the function. ``label=`` lets two signals share one
    axis, which is what a before/after compensator figure needs.
  - ``TappedDelayLineChannel.impulse_response()``. Three scripts built
    an impulse, called the block and sliced off the leading taps to get
    the tap vector an equalizer takes. Sounding a channel with an
    impulse is how its response is measured; it is now one call, and it
    is tested to be exactly the filter the channel applies rather than
    an approximation of it.
  - ``optical.utils.dbm_to_watt`` / ``watt_to_dbm``. Sixteen places
    wrote ``1e-3 * 10 ** (dBm / 10)`` or its inverse inline. The factor
    is 1e-3 and the divisor is 10, not 20; that is exactly the kind of
    thing to write once. ``watt_to_dbm`` refuses a non-positive power
    rather than propagating ``-inf`` into a link budget.

- **The tutorials are now built out of library blocks rather than
  hand-written numpy.** A tutorial exists to show that the library
  works; three of them were partly showing that numpy works, which
  proves nothing and leaves the reader copying code the library already
  ships. The OFDM tutorial drew its frequency-selective channel from an
  exponential profile written out by hand and now takes the 3GPP
  Extended Pedestrian A entry from the catalogue through
  `TappedDelayLineChannel`, sounding it with an impulse to recover the
  taps; the MIMO sphere-decoder study built its channel and noise with
  `@` and `standard_normal` and is now a chain with `FlatMIMOChannel`
  and `AWGN`; the GN model tutorial replaced its own pulse shaping,
  matched filter, gain fit and SNR ratio with `SRRCFilter`,
  `DataAidedComplexGainCompensator` and `compute_effective_snr`, and is
  now one chain reconfigured by `set_params` and driven by `sweep`
  instead of three chains composed by hand. The two profiling scripts
  lost the last hand-rolled channel in the documentation, and gained a
  seed with it -- they were drawing from an unseeded global RNG.

- **The tutorials no longer display their own narrative comments.** Each
  example carried a running commentary that the page repeated
  immediately above it in prose, so every explanation appeared twice and
  the two could drift apart. The commentary now lives in the page only;
  the code keeps its names, its one-line docstrings and the short
  trailing annotations that point at a specific line. 197 comment lines
  came out of ten examples, and the `:lines:` ranges of every page that
  quotes them were remapped in the same change -- which is what the
  coverage guard in `tests/test_docs_references.py` is there to catch.

- Dependencies are now `numpy`, `scipy`, `matplotlib` only. `seaborn`
  (used once, for a KDE plot now implemented with `scipy.stats.gaussian_kde`)
  and `tqdm` (never used) are removed.
- Version numbering restarts above the highest version ever published on PyPI
  (0.91): this development line is `1.0.0.dev0` and will be released as
  `1.0.0`. The previously planned `v0.2.0` would have been a PEP 440
  regression invisible to `pip install -U`.
- CI now tests Python 3.11, 3.12 and 3.13, runs doctests and `ruff`, and
  enforces a coverage ratchet (the measured coverage becomes the floor).
- `requirements.txt` removed; use `pip install .[dev]` extras instead.

### Removed

- `src/comnumpy/optical/chains.py` (empty file).
