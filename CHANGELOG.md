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
| `TrainedBasedPhaseCompensator`, `TrainedBasedComplexGainCompensator`, `TrainedBasedSimpleSynchronizer`, `TrainedBasedFineSynchronizer` | `DataAidedPhaseCompensator`, `DataAidedComplexGainCompensator`, `DataAidedSimpleSynchronizer`, `DataAidedFineSynchronizer` (`DataAidedFIRCompensator` already had the right name) |

### Added (milestones 2-5)

- `comnumpy.optical.wdm` (D44): frozen `WDMGrid` frequency plan
  (absolute hertz, `uniform` and ITU flexible-grid constructors, ASCII
  spectral map, `validate_fs`), plus `WDMMultiplexer` /
  `WDMDemultiplexer` -- the synthesis and analysis pair, on the
  `(..., C, N)` channel axis of D2. Neither block resamples: compose
  them with `Upsampler`/`Downsampler`. `FiberLink` and `DBP` now name
  the multiplexer in their multi-channel rejection instead of pointing
  at something the library did not provide.
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
