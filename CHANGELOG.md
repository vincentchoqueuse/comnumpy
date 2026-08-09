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
| `TrainedBased*(target_data=recorder)` | `target_data=` takes a plain array: tap the reference signal first |

### Added (milestones 2-5)

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

### Milestone 7 (D35, taps, MIMO validation, D5, D3)

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
  (`target_data=` takes a plain array).
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

- `README.md` quick example now runs as written (the previous example called a
  non-existent `SymbolMapper(M=16)` signature and unpacked two return values
  from `Sequential`, which returns one).
- `optical.channels.PhaseNoise` was dead code: it raised `AttributeError` at
  construction (missing `seed` field) and called a non-existent `self.rvs()`.
  It now works and is seedable.
- `core.compensators.TrainedBasedComplexGainCompensator` was not functional
  (invalid dataclass field declarations, `fit()` referencing undefined
  variables). Rewritten; the estimated gain is verified numerically.
- `core.compensators.TrainedBasedPhaseCompensator.__post__init__` typo: the
  initializer was never called. Renamed to `__post_init__`.
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
