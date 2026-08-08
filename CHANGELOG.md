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
