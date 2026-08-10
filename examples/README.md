# Examples

Runnable scripts that **teach**. Each one builds a communication chain,
runs it, prints a metric and plots a figure. Several are the source of
the figures in `docs/` (via `literalinclude`), which is why they are
kept short and linear rather than factored into functions: the point is
that a reader follows the chain from top to bottom.

Run one from its own directory — the scripts save their figures through
relative paths:

```bash
cd examples/simple
python one_shot_awgn.py
```

## The rule: an example does not assert

The repository has three layers, and mixing them is the mistake this
page exists to prevent.

| Layer | Answers | Shape |
|---|---|---|
| `examples/` | *how do I use this?* | plots and prints, **no assertion** |
| `validation/` | *is the result right?* | simulation vs analytical reference, asserted (decision **D7**) |
| `tests/` | *is it still working?* | fast, deterministic, asserted |

So:

- **Never add an `assert` to an example.** A claim of numerical
  correctness belongs in `validation/`, where it is stated against a
  published or closed-form reference and the figure proves it.
- **Never grow an example into a test.** A cheap regression check
  belongs in `tests/`.
- The only guarantee CI makes about this folder is that the scripts
  **still run** — `tests/test_examples_run.py`, run by the `examples`
  job. It checks exit codes, nothing else.

An example that starts wanting an assertion is telling you it is really
a validation script. Move it.

## What is in each folder

| Folder | Subject |
|---|---|
| `simple/` | the core chain: symbol generation, mapping, AWGN, multipath fading, SRRC pulse shaping, phase / CFO / IQ-imbalance compensation, probabilistic shaping, channel coding, chain profiling |
| `ofdm/` | OFDM transmitter and receiver, carrier allocation, cyclic prefix, frequency-domain equalization, PAPR statistics and PAPR reduction |
| `mimo/` | flat and frequency-selective MIMO channels, space-time block codes, ML / linear / OSIC detectors, blind CMA equalization |
| `optical/` | chromatic-dispersion compensation (FIR and least-squares FIR), WDM transmission, and fibre nonlinearity: split-step propagation with digital back-propagation, and the Gaussian Noise model in closed form |
| `nonlinear/` | power-amplifier models — clipper, Rapp, Saleh |

## Runtime

Wall clock measured 2026-08-09, Python 3.11, `MPLBACKEND=Agg`, on an
idle 4-core machine. Where CPU time is also given, the script is
threaded through BLAS/FFT and will degrade sharply on a busy machine.

The scripts marked **slow** are mostly Monte-Carlo sweeps; they are
*skipped* by `tests/test_examples_run.py` — visibly, with their measured
time as the skip reason — not deleted. If you shrink one enough to run
in a few seconds, move it out of the `SLOW` table in that file.

| Script | Time | In the smoke test |
|---|---|---|
| `nonlinear/test_power_amplifier.py` | 0.7 s | yes |
| `optical/CD_compensation_part2.py` | 1.8 s | yes |
| `optical/wdm_transmitter.py` | 1.8 s | yes |
| `simple/one_shot_phase_compensation.py` | 1.9 s | yes |
| `simple/one_shot_awgn.py` | 2.0 s | yes |
| `simple/one_shot_cfo_iq_compensation.py` | 2.4 s | yes |
| `simple/multipath_channels.py` | 3.3 s | yes |
| `mimo/one_shot_CMA.py` | 5.1 s | yes |
| `ofdm/monte_carlo_profiling.py` | 5.8 s | yes |
| `simple/profiling_awgn_ofdm.py` | 5.9 s | yes |
| `ofdm/one_shot_ofdm_papr_reduction.py` | 6.8 s | yes |
| `mimo/one_shot_mimo.py` | 16.0 s | yes |
| `simple/monte_carlo_awgn.py` | 12.4 s | yes |
| `mimo/one_shot_alamouti.py` | 14.4 s | yes |
| `simple/probabilistic_shaping.py` | 14.6 s | yes |
| `ofdm/monte_carlo_ofdm_papr.py` | 19.6 s | yes |
| `simple/channel_coding.py` | 29.7 s | yes |
| `optical/gn_model.py` | 29.7 s | yes — tied for the slowest one kept |
| `mimo/monte_carlo_simulation_1.py` | 32 s | skipped — slow |
| `ofdm/one_shot_ofdm.py` | 49 s (147 s CPU) | skipped — slow |
| `optical/one_shot_NLI.py` | 64 s | skipped — slow |
| `simple/one_shot_srrc_awgn.py` | 113 s (316 s CPU) | skipped — slow |
| `mimo/monte_carlo_simulation_2.py` | 143 s | skipped — slow |
| `optical/CD_compensation_part1.py` | 227 s | skipped — slow |
| `mimo/run_all_scripts.py` | ~350 s | skipped — it is a runner, not an example |
| `optical/NLI_simulation.py` | 700 s | skipped — slow |

The smoke test therefore runs about 95 s of examples.

No example is currently broken. `optical/CD_compensation_part1.py`
was, until 2026-08-09: it built `SymbolMapper([])` / `SymbolDemapper([])`
with an empty alphabet and filled it in later, which stopped working
once the demapper started deriving `k = log2(M)` at construction. It now
builds with the first alphabet of its own sweep. Should another example
break, record it in the `BROKEN` table of `tests/test_examples_run.py`
as an expected failure rather than deleting it — the test then reports an
unexpected success once it is fixed.

## Where the figures go

The scripts write into `docs/**/img/` through hardcoded relative paths
(`img_dir = "../../docs/tutorials/img/"`), which is why they must be run
from their own directory, and why the smoke test runs them inside a
throwaway copy of this folder instead of letting them overwrite the
committed figures.

Several also write into `docs/**/mermaid/`: the chain diagrams the
tutorials display are `chain.to_mermaid()` (decision D33c) rather than
hand-drawn pictures, so a diagram cannot claim something the chain does
not do. `tests/test_examples_run.py` compares what a sandbox run writes
with what is committed, and fails if the two disagree — so after
changing a chain, re-run its example and commit the regenerated `.mmd`.
