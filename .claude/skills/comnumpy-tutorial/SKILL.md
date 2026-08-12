---
name: comnumpy-tutorial
description: Write or revise a comnumpy tutorial page (docs/tutorials/*.rst) and the example script behind it (examples/**/*.py). Use whenever the task is to create, rewrite, restructure or review a tutorial, a teaching example, or the figures and printed tables they contain. Carries the house plan (problem first, then method, then simulation), the writing register, and the rules for using the library rather than reimplementing it.
---

# Writing a comnumpy tutorial

A tutorial is a page under `docs/tutorials/` plus one script under
`examples/`. The page never contains code of its own: it quotes the script
with `literalinclude` line ranges, and the script is the thing that runs.
Everything printed or plotted on the page comes from a real run of that
script, pasted verbatim.

Everything -- code, comments, docstrings, prose -- is in **English**.

## The plan

Always the same order. The reader meets the problem before the machinery.

1. **The problem.** State the system, give the signal model in maths when an
   equation earns its place, implement it, and *show* the damage: a
   constellation that has collapsed, a spectrum with a notch, a histogram
   that is not the law you wanted. A figure here, not a paragraph.
2. **The method.** What answers the problem, the general idea, and one
   equation if it clarifies. No comparison tables of approaches: prose and,
   at most, one formula per strategy.
3. **The implementation and the measurement.** Build the chain, run it, and
   show the numbers -- one operating point first if it reads better, then
   Monte-Carlo curves. Compare against theory or against a closed form
   whenever one exists.
4. **What it costs**, when cost is part of the answer (complexity, run time,
   bandwidth, latency). Measure it, do not assert it.
5. **Conclusion**, then **References**.

Be concise. A tutorial explains a reasoning; it is not documentation for
every argument of every block.

## Use the library

Before writing a helper, look for it. The recurring failure is
reimplementing from scratch what `comnumpy` already does, and the reader
then learns the wrong lesson -- they came to see how the library is used.

- Chains: `Sequential`, `taps=` to observe a signal inside one, `wiring=`
  to feed a data-aided block a reference the chain produces itself,
  `set_params` (dotted or `__`) to change one parameter, `seed` for
  reproducibility, `elapsed_` for the wall time of the last pass,
  `profile_execution_time` for the same broken down by block.
- Never hand-roll a phase correction, an equalizer, a matched filter, a
  detector, a metric or a theoretical curve. There is a block or a
  `compute_*` for it: `DataAidedPhaseCompensator`, `LinearEqualizer`,
  `SRRCFilter`, the `mimo.detectors`, `compute_ser` / `compute_ber` /
  `compute_evm` / `compute_effective_snr` / `compute_papr` /
  `compute_ccdf`, `compute_metric_awgn_theo` and
  `compute_metric_rayleigh_theo` for the closed forms,
  `constellation_capacity` and `bicm_capacity` for the rates.
- Monte-Carlo: `sweep(chain, param, values, metrics, stimulus, seed=)`.
- Figures: `plot_iq`, `plot_error_rate`, `plot_spectrum`, `plot_time`,
  `plot_channel_response`, and the channels' own `info()` / `plot()`.
- If the library makes something awkward, that is a finding about the
  library. Say so and propose the fix rather than working around it in the
  example.

## Structuring the example script

- One `get_*` function per assembly the page talks about, so that what
  varies is an argument. `get_full_chain(n_spans)` beats building a chain
  inline six times.
- **Prefer one chain, transmitter to decision.** A full chain is what the
  reader will write. Cutting the chain into pieces is justified only when a
  Monte-Carlo re-runs an expensive stage that could be run once -- and then
  the page must say so explicitly, with the measurement that justifies it.
- **No inline `for`.** List, dict and set comprehensions, and generator
  expressions inside `join`/`sum`/`np.array`/`tuple`, are banned in tutorial
  scripts. Write the loop out, with a body. This is a hard rule.
- Name blocks (`name="data_tx"`) so that taps, wiring, `set_params` and the
  profile table all read as the same vocabulary.
- Comments in the script explain *why*, and they are part of what the page
  shows -- the `literalinclude` ranges quote them.
- No `plt.show()`; `savefig` into `docs/tutorials/img/`.

## Figures and numbers

- Every figure must answer a question the text has just asked.
- Overlapping curves need different markers or dashes; a theoretical curve
  goes under the measured one, not the reverse.
- Paste the script's real output into `.. code::` blocks. Never invent,
  round differently, or carry a number over from a previous run -- re-run
  and re-paste after any change that can move it. Two tables on one page
  must come from the *same* run, or the same quantity will read 8.9 ms in
  one and 9.1 ms in the other.
- State the estimator's floor when a curve hits one (an SER of 1e-4 from
  8192 symbols is the estimator, not the link).

## Register

Plain, declarative, textbook English. Short sentences. The subject is the
system, not the reader: "the channel varies by 12 dB across the band", not
"you will see that...". Say what is measured and what it means; skip the
enthusiasm. Bold sparingly, on the one number that matters in a paragraph.
Prefer "--" over parentheses for an aside. Admit what is approximate and
what is exact.

## Mechanics

- `literalinclude` ranges must start at a top-level statement or comment
  and must together cover the whole file; `tests/test_tutorial_includes.py`
  and `tests/test_docs_references.py` enforce both. Re-check the ranges
  after *any* edit to the script -- adding one import shifts everything.
- A new page goes in `docs/tutorials/index.rst`, in both the table and the
  toctree; removing one means purging the `:doc:` cross-references other
  pages make to it.
- An example slower than the smoke-test budget goes in `SLOW` in
  `tests/test_examples_run.py`, with the measured time and the date.
- Run before claiming done: the example itself, `ruff check .`,
  `pytest tests/`, and `sphinx-build -W` in `docs/`.
