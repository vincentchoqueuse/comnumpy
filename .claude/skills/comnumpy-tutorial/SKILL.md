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
- Monte-Carlo: `monte_carlo(chain, param, values, metrics, stimulus, seed=)`.
- **A swept result is one dictionary, shown two ways.**
  `data = {"x": snr_dB, "curves": {"ZF": ..., "ML": ...}}` — which is
  already the shape `monte_carlo` returns — then `print_data(data,
  xlabel=…, ylabel=…)` for the table the page pastes and
  `plot_data(data, …)` for the figure. Never hand-roll an aligned
  `print` loop with `:8.4f` widths: the table and the figure must come
  from the same object, or they will eventually come from different
  runs.
- Activate the style sheet once, at the top of the script, right after
  the imports: `style.use()`. The colours and the figure size are
  rcParams and a figure already created keeps the old ones.
- Figures: **draw with matplotlib, decorate with `style.apply(ax, kind)`**.
  A scatter of real against imaginary, a `semilogy` of a rate against an
  SNR: the reader knows those calls, and hiding them behind a wrapper
  teaches nothing. `style.apply(ax, "iq" | "error_rate" | "time" |
  "spectrum")` fills the labels that are still empty, turns on the grid
  and adds the legend; it never touches the data or the scales.
  A kind names the *quantity*. A runtime in ms or an effective SNR in dB
  is none of them: give it its own labels and `ax.grid(True)`, and do not
  borrow `"error_rate"` because the axis happens to be logarithmic.
  Keep `plot_error_rate` for the case it earns: several measured curves
  each paired with its closed form, which it draws in matching colours
  with markers that grow so coincident curves stay legible.
  For the quantities that need computing before drawing, use the
  function: `plot_spectrum`, `plot_welch`, `plot_kde`,
  `plot_channel_response`, `plot_carrier_allocation`. And an object that
  can show itself does: `constellation.plot(ax=)`, `channel.plot()`,
  `channel.info()`.
- If the library makes something awkward, that is a finding about the
  library. Say so and propose the fix rather than working around it in the
  example.

## Structuring the example script

### Naming the chain

Three rules, and they are about where the reader looks and what the
signature promises.

1. **A function only when the chain is built more than once** -- several
   values of a parameter, several variants, a Monte-Carlo. A chain built
   once and used once is written inline: wrapping four blocks in a
   function called once, with no argument, is ceremony, and it puts an
   indirection between the reader and their first chain.
2. **Always `get_<thing>()`**, one prefix, the noun saying what comes
   back. Which noun follows from how the page is organized:

   | the page… | names |
   |---|---|
   | has one chain | `get_chain(...)` |
   | cuts one chain in two | `get_channel()` and `get_receiver()` |
   | compares two structurally different chains | `get_uncoded_chain()`, `get_coded_chain(soft)` |
   | builds a sub-assembly, not a whole chain | the part: `get_transmitter()` |

   Two chains that differ by a block or two are **one** function with a
   parameter, not two functions. Two chains that differ in structure --
   four blocks against six -- are two functions, because merging them
   would hide behind a flag exactly what the page is comparing.
3. **What varies is a parameter, never a module global.** Constants of
   the page (the fibre, the constellation, the roll-off) may stay
   global; anything the page *varies* goes through the signature. A
   function whose signature announces three arguments and silently reads
   eleven more is a closure wearing a `def`, and the reader cannot tell
   from the call site what changed.

### The rest of the script

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
