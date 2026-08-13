# comnumpy tensor conventions

This page is **normative** (architecture decision D2). Every block conforms
to it; a block that needs something else must say so explicitly in its
docstring and validate its input in `prepare()`.

## Canonical layouts

| Layout | Shape | Meaning |
|---|---|---|
| **Serial** | `(..., N)` | samples / bits / symbols on the last axis |
| **Block** | `(..., T, F)` | block index `T` on axis -2, block content `F` on axis -1 |

The Serial ↔ Block conversion is a **pure C-order reshape**
(`Serial2Parallel` / `Parallel2Serial`). Any S/P implementation that
requires a transposition violates this convention.

**Optional structural axes** sit to the left of the core layout, in
physical nesting order: `(batch..., wdm, ant/pol, core)`. The batch is
never a named axis: it is implicit in `...` and carried by broadcasting.

**MIMO layout:** `(..., ant, N)` — antennas on axis -2, time on axis -1.

## Domain (time / frequency)

The FFT changes the *meaning* of axis -1 (time → frequency), never its
*position*. Blocks that change the current domain document it in their
docstring (`FFTProcessor`, `IFFTProcessor`).

## Spectral order

Frequency allocations and masks are described in **physical order**
(signed index, DC at the center). The conversion to FFT order is
explicit, unique, and done with `ifftshift` (decision D16 — the legacy
`shift` parameter of `get_standard_carrier_allocation` is scheduled for
removal with the `CarrierAllocation` object).

## Block categories

Every block belongs to exactly one category, stated in its docstring
("Axes:" line):

1. **Element-wise** — operates pointwise, shape-agnostic
   (`AWGN`, `SymbolMapper`, `HardClipper`).
2. **Axis -1** — hardcodes `axis=-1` and broadcasts over everything else
   (filters, `FFTProcessor`, `IFFTProcessor`, `CyclicPrefixer`).
3. **Declared axis** — requires a specific axis and validates it in
   `prepare()`, raising `ShapeError` with a *got / expected / action*
   message (MIMO blocks on `ant`, `Serial2Parallel`, carrier allocation).

The former `is_mimo` attribute — a shape-guessing mechanism — is removed;
the declared category and `prepare()` validation replace it.

## Batch axes and event axes (D51)

A block's category names its **event axes** — the trailing axes that
make up *one* realization of what the block models: nothing for an
element-wise block, `(N,)` for a filter, `(2, N)` for a polarization
pair, `(ant, N)` for a MIMO frame. Every axis to the left of the event
is a **batch axis**: independent trials, run in one call. The contract
has three families:

1. **Deterministic** blocks broadcast their one configuration over the
   batch — every trial goes through the same taps, the same equalizer
   matrix (`FIRChannel`, `SRRCFilter`, `LinearEqualizer`, the OFDM
   chains). Row `i` of a batched call equals the block applied to row
   `i` alone.
2. **Stochastic** blocks draw **independently per event** — trials must
   not share a realization. `AWGN` draws element-wise; `PhaseNoise`
   declares its event with `per=`: `"pair"` (default — one laser per
   `(2, N)` block, independent across the batch), `"row"`, or
   `"signal"`. A shape the declaration cannot read one way is refused
   with the resolutions named.
   The exception is the block whose realization is **frozen at
   construction** (`PMDEmulator`, `TappedDelayLineChannel`,
   `FlatMIMOChannel`): one seeded draw *is* the configuration, so the
   whole batch sees the same channel — ergodic noise over a fixed
   channel. A new draw per trial is a new instance (or `set_params`),
   and that loop stays visible.
3. **Adaptive** blocks carry **independent state per event** — one
   equalizer per pair, never one state smeared across trials.
   `BlindDualMIMOCompensator` on `(..., 2, N)` adapts one butterfly per
   pair and exposes `H_` with the batch axes in front;
   `BlindPhaseSearchCompensator` tracks one `phase_` trajectory per
   row.

The payoff is Monte-Carlo without a Python loop over trials:
`monte_carlo(chain, param, values, metrics, (n_trials, N), seed=…)`
pools each metric over `n_trials` independent frames per sweep point —
for equal-size trials, exactly the mean of the per-trial values.
`tests/core/test_batch_axes.py` locks one block of each family, and
`tests/test_batch_contract.py` is the **ratchet over the whole
catalogue**: it discovers every `Processor` subclass and fails unless
each is verified batched (`BROADCAST` by row equality, `INDEPENDENT`
by realization inequality, `REFUSES` by the raised error) or exempted
with a written reason. A new block cannot merge without declaring what
a batch means for it.

## Observing signals: taps, not blocks

A chain describes the communication system. Nothing else goes into
`module_list` — no recorder, logger, scope or debugger block. To observe
an intermediate signal, name the block and declare it as a tap:

```python
chain = Sequential([SymbolGenerator(16, name="tx"), SymbolMapper(alphabet),
                    AWGN(snr_dB=15, name="awgn")], taps=["tx", "awgn"])
y = chain(1000)
x_tx = chain.tap("tx")          # signal recorded during the last run
plot_iq(chain.tap("awgn"))      # plotting is a function, not a block
```

Consequences of this rule:

* `repr`, `summary()`, `to_mermaid()`, the JSON export and the block
  indices describe the communication system only.
* A tap costs one dictionary store of a **reference** per tapped block
  (no copy). This relies on the library-wide invariant that **a block
  never mutates its input in place**: `forward()` returns a freshly
  allocated array (or the input untouched). Custom blocks must honour it
  — an in-place block would corrupt earlier taps *and* break chain
  re-entrancy.
* Blocks never hold a live reference to another block. A data-aided
  block takes its reference as a plain array (`reference=x_tx`) when
  the reference is known in advance — a preamble, a training sequence.
* Statistics are functions too: `comnumpy.core.metrics.signal_report(x)`
  returns a dict; the caller decides whether to log it, tabulate it or
  assert on it.

## Feeding an estimator from inside the chain: wiring

When the reference is *produced by the chain itself* (the transmitted
symbols of this very run), a frozen array is wrong: on the second run it
would still hold the first run's data, silently. Declare the edge
instead:

```python
chain = Sequential([
    SymbolGenerator(16, name="tx"),
    SymbolMapper(alphabet, name="ref"),
    ...,
    DataAidedPhaseCompensator(reference=np.zeros(1), name="comp"),
    SymbolDemapper(alphabet),
], wiring={"comp.reference": "ref"})
```

Before `comp` runs, the chain assigns it the signal `ref` produced in the
same pass. Rules:

* the source is tapped automatically — no need to list it in `taps`;
* the source must run **before** the target; a backward edge raises,
  because it would serve the previous run's value;
* the wiring feeds *data*, not structure: the target's `__post_init__` is
  not re-run (use `set_params` for parameters that need a re-precompute);
* like taps, the edge is chain metadata — `module_list` is unchanged, and
  `taps`/`wiring` are carried through the JSON export.

This is the honest, bounded version of the `inputs` field of decision
D31: a second input to a block, declared by the chain. A general DAG is
still out of scope.

**Naming.** The known signal an estimator compares against is the
`reference` — the same word `monte_carlo(reference=...)` uses for the same
idea. Blocks that need one are `DataAided*`, by opposition to the
`Blind*` family; that is the standard pair of the field (Proakis,
§8: *data-aided* vs *blind* estimation).

## Where derived state lives, and how a parameter changes (D50)

A block precomputes in two places, and the split is by *what the value
depends on*:

* **`__post_init__`** — validate, normalize, and precompute whatever
  depends **only on the block's own parameters**: filter taps, an
  alphabet coerced to an array, an allocation mask.
* **`prepare(x)`** — derive whatever depends on the **signal**: its
  length, its rate, its shape, how many paths it carries. Axis
  validation for category-3 blocks happens here.

The direction is asymmetric, which is why there are two places and not
one. Signal-dependent state in `__post_init__` is **wrong** — the
constructor has not seen the signal. Parameter-only state in `prepare()`
is merely **repeated work**; `FiberLink` and `DBP` do this and it is
negligible against the split-step cost, so "derived in `prepare()`" is
not evidence that a value depends on the signal.

**A parameter changes through `Sequential.set_params` (D34), never by
assignment.** `set_params` re-runs `__post_init__` on every block it
touches; a direct assignment does not, and 57 of the 67 blocks that
derive anything derive it there:

```python
chain.set_params(downsampler__L=2)   # rebuilds the anti-alias filter
chain["downsampler"].L = 2           # does not -- the filter stays at 1/4
```

Blocks cannot be frozen to make that assignment raise: `wiring` writes
`reference` on a block *during* the pass, and every estimator writes its
own `theta_`. Mutability is what separates a `Processor` from a value
object such as `FiberSpec` or `Constellation`.

**Writing a new block.** Whatever `__post_init__` computes must be
recomputable by calling `__post_init__` again, with no side effect — that
second call is exactly what `set_params` makes. A block that appends to a
list, opens a file or consumes a generator there breaks `set_params` for
everything downstream of it.

## Error messages

Validation failures raise `comnumpy.ShapeError` and follow the template
*observed, expected, action*:

```
ShapeError: expected shape (..., ant=2, N), got (100,) -- put the antenna
axis on -2 (see CONVENTIONS.md) or fix the channel matrix size.
```
