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
  block takes its reference as a plain array (`target_data=x_tx`) when
  the reference is known in advance — a preamble, a training sequence.

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
    TrainedBasedPhaseCompensator(target_data=np.zeros(1), name="comp"),
    SymbolDemapper(alphabet),
], wiring={"comp.target_data": "ref"})
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
* Statistics are functions too: `comnumpy.core.metrics.signal_report(x)`
  returns a dict; the caller decides whether to log it, tabulate it or
  assert on it.

## Error messages

Validation failures raise `comnumpy.ShapeError` and follow the template
*observed, expected, action*:

```
ShapeError: expected shape (..., ant=2, N), got (100,) -- put the antenna
axis on -2 (see CONVENTIONS.md) or fix the channel matrix size.
```
