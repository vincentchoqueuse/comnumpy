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

## Error messages

Validation failures raise `comnumpy.ShapeError` and follow the template
*observed, expected, action*:

```
ShapeError: expected shape (..., ant=2, N), got (100,) -- put the antenna
axis on -2 (see CONVENTIONS.md) or fix the channel matrix size.
```
