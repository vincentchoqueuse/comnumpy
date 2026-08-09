"""Spectral allocation objects (decisions D15-D21).

An allocation is a frozen :class:`CarrierAllocation` object carrying a
2D time-frequency mask in **physical order** (signed subcarrier index,
DC at the center) plus its metadata and the clause of the standard it
comes from. The conversion to FFT order is explicit and unique
(:meth:`CarrierAllocation.to_fft_order`, decision D16).

Standard allocations come from a registry-backed catalog::

    >>> alloc = get_allocation("802.11a")
    >>> alloc.N_data, alloc.N_pilots
    (48, 4)
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "CarrierType", "CarrierAllocation", "CARRIER_STYLE",
    "band_allocation", "scattered_allocation",
    "get_allocation", "register_allocation", "available_allocations",
]


class CarrierType(IntEnum):
    """Subcarrier roles -- no magic integers (decision D21a)."""
    NULL = 0
    DATA = 1
    PILOT = 2


# Single frozen source for color, glyph, hatch and label (decision D27a):
# the matplotlib figure and the ASCII spectral map read the same row, so
# the two views cannot diverge. Colors from the Okabe-Ito palette; the
# hatch carries the redundancy required by D27c (readable without color).
CARRIER_STYLE: dict[CarrierType, dict[str, str]] = {
    CarrierType.NULL:  {"color": "#BBBBBB", "glyph": ".", "hatch": "",   "label": "null"},
    CarrierType.DATA:  {"color": "#0072B2", "glyph": "#", "hatch": "",   "label": "data"},
    CarrierType.PILOT: {"color": "#D55E00", "glyph": "P", "hatch": "//", "label": "pilot"},
}


@dataclass(frozen=True, slots=True)
class CarrierAllocation:
    r"""A time-frequency subcarrier allocation (decision D15).

    The mask ``carrier_type`` has shape ``(T_p, N_fft)``: one row per
    OFDM symbol of the time period :math:`T_p`, one column per
    subcarrier in **physical order** (signed index :math:`k`, DC at the
    center). Scattered pilot patterns (DVB-T, LTE CRS, NR DM-RS) are 2D
    by nature; a pure band allocation has ``T_p == 1``.

    Parameters
    ----------
    carrier_type : np.ndarray
        ``(T_p, N_fft)`` int8 array of :class:`CarrierType` values, in
        physical order.
    subcarrier_spacing : float, optional, keyword-only
        Subcarrier spacing :math:`\Delta f` in Hz.
    cp_length : int, optional, keyword-only
        Cyclic prefix length in samples.
    standard : str, keyword-only
        Name of the standard this allocation implements. Default "custom".
    reference : str, keyword-only
        Clause of the standard the numbers were copied from.

    Examples
    --------
    >>> mask = np.array([[0, 1, 2, 1, 0]])
    >>> alloc = CarrierAllocation(mask)
    >>> alloc.N_fft, alloc.N_data, alloc.N_pilots, alloc.period
    (5, 2, 1, 1)
    >>> print(alloc.k)
    [-2 -1  0  1  2]
    """
    carrier_type: np.ndarray
    subcarrier_spacing: Optional[float] = field(default=None, kw_only=True)
    cp_length: Optional[int] = field(default=None, kw_only=True)
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self):
        mask = np.atleast_2d(np.asarray(self.carrier_type, dtype=np.int8))
        if mask.ndim != 2:
            raise ValueError(
                f"carrier_type must be a (T_p, N_fft) 2D array, got ndim={mask.ndim}")
        valid = np.isin(mask, [int(v) for v in CarrierType])
        if not np.all(valid):
            bad = np.unique(np.asarray(self.carrier_type)[~valid.reshape(np.shape(self.carrier_type))])
            raise ValueError(
                f"carrier_type contains invalid values {bad.tolist()}; "
                f"allowed: {[int(v) for v in CarrierType]} (CarrierType)")
        # the number of data subcarriers must be constant over the period,
        # otherwise the S/P block size is ill-defined (invariant, D15)
        n_data = np.sum(mask == CarrierType.DATA, axis=1)
        if not np.all(n_data == n_data[0]):
            raise ValueError(
                f"the number of DATA subcarriers must be constant over the "
                f"period, got {n_data.tolist()} per symbol")
        mask.setflags(write=False)
        object.__setattr__(self, "carrier_type", mask)

    # -- properties -----------------------------------------------------
    @property
    def N_fft(self) -> int:
        """FFT size (number of subcarriers)."""
        return self.carrier_type.shape[1]

    @property
    def period(self) -> int:
        """Time period :math:`T_p` of the pattern, in OFDM symbols."""
        return self.carrier_type.shape[0]

    @property
    def N_data(self) -> int:
        """Number of DATA subcarriers per OFDM symbol."""
        return int(np.sum(self.carrier_type[0] == CarrierType.DATA))

    @property
    def N_pilots(self) -> int:
        """Total number of PILOT subcarriers over one period."""
        return int(np.sum(self.carrier_type == CarrierType.PILOT))

    @property
    def k(self) -> np.ndarray:
        """Signed subcarrier indices (physical order, DC at index 0)."""
        N = self.N_fft
        return np.arange(N) - N // 2

    # -- conversions ----------------------------------------------------
    def to_fft_order(self) -> np.ndarray:
        """Return the mask in FFT order.

        Physical order (DC at the center) maps to FFT order (DC first)
        through ``ifftshift`` -- the unique, explicit conversion of
        decision D16 (``fftshift`` is its inverse; the two differ for
        odd ``N_fft``).
        """
        from scipy.fft import ifftshift
        return ifftshift(self.carrier_type, axes=1)

    # -- rendering (decision D21b) --------------------------------------
    def _row_to_glyphs(self, row: np.ndarray) -> str:
        return "".join(CARRIER_STYLE[CarrierType(v)]["glyph"] for v in row)

    def _aggregate_row(self, row: np.ndarray, width: int) -> str:
        """Full-band view for large N_fft: PILOT > DATA > NULL per bucket."""
        edges = np.linspace(0, len(row), width + 1).astype(int)
        glyphs = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            bucket = row[a:b]
            if np.any(bucket == CarrierType.PILOT):
                glyphs.append(CARRIER_STYLE[CarrierType.PILOT]["glyph"])
            elif np.any(bucket == CarrierType.DATA):
                glyphs.append(CARRIER_STYLE[CarrierType.DATA]["glyph"])
            else:
                glyphs.append(CARRIER_STYLE[CarrierType.NULL]["glyph"])
        return "".join(glyphs)

    def __repr__(self) -> str:
        header = self.standard
        if self.subcarrier_spacing is not None:
            header += f" ({self.N_fft}-FFT, df={self.subcarrier_spacing/1e3:g} kHz)"
        else:
            header += f" ({self.N_fft}-FFT)"
        if self.reference:
            header += f"   [{self.reference}]"

        lines = [header]
        max_width = 80
        if self.period == 1:
            row = self.carrier_type[0]
            if self.N_fft <= max_width:
                lines.append(self._row_to_glyphs(row))
            else:
                # aggregation is allowed only for the single-symbol view
                lines.append(self._aggregate_row(row, max_width))
        else:
            # scattered pattern: aggregation would show a uniform band of
            # pilots, so zoom at full resolution instead (decision D21b)
            n_cols = min(self.N_fft, max_width - 4)
            for i in range(min(self.period, 4)):
                lines.append(f"l={i} " + self._row_to_glyphs(self.carrier_type[i][:n_cols]))
            if self.period > 4:
                lines.append(f"... ({self.period - 4} more symbols in the period)")

        stats = f"data {self.N_data} | pilots {self.N_pilots} | " \
                f"nulls {int(np.sum(self.carrier_type == CarrierType.NULL))}"
        extras = []
        if self.subcarrier_spacing is not None:
            extras.append(f"{self.subcarrier_spacing/1e3:g} kHz")
        if self.cp_length is not None:
            extras.append(f"CP {self.cp_length}")
        if extras:
            stats += " | " + ", ".join(extras)
        lines.append(stats)
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:
        """Plot the allocation map; returns the axis (decision D25).

        Colors, hatches and legend labels come from the frozen semantic
        table :data:`CARRIER_STYLE` (decision D27a) -- the same source
        as the ASCII spectral map of ``__repr__``, so the two views
        cannot diverge.
        """
        import matplotlib.pyplot as plt  # local import (D36)
        from matplotlib.patches import Patch, Rectangle
        if ax is None:
            _, ax = plt.subplots()
        for row in range(self.period):
            for j, k in enumerate(self.k):
                ctype = CarrierType(self.carrier_type[row, j])
                style = CARRIER_STYLE[ctype]
                ax.add_patch(Rectangle(
                    (float(k) - 0.5, row - 0.5), 1.0, 1.0,
                    facecolor=style["color"], hatch=style["hatch"],
                    edgecolor="white", linewidth=0.2))
        ax.set_xlim(float(self.k[0]) - 0.5, float(self.k[-1]) + 0.5)
        ax.set_ylim(self.period - 0.5, -0.5)
        ax.set_xlabel("subcarrier index k (physical order)")
        ax.set_ylabel("OFDM symbol in period")
        ax.set_title(f"{self.standard} carrier allocation")
        ax.legend(handles=[
            Patch(facecolor=s["color"], hatch=s["hatch"], label=s["label"])
            for s in CARRIER_STYLE.values()], loc="upper right")
        return ax


# ---------------------------------------------------------------------------
# constructors (decisions D15/D20)
# ---------------------------------------------------------------------------

def _check_expect(alloc: CarrierAllocation, expect: Optional[dict[str, int]]) -> CarrierAllocation:
    """Verify the entry against the numbers copied from the standard (D20)."""
    if expect is None:
        return alloc
    checks = {"data": alloc.N_data, "pilots": alloc.N_pilots}
    for key, expected in expect.items():
        if key not in checks:
            raise ValueError(f"unknown expect key {key!r}; allowed: {sorted(checks)}")
        if checks[key] != expected:
            raise ValueError(
                f"{alloc.standard}: expected {expected} {key} subcarriers "
                f"(from {alloc.reference or 'the standard table'}), the mask "
                f"produces {checks[key]} -- fix the entry")
    return alloc


def band_allocation(N_fft: int, k_used: tuple[int, int],
                    k_pilots: Sequence[int] = (), n_dc: int = 0,
                    expect: Optional[dict[str, int]] = None,
                    **meta: Any) -> CarrierAllocation:
    r"""Build a band allocation from signed subcarrier indices.

    Parameters
    ----------
    N_fft : int
        FFT size.
    k_used : tuple of int
        Inclusive signed range ``(k_min, k_max)`` of occupied subcarriers.
    k_pilots : iterable of int, optional
        Signed indices of pilot subcarriers.
    n_dc : int, optional
        Number of nulled subcarriers centered on DC. Default 0.
    expect : dict, optional
        ``{"data": ..., "pilots": ...}`` copied verbatim from the
        standard's table; checked at construction (decision D20).
    **meta
        Forwarded to :class:`CarrierAllocation` (``subcarrier_spacing``,
        ``cp_length``, ``standard``, ``reference``).

    Examples
    --------
    >>> alloc = band_allocation(8, k_used=(-3, 3), k_pilots=(-2, 2), n_dc=1,
    ...                         expect={"data": 4, "pilots": 2})
    >>> print(alloc.carrier_type[0])
    [0 1 2 1 0 1 2 1]
    """
    mask = np.zeros((1, N_fft), dtype=np.int8)
    k = np.arange(N_fft) - N_fft // 2
    k_min, k_max = k_used
    mask[0, (k >= k_min) & (k <= k_max)] = CarrierType.DATA
    for kp in k_pilots:
        idx = np.flatnonzero(k == kp)
        if idx.size == 0:
            raise ValueError(f"pilot index {kp} outside the FFT grid "
                             f"[{k[0]}, {k[-1]}]")
        mask[0, idx[0]] = CarrierType.PILOT
    if n_dc > 0:
        dc = np.flatnonzero(k == 0)[0]
        start = dc - n_dc // 2
        mask[0, start:start + n_dc] = CarrierType.NULL
    return _check_expect(CarrierAllocation(mask, **meta), expect)


def scattered_allocation(N_fft: int, k_used: tuple[int, int], period: int,
                         rule: Callable[[int, int], bool],
                         expect: Optional[dict[str, int]] = None,
                         **meta: Any) -> CarrierAllocation:
    r"""Build a scattered-pilot allocation from a positional rule.

    Parameters
    ----------
    N_fft : int
        FFT size.
    k_used : tuple of int
        Inclusive signed range of occupied subcarriers.
    period : int
        Time period :math:`T_p` of the pattern, in OFDM symbols.
    rule : callable
        ``rule(l, k) -> bool`` returning True when subcarrier ``k`` of
        symbol ``l`` carries a pilot (``l`` in ``range(period)``, ``k``
        signed). Note: callables are not serializable (documented limit
        of decision D31).
    expect : dict, optional
        Self-check values copied from the standard (decision D20).
    **meta
        Forwarded to :class:`CarrierAllocation`.

    Examples
    --------
    >>> alloc = scattered_allocation(8, k_used=(-4, 3), period=2,
    ...                              rule=lambda l, k: (k + 4 + 2 * l) % 4 == 0)
    >>> print(alloc.carrier_type)
    [[2 1 1 1 2 1 1 1]
     [1 1 2 1 1 1 2 1]]
    """
    mask = np.zeros((period, N_fft), dtype=np.int8)
    k = np.arange(N_fft) - N_fft // 2
    k_min, k_max = k_used
    used = (k >= k_min) & (k <= k_max)
    mask[:, used] = CarrierType.DATA
    for line in range(period):
        for j in np.flatnonzero(used):
            if rule(line, int(k[j])):
                mask[line, j] = CarrierType.PILOT
    return _check_expect(CarrierAllocation(mask, **meta), expect)


# ---------------------------------------------------------------------------
# registry-backed catalog (decision D17)
# ---------------------------------------------------------------------------

_ALLOCATION_REGISTRY: dict[str, Callable[..., CarrierAllocation]] = {}


def register_allocation(name: str):
    """Register a catalog entry; users can add their own standards."""
    def decorator(func: Callable[..., CarrierAllocation]):
        _ALLOCATION_REGISTRY[name] = func
        return func
    return decorator


def get_allocation(standard: str, **kwargs: Any) -> CarrierAllocation:
    """Return a catalog allocation by standard name.

    Examples
    --------
    >>> alloc = get_allocation("802.11a")
    >>> alloc.N_fft, alloc.N_data, alloc.N_pilots
    (64, 48, 4)
    """
    try:
        factory = _ALLOCATION_REGISTRY[standard]
    except KeyError:
        known = ", ".join(sorted(_ALLOCATION_REGISTRY))
        raise KeyError(
            f"unknown standard {standard!r}; known: {known}. "
            f"Register your own with @register_allocation(name).") from None
    return factory(**kwargs)


def available_allocations() -> list[str]:
    """Names accepted by :func:`get_allocation`."""
    return sorted(_ALLOCATION_REGISTRY)


# ---------------------------------------------------------------------------
# catalog entries (decision D20: every entry carries its expect= self-check
# copied from the standard's table, verified at construction).
#
# NOTE: these values were checked for internal consistency (sums match the
# FFT size) and against the tables quoted in `reference`; a line-by-line
# re-validation against the spec text remains a merge condition (D20).
# ---------------------------------------------------------------------------

@register_allocation("802.11a")
def _wifi_11a() -> CarrierAllocation:
    return band_allocation(
        N_fft=64,
        k_used=(-26, 26),                    # 52 occupied subcarriers
        k_pilots=(-21, -7, 7, 21),
        n_dc=1,
        expect={"data": 48, "pilots": 4},    # IEEE 802.11-2020, Table 17-5
        subcarrier_spacing=312.5e3, cp_length=16,
        standard="802.11a", reference="IEEE 802.11-2020, Table 17-5",
    )


@register_allocation("802.11n")
def _wifi_11n() -> CarrierAllocation:
    # HT operation, 20 MHz
    return band_allocation(
        N_fft=64,
        k_used=(-28, 28),                    # 56 occupied subcarriers
        k_pilots=(-21, -7, 7, 21),
        n_dc=1,
        expect={"data": 52, "pilots": 4},    # IEEE 802.11-2020, 19.3.7
        subcarrier_spacing=312.5e3, cp_length=16,
        standard="802.11n", reference="IEEE 802.11-2020, 19.3.7 (HT, 20 MHz)",
    )


@register_allocation("802.11ac-40")
def _wifi_11ac_40() -> CarrierAllocation:
    return band_allocation(
        N_fft=128,
        k_used=(-58, 58),
        k_pilots=(-53, -25, -11, 11, 25, 53),
        n_dc=3,
        expect={"data": 108, "pilots": 6},   # IEEE 802.11-2020, 21.3.7.3
        subcarrier_spacing=312.5e3, cp_length=32,
        standard="802.11ac-40", reference="IEEE 802.11-2020, 21.3.7.3 (VHT, 40 MHz)",
    )


@register_allocation("802.11ac-80")
def _wifi_11ac_80() -> CarrierAllocation:
    return band_allocation(
        N_fft=256,
        k_used=(-122, 122),
        k_pilots=(-103, -75, -39, -11, 11, 39, 75, 103),
        n_dc=3,
        expect={"data": 234, "pilots": 8},   # IEEE 802.11-2020, 21.3.7.3
        subcarrier_spacing=312.5e3, cp_length=64,
        standard="802.11ac-80", reference="IEEE 802.11-2020, 21.3.7.3 (VHT, 80 MHz)",
    )


_LTE_N_RB = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}


@register_allocation("LTE")
def _lte(bandwidth_MHz: float = 10) -> CarrierAllocation:
    # Data-only grid; the cell-specific reference signals (CRS) are a
    # separate scattered pattern (open point of the architecture document).
    if bandwidth_MHz not in _LTE_N_RB:
        raise ValueError(f"LTE bandwidth {bandwidth_MHz} MHz not in "
                         f"{sorted(_LTE_N_RB)}")
    N_RB = _LTE_N_RB[bandwidth_MHz]
    n_sc = 12 * N_RB
    N_fft = 128
    while N_fft < n_sc + 1:  # + DC subcarrier
        N_fft *= 2
    return band_allocation(
        N_fft=N_fft,
        k_used=(-n_sc // 2, n_sc // 2),      # n_sc + 1 indices, DC nulled below
        n_dc=1,
        expect={"data": 12 * N_RB, "pilots": 0},  # 3GPP TS 36.211, 6.12
        subcarrier_spacing=15e3,
        standard="LTE", reference=f"3GPP TS 36.211 (release 17), 6.12 -- "
                                  f"{bandwidth_MHz} MHz, {N_RB} RB",
    )


@register_allocation("5G-NR")
def _nr(mu: int = 0, N_RB: int = 106) -> CarrierAllocation:
    # Data-only grid; DM-RS is a separate scattered pattern. Unlike LTE,
    # NR does not reserve a DC subcarrier (3GPP TS 38.211, 7.4.1).
    if not 0 <= mu <= 4:
        raise ValueError(f"NR numerology mu={mu} outside 0..4")
    n_sc = 12 * N_RB
    N_fft = 128
    while N_fft < n_sc:
        N_fft *= 2
    mask = np.zeros((1, N_fft), dtype=np.int8)
    k = np.arange(N_fft) - N_fft // 2
    mask[0, (k >= -n_sc // 2) & (k <= n_sc // 2 - 1)] = CarrierType.DATA
    alloc = CarrierAllocation(
        mask,
        subcarrier_spacing=15e3 * 2**mu,
        standard="5G-NR",
        reference=f"3GPP TS 38.211 (release 17), 4.4.4 -- mu={mu}, {N_RB} RB",
    )
    return _check_expect(alloc, {"data": 12 * N_RB, "pilots": 0})
