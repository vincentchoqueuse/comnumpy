r"""Wavelength-division multiplexing: the frequency plan and the two blocks.

Three objects, and the split between them is the point:

* :class:`WDMGrid` -- a frozen value object holding *where the channels
  are*, in hertz. It carries no signal and no sampling rate, exactly as
  :class:`~comnumpy.ofdm.allocation.CarrierAllocation` carries a
  subcarrier plan and no OFDM symbol (D15/D18).
* :class:`WDMMultiplexer` -- shifts each channel to its slot and sums.
* :class:`WDMDemultiplexer` -- shifts one slot back to baseband and
  filters.

Naming and units follow the convention shared by the software-radio
libraries (GNU Radio's ``freq_xlating_fir_filter``,
``pfb_synthesizer_ccf`` / ``pfb_channelizer_ccf``): **frequencies are
absolute, in hertz, and every block that needs one takes an explicit
sample rate ``fs``**. Nothing here is normalized behind the user's back.
The multiplexer is the synthesis bank, the demultiplexer the analysis
bank; the optical names are kept because that is what the field calls
them, and the radio names are given here so the equivalence is visible.

Rate conversion is deliberately *not* part of these blocks. A channel
enters the multiplexer already sampled at the composite rate, and leaves
the demultiplexer at that same rate; use
:class:`~comnumpy.core.processors.Upsampler` and
:class:`~comnumpy.core.processors.Downsampler` around them. That is what
keeps a WDM chain readable: one block, one job.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .utils import apply_frequency_response

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

from .utils import itu_grid_frequency

logger = logging.getLogger(__name__)

__all__ = ["WDMGrid", "WDMMultiplexer", "WDMDemultiplexer"]


@dataclass(frozen=True)
class WDMGrid:
    r"""Where the WDM channels sit, in hertz (decision D44).

    Signal Model
    ------------
    A grid of :math:`C` channels is the set of absolute centre
    frequencies :math:`\nu_c` and the occupied bandwidth :math:`B` each
    of them uses. What the blocks actually mix with is the **offset**
    from the composite band centre :math:`\nu_0`:

    .. math::

        f_c = \nu_c - \nu_0, \qquad c = 0, \dots, C-1

    A grid is representable at a sampling rate :math:`f_s` when every
    channel fits inside the band, edges included:

    .. math::

        \max_c |f_c| + \frac{B}{2} \leq \frac{f_s}{2}

    Parameters
    ----------
    frequencies_Hz : sequence of float
        Absolute centre frequencies :math:`\nu_c`, strictly ascending.
    bandwidth_Hz : float, keyword-only
        Occupied bandwidth :math:`B` of one channel, in Hz. Equal for
        all channels: a per-channel bandwidth would make the grid a
        different object (a flexible-grid *allocation*), not this one.
    center_Hz : float, optional, keyword-only
        Composite band centre :math:`\nu_0`, i.e. the frequency the
        sampled waveform sits at. Defaults to the midpoint of the
        outermost channels, which is what minimizes the required
        :math:`f_s`.
    standard : str, keyword-only
        Name of the plan, e.g. ``"ITU-T G.694.1 50 GHz"``.
    reference : str, keyword-only
        Clause the numbers were copied from (same provenance rule as
        D15/D20).

    Raises
    ------
    ValueError
        If fewer than one channel is given, if the frequencies are not
        strictly ascending, if the bandwidth is not positive, or if the
        bandwidth exceeds the tightest channel spacing -- overlapping
        channels cannot be separated by any linear filter, so the grid
        is rejected at construction rather than producing crosstalk
        later.

    References
    ----------
    ITU-T Recommendation G.694.1 (2020), *Spectral grids for WDM
    applications: DWDM frequency grid*.

    G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley,
    2010, Chapter 6.

    Examples
    --------
    >>> grid = WDMGrid.uniform(5, spacing_Hz=50e9, bandwidth_Hz=32e9)
    >>> grid.n_channels, float(grid.offsets_Hz[0]) / 1e9, grid.min_fs / 1e9
    (5, -100.0, 232.0)
    """

    frequencies_Hz: tuple[float, ...]
    bandwidth_Hz: float = field(kw_only=True)
    center_Hz: Optional[float] = field(default=None, kw_only=True)
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        frequencies = tuple(float(nu) for nu in self.frequencies_Hz)
        if not frequencies:
            raise ValueError("a WDM grid needs at least one channel, got none")
        if any(b <= a for a, b in zip(frequencies, frequencies[1:], strict=False)):
            raise ValueError(
                f"expected strictly ascending centre frequencies, got "
                f"{[nu / 1e9 for nu in frequencies]} GHz -- sort the plan, "
                f"the channel order is the array order in the multiplexer")
        if self.bandwidth_Hz <= 0:
            raise ValueError(
                f"expected a positive channel bandwidth, got "
                f"{self.bandwidth_Hz} Hz")
        object.__setattr__(self, "frequencies_Hz", frequencies)
        if self.center_Hz is None:
            object.__setattr__(self, "center_Hz",
                               0.5 * (frequencies[0] + frequencies[-1]))
        else:
            object.__setattr__(self, "center_Hz", float(self.center_Hz))
        spacing = self.spacing_Hz
        if spacing is not None and self.bandwidth_Hz > spacing:
            raise ValueError(
                f"expected a channel bandwidth of at most the tightest "
                f"spacing {spacing / 1e9:.4g} GHz, got "
                f"{self.bandwidth_Hz / 1e9:.4g} GHz -- overlapping channels "
                f"cannot be separated by any filter; widen the spacing or "
                f"narrow the channels")

    # -- properties -----------------------------------------------------
    @property
    def n_channels(self) -> int:
        """Number of channels :math:`C`."""
        return len(self.frequencies_Hz)

    @property
    def offsets_Hz(self) -> np.ndarray:
        r"""Offsets :math:`f_c` from the composite centre, in Hz."""
        return np.asarray(self.frequencies_Hz) - float(self.center_Hz or 0.0)

    @property
    def spacing_Hz(self) -> Optional[float]:
        """Tightest spacing between two neighbours; ``None`` if C == 1."""
        if self.n_channels < 2:
            return None
        return float(np.min(np.diff(self.frequencies_Hz)))

    @property
    def guard_Hz(self) -> Optional[float]:
        """Unused band between two neighbours; ``None`` if C == 1.

        Zero is the Nyquist-WDM limit (channels touching); negative is
        impossible here, ``__post_init__`` rejects it.
        """
        spacing = self.spacing_Hz
        return None if spacing is None else spacing - self.bandwidth_Hz

    @property
    def min_fs(self) -> float:
        r"""Smallest sampling rate carrying the whole comb, in Hz.

        This is :math:`2(\max_c |f_c| + B/2)`, the equality case of the
        condition in the signal model.
        """
        return 2.0 * (float(np.max(np.abs(self.offsets_Hz))) + self.bandwidth_Hz / 2)

    def plot(self, ax: Any = None, cut: Optional[int] = None) -> Any:
        """Draw the grid: one box per channel, on the frequency axis.

        A grid is a layout, so looking at it should not require
        synthesising a signal and estimating its spectrum -- that shows
        the same information through an unnecessary detour, and buries
        the guard band under the roll-off of whatever pulse shape was
        chosen. Each channel is drawn over the bandwidth it occupies, so
        the gaps between the boxes *are* the guard.

        Parameters
        ----------
        ax : matplotlib axis, optional
            Axis to draw on; a new figure is created when omitted
            (decision D25).
        cut : int, optional
            Index of the channel to single out -- the *cut* in the sense
            of a nonlinear-interference model, the one whose noise is
            being counted. Drawn filled where the others are outlined.

        Returns
        -------
        matplotlib axis
            The axis, so the caller keeps control of the figure.
        """
        import matplotlib.pyplot as plt  # local import (D36)
        from matplotlib.patches import Rectangle
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 2.6), layout="constrained")
        offsets = np.asarray(self.offsets_Hz) / 1e9
        width = self.bandwidth_Hz / 1e9
        for index, offset in enumerate(offsets):
            chosen = index == cut
            ax.add_patch(Rectangle(
                (offset - width / 2, 0.0), width, 1.0,
                facecolor="C1" if chosen else "none",
                alpha=0.45 if chosen else 1.0,
                edgecolor="C0", linewidth=1.2))
            if chosen:
                ax.annotate("cut", (offset, 1.06), ha="center", fontsize=9,
                            color="C1")
        span = float(np.max(offsets) - np.min(offsets)) + width
        ax.set_xlim(np.min(offsets) - width, np.max(offsets) + width)
        ax.set_ylim(0.0, 1.35)
        ax.set_yticks([])
        ax.set_xlabel("frequency offset [GHz]")
        guard = "" if self.guard_Hz is None else \
            f", {self.guard_Hz / 1e9:.1f} GHz guard"
        ax.set_title(f"{self.standard} grid: {self.n_channels} channels over "
                     f"{span:.0f} GHz, {width:.1f} GHz each{guard}")
        return ax

    # -- constructors ---------------------------------------------------
    @classmethod
    def uniform(cls, n_channels: int, *, spacing_Hz: float,
                bandwidth_Hz: float, center_Hz: float = 193.1e12,
                standard: str = "custom", reference: str = "") -> "WDMGrid":
        r"""Evenly spaced comb centred on ``center_Hz``.

        The comb is symmetric about the centre, so an odd
        ``n_channels`` puts one channel exactly on it and an even count
        straddles it -- the convention every WDM transmitter uses.

        Parameters
        ----------
        n_channels : int
            Number of channels :math:`C`.
        spacing_Hz : float, keyword-only
            Channel spacing :math:`\Delta \nu` in Hz.
        bandwidth_Hz : float, keyword-only
            Occupied bandwidth :math:`B` of one channel, in Hz.
        center_Hz : float, optional, keyword-only
            Composite centre :math:`\nu_0`. Defaults to 193.1 THz, the
            ITU-T G.694.1 anchor.
        standard, reference : str, optional, keyword-only
            Provenance, forwarded to the constructor.

        Examples
        --------
        >>> grid = WDMGrid.uniform(4, spacing_Hz=50e9, bandwidth_Hz=40e9)
        >>> grid.offsets_Hz / 1e9
        array([-75., -25.,  25.,  75.])
        """
        if n_channels < 1:
            raise ValueError(f"expected at least one channel, got {n_channels}")
        index = np.arange(n_channels) - (n_channels - 1) / 2
        return cls(tuple(center_Hz + index * spacing_Hz),
                   bandwidth_Hz=bandwidth_Hz, center_Hz=center_Hz,
                   standard=standard, reference=reference)

    @classmethod
    def itu(cls, n_indices: Sequence[int], *, bandwidth_Hz: float,
            m: int = 4, center_Hz: Optional[float] = None) -> "WDMGrid":
        r"""Grid built from ITU-T G.694.1 flexible-grid indices.

        A channel is the integer couple :math:`(n, m)` of D19: centre
        :math:`193.1\,\text{THz} + n \times 6.25\,\text{GHz}`, slot
        width :math:`12.5 m` GHz. The fixed 50 GHz grid is ``m = 4``
        with ``n`` a multiple of 8.

        Parameters
        ----------
        n_indices : sequence of int
            Signed channel indices :math:`n`, on the 6.25 GHz
            granularity.
        bandwidth_Hz : float, keyword-only
            Occupied bandwidth :math:`B`. Not the slot width: a channel
            may under-fill its slot, and the guard band is the point of
            the difference.
        m : int, optional, keyword-only
            Slot width multiplier. Default 4, i.e. the 50 GHz slot.
        center_Hz : float, optional, keyword-only
            Composite centre; defaults to the midpoint of the comb.

        Raises
        ------
        ValueError
            If the occupied bandwidth exceeds the slot width, which
            would mean a channel spilling outside the spectrum it was
            allocated.

        Examples
        --------
        >>> grid = WDMGrid.itu([-8, 0, 8], bandwidth_Hz=32e9)
        >>> grid.spacing_Hz / 1e9, grid.guard_Hz / 1e9
        (50.0, 18.0)
        """
        entries = [itu_grid_frequency(int(n), m=m) for n in n_indices]
        slot_Hz = entries[0][1]
        if bandwidth_Hz > slot_Hz:
            raise ValueError(
                f"expected an occupied bandwidth of at most the "
                f"{slot_Hz / 1e9:.4g} GHz slot (m={m}), got "
                f"{bandwidth_Hz / 1e9:.4g} GHz -- raise m or narrow the "
                f"channel, a channel may not spill outside its slot")
        return cls(tuple(center for center, _ in entries),
                   bandwidth_Hz=bandwidth_Hz, center_Hz=center_Hz,
                   standard=f"ITU-T G.694.1 (n, m={m})",
                   reference="ITU-T G.694.1 (2020), section 7")

    # -- checks and display ---------------------------------------------
    def validate_fs(self, fs: float) -> None:
        """Raise if ``fs`` cannot carry the comb without aliasing."""
        if fs < self.min_fs:
            raise ValueError(
                f"expected fs >= {self.min_fs / 1e9:.6g} GHz to carry the "
                f"{self.n_channels}-channel grid (outermost channel at "
                f"{np.max(np.abs(self.offsets_Hz)) / 1e9:.6g} GHz plus half "
                f"a {self.bandwidth_Hz / 1e9:.6g} GHz channel), got "
                f"{fs / 1e9:.6g} GHz -- raise the sampling rate or drop the "
                f"outer channels")

    def __repr__(self) -> str:
        """ASCII spectral map, in the spirit of D21.

        A grid is a picture; printing the tuple of frequencies is not.
        """
        width = 61
        offsets = self.offsets_Hz
        half = max(float(np.max(np.abs(offsets))) + self.bandwidth_Hz / 2, 1.0)
        line = [" "] * width
        for offset in offsets:
            lo = int(round((offset - self.bandwidth_Hz / 2) / half
                           * (width - 1) / 2 + (width - 1) / 2))
            hi = int(round((offset + self.bandwidth_Hz / 2) / half
                           * (width - 1) / 2 + (width - 1) / 2))
            for i in range(max(lo, 0), min(hi, width - 1) + 1):
                line[i] = "#"
            line[max(lo, 0)] = "["
            line[min(hi, width - 1)] = "]"
        guard = self.guard_Hz
        spacing = self.spacing_Hz
        left, right = f"{-half / 1e9:+.4g} GHz", f"{half / 1e9:+.4g} GHz"
        axis = left + " " * max(width - len(left) - len(right), 1) + right
        return (
            f"WDMGrid({self.n_channels} channels, {self.standard})\n"
            f"  {''.join(line)}\n"
            f"  {axis}\n"
            f"  centre {float(self.center_Hz or 0.0) / 1e12:.4f} THz, "
            f"channel {self.bandwidth_Hz / 1e9:.4g} GHz, "
            f"spacing {'n/a' if spacing is None else f'{spacing / 1e9:.4g} GHz'}, "
            f"guard {'n/a' if guard is None else f'{guard / 1e9:.4g} GHz'}\n"
            f"  needs fs >= {self.min_fs / 1e9:.6g} GHz")


def _mixing_phasors(grid: WDMGrid, fs: float, n_samples: int) -> np.ndarray:
    r"""``(C, N)`` array of :math:`e^{j 2 \pi f_c n / f_s}`.

    Warns when an offset is not an integer number of DFT bins: the
    multiplexer then shifts a channel by a fraction of a bin, which is
    a circular shift with leakage rather than a clean relabelling, and
    the round trip stops being exact. The condition is on
    ``f_c * N / fs``, so it is the block length that fixes it.
    """
    offsets = grid.offsets_Hz
    bins = offsets * n_samples / fs
    off_bin = np.abs(bins - np.rint(bins))
    if np.any(off_bin > 1e-9):
        logger.warning(
            "WDM offsets are not integer DFT bins at fs=%.6g Hz over %d "
            "samples (worst channel off by %.3g bin); the shift leaks "
            "across the block edge. Use a block length that is a multiple "
            "of fs/gcd(offsets) to keep the round trip exact.",
            fs, n_samples, float(np.max(off_bin)))
    n = np.arange(n_samples)
    return np.exp(2j * np.pi * np.outer(offsets, n) / fs)


@dataclass(slots=True)
class WDMMultiplexer(Processor):
    r"""Combine per-channel baseband signals into one full-field waveform.

    Signal Model
    ------------
    Each channel is shifted to its slot and the slots are summed:

    .. math::

        x[n] = \sum_{c=0}^{C-1} x_c[n] \,
               e^{j 2 \pi f_c n / f_s}

    with :math:`f_c` the offset of channel :math:`c` from the composite
    band centre (see :class:`WDMGrid`). This is the synthesis bank of a
    software radio (GNU Radio's ``pfb_synthesizer_ccf``), written
    directly rather than through a filter bank because the channels
    already arrive at the composite rate.

    The output is the waveform the fibre propagates: nonlinear
    propagation is meaningful only on this single array, since XPM and
    FWM are interactions *between* the channels and a per-channel array
    can only ever produce SPM.

    Axes: *declared axis* -- expects ``(..., C, N)`` with channels on
    axis −2 and time on axis −1 (the MIMO antenna convention of D2),
    and returns ``(..., N)``.

    Parameters
    ----------
    grid : WDMGrid
        Frequency plan. Its channel order is the order of axis −2.
    fs : float, keyword-only
        Sampling rate :math:`f_s` in Hz, of the input *and* of the
        output -- this block does not resample.
    name : str, optional, keyword-only
        Block name.

    Attributes
    ----------
    phasors_ : np.ndarray
        ``(C, N)`` mixing phasors, built in ``prepare`` because they
        depend on the block length (hence the trailing underscore, D23).

    Raises
    ------
    ShapeError
        If the input is not at least 2D, or if axis −2 does not have
        one entry per channel of the grid.
    ValueError
        If ``fs`` is too low to carry the grid.

    References
    ----------
    G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley,
    2010, Section 6.1.

    Examples
    --------
    >>> grid = WDMGrid.uniform(3, spacing_Hz=50e9, bandwidth_Hz=32e9)
    >>> x = np.ones((3, 64), dtype=complex)
    >>> y = WDMMultiplexer(grid, fs=200e9)(x)
    >>> y.shape
    (64,)
    """

    grid: WDMGrid
    fs: float = field(kw_only=True)
    name: str = field(default="wdm mux", kw_only=True)
    phasors_: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def _warn_if_wider_than_its_slot(self, x: np.ndarray) -> None:
        r"""Say so when a channel does not fit the bandwidth declared for it.

        ``bandwidth_Hz`` is the grid's *occupied* bandwidth, so a pulse
        shaped at roll-off :math:`\rho` needs :math:`R_s(1+\rho)`.
        Writing the symbol rate is the natural mistake and an expensive
        one: the demultiplexer's brick wall then cuts the skirts of the
        pulse, the root-raised-cosine pair stops being Nyquist, and the
        implementation floor of an 11-channel 32 GBd comb at
        :math:`\rho = 0.01` falls from 54.1 dB to 33.5 dB -- for
        **0.15 %** of the energy. It is a coherent effect, so the harm
        is far larger than the fraction suggests, which is why the
        threshold here is low.

        The same measurement rises when the *shaping filter* is short,
        because its truncation sidelobes spill too: on that comb, 0.13 %
        for a clipped grid against 0.0004 % for a correct one at 4097
        taps, but 0.18 % against 0.038 % at 1025 taps. The check
        therefore reports "wider than its slot" without claiming which
        of the two causes it is -- both are real, and the remedy differs.

        The check belongs to the transmitter. At the receiver the same
        measurement is dominated by amplified spontaneous emission,
        which is white and fills the guard band whatever the grid says:
        0.12 % of rejected energy after 700 km against 0.0004 % of
        genuinely clipped signal, on the very comb this was written for.
        """
        spectrum = np.abs(np.fft.fft(x, axis=-1)) ** 2
        freq = np.fft.fftfreq(x.shape[-1], d=1 / self.fs)
        outside = np.abs(freq) > self.grid.bandwidth_Hz / 2
        if not np.any(outside):
            return
        total = np.sum(spectrum, axis=-1)
        spilled = np.sum(spectrum[..., outside], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, spilled / total, 0.0)
        worst = float(np.max(ratio))
        if worst > 1e-3:
            logger.warning(
                "a WDM channel is wider than the %.6g Hz the grid declares "
                "for it: %.3f %% of its energy sits outside, and the "
                "demultiplexer will cut it. Either bandwidth_Hz is too "
                "narrow -- it is the *occupied* bandwidth, so a pulse shaped "
                "at roll-off rho needs Rs*(1+rho), not Rs -- or the shaping "
                "filter is too short and its truncation sidelobes are what "
                "spills. The check cannot tell the two apart, and both are "
                "worth knowing: a truncated Nyquist pair costs far more than "
                "the fraction suggests.",
                self.grid.bandwidth_Hz, 100 * worst)

    def prepare(self, x: np.ndarray) -> None:
        self.grid.validate_fs(self.fs)
        if x.ndim < 2:
            raise ShapeError(
                f"expected a multi-channel signal (..., C, N) with C="
                f"{self.grid.n_channels}, got shape {x.shape} -- a single "
                f"channel still needs its own axis, use x[None, :]")
        if x.shape[-2] != self.grid.n_channels:
            raise ShapeError(
                f"expected {self.grid.n_channels} channels on axis -2 (the "
                f"grid has {self.grid.n_channels}), got {x.shape[-2]} -- "
                f"the grid and the signal must describe the same comb")
        self.phasors_ = _mixing_phasors(self.grid, self.fs, x.shape[-1])
        self._warn_if_wider_than_its_slot(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert self.phasors_ is not None
        return np.sum(x * self.phasors_, axis=-2)


@dataclass(slots=True)
class WDMDemultiplexer(Processor):
    r"""Split a full-field waveform back into per-channel baseband signals.

    Signal Model
    ------------
    Channel :math:`c` is shifted back to baseband and low-pass filtered
    to its own bandwidth:

    .. math::

        y_c[n] = \mathrm{LPF}_{B/2}\left\{
                 y[n] \, e^{-j 2 \pi f_c n / f_s} \right\}

    The filter is the ideal brick wall applied on the DFT of the block,
    the same mask as :class:`~comnumpy.core.filters.BWFilter`: bins with
    :math:`|f| > B/2` are zeroed. Being a DFT mask it is a *circular*
    operation, exact when each offset lands on an integer bin and
    leaky otherwise -- the block warns when it does not.

    **This is channel selection, not matched filtering.** :math:`B` is
    the grid's *occupied* bandwidth, so for a pulse shaped at roll-off
    :math:`\rho` it must be :math:`R_s(1 + \rho)` and not :math:`R_s`;
    setting it to the symbol rate clips the skirts of the pulse, which
    costs far more than it looks -- measured on an 11-channel 32 GBd
    comb at :math:`\rho = 0.01`, an implementation floor of 33.5 dB
    instead of 54.1 dB. That mistake is caught by
    :class:`WDMMultiplexer`, not here: at the receiver the amplified
    spontaneous emission is white and fills the guard band, so measuring
    the energy this mask rejects measures the **noise**, not the
    clipping. Same comb, same grid: 0.12 % of the rejected energy after
    700 km, against 0.0004 % of genuinely clipped signal.

    Conversely, once the mask is wide enough to pass the channel it
    changes nothing: the matched filter downstream does the selecting.
    Measured on the same comb, a mask at :math:`R_s(1+\rho)`, one at the
    full 37.5 GHz slot, and no mask at all give 54.13, 54.14 and
    54.13 dB. That is why there is no filter-shape option here -- a
    root-raised-cosine in this block would be the *receiver's* matched
    filter wearing the demultiplexer's name.

    This is the analysis bank of a software radio (GNU Radio's
    ``pfb_channelizer_ccf``), minus the decimation: the output stays at
    :math:`f_s`, so follow it with
    :class:`~comnumpy.core.processors.Downsampler` when the per-channel
    rate is what you want.

    Axes: *declared axis* -- expects ``(..., N)`` and returns
    ``(..., C, N)``, channels on axis −2; with ``channel=c`` it returns
    ``(..., N)`` for that channel alone.

    Parameters
    ----------
    grid : WDMGrid
        Frequency plan, the same object the multiplexer used.
    fs : float, keyword-only
        Sampling rate :math:`f_s` in Hz, unchanged by this block.
    channel : int, optional, keyword-only
        Extract this channel only, by index into the grid. ``None``
        (default) returns all of them. A receiver that tunes to one
        wavelength is the common case, and returning ``(..., N)`` there
        keeps the rest of the chain shape-agnostic.
    name : str, optional, keyword-only
        Block name.

    Attributes
    ----------
    phasors_ : np.ndarray
        ``(C, N)`` mixing phasors, built in ``prepare`` (D23).
    mask_ : np.ndarray
        ``(N,)`` boolean DFT mask of the channel filter.

    Raises
    ------
    ShapeError
        If the input has no time axis.
    ValueError
        If ``fs`` is too low to carry the grid, or if ``channel`` is not
        a valid index into it.

    References
    ----------
    G. P. Agrawal, *Fiber-Optic Communication Systems*, 4th ed., Wiley,
    2010, Section 6.1.

    Examples
    --------
    >>> grid = WDMGrid.uniform(3, spacing_Hz=50e9, bandwidth_Hz=32e9)
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(3, 256)) + 1j * rng.normal(size=(3, 256))
    >>> y = WDMMultiplexer(grid, fs=200e9)(x)
    >>> WDMDemultiplexer(grid, fs=200e9, channel=1)(y).shape
    (256,)
    """

    grid: WDMGrid
    fs: float = field(kw_only=True)
    channel: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="wdm demux", kw_only=True)
    phasors_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    mask_: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def prepare(self, x: np.ndarray) -> None:
        self.grid.validate_fs(self.fs)
        if self.channel is not None and not (
                0 <= self.channel < self.grid.n_channels):
            raise ValueError(
                f"expected a channel index in [0, {self.grid.n_channels}), "
                f"got {self.channel} -- the grid has "
                f"{self.grid.n_channels} channels")
        if x.ndim < 1:
            raise ShapeError(f"expected a signal (..., N), got shape {x.shape}")
        self.phasors_ = _mixing_phasors(self.grid, self.fs, x.shape[-1])
        freq = np.fft.fftfreq(x.shape[-1], d=1 / self.fs)
        self.mask_ = np.abs(freq) <= self.grid.bandwidth_Hz / 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert self.phasors_ is not None and self.mask_ is not None
        phasors = (self.phasors_ if self.channel is None
                   else self.phasors_[self.channel])
        if self.channel is None:
            mixed = x[..., None, :] * np.conj(phasors)
        else:
            mixed = x * np.conj(phasors)
        return apply_frequency_response(mixed, self.mask_)
