"""Figure style (decision D27b).

Two things live here, and they act at two different moments.

:data:`PATH` and :func:`context` are the **style sheet** -- colours,
fonts, line widths, figure size. Those are matplotlib rcParams, so they
have to be active *before* a figure is created; they cannot be applied
to a line that is already drawn. The sheet ships with the package and is
**never applied at import time** -- importing a library must not mutate
the user's matplotlib state::

    import matplotlib.pyplot as plt
    import comnumpy.style

    plt.style.use(comnumpy.style.PATH)      # global, explicit
    with comnumpy.style.context():          # or scoped
        ...

:func:`apply` is the **per-axis convention**: what an error-rate figure
of this library looks like, what an IQ plane looks like. It runs after
the drawing, on an axis the caller filled themselves::

    fig, ax = plt.subplots()
    line, = ax.semilogy(snr_dB, ser, "o", fillstyle="none", label="16-QAM")
    ax.semilogy(snr_dB, ser_theory, "-", color=line.get_color())
    style.apply(ax, "error_rate")

That is the whole point of the split: the reader sees the two matplotlib
calls that draw the curves, and the library is left with the part that
is a house convention rather than a plotting decision.

Colors come from the Okabe-Ito colorblind-safe palette, and no
information is carried by color alone (decision D27c): semantic tables
such as :data:`comnumpy.ofdm.allocation.CARRIER_STYLE` pair each color
with a glyph and a hatch.
"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:                                   # pragma: no cover
    from matplotlib.axes import Axes

__all__ = ["PATH", "context", "use", "apply", "KINDS"]

PATH: pathlib.Path = pathlib.Path(__file__).parent / "comnumpy.mplstyle"


# The default labels of each kind of figure. A kind is a *sentence about
# the quantities*, not a plotting mode: "error_rate" says the ordinate is
# a probability read on a logarithmic axis, which is why the grid runs on
# both decades and why the abscissa is an SNR unless told otherwise.
KINDS: Dict[str, Dict[str, str]] = {
    "error_rate": {"xlabel": "SNR [dB]", "ylabel": "error rate"},
    "iq": {"xlabel": "real part", "ylabel": "imag part"},
    "time": {"xlabel": "time [s]", "ylabel": "amplitude"},
    "spectrum": {"xlabel": "freq [Hz]", "ylabel": "PSD [dB]"},
}


def use() -> None:
    """Activate the style sheet for every figure created from now on.

    One line at the top of a script, after the imports and before the
    first figure -- the colours and the figure size are rcParams, so a
    figure already created keeps the ones that were active when it was::

        import comnumpy.style as style
        style.use()

    Explicit on purpose: importing this package must not change the
    user's matplotlib state behind their back (decision D27b). Use
    :func:`context` instead to scope the change to a block.
    """
    import matplotlib.pyplot as plt  # local import (D36)
    plt.style.use(str(PATH))


def context():
    """Context manager applying the comnumpy style locally.

    Returns
    -------
    contextlib.AbstractContextManager
        The matplotlib style context for ``with comnumpy.style.context():``.
    """
    import matplotlib.pyplot as plt  # local import (D36)
    return plt.style.context(str(PATH))


def apply(ax: "Axes", kind: str, *, legend: Optional[bool] = None) -> "Axes":
    """Give an axis the house look for one kind of figure.

    What it does
    ------------
    Three things, and deliberately no more:

    * **fills the labels that are still empty**. A label the caller has
      already set is never overwritten -- ``ax.set_ylabel("BER")``
      followed by ``apply(ax, "error_rate")`` keeps ``"BER"``. The
      defaults are in :data:`KINDS`;
    * **turns on the grid**, on both decades when the corresponding axis
      is logarithmic and on the major ticks otherwise;
    * **adds a legend** when the axis carries labelled artists and does
      not have one yet.

    ``kind="iq"`` additionally sets an equal aspect ratio, because a
    constellation read on unequal axes is a different constellation.

    What it does not do
    -------------------
    It never touches the data and never changes a scale. Whether a curve
    is drawn with ``plot`` or ``semilogy``, with markers or without, in
    which colour, is the caller's statement about their measurement, and
    it stays visible in the caller's code. This function decorates what
    is already there.

    The colours, fonts, line widths and figure size are *not* here
    either: those are rcParams, they have to be active before the figure
    exists, and they come from the style sheet -- see :data:`PATH` and
    :func:`context`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to decorate, already drawn on.
    kind : str
        One of the keys of :data:`KINDS`: ``"error_rate"``, ``"iq"``,
        ``"time"`` or ``"spectrum"``.
    legend : bool, optional, keyword-only
        Force the legend on or off. The default, ``None``, adds one when
        the axis has labelled artists and none already.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis, so the call can end a chain of statements.

    Raises
    ------
    ValueError
        If ``kind`` is not one of :data:`KINDS`.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from comnumpy import style
    >>> snr = np.arange(0, 12, 2)
    >>> _, ax = plt.subplots()
    >>> _ = ax.semilogy(snr, 10.0 ** (-snr / 8), "o", label="QPSK")
    >>> _ = style.apply(ax, "error_rate")
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('SNR [dB]', 'error rate')

    A label the caller set is left alone:

    >>> _, ax = plt.subplots()
    >>> _ = ax.semilogy(snr, 10.0 ** (-snr / 8))
    >>> ax.set_ylabel("BER")
    Text(...)
    >>> _ = style.apply(ax, "error_rate")
    >>> ax.get_ylabel()
    'BER'
    """
    from comnumpy.exceptions import ComnumpyError  # local import (D36)

    if kind not in KINDS:
        raise ComnumpyError(
            f"style.apply: unknown kind {kind!r}; expected one of "
            f"{sorted(KINDS)} -- the kind names the quantity being drawn, "
            f"not the plotting command.")

    defaults = KINDS[kind]
    if not ax.get_xlabel():
        ax.set_xlabel(defaults["xlabel"])
    if not ax.get_ylabel():
        ax.set_ylabel(defaults["ylabel"])

    # "both" only means something on a logarithmic axis, where the minor
    # ticks are the intermediate decades a reader interpolates on.
    logarithmic = "log" in (ax.get_xscale(), ax.get_yscale())
    ax.grid(True, which="both" if logarithmic else "major")

    if kind == "iq":
        ax.axis("equal")

    if legend is None:
        labelled, _ = ax.get_legend_handles_labels()
        legend = bool(labelled) and ax.get_legend() is None
    if legend:
        ax.legend()
    return ax
