"""One representation of a result, several ways of showing it.

A sweep produces the same thing every time: an abscissa, and one series
of values per curve. Printing it as a table and plotting it are two
renderings of that one object, and they should not require restating it::

    data = {
        "x": lengths,
        "curves": {
            "Single carrier": runtime["single carrier"],
            "OFDM": runtime["OFDM"],
        },
    }

    print_data(data, xlabel="Frame length", ylabel="Runtime (ms)")
    plot_data(data, xlabel="Frame length", ylabel="Runtime (ms)")

``x`` is the abscissa, ``curves`` maps a name to a series, and every
series has the length of ``x``. That is the whole contract, and it is
checked in one place (:func:`unpack`) so the two renderers cannot
disagree about what a valid result looks like.

It is a plain dictionary of NumPy arrays on purpose. No class to
construct, nothing to import before the data exists, no dependency on
pandas -- and it is already the shape a sweep returns, since
:func:`~comnumpy.monte_carlo.monte_carlo` gives back one array per metric
aligned with the values it swept::

    curves = monte_carlo(chain, "awgn.snr_dB", snr_dB, {"ser": compute_ser},
                         N, reference="tx", seed=1)
    data = {"x": snr_dB, "curves": curves}
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import numpy as np

if TYPE_CHECKING:                                       # pragma: no cover
    from matplotlib.axes import Axes

__all__ = ["unpack", "format_data", "print_data", "plot_data"]

# A column is written in scientific notation as soon as reading it in
# fixed point would be a wall of zeros: five decades of dynamic range is
# what an error-rate sweep covers, and 12.4 next to 0.0000031 is
# unreadable long before that.
_DECADES_BEFORE_SCIENTIFIC = 4.0
# Significant digits kept on the *smallest* value of the column, not the
# largest: the small entries are the ones a fixed number of decimals
# destroys. A column of 8.5, 42.1, 282.3, 2374.0 calibrated on its
# largest gets zero decimals and prints the first entry as "8".
_SIGNIFICANT_DIGITS = 3
_MAX_DECIMALS = 6
_MISSING = "-"


def unpack(data: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Check the contract of a data dictionary and return its two parts.

    Parameters
    ----------
    data : mapping
        ``{"x": abscissa, "curves": {name: values, ...}}``.

    Returns
    -------
    tuple of (np.ndarray, dict of str to np.ndarray)
        The abscissa, and the curves as float arrays.

    Raises
    ------
    ComnumpyError
        If a key is missing or ``curves`` is not a mapping.
    ShapeError
        If a series does not have the length of ``x``.

    Examples
    --------
    >>> x, curves = unpack({"x": [1, 2], "curves": {"a": [3.0, 4.0]}})
    >>> x.tolist(), curves["a"].tolist()
    ([1.0, 2.0], [3.0, 4.0])
    """
    from comnumpy.exceptions import ComnumpyError, ShapeError  # local (D36)

    missing = [key for key in ("x", "curves") if key not in data]
    if missing:
        raise ComnumpyError(
            f"data is missing the key(s) {missing}; expected "
            f"{{'x': abscissa, 'curves': {{name: values}}}} -- pass the "
            f"swept values as 'x' and what was measured as 'curves'.")
    if not isinstance(data["curves"], Mapping):
        raise ComnumpyError(
            f"data['curves'] is a {type(data['curves']).__name__}; expected "
            f"a mapping of name to values, so that each series carries the "
            f"label a table column and a plot legend both need.")

    x = np.asarray(data["x"], dtype=float).ravel()
    curves: Dict[str, np.ndarray] = {}
    for name, values in data["curves"].items():
        series = np.asarray(values, dtype=float).ravel()
        if series.shape != x.shape:
            raise ShapeError(
                f"curve {name!r} has {series.size} values, expected "
                f"{x.size} -- one per point of 'x'. A sweep that lost a "
                f"point silently shifts every value that follows it.")
        curves[str(name)] = series
    return x, curves


def _column_format(values: np.ndarray) -> str:
    """The format string a whole column is written with.

    Per column, never per cell: a column read down must have its decimal
    points in one place, which is the only reason a table beats a list.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "{:.3e}"
    magnitude = np.abs(finite[finite != 0])
    if magnitude.size == 0:                         # every value is zero
        return "{:.0f}"
    largest, smallest = float(magnitude.max()), float(magnitude.min())

    integral = bool(np.all(finite == np.round(finite))) and largest < 1e6
    if integral:
        return "{:.0f}"

    decades = math.log10(largest) - math.log10(smallest)
    if decades > _DECADES_BEFORE_SCIENTIFIC or smallest < 1e-3:
        return "{:.3e}"

    decimals = int(min(max(_SIGNIFICANT_DIGITS - 1
                           - math.floor(math.log10(smallest)), 0),
                       _MAX_DECIMALS))
    # ... and no more than the values actually carry. A column of -6.0,
    # -4.5, 3.0 is exact at one decimal, and "-6.00" only adds a zero the
    # measurement never had.
    while decimals > 0 and np.allclose(np.round(finite, decimals - 1), finite,
                                       rtol=0, atol=0):
        decimals -= 1
    return "{:." + str(decimals) + "f}"


def _cells(values: np.ndarray) -> list[str]:
    template = _column_format(values)
    out = []
    for value in values:
        if math.isnan(value):
            out.append(_MISSING)
        elif math.isinf(value):
            out.append("inf" if value > 0 else "-inf")
        else:
            out.append(template.format(value))
    return out


def _transposed(x: np.ndarray, curves: Dict[str, np.ndarray], *,
                xlabel: str, ylabel: Optional[str]) -> str:
    """One row per curve, one column per point of `x`.

    The whole grid shares one format here rather than one per column: a
    row is what is read across, and a row whose cells change format is
    unreadable. `x` keeps its own, since it is the header.
    """
    stacked = [values for values in curves.values()]
    grid = np.concatenate(stacked) if stacked else np.array([])
    template = _column_format(grid)
    heads = _cells(x)
    body = {}
    for name, values in curves.items():
        row = []
        for value in values:
            row.append(_MISSING if math.isnan(value) else template.format(value))
        body[name] = row

    # list form, not star-args: a degenerate table (no curves, empty x)
    # must still render its header instead of dying inside max()
    name_width = max([len(xlabel)] + [len(name) for name in body])
    cell_width = max([0] + [len(head) for head in heads])
    for row in body.values():
        cell_width = max([cell_width] + [len(cell) for cell in row])

    def line(first: str, cells: list[str]) -> str:
        return (first.ljust(name_width) + "  "
                + "  ".join(cell.rjust(cell_width) for cell in cells)).rstrip()

    lines = []
    if ylabel:
        lines.append(ylabel)
    lines.append(line(xlabel, heads))
    lines.append("-" * (name_width + 2 + len(heads) * (cell_width + 2) - 2))
    for name, row in body.items():
        lines.append(line(name, row))
    return "\n".join(lines)


def format_data(data: Mapping[str, Any], *,
                xlabel: str = "x", ylabel: Optional[str] = None,
                transpose: bool = False) -> str:
    """Render a data dictionary as an aligned text table.

    One column per curve, named after it, plus the abscissa on the left.
    Column widths follow the widest of the header and its cells, every
    column is written in one format chosen from its own dynamic range,
    and everything is right-aligned so the decimal points line up.

    A ``nan`` prints as ``-``: a sweep point with no measurement is a
    hole, and printing ``nan`` in a column of numbers reads as a value.

    ``transpose=True`` turns the table a quarter: one row per curve,
    named on the left, one column per point of ``x``. That is the right
    way round when the curves are many and their names are long -- six
    receivers with names like ``"DBP, 50 steps/span"`` make a table 160
    characters wide as columns, and a readable one as rows.

    Parameters
    ----------
    data : mapping
        ``{"x": abscissa, "curves": {name: values}}``, see :func:`unpack`.
    xlabel : str, optional, keyword-only
        Header of the abscissa column. Default ``"x"``.
    ylabel : str, optional, keyword-only
        What the curves measure, printed as a caption above the table --
        the unit belongs there rather than repeated in every cell.
    transpose : bool, optional, keyword-only
        Curves as rows and ``x`` as columns. Default False.

    Returns
    -------
    str
        The table, without a trailing newline.

    Examples
    --------
    >>> data = {"x": [1000, 2000, 4000, 8000],
    ...         "curves": {"Single carrier": [12.43, 24.81, 49.07, 98.71],
    ...                    "OFDM": [3.21, 5.87, 11.42, 22.31]}}
    >>> print(format_data(data, xlabel="Frame length", ylabel="Runtime (ms)"))
    Runtime (ms)
    Frame length  Single carrier   OFDM
    -----------------------------------
            1000            12.4   3.21
            2000            24.8   5.87
            4000            49.1  11.42
            8000            98.7  22.31

    Each column keeps three significant digits on its *smallest* entry,
    which is why the two columns above do not carry the same number of
    decimals: 12.4 loses nothing a reader wants, and 3.21 would.

    An error rate spans decades, so its column goes to scientific
    notation on its own:

    >>> print(format_data({"x": [0, 5], "curves": {"SER": [7.4e-1, 1.2e-6]}},
    ...                   xlabel="SNR [dB]"))
    SNR [dB]        SER
    -------------------
           0  7.400e-01
           5  1.200e-06

    Turned a quarter, when the names are the long part:

    >>> print(format_data({"x": [-6.0, -4.5, -3.0],
    ...                    "curves": {"dispersion compensation":
    ...                               [15.8, 17.1, 18.2],
    ...                               "DBP, 50 steps/span":
    ...                               [15.9, 17.4, 18.8]}},
    ...                   xlabel="launch power [dBm]", transpose=True))
    launch power [dBm]       -6.0  -4.5  -3.0
    -----------------------------------------
    dispersion compensation  15.8  17.1  18.2
    DBP, 50 steps/span       15.9  17.4  18.8
    """
    x, curves = unpack(data)
    if transpose:
        return _transposed(x, curves, xlabel=xlabel, ylabel=ylabel)

    headers = [xlabel] + list(curves)
    columns = [_cells(x)] + [_cells(values) for values in curves.values()]
    widths = [max([len(header)] + [len(cell) for cell in column])
              for header, column in zip(headers, columns, strict=True)]

    def row(fields: list[str]) -> str:
        return "  ".join(field.rjust(width)
                         for field, width in zip(fields, widths,
                                                 strict=True)).rstrip()

    lines = []
    if ylabel:
        lines.append(ylabel)
    lines.append(row(headers))
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for index in range(x.size):
        lines.append(row([column[index] for column in columns]))
    return "\n".join(lines)


def print_data(data: Mapping[str, Any], *,
               xlabel: str = "x", ylabel: Optional[str] = None,
               transpose: bool = False) -> None:
    """Print what :func:`format_data` renders.

    Parameters
    ----------
    data : mapping
        ``{"x": abscissa, "curves": {name: values}}``, see :func:`unpack`.
    xlabel, ylabel : str, optional, keyword-only
        As in :func:`format_data`.
    transpose : bool, optional, keyword-only
        Curves as rows. As in :func:`format_data`.

    Examples
    --------
    >>> print_data({"x": [0, 1], "curves": {"gain": [1.5, 2.25]}})
    x  gain
    -------
    0  1.50
    1  2.25
    """
    print(format_data(data, xlabel=xlabel, ylabel=ylabel,
                      transpose=transpose))


def plot_data(data: Mapping[str, Any], *,
              xlabel: Optional[str] = None, ylabel: Optional[str] = None,
              kind: Optional[str] = None,
              xscale: Optional[str] = None, yscale: Optional[str] = None,
              ax: Optional["Axes"] = None, **kwargs: Any) -> "Axes":
    """Draw the same data dictionary as a figure.

    One line per curve, labelled with its name -- the table's columns and
    the figure's legend come from the same dictionary, so they cannot
    describe different runs.

    Parameters
    ----------
    data : mapping
        ``{"x": abscissa, "curves": {name: values}}``, see :func:`unpack`.
    xlabel, ylabel : str, optional, keyword-only
        Axis labels.
    kind : str, optional, keyword-only
        Passed to :func:`comnumpy.style.apply` for the decoration; see
        its :data:`~comnumpy.style.KINDS`. Without it the axis still gets
        a grid and a legend, but no default labels.
    xscale, yscale : str, optional, keyword-only
        ``"linear"`` or ``"log"``. Left alone when not given.
    ax : matplotlib.axes.Axes, optional, keyword-only
        Axis to draw on; created when None (decision D25).
    **kwargs
        Forwarded to ``ax.plot`` -- a marker, a line width, whatever the
        page wants for every curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axis, so the caller can keep working on it.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> data = {"x": [0, 1, 2], "curves": {"ZF": [0.1, 0.05, 0.01],
    ...                                    "ML": [0.08, 0.03, 0.004]}}
    >>> ax = plot_data(data, xlabel="SNR [dB]", ylabel="SER", yscale="log")
    >>> len(ax.lines), ax.get_yscale()
    (2, 'log')
    """
    import matplotlib.pyplot as plt          # local import (D36)
    from comnumpy import style               # local import (D36)

    x, curves = unpack(data)
    if ax is None:
        _, ax = plt.subplots()
    for name, values in curves.items():
        ax.plot(x, values, label=name, **kwargs)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        # a rate of zero is a missing measurement, not a point at the
        # bottom of the figure; see plot_error_rate
        ax.set_yscale(yscale, **({"nonpositive": "mask"} if yscale == "log"
                                 else {}))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if kind is not None:
        return style.apply(ax, kind)
    ax.grid(True, which="both" if "log" in (ax.get_xscale(), ax.get_yscale())
            else "major")
    ax.legend()
    return ax
