"""Describe a numerical experiment once, run it, keep what it produced.

A study in this library is almost always the same sentence: *run the
same experiment for several values of one parameter, and keep what each
run measured*. Written by hand, that sentence turns into pre-allocated
arrays, nested loops and an accumulator per metric -- and, too often,
into a run nobody can reproduce because no seed was kept anywhere.

:class:`Experiment` is that sentence as an object::

    experiment = Experiment(config, parameter="snr_dB",
                            values=np.arange(0, 21, 2), seed=42)
    result = experiment.run(simulate)
    result.print()
    result.plot()

``simulate`` is the experiment itself, written by you, called once per
point as ``simulate(config, seed)``: ``config`` is a copy of the
experiment's configuration with the studied parameter set to the point's
value, and ``seed`` is that point's own child seed. It returns a plain
dictionary of numbers -- a BER, an SNR, a runtime, whatever the study
measures -- and the experiment collects each entry into an array aligned
with ``values``.

The result keeps everything the run needs to be believed later: the
parameter and its values, the configuration, the seed actually used
(auto-generated and kept when none was given), the collected data and
the wall time. It renders itself through :mod:`comnumpy.data`, so the
table a page pastes and the figure it shows come from the same object.

This is deliberately not a framework. One parameter, one callable, one
dictionary out: for sweeping a single chain parameter with standard
metrics, :func:`~comnumpy.monte_carlo.monte_carlo` is one call and
remains the right tool. ``Experiment`` is for the study *around* a
chain -- several receivers on one propagation, a detector comparison,
anything where the collection loop was starting to be the longest part
of the script.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from comnumpy.exceptions import ComnumpyError, ShapeError

__all__ = ["Experiment", "ExperimentResult"]

# What a `save=` selection may look like: a single name, a sequence of
# names, or a mapping of name to a keep/drop flag.
SaveSpec = Union[str, Sequence[str], Mapping[str, bool]]


def _is_mapping(candidate: object) -> bool:
    """Runtime guard for values whose annotation already promises it.

    The static types say ``config`` and simulate's return are mappings;
    these checks exist for the caller who did not read them, so they
    must go through a boundary the type-checker cannot argue with.
    """
    return isinstance(candidate, Mapping)


@dataclass(slots=True)
class ExperimentResult:
    """What one :meth:`Experiment.run` produced, and how to look at it.

    Attributes
    ----------
    parameter : str
        The name of the studied parameter.
    values : np.ndarray
        The values it took, one sweep point each.
    seed : int
        The master seed actually used. With it, the same configuration
        reproduces the same result.
    config : dict
        The experimental conditions, as passed to the experiment.
    data : dict of str to np.ndarray
        One array per saved quantity, aligned with ``values``.
    elapsed_ : float
        Wall time of the whole run, in seconds.
    """

    parameter: str
    values: np.ndarray
    seed: int
    config: Dict[str, Any]
    data: Dict[str, np.ndarray]
    elapsed_: float

    def as_data(self) -> Dict[str, Any]:
        """The result in the shape :mod:`comnumpy.data` renders.

        Returns
        -------
        dict
            ``{"x": values, "curves": data}`` -- ready for
            :func:`~comnumpy.data.print_data` and
            :func:`~comnumpy.data.plot_data`.
        """
        return {"x": self.values, "curves": self.data}

    def print(self, *, ylabel: Optional[str] = None,
              transpose: bool = False) -> None:
        """Print the experiment: conditions first, then the table.

        Parameters
        ----------
        ylabel : str, optional, keyword-only
            Caption above the table -- the unit of the collected data.
        transpose : bool, optional, keyword-only
            Data as rows rather than columns, for long names; see
            :func:`~comnumpy.data.format_data`.
        """
        from comnumpy.data import format_data  # local import (D36)

        conditions = []
        for name, value in self.config.items():
            conditions.append(f"{name}={value}")
        print(f"parameter : {self.parameter}, {self.values.size} points")
        print(f"seed      : {self.seed}")
        print(f"config    : {', '.join(conditions) if conditions else '-'}")
        print(f"elapsed   : {self.elapsed_:.1f} s")
        print()
        print(format_data(self.as_data(), xlabel=self.parameter,
                          ylabel=ylabel, transpose=transpose))

    def plot(self, **kwargs: Any) -> Any:
        """Draw the collected data, one curve per saved quantity.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`~comnumpy.data.plot_data`. The abscissa
            label defaults to the parameter's name.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from comnumpy.data import plot_data  # local import (D36)

        kwargs.setdefault("xlabel", self.parameter)
        return plot_data(self.as_data(), **kwargs)


@dataclass(slots=True)
class Experiment:
    r"""The same experiment, run for several values of one parameter.

    Signal Model
    ------------
    An experiment is four statements: the conditions (``config``), the
    studied parameter and the values it takes, and a seed that makes the
    whole run reproducible. :meth:`run` executes the user's ``simulate``
    once per value and collects what it returns.

    Seeding follows decisions D6/D35: the master seed spawns one child
    per point through :class:`numpy.random.SeedSequence`, so the points
    are statistically independent and the whole curve is reproduced by
    the master seed alone. When no seed is given, one is drawn and
    **kept** -- ``experiment.seed`` and the result both carry it, so an
    interesting accident can always be re-run.

    Parameters
    ----------
    config : mapping
        The experimental conditions. Copied at each point, never
        mutated.
    parameter : str, keyword-only
        Name of the studied parameter. It is set in the copy of
        ``config`` that each ``simulate`` call receives, overriding any
        default the configuration carries.
    values : array-like, keyword-only
        The values the parameter takes, one sweep point each.
    seed : int, optional, keyword-only
        Master seed. Default None: a seed is drawn automatically and
        kept.
    save : str, sequence of str, or mapping of str to bool, optional,
           keyword-only
        Which of the quantities returned by ``simulate`` are kept.
        Default None: everything is kept. A mapping keeps the names
        whose value is true, so ``save={"ser": True}`` reads as the
        selection it is.

    Raises
    ------
    ShapeError
        If ``values`` is empty or not one-dimensional.
    ComnumpyError
        If ``config`` is not a mapping.

    Examples
    --------
    >>> def simulate(config, seed):
    ...     rng = np.random.default_rng(seed)
    ...     noise = rng.normal(size=config["n"])
    ...     return {"power": float(np.mean(noise ** 2)),
    ...             "snr_out": config["snr_dB"] - 1.0}
    >>> experiment = Experiment({"n": 4}, parameter="snr_dB",
    ...                         values=[0, 10, 20], seed=42)
    >>> result = experiment.run(simulate)
    >>> result.data["snr_out"]
    array([-1.,  9., 19.])
    >>> result.seed
    42

    The same seed reproduces the same numbers:

    >>> again = Experiment({"n": 4}, parameter="snr_dB",
    ...                    values=[0, 10, 20], seed=42).run(simulate)
    >>> bool(np.all(again.data["power"] == result.data["power"]))
    True
    """

    config: Mapping[str, Any]
    parameter: str = field(kw_only=True)
    values: Any = field(kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    save: Optional[SaveSpec] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not _is_mapping(self.config):
            raise ComnumpyError(
                f"config is a {type(self.config).__name__}; expected a "
                f"mapping of condition name to value -- the experimental "
                f"conditions the result will be read against.")
        self.values = np.asarray(self.values)
        if self.values.ndim != 1 or self.values.size == 0:
            raise ShapeError(
                f"values has shape {self.values.shape}; expected a "
                f"non-empty 1-D array -- one entry per sweep point.")
        if self.seed is None:
            # drawn once and *kept*: reproducibility survives forgetting
            # to choose a seed, which is when it is needed most
            self.seed = int(np.random.SeedSequence().generate_state(1)[0])

    def _kept(self, produced: Sequence[str]) -> list[str]:
        """The names to save, checked against what simulate produced."""
        if self.save is None:
            return list(produced)
        if isinstance(self.save, Mapping):
            wanted = [name for name, keep in self.save.items() if keep]
        elif isinstance(self.save, str):
            wanted = [self.save]
        else:
            wanted = list(self.save)
        unknown = [name for name in wanted if name not in produced]
        if unknown:
            raise ComnumpyError(
                f"save names {unknown}, which simulate does not produce; "
                f"it returned {sorted(produced)}.")
        return wanted

    def run(self, simulate: Callable[[Dict[str, Any], int],
                                     Mapping[str, Any]]) -> ExperimentResult:
        """Execute the experiment and collect its observations.

        Parameters
        ----------
        simulate : callable
            The experiment itself, called once per point as
            ``simulate(config, seed)`` -- ``config`` a copy of the
            conditions with the studied parameter set to the point's
            value, ``seed`` the point's child seed. Must return a
            mapping of quantity name to value, the same names at every
            point.

        Returns
        -------
        ExperimentResult
            The observations, one array per saved quantity, along with
            everything needed to reproduce them.

        Raises
        ------
        ComnumpyError
            If ``simulate`` returns something that is not a mapping, or
            not the same names at every point, or if ``save`` asks for a
            name it never produced.
        """
        assert self.seed is not None  # set by __post_init__
        start = time.perf_counter()
        values = np.asarray(self.values)
        children = np.random.SeedSequence(self.seed).spawn(values.size)

        rows: list[Mapping[str, Any]] = []
        for value, child in zip(values, children, strict=True):
            config = dict(self.config)
            config[self.parameter] = value.item() if value.ndim == 0 else value
            observed = simulate(config, int(child.generate_state(1)[0]))
            if not _is_mapping(observed):
                raise ComnumpyError(
                    f"simulate returned a {type(observed).__name__} at "
                    f"{self.parameter}={value}; expected a mapping of "
                    f"quantity name to value, e.g. "
                    f"{{'ser': ..., 'elapsed_time': ...}}.")
            if rows and set(observed) != set(rows[0]):
                raise ComnumpyError(
                    f"simulate returned {sorted(observed)} at "
                    f"{self.parameter}={value} but {sorted(rows[0])} at "
                    f"the first point; every point must observe the same "
                    f"quantities, or the arrays cannot stay aligned.")
            rows.append(observed)

        data: Dict[str, np.ndarray] = {}
        for name in self._kept(list(rows[0])):
            column = []
            for row in rows:
                column.append(row[name])
            data[name] = np.asarray(column)

        return ExperimentResult(
            parameter=self.parameter, values=values, seed=self.seed,
            config=dict(self.config), data=data,
            elapsed_=time.perf_counter() - start)
