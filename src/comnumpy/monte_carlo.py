"""Parameter sweeps (decision D35).

A Monte-Carlo sweep is a stateless ``for`` loop: reconfigure
(:meth:`~comnumpy.core.generics.Sequential.set_params`, decision D34),
reseed (:meth:`~comnumpy.core.generics.Sequential.seed`, decision D6),
run, collect. The loop below stays visible on purpose -- a student must
be able to read *where* the SNR is swept -- and the caller writes one
line. The Trainer/hook pattern is explicitly rejected by the
architecture document.

Extracted from the identical skeleton of three ``validation/`` scripts
(``core_awgn_ser.py``, ``fec_coding_gain.py``, ``ofdm_awgn_ser.py``),
per the decision's trigger.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any, Optional

import numpy as np

from comnumpy.core.generics import Sequential

__all__ = ["monte_carlo"]

# Every worker runs one sweep point at a time, so a BLAS that spawns one
# thread per core inside each of them oversubscribes the machine by
# n_jobs times and ends up slower than the serial loop. These are read
# when numpy is imported, which is why the pool is started with "spawn":
# a forked child inherits an already-initialized thread pool and ignores
# them.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS")

# Per-worker state: the chain is pickled once when the worker starts
# rather than once per sweep point.
_WORKER: dict[str, Any] = {}


def _start_worker(chain: Sequential, params: list[str],
                  metrics: Mapping[str, Callable[..., Any]], stimulus: Any,
                  reference: Optional[str]) -> None:
    _WORKER.update(chain=chain, params=params, metrics=metrics,
                   stimulus=stimulus, reference=reference)


def _run_point(task: tuple[int, Any, Optional[int]]) -> tuple[int, dict[str, Any]]:
    """One sweep point, in a worker process. Returns its index with it."""
    index, point, child_seed = task
    chain: Sequential = _WORKER["chain"]
    return index, _evaluate(chain, _WORKER["params"], point, child_seed,
                            _WORKER["metrics"], _WORKER["stimulus"],
                            _WORKER["reference"])


def _evaluate(chain: Sequential, params: list[str], point: Any,
              child_seed: Optional[int],
              metrics: Mapping[str, Callable[..., Any]], stimulus: Any,
              reference: Optional[str]) -> dict[str, Any]:
    """Reconfigure, reseed, run, measure -- the four steps of a point."""
    point_values = (point,) if len(params) == 1 else tuple(point)
    chain.set_params(**dict(zip(params, point_values, strict=True)))
    if child_seed is not None:
        chain.seed(child_seed)

    y = chain(stimulus)

    target = chain.tap(reference) if reference is not None else None
    return {name: (metric(target, y) if target is not None else metric(y))
            for name, metric in metrics.items()}


def monte_carlo(chain: Sequential,
                param: str | Sequence[str],
                values: Sequence[Any],
                metrics: Mapping[str, Callable[..., Any]],
                stimulus: Any,
                *,
                reference: Optional[str] = None,
                seed: Optional[int] = None,
                n_jobs: int = 1) -> dict[str, np.ndarray]:
    """Run a chain over a range of parameter values and collect metrics.

    Parameters
    ----------
    chain : Sequential
        The chain to sweep. It is reconfigured in place.
    param : str or sequence of str
        Dotted parameter path(s), e.g. ``"noise.snr_dB"`` (see
        :meth:`Sequential.block_ids`). With several paths, each entry of
        ``values`` provides one value per path (zip semantics; the
        cartesian product is out of scope, per the decision).
    values : sequence
        Parameter values, one sweep point each.
    metrics : mapping of str to callable
        Metrics collected at every point. Called as ``metric(target, y)``
        when ``reference`` names a tapped block, else ``metric(y)``.
        Smaller-is-better quantities (BER, SER, EVM); there is no
        ``score()`` (decision D24).
    stimulus
        Input passed to the chain at every point (e.g. a symbol count).
    reference : str, optional, keyword-only
        Block id whose tapped output is the metric target. The tap is
        declared on the chain if it is not already
        (:attr:`Sequential.taps`).
    seed : int, optional, keyword-only
        Master seed; every sweep point gets an independent child seed
        through the chain (decisions D6/D35), so the whole curve is
        reproducible.
    n_jobs : int, optional, keyword-only
        Sweep points to run at once. Default 1, the plain loop below.
        ``-1`` uses every core. See *Running points in parallel*.

    Returns
    -------
    dict of str to np.ndarray
        One array per metric, aligned with ``values``.

    Raises
    ------
    ValueError
        If ``n_jobs`` is not 1 and no ``seed`` was given, or if the chain
        or a metric cannot be sent to a worker process.

    Notes
    -----
    **Running points in parallel.** The points of a sweep are
    independent by construction -- each one reconfigures and reseeds the
    chain from scratch -- so they can be run at the same time, and
    ``n_jobs`` says how many at once. Three consequences are worth
    knowing before turning it on.

    *The result does not change.* Each point already draws from its own
    child seed, so a curve computed with ``n_jobs=4`` is **identical**
    to the serial one, value for value, not merely statistically
    equivalent. This is asserted in the test suite. It is also why
    ``seed=`` becomes mandatory: without it, what a point draws would
    depend on which worker picked it up and in which order.

    *A worker is a process, so everything it needs is pickled* -- the
    chain, the metrics and the stimulus. A metric written as a ``lambda``
    or as a closure cannot cross that boundary; define it at module level
    instead. The error says so rather than letting ``pickle`` explain it.

    *A stateful chain should stay serial.* Each worker holds its own copy
    of the chain, so a block that carries something between points (an
    adaptive equalizer, an estimator fitted at the previous point) sees a
    different history than it would in the loop. Sweeps of such chains
    are order-dependent to begin with; this only makes it visible.

    *It is not free.* Starting a worker and pickling the chain into it
    costs about 130 ms, once per worker, so the sweep has to be long
    enough to pay for it. Measured on four cores, a 24-point sweep of a
    sphere-decoded 4x4 link:

    ==================  ==========  ==========  ==========
    cost of one point   n_jobs=1    n_jobs=2    n_jobs=4
    ==================  ==========  ==========  ==========
    19 ms               0.46 s      0.51 s      0.39 s
    222 ms              5.34 s      2.99 s      1.99 s
    ==================  ==========  ==========  ==========

    Below ~20 ms a point the pool costs more than it saves; at ~200 ms
    it returns 2.7x on four cores. The values are identical in every
    column.

    Examples
    --------
    >>> from comnumpy import AWGN, Sequential, SymbolGenerator
    >>> from comnumpy.core.metrics import compute_ser
    >>> chain = Sequential([SymbolGenerator(4), AWGN(snr_dB=0, name="noise")])
    >>> out = monte_carlo(chain, "noise.snr_dB", [0, 10],
    ...                   {"power": lambda y: float(np.mean(np.abs(y)**2))},
    ...                   stimulus=1000, seed=1)
    >>> print(out["power"].shape)
    (2,)
    """
    params = [param] if isinstance(param, str) else list(param)
    if reference is not None and reference not in (chain.taps or []):
        chain.taps = (chain.taps or []) + [reference]
    seeds = (np.random.SeedSequence(seed).spawn(len(values))
             if seed is not None else [None] * len(values))

    children = [None if child is None else int(child.generate_state(1)[0])
                for child in seeds]

    if n_jobs != 1:
        collected = _sweep_parallel(chain, params, values, children, metrics,
                                    stimulus, reference, n_jobs)
    else:
        collected = [_evaluate(chain, params, point, child, metrics, stimulus,
                               reference)
                     for point, child in zip(values, children, strict=True)]

    return {name: np.asarray([point[name] for point in collected])
            for name in metrics}


def _sweep_parallel(chain: Sequential, params: list[str],
                    values: Sequence[Any], children: list[Optional[int]],
                    metrics: Mapping[str, Callable[..., Any]], stimulus: Any,
                    reference: Optional[str],
                    n_jobs: int) -> list[dict[str, Any]]:
    """The same points, on several processes. Same values, out of order."""
    if any(child is None for child in children):
        raise ValueError(
            f"monte_carlo(n_jobs={n_jobs}) needs seed= as well: the points are "
            f"then drawn from independent child seeds and the curve is the "
            f"same as the serial one. Without it, what a point draws would "
            f"depend on which worker ran it.")
    for label, payload in (("chain", chain), ("metrics", dict(metrics))):
        try:
            pickle.dumps(payload)
        except Exception as error:            # noqa: BLE001 -- re-raised
            raise ValueError(
                f"monte_carlo(n_jobs={n_jobs}) runs the points in worker "
                f"processes, and the {label} cannot be sent to one: "
                f"{error}. A lambda or a closure cannot be pickled -- "
                f"define the metric at module level, or keep n_jobs=1."
            ) from error

    workers = os.cpu_count() or 1 if n_jobs < 0 else n_jobs
    workers = max(1, min(workers, len(values)))
    tasks = [(index, point, child)
             for index, (point, child) in enumerate(zip(values, children,
                                                        strict=True))]
    results: list[Any] = [None] * len(tasks)
    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        try:
            with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_start_worker,
                    initargs=(chain, params, metrics, stimulus, reference),
            ) as pool:
                for index, values_at_point in pool.map(_run_point, tasks):
                    results[index] = values_at_point
        except BrokenExecutor as error:
            # By far the most common cause, and the one whose native
            # message ("A process in the process pool was terminated
            # abruptly") explains nothing: a worker re-imports the
            # module it was started from, so a script whose body is not
            # under a __main__ guard runs itself again inside every
            # worker.
            raise RuntimeError(
                "monte_carlo(n_jobs>1) starts worker processes, and each one "
                "re-imports the module it was started from. If you are "
                "calling this from a script, its body must sit under\n\n"
                "    if __name__ == \"__main__\":\n\n"
                "or the workers run the script again instead of the "
                "sweep (a notebook or an interactive session needs "
                f"nothing). The pool reported: {error}") from error
    finally:
        for name, previous in saved.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    return results
