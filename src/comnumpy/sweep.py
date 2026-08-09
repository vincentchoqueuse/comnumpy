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

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from comnumpy.core.generics import Sequential

__all__ = ["sweep"]


def sweep(chain: Sequential,
          param: str | Sequence[str],
          values: Sequence[Any],
          metrics: Mapping[str, Callable[..., Any]],
          stimulus: Any,
          *,
          reference: Optional[str] = None,
          seed: Optional[int] = None) -> dict[str, np.ndarray]:
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
        when ``reference`` names a Recorder block, else ``metric(y)``.
        Smaller-is-better quantities (BER, SER, EVM); there is no
        ``score()`` (decision D24).
    stimulus
        Input passed to the chain at every point (e.g. a symbol count).
    reference : str, optional, keyword-only
        Block id of a Recorder inside the chain whose recorded data is
        the metric target.
    seed : int, optional, keyword-only
        Master seed; every sweep point gets an independent child seed
        through the chain (decisions D6/D35), so the whole curve is
        reproducible.

    Returns
    -------
    dict of str to np.ndarray
        One array per metric, aligned with ``values``.

    Examples
    --------
    >>> from comnumpy import AWGN, Recorder, Sequential, SymbolGenerator
    >>> from comnumpy.core.metrics import compute_ser
    >>> chain = Sequential([SymbolGenerator(4), Recorder(name="tx"), AWGN(snr_dB=0, name="noise")])
    >>> out = sweep(chain, "noise.snr_dB", [0, 10], {"power": lambda y: float(np.mean(np.abs(y)**2))},
    ...             stimulus=1000, seed=1)
    >>> print(out["power"].shape)
    (2,)
    """
    params = [param] if isinstance(param, str) else list(param)
    seeds = (np.random.SeedSequence(seed).spawn(len(values))
             if seed is not None else [None] * len(values))

    results: dict[str, list[Any]] = {name: [] for name in metrics}
    for point, child in zip(values, seeds, strict=True):
        point_values = (point,) if len(params) == 1 else tuple(point)
        chain.set_params(**dict(zip(params, point_values, strict=True)))
        if child is not None:
            chain.seed(int(child.generate_state(1)[0]))

        y = chain(stimulus)

        target = chain[reference].get_data() if reference is not None else None
        for name, metric in metrics.items():
            value = metric(target, y) if target is not None else metric(y)
            results[name].append(value)

    return {name: np.asarray(collected) for name, collected in results.items()}
