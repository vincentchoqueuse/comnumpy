"""Analytical reference for convolutional codes (decisions D4, D7).

The distance spectrum is **enumerated from the trellis of the code
itself**, never transcribed from a textbook table: the same
``reg = (b << (K - 1)) | state`` recursion that
:class:`~comnumpy.fec.convolutional.ConvolutionalEncoder` runs forward is
walked breadth-first here, one trellis step per iteration, to collect the
error events. The published tables then serve as a cross-check, not as a
source -- the enumerator returns :math:`d_{\\mathrm{free}} = 10` for the
NASA ``(133, 171)`` code because the trellis says so.

That spectrum feeds the union bound, which is the analytical curve a
soft-decision Viterbi decoder is measured against
(``validation/fec_union_bound.py``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# the trellis is built once, in the encoder: this module walks that table,
# it does not rebuild one (a second construction would be a second truth)
from comnumpy.fec.convolutional import (
    _output_table, _parse_generators)  # pyright: ignore[reportPrivateUsage]

__all__ = ["DistanceSpectrum", "distance_spectrum", "union_bound_ber"]

# safety net for the automatic growth of ``d_max``: a rate-1/n code whose
# free distance exceeds this has a trellis far outside the teaching range
_D_MAX_LIMIT = 256


@dataclass(frozen=True, slots=True)
class DistanceSpectrum:
    r"""Weight enumerator of the error events of a rate :math:`1/n` code.

    Signal Model
    ------------
    An *error event* is a trellis path that leaves the all-zero path at
    time 0 (input bit 1), stays away from the all-zero state, and merges
    back into it later. Collecting these paths by output weight
    :math:`d` and input weight :math:`i` gives the transfer function of
    the code,

    .. math::

        T(D, N) = \sum_{d \ge d_{\mathrm{free}}} \sum_{i \ge 1}
                  a_{d,i} \, D^{d} N^{i},

    of which this object stores the two marginals used by every
    performance bound:

    .. math::

        a_d = \sum_i a_{d,i}, \qquad
        \beta_d = \sum_i i \, a_{d,i}
                = \left. \frac{\partial T}{\partial N} \right|_{N = 1}
                  \text{, coefficient of } D^d

    :math:`a_d` counts the error events of output weight :math:`d`;
    :math:`\beta_d` counts the information bit errors they carry, which
    is what :func:`union_bound_ber` weighs.

    Parameters
    ----------
    g : tuple of int
        Generator polynomials :math:`g_i` in octal, as passed to
        :class:`~comnumpy.fec.convolutional.ConvolutionalEncoder`.
    K : int
        Constraint length :math:`K` implied by ``g``.
    k : int
        Number of information bits per trellis step :math:`k`; always 1
        for the rate :math:`1/n` codes this module enumerates.
    rate : float
        Code rate :math:`R = k/n`.
    d_free : int
        Free distance :math:`d_{\mathrm{free}}`, the smallest output
        weight of an error event.
    distances : np.ndarray of int
        The weights :math:`d` covered, ``d_free`` through ``d_max``
        inclusive (gaps are kept, with a zero coefficient: a code whose
        events all have even weight must show it).
    a_d : np.ndarray of int
        Number of error events :math:`a_d`, aligned with ``distances``.
    beta_d : np.ndarray of int
        Cumulated information weight :math:`\beta_d`, aligned with
        ``distances``.

    References
    ----------
    A. J. Viterbi, "Convolutional codes and their performance in
    communication systems," IEEE Trans. Commun. Technol. 19(5), 1971
    (transfer function :math:`T(D, N)` and its derivative);
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 8.2.

    Examples
    --------
    >>> spectrum = distance_spectrum((0o5, 0o7), n_terms=3)
    >>> print(spectrum.d_free, spectrum.rate)
    5 0.5
    >>> print(spectrum.distances, spectrum.a_d, spectrum.beta_d)
    [5 6 7] [1 2 4] [ 1  4 12]
    """
    g: tuple[int, ...]
    K: int
    k: int
    rate: float
    d_free: int
    distances: npt.NDArray[np.int64]
    a_d: npt.NDArray[np.int64]
    beta_d: npt.NDArray[np.int64]


def _branch_weight(g: tuple[int, ...], K: int) -> np.ndarray:
    """Output Hamming weight of every trellis branch, indexed by register."""
    return _output_table(g, K).sum(axis=-1).astype(np.int64)


def _zero_weight_cycle(weight: np.ndarray, K: int) -> bool:
    """True if the nonzero states carry a cycle of zero-output-weight branches.

    Such a cycle is the signature of a catastrophic encoder: it lets an
    error event accumulate input weight without accumulating output
    weight, so :math:`a_d` and :math:`\\beta_d` are infinite. Detected by
    peeling away the states that have no zero-weight successor; whatever
    survives lies on a cycle.
    """
    S = 2 ** (K - 1)
    successors = {s: [] for s in range(1, S)}
    for s in range(1, S):
        for b in (0, 1):
            reg = (b << (K - 1)) | s
            next_state = reg >> 1
            if next_state != 0 and weight[reg] == 0:
                successors[s].append(next_state)

    alive = set(successors)
    peeled = True
    while peeled:
        peeled = False
        for s in list(alive):
            if not any(t in alive for t in successors[s]):
                alive.discard(s)
                peeled = True
    return bool(alive)


def _enumerate_error_events(weight: np.ndarray, K: int,
                            d_max: int) -> tuple[list[int], list[int]]:
    """Breadth-first count of the error events of output weight <= ``d_max``.

    One iteration = one trellis step. ``count[s][d]`` is the number of
    paths of the current length that reached state ``s`` with output
    weight ``d`` without touching state 0; ``info[s][d]`` is their
    cumulated input weight. A path that steps into state 0 is a finished
    error event and is banked into ``a`` / ``beta``. Paths heavier than
    ``d_max`` are pruned, which is what makes the frontier die out.
    """
    S = 2 ** (K - 1)
    a = [0] * (d_max + 1)
    beta = [0] * (d_max + 1)
    count = [[0] * (d_max + 1) for _ in range(S)]
    info = [[0] * (d_max + 1) for _ in range(S)]

    # divergence: input bit 1 from the all-zero state
    reg = 1 << (K - 1)
    if weight[reg] <= d_max:
        count[reg >> 1][weight[reg]] = 1
        info[reg >> 1][weight[reg]] = 1

    while any(any(row) for row in count):
        next_count = [[0] * (d_max + 1) for _ in range(S)]
        next_info = [[0] * (d_max + 1) for _ in range(S)]
        for s in range(1, S):          # state 0 is the merge, never a start
            for d, paths in enumerate(count[s]):
                if not paths:
                    continue
                bits = info[s][d]
                for b in (0, 1):
                    reg = (b << (K - 1)) | s
                    d_next = d + int(weight[reg])
                    if d_next > d_max:
                        continue       # pruning: this branch can only grow
                    state = reg >> 1
                    if state == 0:     # merged back: the event is complete
                        a[d_next] += paths
                        beta[d_next] += bits + b * paths
                    else:
                        next_count[state][d_next] += paths
                        next_info[state][d_next] += bits + b * paths
        count, info = next_count, next_info
    return a, beta


def distance_spectrum(g: Sequence[int] = (0o133, 0o171), *,
                      n_terms: int = 6,
                      d_max: int | None = None) -> DistanceSpectrum:
    r"""Enumerate the distance spectrum of a rate :math:`1/n` convolutional code.

    Signal Model
    ------------
    The encoder register content is :math:`\mathrm{reg} = (b \ll (K-1))
    \,|\, s` for input bit :math:`b` in state :math:`s`, the very
    recursion run by
    :class:`~comnumpy.fec.convolutional.ConvolutionalEncoder`; the branch
    output weight is :math:`w(\mathrm{reg}) = \sum_{i} g_i \cdot
    \mathrm{reg} \bmod 2`. An *error event* is a path
    :math:`0 \to s_1 \to \dots \to s_L \to 0` with :math:`b = 1` on the
    first branch and :math:`s_\ell \ne 0` in between. Because the code is
    linear, comparing every path pair reduces to comparing every path to
    the all-zero one, so these events carry the whole error behaviour.

    Counting them by output weight :math:`d` and input weight yields the
    coefficients of the transfer function :math:`T(D, N)`:

    .. math::

        a_d = \#\{\text{error events of output weight } d\}, \qquad
        \beta_d = \sum_{\text{events of weight } d}
                  w_{\mathrm{in}}(\text{event})

    and :math:`d_{\mathrm{free}} = \min \{d : a_d > 0\}`. The search is a
    breadth-first walk of the trellis pruned at ``d_max`` (one iteration
    per trellis step); the walk terminates because, absent a zero-weight
    cycle, every long path eventually exceeds ``d_max``.

    Axes: *not a signal transform* -- a pure function of the code
    parameters, returning arrays indexed by output weight.

    Parameters
    ----------
    g : tuple of int
        Generator polynomials :math:`g_i` in octal (MSB = current input
        tap), identical to the encoder's. Default ``(0o133, 0o171)``,
        the NASA/Voyager :math:`K = 7` code.
    n_terms : int, optional, keyword-only
        Number of *non-zero* coefficients :math:`a_d` wanted; ``d_max``
        is grown until that many are found. Ignored when ``d_max`` is
        given. Default 6.
    d_max : int, optional, keyword-only
        Largest output weight :math:`d` enumerated. Larger means a
        tighter :func:`union_bound_ber` at low SNR and a longer run.

    Returns
    -------
    DistanceSpectrum
        The coefficients :math:`a_d` and :math:`\beta_d` for
        :math:`d_{\mathrm{free}} \le d \le d_{\max}`.

    Raises
    ------
    ValueError
        If ``g`` is catastrophic -- a zero-output-weight cycle makes
        :math:`a_d` infinite -- or if no error event is found below
        ``d_max``.

    References
    ----------
    A. J. Viterbi, "Convolutional codes and their performance in
    communication systems," IEEE Trans. Commun. Technol. 19(5),
    pp. 751-772, 1971 (transfer function of the state diagram);
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 8.2.

    Examples
    --------
    >>> spectrum = distance_spectrum((0o5, 0o7), n_terms=4)
    >>> print(spectrum.d_free, spectrum.a_d, spectrum.beta_d)
    5 [1 2 4 8] [ 1  4 12 32]
    >>> nasa = distance_spectrum(n_terms=3)   # (133, 171), K = 7
    >>> print(nasa.d_free, nasa.distances, nasa.a_d)
    10 [10 11 12 13 14] [ 11   0  38   0 193]
    """
    g, K = _parse_generators(g)
    weight = _branch_weight(g, K)
    if _zero_weight_cycle(weight, K):
        raise ValueError(
            f"distance_spectrum: generators {tuple(oct(gi) for gi in g)} are "
            f"catastrophic (the trellis has a cycle of zero output weight, so "
            f"a finite number of channel errors can cause infinitely many "
            f"information errors, and a_d is infinite) -- pick generators "
            f"whose polynomials are coprime, e.g. (0o5, 0o7) for K=3.")

    if d_max is not None:
        a, beta = _enumerate_error_events(weight, K, int(d_max))
    else:
        d_max = 2 * K
        while True:
            a, beta = _enumerate_error_events(weight, K, d_max)
            if sum(1 for c in a if c) >= n_terms:
                break
            if d_max > _D_MAX_LIMIT:
                raise ValueError(
                    f"distance_spectrum: fewer than n_terms={n_terms} non-zero "
                    f"coefficients below d_max={_D_MAX_LIMIT} for generators "
                    f"{tuple(oct(gi) for gi in g)} -- pass d_max explicitly if "
                    f"the code really has such a sparse spectrum.")
            d_max *= 2
        # trim to the requested number of non-zero coefficients
        nonzero = [d for d, c in enumerate(a) if c]
        d_max = nonzero[n_terms - 1]

    support = [d for d, c in enumerate(a) if c]
    if not support:
        raise ValueError(
            f"distance_spectrum: no error event of output weight <= {d_max} "
            f"for generators {tuple(oct(gi) for gi in g)} -- increase d_max.")
    d_free = support[0]
    kept = slice(d_free, d_max + 1)
    return DistanceSpectrum(
        g=g, K=K, k=1, rate=1.0 / len(g), d_free=d_free,
        distances=np.arange(d_free, d_max + 1, dtype=np.int64),
        a_d=np.array(a[kept], dtype=np.int64),
        beta_d=np.array(beta[kept], dtype=np.int64),
    )


def union_bound_ber(spectrum: DistanceSpectrum,
                    ebn0_dB: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""Union bound on the bit error rate of soft-decision Viterbi decoding over AWGN.

    Signal Model
    ------------
    Antipodal signalling over AWGN, :math:`y[n] = x[n] + b[n]` with
    :math:`x[n] = \pm\sqrt{E_s}` and :math:`b[n] \sim \mathcal{N}(0,
    N_0/2)`, decoded by a maximum-likelihood (soft) Viterbi decoder. Two
    codewords differing in :math:`d` coded bits are confused with
    probability :math:`Q(\sqrt{2 d R E_b/N_0})`; summing that pairwise
    probability over every error event, each weighted by the number
    :math:`\beta_d` of information bits it flips, over-counts the
    overlapping events and therefore bounds the bit error rate:

    .. math::

        P_b \le \frac{1}{k} \sum_{d \ge d_{\mathrm{free}}} \beta_d \,
                Q\!\left(\sqrt{2 d R \frac{E_b}{N_0}}\right),
        \qquad
        Q(u) = \frac{1}{\sqrt{2\pi}} \int_u^{\infty} e^{-t^2/2}\, dt

    Two honest caveats, both visible in ``validation/fec_union_bound.py``:
    the sum is **truncated** at the ``d_max`` of ``spectrum``, so it is a
    genuine upper bound only where its tail is negligible; and the full
    series only converges above the code's cutoff rate. Below roughly
    3 dB the truncated sum keeps growing with ``d_max`` and usually
    exceeds 1 -- a value :math:`> 1` is not a tight bound, it is no
    information at all. Above 4 dB the terms fall geometrically, the
    truncation error is negligible, and the bound is within a small
    factor of the simulated curve.

    Axes: *element-wise* -- a scalar conversion applied pointwise to an
    array of operating points.

    Parameters
    ----------
    spectrum : DistanceSpectrum
        Weight enumerator of the code, from :func:`distance_spectrum`;
        supplies :math:`\beta_d`, the weights :math:`d`, the rate
        :math:`R` and the number of inputs per step :math:`k`.
    ebn0_dB : float or np.ndarray
        Information-bit-energy-to-noise ratio :math:`E_b/N_0` in dB
        (energy per *information* bit, so the rate loss is already
        accounted for).

    Returns
    -------
    np.ndarray
        Upper bound on :math:`P_b`, same shape as ``ebn0_dB``. Values
        above 1 are returned as computed, not clipped: a bound that
        exceeds 1 must look useless, because it is.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 8.2 (union bound for soft-decision
    decoding, eq. 8.2-22); A. J. Viterbi, "Convolutional codes and their
    performance in communication systems," IEEE Trans. Commun. Technol.
    19(5), 1971.

    Examples
    --------
    >>> spectrum = distance_spectrum((0o5, 0o7), d_max=26)   # K = 3
    >>> print(f"{float(union_bound_ber(spectrum, 5.0)):.3e}")
    9.171e-05
    >>> print(f"{float(union_bound_ber(distance_spectrum(d_max=30), 5.0)):.3e}")
    4.427e-07
    """
    from scipy.stats import norm    # local import (D36)

    gamma_b = 10.0 ** (np.asarray(ebn0_dB, dtype=float) / 10.0)
    d = spectrum.distances.astype(float)[:, None]
    beta = spectrum.beta_d.astype(float)[:, None]
    argument = np.sqrt(2.0 * d * spectrum.rate * np.atleast_1d(gamma_b).ravel())
    bound = np.sum(beta * norm.sf(argument), axis=0) / spectrum.k
    return bound.reshape(gamma_b.shape)
