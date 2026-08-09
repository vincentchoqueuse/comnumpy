"""Convolutional coding: encoder and Viterbi decoder (decision D4).

Batch-vectorized implementations: the only Python loop is over time
(the trellis recursion is irreducible); everything across batch and
states is numpy. Branch metrics are computed inside the time loop, so
memory stays bounded (architecture document, section 8).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

__all__ = ["ConvolutionalEncoder", "ViterbiDecoder"]


def _parse_generators(g: Sequence[int]) -> tuple[tuple[int, ...], int]:
    """Constraint length K and generator taps from octal generators."""
    g = tuple(int(gi) for gi in g)
    if len(g) < 2:
        raise ValueError(f"need at least 2 generator polynomials, got {g}")
    K = max(gi.bit_length() for gi in g)
    if K < 2:
        raise ValueError(f"generators {g} imply a constraint length K={K} < 2")
    return g, K


def _output_table(g: Sequence[int], K: int) -> np.ndarray:
    """(2**K, n) table: coded bits for every register content."""
    regs = np.arange(2**K)[:, None]
    taps = np.array(g)[None, :]
    ands = regs & taps
    # parity of (reg & g_i): popcount modulo 2
    bits = np.zeros_like(ands)
    for shift in range(K):
        bits ^= (ands >> shift) & 1
    return bits.astype(np.int8)


@dataclass(slots=True)
class ConvolutionalEncoder(Processor):
    r"""Rate :math:`1/n` convolutional encoder.

    Signal Model
    ------------
    With input bits :math:`x[m]` and generator polynomials
    :math:`g_1, \dots, g_n` of constraint length :math:`K`, the coded
    stream interleaves the :math:`n` parity outputs:

    .. math::

        y[nm + i] = \bigoplus_{l=0}^{K-1} g_{i}[l] \, x[m - l],
        \qquad i = 0, \dots, n-1

    The default generators :math:`(133, 171)_8` with :math:`K = 7` are
    the standard NASA/Voyager code (free distance 10). When
    ``terminated`` is True, :math:`K - 1` zero tail bits flush the
    register so the decoder can end in the all-zero state.

    Axes: *axis -1* -- bits on the last axis, broadcast over the rest;
    ``(..., N)`` maps to ``(..., n (N + K - 1))`` when terminated.

    Parameters
    ----------
    g : tuple of int
        Generator polynomials :math:`g_i` in octal (MSB = current input
        tap). Default ``(0o133, 0o171)``.
    terminated : bool, keyword-only
        Append :math:`K - 1` zero tail bits. Default True.
    name : str, keyword-only
        Name of the processor. Default ``"conv_encoder"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 8.2.

    Examples
    --------
    >>> encoder = ConvolutionalEncoder((0o5, 0o7))  # K=3, rate 1/2
    >>> bits = np.array([1, 0, 1, 1])
    >>> print(encoder(bits))
    [1 1 0 1 0 0 1 0 1 0 1 1]
    """
    g: tuple[int, ...] = (0o133, 0o171)
    terminated: bool = field(default=True, kw_only=True)
    name: str = field(default="conv_encoder", kw_only=True)
    # precomputed trellis tables (parametric, allowed in __post_init__)
    K: int = field(init=False, repr=False)
    n_streams: int = field(init=False, repr=False)
    out_table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.g, self.K = _parse_generators(self.g)
        self.n_streams = len(self.g)
        self.out_table = _output_table(self.g, self.K)

    @property
    def rate(self) -> float:
        """Code rate :math:`1/n` (tail bits not counted)."""
        return 1.0 / self.n_streams

    def forward(self, x: np.ndarray) -> np.ndarray:
        bits = np.asarray(x).astype(np.int64)
        n_tail = self.K - 1 if self.terminated else 0
        if n_tail:
            pad = np.zeros(bits.shape[:-1] + (n_tail,), dtype=np.int64)
            bits = np.concatenate([bits, pad], axis=-1)

        T = bits.shape[-1]
        y = np.empty(bits.shape[:-1] + (T, self.n_streams), dtype=np.int8)
        state = np.zeros(bits.shape[:-1], dtype=np.int64)
        for t in range(T):  # time loop only; batch stays vectorized (D4)
            reg = (bits[..., t] << (self.K - 1)) | state
            y[..., t, :] = self.out_table[reg]
            state = reg >> 1
        return y.reshape(bits.shape[:-1] + (T * self.n_streams,))


@dataclass(slots=True)
class ViterbiDecoder(Processor):
    r"""Maximum-likelihood decoder for a rate :math:`1/n` convolutional code.

    Signal Model
    ------------
    The decoder finds the information sequence :math:`\hat{x}` whose
    coded sequence :math:`c(\hat{x})` minimizes the path metric

    .. math::

        \hat{x} = \arg\min_{x} \sum_{m, i} d\big(r[nm+i],\, c_i[m]\big)

    where :math:`d` is the Hamming distance for hard decisions
    (``soft=False``, input bits in :math:`\{0, 1\}`) and the max-log
    metric :math:`d(r, c) = r \, c` for soft decisions (``soft=True``,
    input log-likelihood ratios :math:`r = \log P(0)/P(1)`, as produced
    by ``SymbolDemapper(..., soft=True)``).

    Axes: *axis -1* -- coded samples on the last axis, batch-vectorized
    Viterbi recursion with a Python loop over time only (decision D4);
    ``(..., n (N + K - 1))`` maps back to ``(..., N)`` when terminated.

    Parameters
    ----------
    g : tuple of int
        Generator polynomials :math:`g_i` in octal; must match the
        encoder. Default ``(0o133, 0o171)``.
    soft : bool, keyword-only
        Interpret the input as LLRs instead of hard bits. Default False.
    terminated : bool, keyword-only
        Assume :math:`K - 1` zero tail bits (decoding ends in state 0).
        Default True.
    name : str, keyword-only
        Name of the processor. Default ``"viterbi"``.

    Raises
    ------
    ShapeError
        If the coded length is not a multiple of :math:`n`, or shorter
        than the tail.

    References
    ----------
    A. J. Viterbi, "Error bounds for convolutional codes and an
    asymptotically optimum decoding algorithm," IEEE Trans. Inf.
    Theory 13(2), 1967; Proakis & Salehi, 5th ed., Section 8.2.

    Examples
    --------
    >>> encoder = ConvolutionalEncoder((0o5, 0o7))
    >>> decoder = ViterbiDecoder((0o5, 0o7))
    >>> coded = encoder(np.array([1, 0, 1, 1]))
    >>> coded[2] ^= 1  # one channel error
    >>> print(decoder(coded))
    [1 0 1 1]
    """
    g: tuple[int, ...] = (0o133, 0o171)
    soft: bool = field(default=False, kw_only=True)
    terminated: bool = field(default=True, kw_only=True)
    name: str = field(default="viterbi", kw_only=True)
    # precomputed trellis tables (parametric, allowed in __post_init__)
    K: int = field(init=False, repr=False)
    n_streams: int = field(init=False, repr=False)
    expected: np.ndarray = field(init=False, repr=False)
    pred_state: np.ndarray = field(init=False, repr=False)
    pred_input: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.g, self.K = _parse_generators(self.g)
        self.n_streams = len(self.g)
        S = 2 ** (self.K - 1)
        out_table = _output_table(self.g, self.K)

        # arrival-state view of the trellis: state s' has two predecessor
        # registers reg = 2 s' + j, j in {0, 1}
        s_prime = np.arange(S)[:, None]
        regs = 2 * s_prime + np.array([0, 1])[None, :]     # (S, 2)
        self.pred_state = regs & (S - 1)                    # (S, 2)
        self.pred_input = regs >> (self.K - 1)              # (S, 2)
        self.expected = out_table[regs]                     # (S, 2, n) coded bits

    def _branch_cost(self, r_t: np.ndarray) -> np.ndarray:
        """(batch..., S, 2) cost of every branch for one received n-tuple."""
        r = r_t[..., None, None, :]          # (batch..., 1, 1, n)
        c = self.expected                    # (S, 2, n)
        if self.soft:
            # max-log metric: choosing bit 1 costs the LLR
            return np.sum(np.where(c == 1, r, 0.0), axis=-1)
        return np.sum(np.abs(r - c), axis=-1)

    def forward(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)
        n = self.n_streams
        if r.shape[-1] % n != 0:
            raise ShapeError(
                f"ViterbiDecoder expects a coded length multiple of n={n}, "
                f"got {r.shape} -- check the encoder rate.")
        T = r.shape[-1] // n
        n_tail = self.K - 1 if self.terminated else 0
        if T <= n_tail:
            raise ShapeError(
                f"coded length {r.shape[-1]} is shorter than the tail "
                f"({n_tail} steps of {n} bits) -- nothing to decode.")
        r = r.reshape(r.shape[:-1] + (T, n)).astype(float)

        batch_shape = r.shape[:-2]
        S = 2 ** (self.K - 1)
        INF = np.inf
        pm = np.full(batch_shape + (S,), INF)
        pm[..., 0] = 0.0                       # encoder starts in state 0
        decisions = np.empty(batch_shape + (T, S), dtype=np.uint8)

        for t in range(T):  # time loop only; batch and states vectorized (D4)
            bm = self._branch_cost(r[..., t, :])            # (..., S, 2)
            cand = pm[..., self.pred_state] + bm            # (..., S, 2)
            decisions[..., t, :] = np.argmin(cand, axis=-1)
            pm = np.min(cand, axis=-1)

        # traceback, vectorized over the batch
        state = (np.zeros(batch_shape, dtype=np.int64) if self.terminated
                 else np.argmin(pm, axis=-1))
        bits = np.empty(batch_shape + (T,), dtype=np.int64)
        batch_index = np.ix_(*[np.arange(s) for s in batch_shape]) if batch_shape else ()
        for t in range(T - 1, -1, -1):
            winner = decisions[..., t, :][batch_index + (state,)] if batch_shape \
                else decisions[t, state]
            bits[..., t] = self.pred_input[state, winner]
            state = self.pred_state[state, winner]

        return bits[..., :T - n_tail] if n_tail else bits
