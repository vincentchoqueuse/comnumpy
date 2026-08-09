"""LDPC coding: systematic encoder and min-sum decoder (decision D5).

Same vectorization philosophy as the convolutional module (D4): the
decoder state lives in ``(batch, n_edges)`` arrays, message-passing
updates are segmented numpy reductions (``ufunc.reduceat``) over the
edges of the Tanner graph, and the only Python loop is over decoding
iterations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

__all__ = ["LDPCEncoder", "LDPCDecoder", "make_gallager_parity_check"]


def _rref_gf2(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced row echelon form of ``H`` over GF(2).

    Returns the ``(rank, n)`` reduced matrix (dependent rows dropped)
    and the pivot column indices. Non-pivot columns are the free
    (information) positions of the systematic encoding.
    """
    A = (np.asarray(H) % 2).astype(np.uint8).copy()
    m, n = A.shape
    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        if r == m:
            break
        rows = np.nonzero(A[r:, c])[0]
        if rows.size == 0:
            continue
        p = r + int(rows[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool).copy()
        mask[r] = False
        A[mask] ^= A[r]
        pivot_cols.append(c)
        r += 1
    return A[:r], np.array(pivot_cols, dtype=np.int64)


def make_gallager_parity_check(n: int, d_v: int = 3, d_c: int = 6,
                               seed: Optional[int] = None) -> np.ndarray:
    r"""Regular parity-check matrix from the Gallager ensemble.

    Signal Model
    ------------
    The :math:`(d_v, d_c)`-regular matrix stacks :math:`d_v` bands of
    :math:`n / d_c` rows. The first band places its ones in consecutive
    groups of :math:`d_c` columns; every other band is a random column
    permutation of it, so each column has weight :math:`d_v` and each
    row weight :math:`d_c`. The design rate is
    :math:`R = 1 - d_v / d_c` (the true rate is slightly higher when
    the stacked rows are linearly dependent, which the encoder detects).

    Parameters
    ----------
    n : int
        Code length (number of columns). Must be a multiple of ``d_c``.
    d_v : int, optional
        Column (variable node) weight. Default 3.
    d_c : int, optional
        Row (check node) weight. Default 6.
    seed : int, optional
        Seed for the band permutations.

    Returns
    -------
    np.ndarray
        Parity-check matrix of shape ``(n d_v / d_c, n)``, dtype uint8.

    References
    ----------
    R. G. Gallager, "Low-density parity-check codes," IRE Trans. Inf.
    Theory 8(1), 1962.

    Examples
    --------
    >>> H = make_gallager_parity_check(12, d_v=3, d_c=6, seed=0)
    >>> print(H.shape, int(H.sum(axis=0)[0]), int(H.sum(axis=1)[0]))
    (6, 12) 3 6
    """
    if n % d_c != 0:
        raise ValueError(f"n={n} must be a multiple of d_c={d_c}")
    if not 0 < d_v < d_c:
        raise ValueError(f"need 0 < d_v < d_c, got d_v={d_v}, d_c={d_c}")
    rng = np.random.default_rng(seed)
    band = np.zeros((n // d_c, n), dtype=np.uint8)
    for i in range(n // d_c):
        band[i, i * d_c:(i + 1) * d_c] = 1
    blocks = [band] + [band[:, rng.permutation(n)] for _ in range(d_v - 1)]
    return np.vstack(blocks)


@dataclass(slots=True)
class LDPCEncoder(Processor):
    r"""Systematic encoder for an LDPC code given its parity-check matrix.

    Signal Model
    ------------
    The codeword :math:`c \in \{0,1\}^n` satisfies :math:`H c = 0`
    over GF(2). Bringing :math:`H` to reduced row echelon form
    partitions the columns into :math:`r` pivot (parity) positions and
    :math:`k = n - r` free (information) positions; the information
    bits :math:`u` are copied to the free positions and each parity is

    .. math::

        c_{p_i} = \bigoplus_{j \in \mathrm{free}} \tilde{H}_{ij} \, u_j

    where :math:`\tilde{H}` is the reduced matrix. Linearly dependent
    rows of :math:`H` are detected and ignored, so :math:`k` can exceed
    the design dimension :math:`n - m`.

    Axes: *axis -1* -- information bits on the last axis, broadcast
    over the rest; ``(..., k)`` maps to ``(..., n)``.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix of shape ``(m, n)`` (entries 0/1); must
        match the decoder.
    name : str, keyword-only
        Name of the processor. Default ``"ldpc_encoder"``.

    References
    ----------
    T. Richardson, R. Urbanke, *Modern Coding Theory*, Cambridge
    University Press, 2008, Section 3.2 (encoding via echelon form).

    Examples
    --------
    >>> H = np.array([[1, 1, 0, 1, 1, 0, 0],
    ...               [1, 0, 1, 1, 0, 1, 0],
    ...               [0, 1, 1, 1, 0, 0, 1]])   # (7, 4) Hamming code
    >>> encoder = LDPCEncoder(H)
    >>> c = encoder(np.array([1, 0, 1, 1]))
    >>> print(c)
    [0 0 1 0 0 1 1]
    >>> print(H @ c % 2)
    [0 0 0]
    """
    H: np.ndarray
    name: str = field(default="ldpc_encoder", kw_only=True)
    # systematic structure (parametric, allowed in __post_init__)
    n: int = field(init=False, repr=False)
    k: int = field(init=False, repr=False)
    pivot_cols: np.ndarray = field(init=False, repr=False)
    free_cols: np.ndarray = field(init=False, repr=False)
    _P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.H = (np.asarray(self.H) % 2).astype(np.uint8)
        rref, self.pivot_cols = _rref_gf2(self.H)
        self.n = self.H.shape[1]
        self.free_cols = np.setdiff1d(np.arange(self.n), self.pivot_cols)
        self.k = int(self.free_cols.size)
        self._P = rref[:, self.free_cols]         # (rank, k)

    @property
    def rate(self) -> float:
        """Code rate :math:`k / n` (true rank, not design rate)."""
        return self.k / self.n

    def forward(self, x: np.ndarray) -> np.ndarray:
        bits = np.asarray(x).astype(np.uint8)
        if bits.shape[-1] != self.k:
            raise ShapeError(
                f"LDPCEncoder expects {self.k} information bits on the "
                f"last axis, got {bits.shape} -- k = n - rank(H).")
        c = np.zeros(bits.shape[:-1] + (self.n,), dtype=np.uint8)
        c[..., self.free_cols] = bits
        c[..., self.pivot_cols] = (bits @ self._P.T) % 2
        return c


@dataclass(slots=True)
class LDPCDecoder(Processor):
    r"""Min-sum belief-propagation decoder for an LDPC code.

    Signal Model
    ------------
    Messages flow on the edges of the Tanner graph of :math:`H`. With
    channel log-likelihood ratios :math:`L_j = \log P(c_j{=}0)/P(c_j{=}1)`
    (the ``SymbolDemapper(soft=True)`` convention), the min-sum check
    update for the message from check :math:`i` to variable :math:`j` is

    .. math::

        m_{i \to j} = \alpha
        \prod_{j' \in N(i) \setminus j} \mathrm{sign}(m_{j' \to i})
        \min_{j' \in N(i) \setminus j} |m_{j' \to i}|

    and the variable update is
    :math:`m_{j \to i} = L_j + \sum_{i' \in N(j) \setminus i} m_{i' \to j}`.
    Iterations stop early when the hard decision satisfies every parity
    check. All messages live in ``(batch, n_edges)`` arrays; the check
    and variable updates are segmented reductions over edges sorted by
    check and by variable (decision D5) -- the only Python loop is over
    iterations.

    Axes: *axis -1* -- channel LLRs on the last axis, broadcast over
    the rest; ``(..., n)`` maps to ``(..., k)`` (``output="info"``) or
    ``(..., n)`` (``output="codeword"``).

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix of shape ``(m, n)``; must match the
        encoder. Every row and column must contain at least one 1.
    n_iter : int, keyword-only
        Maximum number of message-passing iterations. Default 25.
    alpha : float, keyword-only
        Normalization factor of the check update (normalized min-sum).
        Default 1.0 (plain min-sum); 0.75 is a common choice that
        recovers most of the sum-product gap.
    output : {"info", "codeword"}, keyword-only
        Return the information bits at the systematic positions
        (default) or the full hard-decision codeword.
    name : str, keyword-only
        Name of the processor. Default ``"ldpc_decoder"``.

    Raises
    ------
    ShapeError
        If the last axis of the input does not match the code length.

    References
    ----------
    R. G. Gallager, "Low-density parity-check codes," IRE Trans. Inf.
    Theory 8(1), 1962; J. Chen, M. Fossorier, "Near optimum universal
    belief propagation based decoding of low-density parity check
    codes," IEEE Trans. Commun. 50(3), 2002 (normalized min-sum).

    Examples
    --------
    >>> H = np.array([[1, 1, 0, 1, 1, 0, 0],
    ...               [1, 0, 1, 1, 0, 1, 0],
    ...               [0, 1, 1, 1, 0, 0, 1]])   # (7, 4) Hamming code
    >>> c = LDPCEncoder(H)(np.array([1, 0, 1, 1]))
    >>> llr = 4.0 * (1.0 - 2.0 * c)   # log P(0)/P(1), positive = bit 0
    >>> llr[1] = -1.0                 # one weak channel error
    >>> decoder = LDPCDecoder(H)
    >>> print(decoder(llr))
    [1 0 1 1]
    """
    H: np.ndarray
    n_iter: int = field(default=25, kw_only=True)
    alpha: float = field(default=1.0, kw_only=True)
    output: Literal["info", "codeword"] = field(default="info", kw_only=True)
    name: str = field(default="ldpc_decoder", kw_only=True)
    # Tanner graph in edge form (parametric, allowed in __post_init__)
    n: int = field(init=False, repr=False)
    free_cols: np.ndarray = field(init=False, repr=False)
    _var_idx: np.ndarray = field(init=False, repr=False)
    _c_starts: np.ndarray = field(init=False, repr=False)
    _c_counts: np.ndarray = field(init=False, repr=False)
    _v_order: np.ndarray = field(init=False, repr=False)
    _v_starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.H = (np.asarray(self.H) % 2).astype(np.uint8)
        m, self.n = self.H.shape
        _, pivot_cols = _rref_gf2(self.H)
        self.free_cols = np.setdiff1d(np.arange(self.n), pivot_cols)

        check_idx, self._var_idx = np.nonzero(self.H)  # sorted by check
        self._c_counts = np.bincount(check_idx, minlength=m)
        v_counts = np.bincount(self._var_idx, minlength=self.n)
        if (self._c_counts == 0).any() or (v_counts == 0).any():
            raise ValueError(
                "LDPCDecoder: H has an empty row or column -- every "
                "check and every variable must touch at least one edge.")
        self._c_starts = np.concatenate(
            ([0], np.cumsum(self._c_counts)[:-1]))
        self._v_order = np.argsort(self._var_idx, kind="stable")
        self._v_starts = np.concatenate(([0], np.cumsum(v_counts)[:-1]))

    def _check_update(self, m_vc: np.ndarray) -> np.ndarray:
        """Min-sum extrinsic check-to-variable messages, (batch, E)."""
        starts, counts = self._c_starts, self._c_counts
        s = np.where(m_vc < 0, -1.0, 1.0)
        sign_all = np.repeat(np.multiply.reduceat(s, starts, axis=1),
                             counts, axis=1)
        A = np.abs(m_vc)
        min1 = np.repeat(np.minimum.reduceat(A, starts, axis=1),
                         counts, axis=1)
        is_min = A == min1
        n_min = np.repeat(np.add.reduceat(is_min, starts, axis=1),
                          counts, axis=1)
        A_masked = np.where(is_min, np.inf, A)
        min2 = np.repeat(np.minimum.reduceat(A_masked, starts, axis=1),
                         counts, axis=1)
        # excluding oneself: the unique minimum sees the second minimum,
        # everyone else (including duplicated minima) sees the minimum
        ext_abs = np.where(is_min & (n_min == 1), min2, min1)
        return self.alpha * (sign_all * s) * ext_abs

    def forward(self, x: np.ndarray) -> np.ndarray:
        llr = np.asarray(x, dtype=float)
        if llr.shape[-1] != self.n:
            raise ShapeError(
                f"LDPCDecoder expects {self.n} channel LLRs on the last "
                f"axis, got {llr.shape} -- one LLR per codeword bit.")
        batch_shape = llr.shape[:-1]
        L = llr.reshape(-1, self.n)                       # (B, n)
        var_idx, v_order, v_starts = (self._var_idx, self._v_order,
                                      self._v_starts)

        m_vc = L[:, var_idx]                              # (B, E)
        hard = (L < 0).astype(np.uint8)
        for _ in range(self.n_iter):  # iteration loop only (D5)
            m_cv = self._check_update(m_vc)
            per_var = np.add.reduceat(m_cv[:, v_order], v_starts, axis=1)
            L_total = L + per_var                         # (B, n)
            hard = (L_total < 0).astype(np.uint8)
            parity = np.add.reduceat(hard[:, var_idx],
                                     self._c_starts, axis=1) % 2
            if not parity.any():
                break
            m_vc = L_total[:, var_idx] - m_cv

        hard = hard.reshape(batch_shape + (self.n,))
        if self.output == "codeword":
            return hard
        return hard[..., self.free_cols]
