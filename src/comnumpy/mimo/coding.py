r"""Space-time block codes: spending antennas on reliability or on rate.

A MIMO channel offers two things that cannot both be taken in full:
**diversity**, the number of independent fading coefficients an error
must defeat, and **multiplexing**, the number of symbols carried per
channel use. A space-time code is the choice of where to sit between
them, and this module describes codes in the one form that makes that
choice checkable.

**Every code here is a linear dispersion code.** A codeword occupying
:math:`T` channel uses on :math:`N_t` antennas is

.. math::

    \mathbf{G}(\mathbf{s}) = \sum_{k=1}^{K}
    \left( \mathbf{A}_k s_k + \mathbf{B}_k s_k^{*} \right)

with fixed matrices :math:`\mathbf{A}_k, \mathbf{B}_k` of size
:math:`N_t \times T`. Alamouti, the orthogonal designs of Tarokh *et
al.*, spatial multiplexing and the Golden code are all of this form; the
conjugate term is what lets an orthogonal design exist at all.

**Why that form and not a table of matrices.** Writing
:math:`s_k = u_k + \mathrm{j} v_k` makes the received block a *real*
linear function of the :math:`2K` real unknowns,

.. math::

    \begin{bmatrix} \Re\{\mathrm{vec}\,\mathbf{Y}\} \\
                    \Im\{\mathrm{vec}\,\mathbf{Y}\} \end{bmatrix}
    = \mathbf{M}(\mathbf{H})
      \begin{bmatrix} \mathbf{u} \\ \mathbf{v} \end{bmatrix}
      + \text{noise}

and :math:`\mathbf{M}(\mathbf{H})` -- the *equivalent channel*,
:meth:`SpaceTimeCode.equivalent_channel` -- is where every property of
the code becomes a statement about a matrix. A design is orthogonal
exactly when

.. math::

    \mathbf{M}^{\mathsf{T}} \mathbf{M}
    = c \left\|\mathbf{H}\right\|_F^2 \, \mathbf{I}_{2K}

which is checked **at construction** for every code declaring itself
orthogonal (D20), so a mistyped sign in a dispersion matrix cannot ship.
That same identity is why an orthogonal code is decoded by a matched
filter and not by a search: the maximum-likelihood estimate is
:math:`\mathbf{M}^{\mathsf{T}} \mathbf{y} / (c\|\mathbf{H}\|_F^2)`,
exactly. The constant :math:`c` is a property of the design -- 1 for
Alamouti, 2 for the rate-1/2 designs that repeat each symbol over a
conjugated half -- and is measured with the identity rather than assumed.

Codes are taken from a registry, as constellations are taken from
:func:`~comnumpy.core.utils.get_alphabet`::

    code = get_code("alamouti")
    chain = Sequential([SymbolGenerator(4), SymbolMapper(alphabet),
                        SpaceTimeEncoder(code), FlatMIMOChannel(H), ...])

References
----------
S. M. Alamouti, "A simple transmit diversity technique for wireless
communications", IEEE J. Sel. Areas Commun., vol. 16, no. 8,
pp. 1451-1458, Oct. 1998; V. Tarokh, H. Jafarkhani, A. R. Calderbank,
"Space-time block codes from orthogonal designs", IEEE Trans. Inf.
Theory, vol. 45, no. 5, pp. 1456-1467, Jul. 1999; B. Hassibi,
B. M. Hochwald, "High-rate codes that are linear in space and time",
IEEE Trans. Inf. Theory, vol. 48, no. 7, pp. 1804-1824, Jul. 2002
(linear dispersion); J.-C. Belfiore, G. Rekaya, E. Viterbo, "The Golden
code: a 2x2 full-rate space-time code with nonvanishing determinants",
IEEE Trans. Inf. Theory, vol. 51, no. 4, pp. 1432-1436, Apr. 2005;
V. Tarokh, N. Seshadri, A. R. Calderbank, "Space-time codes for high
data rate wireless communication: performance criterion and code
construction", IEEE Trans. Inf. Theory, vol. 44, no. 2, pp. 744-765,
Mar. 1998 (the rank and determinant criteria).

Examples
--------
>>> import numpy as np
>>> code = get_code("alamouti")
>>> code.n_tx, code.n_slots, code.n_symbols, code.rate
(2, 2, 2, 1.0)
>>> print(np.round(code.encode(np.array([1 + 0j, 2 + 1j])), 3))
[[ 1.+0.j -2.+1.j]
 [ 2.+1.j  1.+0.j]]
>>> code.is_orthogonal
True
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

__all__ = [
    "SpaceTimeCode", "get_code", "register_code", "available_codes",
    "SpaceTimeEncoder", "SpaceTimeDecoder", "coding_gain",
]

_CODE_REGISTRY: dict[str, Callable[..., "SpaceTimeCode"]] = {}

# How the codes are printed in their papers: one list of terms per entry
# of the codeword, each term being (symbol index, coefficient, is it the
# conjugate of that symbol).
Term = tuple[int, complex, bool]
CodewordTable = Sequence[Sequence[Sequence[Term]]]

# The orthogonality identity is checked in floating point; the matrices
# hold halves and 1/sqrt(2), so the residual is a few ulps.
_ORTHOGONALITY_ATOL = 1e-12


@dataclass(frozen=True, slots=True)
class SpaceTimeCode:
    r"""A space-time block code, as its linear dispersion matrices.

    Signal Model
    ------------
    The codeword transmitted over :math:`T` channel uses is

    .. math::

        \mathbf{G}(\mathbf{s}) = \sum_{k=1}^{K}
        \left(\mathbf{A}_k s_k + \mathbf{B}_k s_k^{*}\right)
        \in \mathbb{C}^{N_t \times T}

    and the receiver observes
    :math:`\mathbf{Y} = \mathbf{H}\mathbf{G}(\mathbf{s}) + \mathbf{V}`,
    with :math:`\mathbf{V}` the noise (:math:`\mathbf{B}_k` being taken
    by the conjugate dispersion matrices above).
    The **rate** is :math:`K/T` symbols per channel use, the **diversity
    order** is the minimum rank of
    :math:`\mathbf{G}(\mathbf{s}) - \mathbf{G}(\mathbf{s}')` over
    distinct symbol vectors (the rank criterion), and a code reaching
    :math:`N_t` there is *full diversity*.

    Axes: *declared axis* -- codewords are ``(n_tx, n_slots)``, antennas
    on axis -2, as everywhere else in the library (D2).

    Parameters
    ----------
    name : str
        Registry name.
    dispersion_direct : np.ndarray
        The matrices :math:`\mathbf{A}_k`, stacked as
        ``(K, n_tx, n_slots)``.
    dispersion_conjugate : np.ndarray
        The matrices :math:`\mathbf{B}_k`, same shape. Zero for a code
        that does not conjugate, such as spatial multiplexing or the
        Golden code.
    orthogonal : bool, optional, keyword-only
        Declares the design orthogonal. It is **verified** at
        construction, not trusted (D20).
    reference : str, optional, keyword-only
        Where the code comes from.

    Raises
    ------
    ShapeError
        If the two dispersion stacks disagree in shape.
    ValueError
        If ``orthogonal=True`` and the design is not orthogonal.

    References
    ----------
    B. Hassibi, B. M. Hochwald, IEEE Trans. Inf. Theory 48(7), 2002.

    Examples
    --------
    >>> zero = np.zeros((1, 1, 1))
    >>> identity = np.ones((1, 1, 1))
    >>> trivial = SpaceTimeCode("siso", identity, zero, orthogonal=True)
    >>> trivial.rate
    1.0
    """

    name: str
    dispersion_direct: np.ndarray
    dispersion_conjugate: np.ndarray
    orthogonal: bool = field(default=False, kw_only=True)
    reference: str = field(default="", kw_only=True)
    # measured at construction for an orthogonal design (D20): the c of
    # M^T M = c |H|_F^2 I, which the matched filter divides by
    orthogonality_gain: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        direct = np.asarray(self.dispersion_direct, dtype=complex)
        conjugate = np.asarray(self.dispersion_conjugate, dtype=complex)
        if direct.shape != conjugate.shape:
            raise ShapeError(
                f"the two dispersion stacks describe the same code, so "
                f"they have the same shape: got {direct.shape} for the "
                f"direct term and {conjugate.shape} for the conjugate "
                f"one. Pass zeros for a code that does not conjugate.")
        if direct.ndim != 3:
            raise ShapeError(
                f"dispersion matrices are stacked as (K, n_tx, n_slots), "
                f"got an array of shape {direct.shape}.")
        object.__setattr__(self, "dispersion_direct", direct)
        object.__setattr__(self, "dispersion_conjugate", conjugate)
        if self.orthogonal:
            object.__setattr__(self, "orthogonality_gain",
                               self._verify_orthogonality())

    def _verify_orthogonality(self) -> float:
        """D20: a declared orthogonal design proves it before it ships.

        The gain ``c`` is a property of the design, not something to
        assume: a codeword repeating each symbol over two halves, as
        Tarokh's rate-1/2 designs do, collects twice the energy Alamouti
        does. It is measured here and reused by the decoder, so the two
        cannot disagree.
        """
        rng = np.random.default_rng(0)
        gains = []
        for n_rx in (1, 2, 3):
            shape = (n_rx, self.n_tx)
            channel = (rng.normal(size=shape) + 1j * rng.normal(size=shape))
            matrix = self.equivalent_channel(channel)
            gram = matrix.T @ matrix
            energy = float(np.sum(np.abs(channel) ** 2))
            gain = float(gram[0, 0]) / energy
            residual = float(np.max(np.abs(
                gram - gain * energy * np.eye(2 * self.n_symbols))))
            if residual > _ORTHOGONALITY_ATOL * max(energy, 1.0):
                raise ValueError(
                    f"the code {self.name!r} is declared orthogonal but "
                    f"its equivalent channel has M^T M off a multiple of "
                    f"the identity by {residual:.3e} with {n_rx} receive "
                    f"antenna(s). An orthogonal design must satisfy that "
                    f"identity exactly: check the signs and the "
                    f"conjugations of its dispersion matrices.")
            gains.append(gain)
        if max(gains) - min(gains) > _ORTHOGONALITY_ATOL * max(gains):
            raise ValueError(
                f"the orthogonality gain of {self.name!r} depends on the "
                f"channel ({gains}), which no orthogonal design does.")
        return gains[0]

    @property
    def n_symbols(self) -> int:
        """:math:`K`, symbols carried by one codeword."""
        return int(self.dispersion_direct.shape[0])

    @property
    def n_tx(self) -> int:
        """:math:`N_t`, transmit antennas."""
        return int(self.dispersion_direct.shape[1])

    @property
    def n_slots(self) -> int:
        """:math:`T`, channel uses spanned by one codeword."""
        return int(self.dispersion_direct.shape[2])

    @property
    def rate(self) -> float:
        """:math:`K/T`, symbols per channel use."""
        return self.n_symbols / self.n_slots

    @property
    def is_orthogonal(self) -> bool:
        """Whether the design was declared -- and verified -- orthogonal."""
        return self.orthogonal

    def info(self) -> dict[str, Any]:
        r"""What the code is, as a dictionary ready to print.

        The same service the channels and
        :class:`~comnumpy.core.utils.Constellation` offer: the facts a
        page would otherwise restate in prose, read off the object that
        holds them.

        Returns
        -------
        dict
            ``name``, ``n_tx``, ``n_slots``, ``n_symbols``, ``rate``,
            ``orthogonal``, ``orthogonality_gain`` -- the constant
            :math:`c` of :math:`M^T M = c \|H\|_F^2 I` that the matched
            filter divides by, zero for a non-orthogonal design -- and
            ``reference``.

        Examples
        --------
        >>> info = get_code("alamouti").info()
        >>> info["n_tx"], info["rate"], info["orthogonal"]
        (2, 1.0, True)
        """
        return {"name": self.name,
                "n_tx": self.n_tx,
                "n_slots": self.n_slots,
                "n_symbols": self.n_symbols,
                "rate": self.rate,
                "orthogonal": self.orthogonal,
                "orthogonality_gain": self.orthogonality_gain,
                "reference": self.reference}

    def encode(self, symbols: np.ndarray) -> np.ndarray:
        r"""Build codewords from symbols.

        Parameters
        ----------
        symbols : np.ndarray
            ``(..., K)`` or ``(..., n * K)``: the last axis is read in
            blocks of :math:`K`.

        Returns
        -------
        np.ndarray
            ``(..., n_tx, n * n_slots)``, antennas on axis -2.
        """
        values = np.asarray(symbols, dtype=complex)
        if values.shape[-1] % self.n_symbols:
            raise ShapeError(
                f"{self.name!r} carries {self.n_symbols} symbols per "
                f"codeword and got {values.shape[-1]} along the last "
                f"axis, which is not a multiple of it.")
        blocks = values.reshape(values.shape[:-1] + (-1, self.n_symbols))
        # (..., n, K) x (K, n_tx, T) -> (..., n, n_tx, T)
        words = (np.einsum("...k,kij->...ij", blocks, self.dispersion_direct)
                 + np.einsum("...k,kij->...ij", np.conj(blocks),
                             self.dispersion_conjugate))
        # concatenate the codewords along time
        moved = np.moveaxis(words, -2, -3)          # (..., n_tx, n, T)
        return moved.reshape(moved.shape[:-2] + (-1,))

    def equivalent_channel(self, H: np.ndarray) -> np.ndarray:
        r"""The real matrix :math:`\mathbf{M}(\mathbf{H})` of the module docstring.

        Parameters
        ----------
        H : np.ndarray
            Channel matrix, ``(n_rx, n_tx)``.

        Returns
        -------
        np.ndarray
            Real matrix of shape ``(2 * n_rx * n_slots, 2 * K)``, mapping
            :math:`[\mathbf{u}; \mathbf{v}]` onto the stacked real and
            imaginary parts of the received block.
        """
        channel = np.asarray(H, dtype=complex)
        if channel.ndim != 2 or channel.shape[1] != self.n_tx:
            raise ShapeError(
                f"{self.name!r} transmits on {self.n_tx} antennas, so the "
                f"channel is (n_rx, {self.n_tx}); got {channel.shape}.")
        # G is real-linear in (u, v) with these complex generators
        for_real = self.dispersion_direct + self.dispersion_conjugate
        for_imag = 1j * (self.dispersion_direct - self.dispersion_conjugate)
        columns = []
        for generators in (for_real, for_imag):
            for index in range(self.n_symbols):
                block = channel @ generators[index]       # (n_rx, T)
                columns.append(block.reshape(-1))
        complex_matrix = np.stack(columns, axis=1)
        return np.vstack([np.real(complex_matrix),
                          np.imag(complex_matrix)])

    def minimum_rank(self, alphabet: np.ndarray) -> int:
        r"""Diversity order by the rank criterion, over one alphabet.

        The minimum rank of :math:`\mathbf{G}(\mathbf{s}) -
        \mathbf{G}(\mathbf{s}')` over distinct symbol vectors. It equals
        :math:`N_t` for a full-diversity code and 1 for spatial
        multiplexing, and the error probability of the code decays as
        :math:`\mathrm{SNR}^{-r N_r}` with :math:`r` that rank.

        Exhaustive over :math:`|\mathcal{A}|^{K}` pairs of a *difference*
        vector, so it is meant for small alphabets and short codes.
        """
        points = np.asarray(alphabet, dtype=complex).ravel()
        differences = np.unique(np.round(
            (points[:, None] - points[None, :]).ravel(), 12))
        smallest = self.n_tx
        for pattern in _nonzero_patterns(differences, self.n_symbols):
            word = self.encode(pattern)
            smallest = min(smallest,
                           int(np.linalg.matrix_rank(word, tol=1e-9)))
            if smallest == 1:
                break
        return smallest


def _nonzero_patterns(values: np.ndarray, length: int):
    """Every non-zero vector of ``length`` entries drawn from ``values``."""
    from itertools import product
    for combination in product(values, repeat=length):
        pattern = np.array(combination, dtype=complex)
        if np.any(pattern != 0):
            yield pattern


def coding_gain(code: SpaceTimeCode, alphabet: np.ndarray) -> float:
    r"""Determinant criterion: the coding gain of a full-diversity code.

    Signal Model
    ------------
    When the rank criterion is met with full rank :math:`N_t`, the
    pairwise error probability is governed by

    .. math::

        \delta = \min_{\mathbf{s} \neq \mathbf{s}'}
        \det\left[ \left(\mathbf{G} - \mathbf{G}'\right)
        \left(\mathbf{G} - \mathbf{G}'\right)^{H} \right]

    and a code with :math:`\delta` bounded away from zero as the
    constellation grows is said to have a *non-vanishing determinant* --
    the property the Golden code was built for.

    Axes: *element-wise* -- takes a code and a 1-D alphabet.

    Parameters
    ----------
    code : SpaceTimeCode
        The code to measure.
    alphabet : np.ndarray
        Constellation the symbols are drawn from.

    Returns
    -------
    float
        :math:`\delta`, zero if the code is not full diversity.

    References
    ----------
    V. Tarokh, N. Seshadri, A. R. Calderbank, IEEE Trans. Inf. Theory
    44(2), 1998 (determinant criterion); Belfiore, Rekaya, Viterbo,
    IEEE Trans. Inf. Theory 51(4), 2005.

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> qpsk = get_alphabet("QAM", 4)
    >>> print(round(coding_gain(get_code("alamouti"), qpsk), 6))
    4.0
    """
    points = np.asarray(alphabet, dtype=complex).ravel()
    differences = np.unique(np.round(
        (points[:, None] - points[None, :]).ravel(), 12))
    smallest = np.inf
    for pattern in _nonzero_patterns(differences, code.n_symbols):
        word = code.encode(pattern)
        gram = word @ np.conj(word.T)
        smallest = min(smallest, float(np.abs(np.linalg.det(gram))))
    return float(smallest)


def register_code(name: str) -> Callable[[Callable[..., SpaceTimeCode]],
                                         Callable[..., SpaceTimeCode]]:
    """Register a code factory; users can add their own.

    Parameters
    ----------
    name : str
        Name the factory answers to in :func:`get_code`.

    Examples
    --------
    >>> @register_code("repetition-2")
    ... def _repetition():
    ...     direct = np.zeros((1, 2, 2), dtype=complex)
    ...     direct[0, 0, 0] = direct[0, 1, 1] = 1
    ...     return SpaceTimeCode("repetition-2", direct,
    ...                          np.zeros_like(direct))
    >>> get_code("repetition-2").rate
    0.5
    """
    def decorator(func: Callable[..., SpaceTimeCode]
                  ) -> Callable[..., SpaceTimeCode]:
        _CODE_REGISTRY[name] = func
        return func
    return decorator


def available_codes() -> list[str]:
    """Names :func:`get_code` answers to.

    Examples
    --------
    >>> "alamouti" in available_codes()
    True
    """
    return sorted(_CODE_REGISTRY)


def get_code(name: str, **kwargs: object) -> SpaceTimeCode:
    """Return a catalog code by name, as ``get_alphabet`` returns a constellation.

    Parameters
    ----------
    name : str
        Entry name; see :func:`available_codes`.
    **kwargs
        Forwarded to the entry factory.

    Returns
    -------
    SpaceTimeCode
        The code.

    Raises
    ------
    KeyError
        If no entry answers to that name.

    Examples
    --------
    >>> get_code("alamouti").n_tx
    2
    >>> get_code("spatial-multiplexing", n_tx=4).rate
    4.0
    """
    if name not in _CODE_REGISTRY:
        raise KeyError(
            f"unknown space-time code {name!r}, known codes are "
            f"{available_codes()}. Register your own with "
            f"@register_code, or build a SpaceTimeCode from its "
            f"dispersion matrices directly.")
    return _CODE_REGISTRY[name](**kwargs)


def _from_table(name: str, table: CodewordTable, n_symbols: int, *,
                orthogonal: bool = False,
                reference: str = "") -> SpaceTimeCode:
    """Build the dispersion stacks from a codeword written entry by entry.

    ``table[i][t]`` lists the terms of :math:`G_{i,t}` as
    ``(symbol index, coefficient, conjugated)``, which is how these
    codes are printed in their papers.
    """
    n_tx, n_slots = len(table), len(table[0])
    direct = np.zeros((n_symbols, n_tx, n_slots), dtype=complex)
    conjugate = np.zeros_like(direct)
    for row, entries in enumerate(table):
        for slot, terms in enumerate(entries):
            for index, coefficient, is_conjugate in terms:
                target = conjugate if is_conjugate else direct
                target[index, row, slot] += coefficient
    return SpaceTimeCode(name, direct, conjugate, orthogonal=orthogonal,
                         reference=reference)


@register_code("alamouti")
def _alamouti() -> SpaceTimeCode:
    r"""The 2x1 scheme, rate 1, full diversity, exactly ML by matched filter."""
    table: list[list[list[Term]]] = [
        [[(0, 1, False)], [(1, -1, True)]],
        [[(1, 1, False)], [(0, 1, True)]],
    ]
    return _from_table(
        "alamouti", table, 2, orthogonal=True,
        reference="S. M. Alamouti, IEEE JSAC 16(8), 1998")


@register_code("ostbc-3-1/2")
def _ostbc_g3() -> SpaceTimeCode:
    """Tarokh's G3: 3 antennas, rate 1/2, real and conjugate halves."""
    signs = [
        [(0, 1), (1, -1), (2, -1), (3, -1)],
        [(1, 1), (0, 1), (3, 1), (2, -1)],
        [(2, 1), (3, -1), (0, 1), (1, 1)],
    ]
    table: list[list[list[Term]]] = [
        [[(index, complex(sign), False)] for index, sign in row]
        + [[(index, complex(sign), True)] for index, sign in row]
        for row in signs]
    return _from_table(
        "ostbc-3-1/2", table, 4, orthogonal=True,
        reference="V. Tarokh et al., IEEE Trans. IT 45(5), 1999, G_3")


@register_code("ostbc-4-1/2")
def _ostbc_g4() -> SpaceTimeCode:
    """Tarokh's G4: 4 antennas, rate 1/2."""
    signs = [
        [(0, 1), (1, -1), (2, -1), (3, -1)],
        [(1, 1), (0, 1), (3, 1), (2, -1)],
        [(2, 1), (3, -1), (0, 1), (1, 1)],
        [(3, 1), (2, 1), (1, -1), (0, 1)],
    ]
    table: list[list[list[Term]]] = [
        [[(index, complex(sign), False)] for index, sign in row]
        + [[(index, complex(sign), True)] for index, sign in row]
        for row in signs]
    return _from_table(
        "ostbc-4-1/2", table, 4, orthogonal=True,
        reference="V. Tarokh et al., IEEE Trans. IT 45(5), 1999, G_4")


def _rate_three_quarter_rows() -> list[list[list[Term]]]:
    r"""The three rows shared by Tarokh's H_3 and H_4.

    The last two entries mix a symbol with its conjugate:
    :math:`(-x_1 - x_1^{*} + x_2 - x_2^{*})/2` is
    :math:`-\Re\{x_1\} + \mathrm{j}\Im\{x_2\}`, which is why these
    designs need the conjugate dispersion term.
    """
    half = 1 / np.sqrt(2)
    return [
        [[(0, 1, False)], [(1, -1, True)],
         [(2, half, True)], [(2, half, True)]],
        [[(1, 1, False)], [(0, 1, True)],
         [(2, half, True)], [(2, -half, True)]],
        [[(2, half, False)], [(2, half, False)],
         [(0, -0.5, False), (0, -0.5, True),
          (1, 0.5, False), (1, -0.5, True)],
         [(1, 0.5, False), (1, 0.5, True),
          (0, 0.5, False), (0, -0.5, True)]],
    ]


@register_code("ostbc-3-3/4")
def _ostbc_h3() -> SpaceTimeCode:
    """Tarokh's H3: 3 antennas, rate 3/4 -- three symbols in four slots."""
    return _from_table(
        "ostbc-3-3/4", _rate_three_quarter_rows(), 3, orthogonal=True,
        reference="V. Tarokh et al., IEEE Trans. IT 45(5), 1999, H_3")


@register_code("ostbc-4-3/4")
def _ostbc_h4() -> SpaceTimeCode:
    r"""H4: 4 antennas, rate 3/4, i.e. H3 with a fourth antenna.

    Provenance, stated because it is not a plain transcription: the
    three first rows are H3 as printed in the paper, and the fourth row
    is the completion its structure allows -- :math:`\pm x_3/\sqrt{2}`
    then two half-sums of :math:`x_1, x_2` and their conjugates. Which
    signs those halves carry is *determined*: over the 256 sign patterns
    of that structure, exactly two make the design orthogonal, and they
    differ by a global sign on the row, hence describe the same code.
    The orthogonality identity is therefore what fixes the convention
    here, not a reading of the table.
    """
    half = 1 / np.sqrt(2)
    fourth: list[list[Term]] = [
        [(2, half, False)], [(2, -half, False)],
        [(0, 0.5, False), (0, -0.5, True),
         (1, -0.5, False), (1, -0.5, True)],
        [(0, -0.5, False), (0, -0.5, True),
         (1, -0.5, False), (1, 0.5, True)],
    ]
    return _from_table(
        "ostbc-4-3/4", _rate_three_quarter_rows() + [fourth], 3,
        orthogonal=True,
        reference="V. Tarokh et al., IEEE Trans. IT 45(5), 1999, H_4 "
                  "(sign convention fixed by the orthogonality identity)")


@register_code("spatial-multiplexing")
def _spatial_multiplexing(n_tx: int = 2) -> SpaceTimeCode:
    """V-BLAST: one symbol per antenna per slot. Rate n_tx, no diversity."""
    direct = np.zeros((n_tx, n_tx, 1), dtype=complex)
    for index in range(n_tx):
        direct[index, index, 0] = 1.0
    return SpaceTimeCode(
        "spatial-multiplexing", direct, np.zeros_like(direct),
        reference="G. J. Foschini, Bell Labs Tech. J. 1(2), 1996")


@register_code("golden")
def _golden() -> SpaceTimeCode:
    r"""The 2x2 full-rate, full-diversity code with non-vanishing determinant."""
    theta = (1 + np.sqrt(5)) / 2
    theta_bar = (1 - np.sqrt(5)) / 2
    alpha = 1 + 1j * (1 - theta)
    alpha_bar = 1 + 1j * (1 - theta_bar)
    scale = 1 / np.sqrt(5)
    table: list[list[list[Term]]] = [
        [[(0, scale * alpha, False), (1, scale * alpha * theta, False)],
         [(2, scale * alpha, False), (3, scale * alpha * theta, False)]],
        [[(2, 1j * scale * alpha_bar, False),
          (3, 1j * scale * alpha_bar * theta_bar, False)],
         [(0, scale * alpha_bar, False),
          (1, scale * alpha_bar * theta_bar, False)]],
    ]
    return _from_table(
        "golden", table, 4,
        reference="Belfiore, Rekaya, Viterbo, IEEE Trans. IT 51(4), 2005")


@dataclass(slots=True)
class SpaceTimeEncoder(Processor):
    r"""Map a symbol stream onto antennas and time slots.

    Signal Model
    ------------
    Blocks of :math:`K` symbols become codewords
    :math:`\mathbf{G}(\mathbf{s})` of :math:`N_t \times T`, concatenated
    along time. A stream of :math:`N` symbols therefore leaves as
    :math:`(N_t, N T / K)` samples: the code *changes the number of
    channel uses*, by the factor :math:`T/K = 1/\text{rate}`.

    Axes: *declared axis* -- output carries antennas on axis -2 (D2), so
    it feeds :class:`~comnumpy.mimo.channels.FlatMIMOChannel` directly.

    Parameters
    ----------
    code : SpaceTimeCode
        The code, usually from :func:`get_code`.
    name : str, optional, keyword-only
        Block name.

    Raises
    ------
    ShapeError
        If the symbol count is not a multiple of :math:`K`.

    References
    ----------
    S. M. Alamouti, IEEE JSAC 16(8), 1998; V. Tarokh et al., IEEE Trans.
    Inf. Theory 45(5), 1999.

    Examples
    --------
    >>> encoder = SpaceTimeEncoder(get_code("alamouti"))
    >>> symbols = np.array([1 + 0j, 2 + 1j, -1 + 0j, 0 + 1j])
    >>> encoder(symbols).shape                      # 4 symbols -> 2x4
    (2, 4)
    """

    code: SpaceTimeCode
    name: str = field(default="space-time encoder", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.code.encode(np.asarray(X))


@dataclass(slots=True)
class SpaceTimeDecoder(Processor):
    r"""Recover the symbols of an orthogonal design, exactly.

    Signal Model
    ------------
    With the equivalent channel :math:`\mathbf{M}(\mathbf{H})` of the
    module docstring, the received block reads
    :math:`\mathbf{y} = \mathbf{M}[\mathbf{u};\mathbf{v}] + \mathbf{b}`
    with white real noise. When the design is orthogonal,
    :math:`\mathbf{M}^{\mathsf{T}}\mathbf{M} = c\|\mathbf{H}\|_F^2
    \mathbf{I}`, so the least-squares solution is a **matched filter**

    .. math::

        \begin{bmatrix}\widehat{\mathbf{u}} \\ \widehat{\mathbf{v}}
        \end{bmatrix}
        = \frac{\mathbf{M}^{\mathsf{T}} \mathbf{y}}
               {c \left\|\mathbf{H}\right\|_F^2}

    and it is *the* maximum-likelihood estimate, not an approximation of
    it: this is the whole point of orthogonal designs, and the reason
    they cost :math:`O(K)` instead of :math:`O(|\mathcal{A}|^K)`. Each
    symbol comes out with an SNR multiplied by
    :math:`c\|\mathbf{H}\|_F^2`, i.e. with diversity :math:`N_t N_r`;
    the constant :math:`c` belongs to the design and is measured when it
    is built, not assumed.

    A non-orthogonal code has no such shortcut, and this block refuses it
    rather than returning a zero-forcing answer under a name that
    promises optimality; :meth:`SpaceTimeCode.equivalent_channel` hands
    the problem to the detectors of
    :mod:`comnumpy.mimo.detectors` instead.

    Axes: *declared axis* -- input carries antennas on axis -2.

    Parameters
    ----------
    code : SpaceTimeCode
        The code used at the transmitter; must be orthogonal.
    H : np.ndarray, optional, keyword-only
        Channel matrix ``(n_rx, n_tx)``, assumed known at the receiver.
    name : str, optional, keyword-only
        Block name.

    Raises
    ------
    ValueError
        If the code is not orthogonal, or if ``H`` was not set.
    ShapeError
        If the number of received samples is not a multiple of the
        code's ``n_slots``.

    References
    ----------
    V. Tarokh, H. Jafarkhani, A. R. Calderbank, IEEE Trans. Inf. Theory
    45(5), 1999, Section III (linear decoding of orthogonal designs).

    Examples
    --------
    >>> code = get_code("alamouti")
    >>> H = np.array([[1 + 0j, 0.5 - 0.2j]])
    >>> symbols = np.array([1 + 1j, -1 + 0j])
    >>> received = H @ code.encode(symbols)
    >>> print(np.round(SpaceTimeDecoder(code, H=H)(received), 6) + 0.0)
    [ 1.+1.j -1.+0.j]
    """

    code: SpaceTimeCode
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    name: str = field(default="space-time decoder", kw_only=True)

    def __post_init__(self) -> None:
        if not self.code.is_orthogonal:
            raise ValueError(
                f"SpaceTimeDecoder decodes orthogonal designs by matched "
                f"filtering, which is exactly ML for them, and "
                f"{self.code.name!r} is not one. Build its equivalent "
                f"channel with code.equivalent_channel(H) and hand it to "
                f"a detector of comnumpy.mimo.detectors.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.H is None:
            raise ValueError(
                f"SpaceTimeDecoder needs the channel matrix H and none "
                f"was set. Got H=None, expected an (n_rx, {self.code.n_tx})"
                f" array -- pass it at construction or assign "
                f"decoder.H once the channel is known.")
        received = np.asarray(X, dtype=complex)
        code = self.code
        if received.shape[-1] % code.n_slots:
            raise ShapeError(
                f"{code.name!r} spans {code.n_slots} channel uses per "
                f"codeword and got {received.shape[-1]} samples, which "
                f"is not a multiple of it.")
        matrix = code.equivalent_channel(self.H)
        energy = (code.orthogonality_gain
                  * float(np.sum(np.abs(np.asarray(self.H)) ** 2)))
        n_words = received.shape[-1] // code.n_slots
        blocks = received.reshape(received.shape[:-1]
                                  + (n_words, code.n_slots))
        blocks = np.moveaxis(blocks, -2, -3)       # (..., n, n_rx, T)
        flat = blocks.reshape(blocks.shape[:-2] + (-1,))
        stacked = np.concatenate([np.real(flat), np.imag(flat)], axis=-1)
        estimate = stacked @ matrix / energy       # (..., n, 2K)
        half = code.n_symbols
        symbols = estimate[..., :half] + 1j * estimate[..., half:]
        return symbols.reshape(symbols.shape[:-2] + (-1,))
