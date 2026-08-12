from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector
from .utils import plot_alphabet

__all__ = ["SymbolMapper", "SymbolDemapper"]

if TYPE_CHECKING:  # matplotlib stays out of the import path (D36)
    from matplotlib.axes import Axes


@dataclass(slots=True)
class SymbolMapper(Processor):
    r"""Symbol mapper converting integer indices to constellation symbols.

    Signal Model
    ------------
    Each integer input :math:`x[n]` selects one symbol of the alphabet
    :math:`\mathcal{M} = \{s_0, s_1, \ldots, s_{M-1}\}`:

    .. math::

       y[n] = s_{x[n]}, \qquad x[n] \in \{0, 1, \ldots, M-1\}

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation alphabet :math:`\mathcal{M} = \{s_0, \ldots, s_{M-1}\}`
        as a 1-D array of :math:`M` complex symbols.
    name : str, optional, keyword-only
        Name of the symbol mapper instance. Default is ``"Symbol Mapper"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 3.

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> mapper = SymbolMapper(get_alphabet("QAM", 4))
    >>> y = mapper(np.array([0, 3, 2, 1]))
    >>> print(np.round(y, 2))
    [-0.71+0.71j  0.71-0.71j  0.71+0.71j -0.71-0.71j]
    """
    alphabet: np.ndarray
    name: str = field(default="Symbol Mapper", kw_only=True)

    def __post_init__(self):
        # accept anything array-like, so a Constellation (or a list) can
        # be handed to the block the same way an ndarray is
        self.alphabet = np.asarray(self.alphabet)

    def get_alphabet(self):
        return self.alphabet

    def plot(self, ax: Optional["Axes"] = None,
             title: str = "Symbol Constellation") -> "Axes":
        return plot_alphabet(self.alphabet, ax=ax, title=title)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = self.alphabet[X]
        return Y


@dataclass(slots=True)
class SymbolDemapper(Processor):
    r"""Symbol demapper: minimum-distance (hard) decisions or bit LLRs (soft).

    Signal Model
    ------------
    With ``soft=False`` (default), each input sample :math:`x[n]` is
    mapped to the index of the nearest symbol of the alphabet
    :math:`\mathcal{M} = \{s_0, s_1, \ldots, s_{M-1}\}`:

    .. math::

       y[n] = \arg \min_{m \in \{0, 1, \ldots, M-1\}} \left|x[n] - s_m\right|^2

    With ``soft=True``, the demapper outputs the max-log log-likelihood
    ratio of every bit :math:`i` of the symbol index (MSB first,
    :math:`k = \log_2 M` bits per symbol, decision D12):

    .. math::

       L_i[n] = \frac{1}{\sigma^2} \left(
       \min_{m \,:\, b_i(m) = 1} |x[n] - s_m|^2
       - \min_{m \,:\, b_i(m) = 0} |x[n] - s_m|^2 \right)

    so :math:`L_i > 0` favors bit 0 -- the convention expected by
    ``ViterbiDecoder(soft=True)``. This is the inverse operation of
    :class:`SymbolMapper`.

    Axes: *element-wise* -- applied pointwise, shape-agnostic; with
    ``soft=True`` the last axis grows from :math:`N` symbols to
    :math:`k N` LLRs (bits of a symbol contiguous, MSB first).

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation alphabet :math:`\mathcal{M} = \{s_0, \ldots, s_{M-1}\}`
        as a 1-D array of :math:`M` complex symbols; the bits of a symbol
        are the binary representation of its index.
    soft : bool, keyword-only
        Output bit LLRs instead of hard symbol decisions. Default False.
    sigma2 : float, keyword-only
        Noise variance :math:`\sigma^2` scaling the LLRs. Default 1.0
        (the scaling does not affect Viterbi decoding).
    name : str, optional, keyword-only
        Name of the symbol demapper instance. Default is ``"Symbol Demapper"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 4; P. Robertson, E. Villebrun,
    P. Hoeher, "A comparison of optimal and sub-optimal MAP decoding
    algorithms operating in the log domain," ICC 1995 (max-log
    approximation).

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> demapper = SymbolDemapper(get_alphabet("QAM", 4))
    >>> x = np.array([-0.7+0.7j, 0.7-0.7j, 0.7+0.7j, -0.7-0.7j])
    >>> print(demapper(x))
    [0 3 2 1]
    >>> soft = SymbolDemapper(get_alphabet("QAM", 4), soft=True)
    >>> print(np.round(soft(x[:2]), 2))  # LLR > 0 favors bit 0
    [ 1.98  1.98 -1.98 -1.98]
    """
    alphabet: np.ndarray
    soft: bool = field(default=False, kw_only=True)
    sigma2: float = field(default=1.0, kw_only=True)
    name: str = field(default="Symbol Demapper", kw_only=True)
    # precomputed bit patterns of the alphabet indices (parametric)
    _bits: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.alphabet = np.asarray(self.alphabet)
        M = len(self.alphabet)
        k = int(np.log2(M))
        if 2**k != M:
            raise ValueError(
                f"soft demapping needs an alphabet size that is a power of "
                f"two, got M={M}")
        indices = np.arange(M)
        # (M, k) bit table, MSB first -- same convention as sym_2_bin
        self._bits = ((indices[:, None] >> np.arange(k - 1, -1, -1)) & 1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not self.soft:
            s, _ = hard_projector(X, self.alphabet)
            return s

        d2 = np.abs(X[..., None] - self.alphabet) ** 2   # (..., M)
        k = self._bits.shape[1]
        llr = np.empty(X.shape + (k,))
        for i in range(k):
            mask1 = self._bits[:, i] == 1
            d2_bit0 = np.min(d2[..., ~mask1], axis=-1)
            d2_bit1 = np.min(d2[..., mask1], axis=-1)
            llr[..., i] = (d2_bit1 - d2_bit0) / self.sigma2
        return llr.reshape(X.shape[:-1] + (X.shape[-1] * k,))

