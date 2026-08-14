import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from comnumpy.core.generics import Processor

__all__ = ["SymbolGenerator", "GaussianGenerator"]


@dataclass(slots=True)
class SymbolGenerator(Processor):
    r"""Generator of independent and identically distributed (IID) integer symbols.

    Signal Model
    ------------
    Each output symbol :math:`y[n]` is drawn independently from the
    integer alphabet :math:`\{0, 1, \ldots, M-1\}`, uniformly by default:

    .. math::

        \Pr\left(y[n] = m\right) = \frac{1}{M}, \qquad
        m \in \{0, 1, \ldots, M-1\}

    A source is not always uniform, though: probabilistic shaping sends
    inner constellation points more often than outer ones, and
    ``distribution=`` gives that source directly,

    .. math::

        \Pr\left(y[n] = m\right) = p_m

    which is what :func:`~comnumpy.core.shaping.maxwell_boltzmann`
    returns. Note what this block is *not*: an i.i.d. draw from
    :math:`P_X` is the idealization a distribution matcher approaches,
    not what it produces -- a matcher maps uniform bits onto a finite set
    of sequences, so its output is neither independent nor exactly
    :math:`P_X` on a finite block (see
    :class:`~comnumpy.core.shaping.DistributionMatcher`). Use this block
    to study what a distribution is worth, and the matcher to produce it.

    This is a source block: the call argument is not an input signal
    :math:`x` but the requested output size.

    Axes: *element-wise* -- each symbol is drawn independently; the output
    shape is the requested size (int or tuple).

    Parameters
    ----------
    M : int
        Alphabet size :math:`M`; symbols are drawn from
        :math:`\{0, 1, \ldots, M-1\}`.
    distribution : np.ndarray, optional, keyword-only
        Probabilities :math:`p_m`, of length :math:`M`. Default (None)
        is the uniform law.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the generator instance. Default is ``"generator"``.

    Raises
    ------
    ValueError
        If ``distribution`` does not have one non-negative probability
        per symbol, summing to one.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 1; G. Böcherer, F. Steiner, P. Schulte,
    IEEE Trans. Commun., vol. 63, no. 12, 2015 (non-uniform signaling).

    Examples
    --------
    >>> generator = SymbolGenerator(M=4, seed=42)
    >>> symbols_1D = generator(5)  # Generates a 1D array of size 5
    >>> print(symbols_1D)
    [0 3 2 1 1]
    >>> symbols_2D = generator((2, 3))  # Generates a 2D array of shape (2, 3)
    >>> print(symbols_2D)
    [[3 0 2]
     [0 0 2]]

    A shaped source, drawn from a Maxwell-Boltzmann law:

    >>> from comnumpy.core.shaping import maxwell_boltzmann
    >>> pam = np.arange(-3, 4, 2).astype(float)
    >>> shaped = SymbolGenerator(4, distribution=maxwell_boltzmann(
    ...     pam, entropy=1.5), seed=0)
    >>> print(np.bincount(shaped(10000)) / 10000)
    [0.0557 0.4433 0.4457 0.0553]
    """
    M: int
    distribution: Optional[np.ndarray] = field(default=None, kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="generator", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if self.distribution is not None:
            self.distribution = np.asarray(self.distribution,
                                           dtype=float).ravel()
            if self.distribution.size != self.M:
                raise ValueError(
                    f"the distribution has {self.distribution.size} "
                    f"probabilities and the alphabet {self.M} symbols -- "
                    f"one probability per symbol is expected.")
            if np.any(self.distribution < 0) or not np.isclose(
                    float(np.sum(self.distribution)), 1.0, atol=1e-9):
                raise ValueError(
                    f"a distribution is non-negative and sums to one, got "
                    f"a sum of {float(np.sum(self.distribution))} and a "
                    f"minimum of {float(np.min(self.distribution))}.")
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X: object) -> np.ndarray:
        # np.integer too: sizes come out of shape arithmetic. bool is an
        # int subclass and would slip through as 0 or 1.
        if isinstance(X, (int, np.integer)) and not isinstance(X, bool):
            size = (int(X),)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        if self.distribution is None:
            return self.rng.integers(self.M, size=size)
        return self.rng.choice(self.M, size=size, p=self.distribution)


@dataclass(slots=True)
class GaussianGenerator(Processor):
    r"""Generator of IID circularly-symmetric complex Gaussian samples.

    Signal Model
    ------------
    Each output sample :math:`y[n]` is drawn independently from a
    zero-mean circular complex Gaussian distribution of variance
    :math:`\sigma^2`:

    .. math::

        y[n] \sim \mathcal{CN}\left(0, \sigma^2\right)

    This is a source block: the call argument is not an input signal
    :math:`x` but the requested output size.

    Axes: *element-wise* -- each sample is drawn independently; the output
    shape is the requested size (int or tuple).

    Parameters
    ----------
    sigma2 : float
        Variance :math:`\sigma^2` of the complex Gaussian distribution.
        Default is 1.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the generator instance. Default is ``"gaussian_generator"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 2.

    Examples
    --------
    >>> generator = GaussianGenerator(sigma2=1.0, seed=42)
    >>> symbols_1D = generator(5)  # Generates a 1D array of 5 complex symbols
    >>> print(np.round(symbols_1D, 3))
    [ 0.215-0.921j -0.735+0.09j   0.531-0.224j  0.665-0.012j -1.38 -0.603j]
    >>> symbols_2D = generator((3, 3))  # Generates a 2D array of shape (3, 3)
    >>> symbols_2D.shape
    (3, 3)
    """
    sigma2: float = 1
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="gaussian_generator", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X: object) -> np.ndarray:
        # np.integer too: sizes come out of shape arithmetic. bool is an
        # int subclass and would slip through as 0 or 1.
        if isinstance(X, (int, np.integer)) and not isinstance(X, bool):
            size = (int(X),)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        scale = np.sqrt(self.sigma2/2)
        Y = self.rng.normal(0, scale=scale, size=size) + 1j*self.rng.normal(0, scale=scale, size=size)
        return Y
