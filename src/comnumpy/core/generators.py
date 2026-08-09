import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from comnumpy.core.generics import Processor


@dataclass(slots=True)
class SymbolGenerator(Processor):
    r"""Generator of independent and identically distributed (IID) uniform integer symbols.

    Signal Model
    ------------
    Each output symbol :math:`y[n]` is drawn independently and uniformly
    from the integer alphabet :math:`\{0, 1, \ldots, M-1\}`:

    .. math::

        \Pr\left(y[n] = m\right) = \frac{1}{M}, \qquad
        m \in \{0, 1, \ldots, M-1\}

    This is a source block: the call argument is not an input signal
    :math:`x` but the requested output size.

    Axes: *element-wise* -- each symbol is drawn independently; the output
    shape is the requested size (int or tuple).

    Parameters
    ----------
    M : int
        Alphabet size :math:`M`; symbols are drawn from
        :math:`\{0, 1, \ldots, M-1\}`.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the generator instance. Default is ``"generator"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 1.

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
    """
    M: int
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="generator", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X: object) -> np.ndarray:
        if isinstance(X, int):
            size = (X,)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        Y = self.rng.integers(self.M, size=size)
        return Y


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
        if isinstance(X, int):
            size = (X,)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        scale = np.sqrt(self.sigma2/2)
        Y = self.rng.normal(0, scale=scale, size=size) + 1j*self.rng.normal(0, scale=scale, size=size)
        return Y
