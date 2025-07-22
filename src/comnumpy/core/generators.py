import numpy as np
from dataclasses import dataclass
from comnumpy.core.generics import Processor


@dataclass
class SymbolGenerator(Processor):
    r"""
    A generator for creating independent and identically distributed (IID) symbols.

    This class generates a sequence of integer randomly chosen uniformly from the set :math:`\{0, 1, ..., M-1\}`.
    

    Signal Model
    ------------

    Each symbol :math:`y[n]` is drawn independently from the same distribution :

    .. math ::

        p(y[n]=m) = \begin{cases}
            \frac{1}{M} & \text{if } m \in \{0, 1, \ldots, M-1 \}, \\
            0 & \text{otherwise}.
            \end{cases}

    Attributes
    ----------
    M : int
        The size of the alphabet.
    is_mimo : bool, optional
        Indicates whether the generator is used in a MIMO context (default: True).
    seed : int, optional
        Seed for the random number generator, for reproducibility.
        Default is None, which initializes the generator without a fixed seed.
    name : str, optional
        Name of the generator instance (default: "generator").

    Examples
    --------
    >>> generator = SymbolGenerator(M=4, seed=42)
    >>> symbols_1D = generator(5)  # Generates a 1D array of size 5
    >>> print(symbols_1D)
    [3 2 0 3 3]
    >>> symbols_2D = generator((3, 3))  # Generates a 2D array of shape (3, 3)
    >>> print(symbols_2D)
    [[2 1 2]
     [3 2 1]
     [0 0 2]]
    """
    M: int
    seed: int = None
    name: str = "generator"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X):
        if isinstance(X, int):
            size = (X,)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")
        
        Y = self.rng.integers(self.M, size=size)
        return Y


@dataclass
class GaussianGenerator(Processor):
    r"""
    A generator for creating Gaussian-distributed symbols.

    This class generates a sequence of symbols drawn from a Gaussian distribution with mean 0 and standard deviation `sigma`.

    Signal Model
    ------------

    Each symbol :math:`\mathbf{y}[n]` is drawn independently from a Gaussian distribution:

    .. math ::

        \mathbf{y}[n] \sim \mathcal{N}_c(0, \sigma^2)

    Attributes
    ----------
    sigma2 : float
        The variance of the Gaussian distribution.
    seed : int, optional
        Seed for the random number generator, for reproducibility.
        Default is None, which initializes the generator without a fixed seed.
    name : str, optional
        Name of the generator instance (default: "gaussian_generator").

    Examples
    --------
    >>> generator = GaussianGenerator(sigma2=1.0, seed=42)
    >>> symbols_1D = generator(5)  # Generates a 1D array of size 5
    >>> print(symbols_1D)
    [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]
    >>> symbols_2D = generator((3, 3))  # Generates a 2D array of shape (3, 3)
    >>> print(symbols_2D)
    [[ 0.49671415 -0.1382643   0.64768854]
     [ 1.52302986 -0.23415337  1.57921282]
     [ 0.76743473 -0.46947439  0.54256004]]
    """
    sigma2: float = 1
    seed: int = None
    name: str = "gaussian_generator"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X):
        if isinstance(X, int):
            size = (X,)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        scale = np.sqrt(self.sigma2/2)
        Y = self.rng.normal(0, scale=scale, size=size) + 1j*self.rng.normal(0, scale=scale, size=size)
        return Y
