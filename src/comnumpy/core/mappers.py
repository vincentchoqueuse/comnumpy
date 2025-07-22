import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector
from .utils import plot_alphabet


@dataclass
class SymbolMapper(Processor):
    r"""
    Symbol Mapper for converting digital data to symbols based on a predefined alphabet.

    This class maps an array of integers (representing digital data) to complex symbols according to a specified alphabet.
    It's used in digital communication systems where digital bits need to be mapped to symbols for modulation.

    Signal Model
    ------------

    .. math::

       y[n] = f_{\mathcal{M}}(x[n])

    where :
     
    * :math:`f_{\mathcal{M}}(.)` is a mapper function that converts an integer :math:`\{0, \cdots, M-1\}` into a symbol belonging to an alphabet :math:`\mathcal{M}`,
    * :math:`\mathcal{M}=\{s_0, \cdots, s_{M-1}\}` corresponds to the alphabet.

    Attributes
    ----------
    alphabet : np.ndarray
        An array of complex symbols representing the modulation alphabet.
    name : str
        Name of the symbol mapper instance. Default is "SymbolMapper".

    Examples
    --------
    >>> from comnumpy.core.generators import SymbolGenerator
    >>> from comnumpy.core.mappers import SymbolMapper
    >>> from comnumpy.core.utils import get_alphabet
    >>> M = 4
    >>> alphabet = get_alphabet("QAM", M)
    >>> generator = SymbolGenerator(M=M, seed=42)
    >>> mapper = SymbolMapper(alphabet)
    >>> input = generator(5)
    >>> print(input)
    [0 3 2 1 1]
    >>> symbols = mapper(input)
    >>> print(symbols)
    [-0.70710678+0.70710678j  0.70710678-0.70710678j  0.70710678+0.70710678j -0.70710678-0.70710678j -0.70710678-0.70710678j]
    """
    alphabet: np.ndarray
    is_mimo: bool = True
    name: str = "Symbol Mapper"

    def get_alphabet(self):
        return self.alphabet

    def plot(self, num=None, title="Symbol Constellation"):
        plot_alphabet(self.alphabet, num=num, title=title)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = self.alphabet[X]
        return Y


@dataclass
class SymbolDemapper(Processor):
    r"""
    Symbol Demapper for converting symbols to digital data based on a predefined alphabet.

    This class demaps complex symbols to the nearest symbols in a specified alphabet, effectively performing the inverse operation of a symbol mapper.

    Signal Model
    ------------

    .. math::

       y[n] = \mathcal{P}_{\mathcal{M}}(x[n]) = \arg \min_{m \in \{0, \cdots, M-1\}} |x[n]-s_m|^2

    * :math:`\mathcal{P}_{\mathcal{M}}(.)` corresponds to the orthogonal projector into the constellation :math:`\mathcal{M}`
    * :math:`\mathcal{M}=\{s_0, \cdots, s_{M-1}\}` corresponds to the alphabet.


    Attributes
    ----------
    alphabet : np.ndarray
        An array of complex symbols representing the modulation alphabet.
    name : str
        Name of the symbol demapper instance. Default is "SymbolDemapper".

    Examples
    --------

    >>> import numpy as np
    >>> from comnumpy.core.mappers import SymbolDemapper
    >>> from comnumpy.core.utils import get_alphabet
    >>> M = 4
    >>> alphabet = get_alphabet("QAM", M)
    >>> demapper = SymbolDemapper(alphabet)
    >>> symbols = np.array([-0.70710678+0.70710678j, 0.70710678-0.70710678j, 0.70710678+0.70710678j, -0.70710678-0.70710678j, -0.70710678-0.70710678j])
    >>> output = demapper(symbols)
    >>> print(output)
    [0 3 2 1 1]
    """
    alphabet: np.ndarray
    name: str = "Symbol Demapper"

    def forward(self, X: np.ndarray) -> np.ndarray:
        s, x = hard_projector(X, self.alphabet)
        return s

