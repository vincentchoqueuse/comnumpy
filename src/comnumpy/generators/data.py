from .core import Generator
import numpy as np


class Symbol_Generator(Generator):
    """
    Independent and identically distributed (IID) symbol generator.

    Parameters
    ----------
    N : int
        The number of symbols to generate.
    alphabet : list or array-like
        A list or array of potential symbols from which to generate.
    seed : int, optional
        Seed for the random number generator, for reproducibility. 
        Default is None, which initializes the generator without a fixed seed.
    """
    
    def __init__(self, alphabet, seed=None, name="generator"):
        self.rng = np.random.default_rng(seed)
        self.alphabet = alphabet
        self.name = name

    def forward(self, N):
        index = self.rng.integers(len(self.alphabet), size=N)
        y = self.alphabet[index]
        return y
