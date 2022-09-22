import numpy as np
from .core import Processor
import numpy.linalg as LA
import matplotlib.pyplot as plt
import itertools


def hard_projector(z, alphabet):
    N = len(z)
    z_0 = np.atleast_2d(np.ravel(z))
    error = np.abs(np.transpose(z_0) - alphabet)**2
    index = np.argmin(error,axis=1)
    s = index.astype(int)
    x = alphabet[s]
    return s, x


def soft_projector(z, alphabet, sigma2, kernel = None):
    alphabet = alphabet.reshape(1, -1)
    z = z.reshape(-1, 1)
    term1 = np.exp(-(1/sigma2) * np.abs(alphabet - z)**2)

    if kernel is None: 
        kernel = alphabet

    num = np.sum(kernel * term1, axis=1)
    den = np.sum(term1, axis=1)
    return num/den

