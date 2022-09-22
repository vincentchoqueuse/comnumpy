import os.path as path
import numpy as np
from scipy.stats import randint
from .core import Processor
from .functional import hard_projector

def get_alphabet(modulation, order, type="gray", norm=True):
    # extract alphabet
    pathname = path.dirname(path.abspath(__file__))
    filename = "{}/data/{}_{}_{}.csv".format(pathname,modulation,order,type)
    data = np.loadtxt(filename,delimiter=',',skiprows=1)
    alphabet = data[:,1]+1j*data[:,2]

    if norm == True :
        alphabet = alphabet/np.sqrt(np.mean(np.abs(alphabet)**2))

    return alphabet


def sym_2_bin(sym,width=4):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice],width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)


class Modulator(Processor):

    def __init__(self,alphabet):
        super().__init__()
        self._alphabet = alphabet

    def get_alphabet(self):
        return self._alphabet

    def forward(self,x):
        Y = self._alphabet[x]
        return Y


class Demodulator(Processor):

    def __init__(self,alphabet):
        super().__init__()
        self._alphabet = alphabet

    def forward(self,x):
        s, x = hard_projector(x, self._alphabet)
        return s