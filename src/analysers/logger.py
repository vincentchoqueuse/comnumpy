import numpy as np
from .core import Analyser


class Logger(Analyser):

    def __init__(self, num=None, name="logger"):
        self.name = name
        self.num = num

    def analyse(self, x):
        print("Data logger ({}): {}".format(self.name, x))

class Power_Reporter(Analyser):

    def __init__(self, num=None, verbose=True, name="power"):
        self.name = name
        self.num = num
        self.verbose = verbose

    def analyse(self, x):
        if self.verbose:
            P = np.mean(np.abs(x)**2)
            print("Power reporter ({}): {}".format(self.name, P))

class IQ_Reporter(Analyser):

    def __init__(self, num=None, verbose=True, name="power"):
        self.name = name
        self.num = num
        self.verbose = verbose

    def analyse(self, x):
        if self.verbose:
            R_xxc = np.mean(np.abs(x)**2)
            R_xx = np.mean(x**2)
            print("IQ reporter ({}): R_xxc={} R_xx={}".format(self.name, R_xxc, R_xx))

