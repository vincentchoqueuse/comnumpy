import numpy as np
from .core import Analyser


class Logger(Analyser):

    """
    This class implements a basic Logger that let the signal pass trough.
    """

    def __init__(self, num=None, name="logger"):
        self.name = name
        self.num = num

    def forward(self, x):
        print("Data logger ({}): {}".format(self.name, x))
        return x

class Power_Reporter(Analyser):

    """
    This class implements a basic Power Reporter that let the signal pass trough.
    """

    def __init__(self, num=None, verbose=True, name="power"):
        self.name = name
        self.num = num
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            P = np.mean(np.abs(x)**2)
            print("Power reporter ({}): {}".format(self.name, P))
        return x

class IQ_Reporter(Analyser):

    """
    This class implements a basic IQ reporter that let the signal pass trough.
    """

    def __init__(self, num=None, verbose=True, name="power"):
        self.name = name
        self.num = num
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            R_xxc = np.mean(np.abs(x)**2)
            R_xx = np.mean(x**2)
            print("IQ reporter ({}): R_xxc={} R_xx={}".format(self.name, R_xxc, R_xx))
        return x

