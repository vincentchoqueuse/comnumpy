from .core import Processor
import numpy as np
from scipy import signal


class Delay_Compensator(Processor):

    def __init__(self, delay):
        self.delay = delay

    def forward(self,x):
        return x[self.delay:]


class Extractor(Processor):

    def __init__(self, N_start=0, N=None, name="extractor"):
        self.N_start = N_start
        self.N = N
        self.name = name

    def forward(self, x):

        if self.N is None:
            y = x[self.N_start:]
        else:
            y = x[self.N_start:self.N_start+self.N]
        return y


class Resample(Processor):

    def __init__(self, up, down, name="resampler"):
        self.up = up
        self.down = down
        self.name = name

    def forward(self, x):
        y = signal.resample_poly(x, self.up, self.down)
        return y
   

class Serial_2_Parallel(Processor):

    def __init__(self, N_sc, order="F", name="S2P"):
        if not (isinstance(N_sc, int) and N_sc > 0):
            raise ValueError("N_sc must be a positive integer.")
        
        self.N_sc = N_sc
        self.order = order
        self.name = name

    def forward(self, x):
        N_sc = self.N_sc
        N = len(x)
        M = int(np.ceil(N/N_sc))
        x_padded = np.zeros(self.N_sc*M, dtype=x.dtype)
        x_padded[:N] = x
        X = x_padded.reshape((N_sc, M), order=self.order)
        return X

class Parallel_2_Serial(Processor):

    def __init__(self, order="F", name="P2S"):
        self.order = order
        self.name = name

    def forward(self, X):
        x = np.ravel(X, order=self.order)
        return x