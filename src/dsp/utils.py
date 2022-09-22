from .core import Processor
from scipy import signal


class Amplificator(Processor):

    def __init__(self, amp=1):
        self.amp = amp

    def forward(self,x):
        return self.amp*x


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
        

