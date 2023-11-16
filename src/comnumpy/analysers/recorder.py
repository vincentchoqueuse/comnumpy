from .core import Analyser


class Recorder(Analyser):

    """
    This class implements a basic Recorder that let the signal pass trough.
    """

    def __init__(self, sample_intervals = None, name="recorder"):
        self.data = None
        self.sample_intervals = sample_intervals
        self.name = name 

    def get_data(self):
        return self.data

    def forward(self, x):
        self.data = x
        if self.sample_intervals:
            n_min, n_max = self.sample_intervals
            self.data = x[n_min:n_max]
        else:
            self.data = x
        return x

