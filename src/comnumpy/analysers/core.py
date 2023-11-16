class Analyser():

    """
    This class implements a basic Analyser. This call analyses the input signal and returns the signal unchanged to the output
    """

    def __init__(self):
        self.name = "generic visualizer"

    def forward(self, x):
        return x

    def to_dict(self):
        return None

    def __call__(self, x):
        y = self.forward(x)
        return y