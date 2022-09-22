class Metric():

    def __init__(self):
        self.name = "generic processor"

    def compute(self,x):
        return x

    def __call__(self,x):
        self.compute(x)
        return x

