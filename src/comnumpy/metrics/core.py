class Metric():

    def __init__(self):
        self.name = "generic metric"

    def compute(self,x):
        return x
    
    def to_dict(self):
        return None

    def __call__(self,x):
        self.compute(x)
        return x

