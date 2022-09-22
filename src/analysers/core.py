class Analyser():

    def __init__(self):
        self.name = "generic visualizer"

    def analyse(self,x):
        pass

    def __call__(self,x):
        self.analyse(x)
        return x