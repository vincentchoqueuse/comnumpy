class Generator():

    def __init__(self):
        self.name = "generic generator"

    def forward(self,x):
        return x

    def __call__(self, x=None):
        return self.forward(x)