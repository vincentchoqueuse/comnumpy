class Generator():

    def __init__(self):
        self.name = "generic generator"

    def forward(self, x):
        return x

    def to_dict(self):
        return None

    def __call__(self, x):
        return self.forward(x)