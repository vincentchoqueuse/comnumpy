class Processor():

    def __init__(self):
        self.name = "generic processor"

    def forward(self,x):
        return x

    def get_delay(self):
        return 0

    def to_dict(self):
        return None

    def __call__(self,x):
        return self.forward(x)

