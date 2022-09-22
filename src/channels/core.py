class Channel():

    def __init__(self):
        self.name = "generic channel"

    def forward(self,X):
        return X

    def get_delay(self):
        return 0

    def __call__(self,X):
        return self.forward(X)

