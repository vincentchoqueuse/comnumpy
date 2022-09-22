import numpy as np
import matplotlib.pyplot as plt
from .core import Analyser


class Recorder(Analyser):

    def __init__(self, name="recorder"):
        self.data = None
        self.name = name 

    def get_data(self):
        return self.data

    def forward(self, x):
        self.data = x
        return x


