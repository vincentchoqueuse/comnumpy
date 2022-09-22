import numpy as np
from .core import Metric


class Zero_Crossing(Metric):

    def __init__(self,name="name"):
        self.name = name 

    def compute(self, x):
        zero_crossings = np.where(np.diff(np.sign(x)))[0]
        print(zero_crossings)