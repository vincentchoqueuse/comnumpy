import numpy as np
from .core import Metric


class Metric_Recorder(Metric):

    def __init__(self, metric_fn, params={}, name="name"):
        self.metric_fn = metric_fn
        self.params = params
        self.name = name 

    def get_data(self):
        return self.data
    
    def to_json(self):
        return {}

    def compute(self, x):

        if x.ndim == 1:
            value = self.metric_fn(x, **self.params)

        if x.ndim == 2:
            N, L = x.shape
            value = []
            for l in range(L):
                value_temp = self.metric_fn(x[:, l], **self.params)
                value.append(value_temp)

        self.data = value