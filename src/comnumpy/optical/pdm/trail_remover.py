import numpy as np
from comnumpy.core.processors import Processor
from dataclasses import dataclass, field

@dataclass
class TrailRemover(Processor):
    r"""Remove trailing samples from a signal.
    
    This processor removes a specified number of samples from the end
    of the input signal, useful for removing edge effects or transients.
    
    Parameters
    ----------
    trail : int
        Number of samples to remove from the end of the signal
    name : str, optional
        Processor name (default: "trail remover")
    """
    
    trail: int
    name: str = "trail remover"
    
    def __post_init__(self):
        super().__init__()

    def forward(self, x):
        y = x[:-self.trail]
        return y