from dataclasses import dataclass
from typing import Optional, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Processor  # À garder selon ton infra réelle


@dataclass
class FFTMonitor(Processor):
    """
    A monitoring tool for visualizing the amplitude of frequency-domain signals 
    (e.g., before an IFFT or after a FFT block in an OFDM system).

    Attributes
    ----------
    reduction : Optional[str]
        Method used to reduce the 2D input matrix before plotting.
        - "mean" : Computes the average amplitude across all OFDM symbols.
        - None   : Plots each symbol’s amplitude individually (superimposed).
        
    title : str
        Title of the plot.
    
    name : str
        Identifier for the monitor instance.
    """

    reduction: Optional[Literal["mean"]] = "mean"
    title: str = "IFFT_Monitor"
    name: str = "Ifft_monitor"

    def get_reduction(self, X: np.ndarray) -> Union[np.ndarray, float]:
        amplitudes = np.abs(X)
        if self.reduction is None:
            return amplitudes  # superimpose all
        elif self.reduction == "mean":
            return np.mean(amplitudes, axis=1)
        else:
            raise ValueError("Invalid reduction option. Choose None or 'mean'.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        values_to_plot = self.get_reduction(x)
        subcarrier_indices = np.arange(-x.shape[0] // 2, x.shape[0] // 2)

        plt.figure(figsize=(8, 6))

        if self.reduction is None:
            for col in range(values_to_plot.shape[1]):
                plt.stem(subcarrier_indices, values_to_plot[:, col])
        else:
            plt.stem(subcarrier_indices, values_to_plot)

        plt.title(self.title)
        return x
