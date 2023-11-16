from .core import Processor
import numpy as np


class Simple_Synchronizer(Processor):

    """ Implements a simple synchronizer using cross-correlation to determine time delay and scaling between signals.
        
        Parameters
        ----------
        recorder_preamble : ndarray
            The reference preamble signal to which the input signals will be synchronized.
        scale_correction : bool, optional
            If True, applies a scaling correction based on the peak of the cross-correlation. Default is True.
        save_cross_corr : bool, optional
            If True, saves the computed cross-correlation and the associated lag vector. Default is True.
        name : str, optional
            Name of the synchronizer instance. Default is "synchronizer".
    """

    def __init__(self, recorder_preamble, scale_correction=True, save_cross_corr=True, name="synchronizer"):
        self.recorder_preamble = recorder_preamble
        self.scale_correction = scale_correction
        self.save_cross_correlation = save_cross_corr
        
        self.delay = None 
        self.scale = 1
        self.cross_corr = None 
        self.n_vect = None
        self.name = name 
        
    def fit(self, x, x_preamble):
        N = len(x)
        N_preamble = len(x_preamble)

        x_preamble_padded = np.zeros(N, dtype=x.dtype)
        x_preamble_padded[:N_preamble] = x_preamble

        # compute cross correlation
        cross_corr = (1/N_preamble)*np.correlate(x,  x_preamble_padded, mode='full')
        n_vect = np.arange(len(cross_corr)) - (N - 1)

        # Find the time delay: the index of the maximum cross-correlation minus the length of x minus 1
        index_max = np.argmax(np.abs(cross_corr)**2)
        value_max = cross_corr[index_max]
        
        self.delay = n_vect[index_max]    
        if self.scale_correction:
            self.scale = value_max

        # save correlation if needed
        if self.save_cross_correlation:
            self.cross_corr = cross_corr
            self.n_vect = n_vect


    def forward(self, x):
        x_preamble = self.recorder_preamble.get_data()
        self.fit(x, x_preamble)
        y = self.scale*x[self.delay:]
        return y


