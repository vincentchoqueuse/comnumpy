from numpy.random import randn
import numpy as np
from .core import Channel


class IQ_imbalance(Channel):

    r"""
    IQ_imbalance()
    Physical Layer for IQ imbalance impairments. IQ imbalance is  modeled in the complex domain as
    .. math::
        x_l[n] = \alpha x_{l-1}[n]+ \beta x_{l-1}^*[n]
    where :math:`n=0,1\cdots,N-1` and :math:`(\alpha,\beta) \in \mathbb{C}^2.
    Parameters
    ----------
    params : numpy array
        A numpy array of size 4 containing the IQ imbalance parameters
    Returns
    -------
    None
    """

    def __init__(self, alpha, beta, name="iq_impairment"):
        self.name = name
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        y = self.alpha*x + self.beta*np.conj(x)
        return y

 
class CFO(Channel):
    r"""
    CFO()
    The effect of a residual Carrier Frequency Offset (CFO) is usually modeled as \cite{YAO05}
    .. math::
        x_l[n] =x_{l-1}[n]e^{j\omega n}
    where :math:`\omega` corresponds to the normalized residual carrier offset (in rad/samples). The CFO layer only depends on the layer parameter $\boldsymbol\theta = \omega$.
    Parameters
    ----------
    params : numpy array
        Numpy array of length 1 containing the CFO parameters
    Returns
    -------
    y :  numpy array
         Output signal
    """
    def __init__(self, cfo, name="CFO"):
        self.cfo = cfo
        self.name = name

    def forward(self, x):
        N = len(x)
        n_vect = np.arange(N)
        y = x*np.exp(1j*self.cfo*n_vect)
        return y

