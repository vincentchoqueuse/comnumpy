import numpy as np
from dataclasses import dataclass
from comnumpy.core.generics import Processor


@dataclass
class RappAmplifier(Processor):
    r"""
    A class representing a Solid State Power Amplifier with nonlinear characteristics.

    Signal Model
    ------------

    The general form of the Rapp input-output relationship can be used for memoryless nonlinearity and it is described by [1, 4.45]:

    .. math ::

        y[n] = g \left( 1 + \left( \frac{|x[n]|}{a_\mathrm{sat}} \right)^{2l} \right)^{-\frac{1}{2l}} x[n]

    where:

    * :math:`A_{\text{sat}}` is the input saturation value,
    * :math:`l` is the Magnitude smoothness factor,
    * :math:`g` is the small-signal gain,

    Attributes
    ----------
    a_sat : float
        The input saturation value.
    l : int, optional
        The magnitude smoothness factor (default: 2).
    g_ss : float, optional
        The small-signal gain (default is 1).
    name : str, optional
        The name of the amplifier (default is "Rapp Amplifier").

    References
    ----------
    * Ghannouchi, Fadhel M., Oualid Hammi, and Mohamed Helaoui.
      Behavioral modeling and predistortion of wideband wireless transmitters. John Wiley & Sons, 2015.
    """
    a_sat: float
    l: int = 2
    g_ss: float = 1
    name: str = "Rapp Amplifier"

    def forward(self, x: np.ndarray) -> np.ndarray:
        coef = self.g_ss * (1+(np.abs(x) / self.a_sat)**(2*self.l))**(-1/(2*self.l))
        y = coef * x
        return y

@dataclass
class SalehAmplifier(Processor):
    r"""
    A class representing a Saleh model of a Traveling Wave Tube Amplifier (TWTA).

    This model characterizes the nonlinear behavior of a TWTA. This includes both AM/AM and AM/PM conversions.

    Signal Model
    ------------

    The conversions are defined as:

    .. math::

        y[n] = x[n] \cdot G(x[n])  e^{j \Phi(x[n])}

    * AM/AM Conversion:

    .. math::

        G(x) = \frac{\alpha_a }{1 + \beta_a \frac{|x|^2}{V_{sat}^2}}

    * AM/PM Conversion:

    .. math::

        \Phi(x) = \frac{\alpha_{\phi} \cdot \frac{|x|^2}{V_{sat}^2}}{1 + \beta_{\phi} \cdot \frac{|x|^2}{V_{sat}^2}}

    Attributes
    ----------
    a_sat : float, optional
        The input saturation value (default is 1).
    alpha_am : float, optional
        The AM/AM conversion parameter (default is 1.9638).
    beta_am : float, optional
        The AM/AM conversion parameter (default is 0.9945).
    alpha_pm : float, optional
        The AM/PM conversion parameter (default is 2.5293).
    beta_pm : float, optional
        The AM/PM conversion parameter (default is 2.8168).
    name : str, optional
        The name of the amplifier (default is "Saleh Amplifier").
    """
    a_sat: float = 1
    alpha_am: float = 1.9638
    beta_am: float = 0.9945
    alpha_pm: float = 2.5293
    beta_pm: float = 2.8168
    name: str = "Saleh Amplifier"

    def forward(self, x: np.ndarray) -> np.ndarray:
        a_norm = np.abs(x/self.a_sat)
        G = self.alpha_am / (1+self.beta_am*(a_norm**2))
        phi = self.alpha_pm * (a_norm**2)/(1+self.beta_pm*(a_norm**2))
        return x * G * np.exp(1j*phi)
