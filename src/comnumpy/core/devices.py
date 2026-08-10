import numpy as np
from dataclasses import dataclass, field
from comnumpy.core.generics import Processor

__all__ = ["RappAmplifier", "SalehAmplifier"]


@dataclass(slots=True)
class RappAmplifier(Processor):
    r"""Rapp model of a memoryless solid-state power amplifier (SSPA).

    Signal Model
    ------------
    The Rapp model is a purely AM/AM memoryless nonlinearity: the
    amplitude is compressed, the phase is left untouched.

    .. math::

        y[n] = g \left( 1 + \left( \frac{|x[n]|}{a_\mathrm{sat}}
               \right)^{2l} \right)^{-\frac{1}{2l}} x[n]

    Equivalently, the AM/AM characteristic on the envelope
    :math:`r = |x[n]|` is

    .. math::

        |y[n]| = \frac{g \, r}
                      {\left( 1 + \left( r / a_\mathrm{sat}
                       \right)^{2l} \right)^{1/(2l)}}, \qquad
        \arg y[n] = \arg x[n]

    which is linear with slope :math:`g` for :math:`r \ll a_\mathrm{sat}`
    and saturates at :math:`g \, a_\mathrm{sat}` for
    :math:`r \gg a_\mathrm{sat}`. The smoothness factor :math:`l`
    controls how abruptly the transition happens: :math:`l \to \infty`
    tends to a hard clipper at :math:`a_\mathrm{sat}`, while small
    :math:`l` gives a soft, early compression.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    a_sat : float
        Input saturation amplitude :math:`a_\mathrm{sat}`, in the same
        unit as :math:`|x[n]|`.
    l : int, optional, keyword-only
        Magnitude smoothness factor :math:`l`. Default is 2.
    g_ss : float, optional, keyword-only
        Small-signal (linear-region) gain :math:`g`. Default is 1.
    name : str, optional, keyword-only
        Name of the amplifier instance. Default is
        ``"Rapp Amplifier"``.

    References
    ----------
    C. Rapp, "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal for a
    digital sound broadcasting system," in *Proc. Second European
    Conference on Satellite Communications (ECSC-2)*, Liege, Belgium,
    Oct. 1991, pp. 179-184.

    F. M. Ghannouchi, O. Hammi, M. Helaoui, *Behavioral Modeling and
    Predistortion of Wideband Wireless Transmitters*, John Wiley & Sons,
    2015, Eq. (4.45).

    Examples
    --------
    >>> amp = RappAmplifier(1.0, l=2)
    >>> print(np.round(amp(np.array([0.5, 1.0, 4.0])), 4))
    [0.4925 0.8409 0.999 ]
    """
    a_sat: float
    l: int = field(default=2, kw_only=True)
    g_ss: float = field(default=1, kw_only=True)
    name: str = field(default="Rapp Amplifier", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        coef = self.g_ss * (1+(np.abs(x) / self.a_sat)**(2*self.l))**(-1/(2*self.l))
        y = coef * x
        return y

@dataclass(slots=True)
class SalehAmplifier(Processor):
    r"""Saleh model of a memoryless travelling-wave tube amplifier (TWTA).

    Signal Model
    ------------
    Unlike the Rapp model, the Saleh model carries an AM/PM conversion:
    the amplifier turns an amplitude variation into a phase rotation,
    which is the dominant impairment of a TWTA. Writing the normalized
    input envelope :math:`r[n] = |x[n]| / a_\mathrm{sat}`,

    .. math::

        y[n] = x[n] \, G\left(r[n]\right) \,
               e^{j \Phi\left(r[n]\right)}

    with the AM/AM conversion

    .. math::

        G(r) = \frac{\alpha_a}{1 + \beta_a r^2}
        \qquad \Longrightarrow \qquad
        |y[n]| = a_\mathrm{sat} \, \frac{\alpha_a \, r[n]}
                                        {1 + \beta_a \, r^2[n]}

    and the AM/PM conversion

    .. math::

        \Phi(r) = \frac{\alpha_\phi \, r^2}{1 + \beta_\phi \, r^2}

    The output amplitude peaks at :math:`r = 1 / \sqrt{\beta_a}`, where
    it reaches :math:`a_\mathrm{sat} \alpha_a / (2 \sqrt{\beta_a})`, and
    then *decreases* -- a TWTA is driven back past saturation, unlike the
    monotone Rapp SSPA. The phase shift grows monotonically and
    saturates at :math:`\alpha_\phi / \beta_\phi` radians. The default
    coefficients are those fitted by Saleh to a measured TWTA.

    Axes: *element-wise* -- applied pointwise, shape-agnostic. The
    output is always complex, because of the AM/PM term.

    Parameters
    ----------
    a_sat : float, optional
        Input saturation amplitude :math:`a_\mathrm{sat}` used to
        normalize the envelope, :math:`r[n] = |x[n]| / a_\mathrm{sat}`.
        Default is 1.
    alpha_am : float, optional, keyword-only
        AM/AM small-signal gain :math:`\alpha_a`. Default is 1.9638.
    beta_am : float, optional, keyword-only
        AM/AM compression coefficient :math:`\beta_a`. Default is
        0.9945.
    alpha_pm : float, optional, keyword-only
        AM/PM conversion coefficient :math:`\alpha_\phi` in rad.
        Default is 2.5293.
    beta_pm : float, optional, keyword-only
        AM/PM saturation coefficient :math:`\beta_\phi`. Default is
        2.8168.
    name : str, optional, keyword-only
        Name of the amplifier instance. Default is
        ``"Saleh Amplifier"``.

    References
    ----------
    A. A. M. Saleh, "Frequency-independent and frequency-dependent
    nonlinear models of TWT amplifiers," *IEEE Transactions on
    Communications*, vol. 29, no. 11, pp. 1715-1720, Nov. 1981.

    Examples
    --------
    >>> y = SalehAmplifier()(np.array([0.5, 1.0, 2.0], dtype=complex))
    >>> print(np.round(np.abs(y), 4))     # AM/AM: peaks near r = 1, then falls
    [0.7864 0.9846 0.789 ]
    >>> print(np.round(np.angle(y), 4))   # AM/PM: growing phase rotation (rad)
    [0.371  0.6627 0.8247]
    """
    a_sat: float = 1
    alpha_am: float = field(default=1.9638, kw_only=True)
    beta_am: float = field(default=0.9945, kw_only=True)
    alpha_pm: float = field(default=2.5293, kw_only=True)
    beta_pm: float = field(default=2.8168, kw_only=True)
    name: str = field(default="Saleh Amplifier", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        a_norm = np.abs(x/self.a_sat)
        G = self.alpha_am / (1+self.beta_am*(a_norm**2))
        phi = self.alpha_pm * (a_norm**2)/(1+self.beta_pm*(a_norm**2))
        return x * G * np.exp(1j*phi)
