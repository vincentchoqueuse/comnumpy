import numpy as np
import numpy.linalg as LA
from comnumpy.core.generics import Processor
from scipy import signal
from scipy.linalg import toeplitz
from scipy import special
from dataclasses import dataclass, field
from typing import ClassVar, List
from .utils import compute_beta2
from .constants import WAVELENGTH, CD_COEFFICIENT, SPEED_OF_LIGHT


@dataclass(slots=True)
class ChromaticDispersionFIRCompensator(Processor):
    r"""Static FIR compensator of chromatic dispersion (truncated inverse impulse response, Savory).

    Signal Model
    ------------
    A fiber span of length :math:`z` acts on the complex envelope as the
    all-pass dispersive filter

    .. math::

        H_{\mathrm{CD}}(\omega) = e^{j \frac{\beta_2}{2} z \omega^2}

    where :math:`\omega` is the angular frequency in rad/s and
    :math:`\beta_2` the group velocity dispersion. The compensator
    approximates its inverse. Sampling at :math:`f_s` and using the
    normalized pulsation :math:`\Omega = \omega / f_s \in [-\pi, \pi]`
    (rad/sample), the target response is the quadratic-phase all-pass

    .. math::

        H_c\!\left(e^{j\Omega}\right) =
        e^{-j \frac{\beta_2}{2} z f_s^2 \Omega^2} = e^{j K \Omega^2},
        \qquad K = -\frac{\beta_2 z f_s^2}{2}

    with the dimensionless dispersion :math:`K > 0`. Its inverse DTFT is
    the Fresnel (chirped) sequence

    .. math::

        h[n] = \sqrt{\frac{j}{4 \pi K}} \, e^{-j \frac{n^2}{4 K}}

    truncated to the span imposed by the maximum group delay
    :math:`2 \pi K` of the all-pass, that is :math:`|n| \le
    \lfloor 2 \pi K \rfloor`, hence :math:`N = 2 \lfloor 2 \pi K \rfloor + 1`
    taps. The block applies the full linear convolution

    .. math::

        y[n] = \sum_{l} h[l] \, x[n - l]

    so the output is :math:`N - 1` samples longer than the input and
    carries a group delay of :math:`(N - 1) / 2` samples.

    Units follow the module convention: :math:`\beta_2 = -10^3 D
    \lambda^2 / (2 \pi c)` is returned in ps^2/km by
    :func:`comnumpy.optical.utils.compute_beta2` and converted internally
    to s^2/km (factor :math:`10^{-24}`), :math:`z` is in km and
    :math:`f_s` in Hz, which makes :math:`K` dimensionless.

    Axes: *declared axis* -- operates on a 1D serial signal ``(N_x,)``
    (``scipy.signal.convolve`` in ``"full"`` mode).

    Parameters
    ----------
    z : float
        Fiber length :math:`z` in km.
    fs : float, optional, keyword-only
        Sampling frequency :math:`f_s` in Hz. Default is 1.
    name : str, optional, keyword-only
        Name of the processor instance. Default is
        ``"fir cd compensator"``.

    Attributes
    ----------
    h : np.ndarray
        Filter taps :math:`h[n]` for :math:`|n| \le \lfloor 2 \pi K
        \rfloor`, built in ``__post_init__``.
    K : float
        Dimensionless dispersion :math:`K = -\beta_2 z f_s^2 / 2`.
    lamb : float
        Class-level wavelength :math:`\lambda` in nm (``WAVELENGTH``).
    D : float
        Class-level dispersion coefficient :math:`D` in ps/nm/km
        (``CD_COEFFICIENT``).
    c : float
        Class-level speed of light :math:`c` in m/s (``SPEED_OF_LIGHT``).

    References
    ----------
    S. J. Savory, "Digital filters for coherent optical receivers,"
    Optics Express, vol. 16, no. 2, pp. 804-817, 2008 (Eqs. 4-11).

    Examples
    --------
    Dispersion spreads an isolated pulse; the compensator recompresses it.

    >>> from comnumpy.optical.channels import ChromaticDispersion
    >>> x = np.zeros(512, dtype=complex)
    >>> x[256] = 1.0
    >>> y = ChromaticDispersion(1000.0, fs=20e9)(x)
    >>> comp = ChromaticDispersionFIRCompensator(1000.0, fs=20e9)
    >>> delay = (len(comp.h) - 1) // 2
    >>> x_hat = comp(y)[delay:delay + 512]
    >>> print(len(comp.h), round(float(np.abs(y).max()), 3), round(float(np.abs(x_hat[256])), 3))
    55 0.162 0.975
    """
    lamb: ClassVar[float] = WAVELENGTH
    D: ClassVar[float] = CD_COEFFICIENT
    c: ClassVar[float] = SPEED_OF_LIGHT

    z: float  # in km
    fs: float = field(default=1.0, kw_only=True)
    name: str = field(default="fir cd compensator", kw_only=True)
    # internal state (declared for slots, D40a)
    h: np.ndarray = field(init=False, repr=False, default_factory=lambda: None)
    K: float = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        beta2_ps2_per_km = compute_beta2(self.lamb, self.D, self.c)
        beta2 = ((10**-12)**2)*beta2_ps2_per_km  # convert into s^2/km
        K = - beta2 * self.z * (self.fs**2) / 2
        N = int(2 * np.floor(2 * K * np.pi) + 1)
        bound = int(np.floor(N / 2))
        n_vect = np.arange(-bound, bound + 1)
        coef = np.sqrt(1j / (4 * K * np.pi))
        self.h = coef * np.exp(-1j * (n_vect**2) / (4 * K))
        self.K = K

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = signal.convolve(x, self.h, mode='full')
        return y


@dataclass(slots=True)
class ChromaticDispersionLSFIRCompensator(Processor):
    r"""Least-squares FIR compensator of chromatic dispersion (optimal design over a frequency band).

    Signal Model
    ------------
    The target response is the same quadratic-phase all-pass as in
    :class:`ChromaticDispersionFIRCompensator`, i.e. the inverse of the
    dispersive response :math:`e^{j (\beta_2 / 2) z \omega^2}` of the
    fiber, written with the normalized pulsation
    :math:`\Omega = \omega / f_s \in [-\pi, \pi]`:

    .. math::

        H_c\!\left(e^{j\Omega}\right) = e^{j K \Omega^2},
        \qquad K = -\frac{\beta_2 z f_s^2}{2}

    Rather than truncating the inverse DTFT of :math:`H_c`, the
    :math:`N` taps :math:`h[n]`, :math:`|n| \le (N-1)/2`, minimize the
    integrated squared error over the design band
    :math:`[\Omega_1, \Omega_2]`:

    .. math::

        \min_{h} \int_{\Omega_1}^{\Omega_2}
        \left| \sum_{l} h[l] e^{-j l \Omega} - e^{j K \Omega^2} \right|^2 d\Omega

    The normal equations of this quadratic problem read :math:`Q h = d`,
    with the Hermitian Toeplitz Gram matrix and the cross-correlation
    vector

    .. math::

        Q_{m,l} = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2}
        e^{-j (l - m) \Omega} d\Omega, \qquad
        d[n] = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2}
        e^{j K \Omega^2} e^{j n \Omega} d\Omega

    The implemented closed form of :math:`d[n]` is the one obtained for
    the full band :math:`[-\pi, \pi]`:

    .. math::

        d[n] = \frac{e^{-j\left(\frac{n^2}{4K} + \frac{3\pi}{4}\right)}}{4\sqrt{\pi K}}
        \left[ \mathrm{erf}\!\left(\kappa (2 \pi K - n)\right)
             + \mathrm{erf}\!\left(\kappa (2 \pi K + n)\right) \right],
        \qquad \kappa = \frac{e^{j 3 \pi / 4}}{2 \sqrt{K}}

    and the system is solved as :math:`h = (Q + \epsilon I)^{-1} d`, the
    ridge term :math:`\epsilon` regularizing the near-singular Gram
    matrix. As the band widens and :math:`K` grows, :math:`Q \to I` and
    :math:`d[n] \to \sqrt{j / (4 \pi K)} e^{-j n^2 / (4K)}`, i.e. the
    design falls back on the truncated (Savory) filter. The block then
    applies the full linear convolution

    .. math::

        y[n] = \sum_{l} h[l] \, x[n - l]

    whose output is :math:`N - 1` samples longer than the input, with a
    group delay of :math:`(N - 1) / 2` samples. Units are those of the
    module: :math:`\beta_2` in ps^2/km from
    :func:`comnumpy.optical.utils.compute_beta2`, converted internally to
    s^2/km, :math:`z` in km and :math:`f_s` in Hz, so that :math:`K` is
    dimensionless.

    Axes: *declared axis* -- operates on a 1D serial signal ``(N_x,)``
    (``scipy.signal.convolve`` in ``"full"`` mode).

    Parameters
    ----------
    z : float
        Fiber length :math:`z` in km.
    N : int
        Filter length :math:`N` in taps; must be odd so that the support
        :math:`|n| \le (N-1)/2` is symmetric.
    fs : float, optional, keyword-only
        Sampling frequency :math:`f_s` in Hz. Default is 1.
    w_vect : List[float], optional, keyword-only
        Design band :math:`[\Omega_1, \Omega_2]` in rad/sample. Default
        is ``[-np.pi, np.pi]`` (full band).
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"optimal"``.

    Attributes
    ----------
    h : np.ndarray
        Least-squares taps :math:`h[n]`, solution of :math:`Q h = d`.
    K : float
        Dimensionless dispersion :math:`K = -\beta_2 z f_s^2 / 2`.
    epsilon : float
        Class-level ridge term :math:`\epsilon` added to :math:`Q`
        before inversion. Default is ``1e-14``.
    lamb : float
        Class-level wavelength :math:`\lambda` in nm (``WAVELENGTH``).
    D : float
        Class-level dispersion coefficient :math:`D` in ps/nm/km
        (``CD_COEFFICIENT``).
    c : float
        Class-level speed of light :math:`c` in m/s (``SPEED_OF_LIGHT``).

    Raises
    ------
    ValueError
        If the filter length ``N`` is even.

    References
    ----------
    * A. Eghbali, H. Johansson, O. Gustafsson, S. J. Savory, "Optimal
      least-squares FIR digital filters for compensation of chromatic
      dispersion in digital coherent optical receivers," Journal of
      Lightwave Technology, vol. 32, no. 8, pp. 1449-1456, 2014.
    * S. J. Savory, "Digital filters for coherent optical receivers,"
      Optics Express, vol. 16, no. 2, pp. 804-817, 2008.

    Examples
    --------
    Same round trip as the truncated design, at equal filter length.

    >>> from comnumpy.optical.channels import ChromaticDispersion
    >>> x = np.zeros(512, dtype=complex)
    >>> x[256] = 1.0
    >>> y = ChromaticDispersion(1000.0, fs=20e9)(x)
    >>> comp = ChromaticDispersionLSFIRCompensator(1000.0, 55, fs=20e9)
    >>> delay = (comp.N - 1) // 2
    >>> x_hat = comp(y)[delay:delay + 512]
    >>> print(round(float(np.abs(x_hat[256])), 3))
    0.977
    """
    lamb: ClassVar[float] = WAVELENGTH
    D: ClassVar[float] = CD_COEFFICIENT
    c: ClassVar[float] = SPEED_OF_LIGHT
    epsilon: ClassVar[float] = 1e-14

    z: float
    N: int
    fs: float = field(default=1.0, kw_only=True)
    w_vect: List[float] = field(default_factory=lambda: [-np.pi, np.pi], kw_only=True)
    name: str = field(default="optimal", kw_only=True)
    # internal state (declared for slots, D40a)
    h: np.ndarray = field(init=False, repr=False, default_factory=lambda: None)
    K: float = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        if self.N % 2 == 0:
            raise ValueError(f"The value of N must be odd (current={self.N})")

        beta2_ps2_per_km = compute_beta2(self.lamb, self.D, self.c)
        beta2 = ((10**-12)**2)*beta2_ps2_per_km  # convert into s^2/km
        K = -beta2 * self.z * (self.fs**2) / 2
        Omega_1, Omega_2 = self.w_vect

        # Construct Matrix Q
        q_row = np.zeros(self.N, dtype=complex)
        q_col = np.zeros(self.N, dtype=complex)
        q_row[0] = q_col[0] = (Omega_2 - Omega_1) / (2 * np.pi)
        for m in range(1, self.N):
            coef = 1 / (2j * np.pi * m)
            q_row[m] = coef * (np.exp(-1j * m * Omega_1) - np.exp(-1j * m * Omega_2))
        Q = toeplitz(q_col, q_row)

        # Construct vector d
        bound = self.N // 2
        n_vect = np.arange(-bound, bound + 1)
        coef1 = 1 / (4 * np.sqrt(np.pi * K))
        coef2 = np.exp(1j * 3 * np.pi / 4) / (2 * np.sqrt(K))
        d_vect = np.zeros(len(n_vect), dtype=complex)

        for idx, n in enumerate(n_vect):
            term1 = coef2 * (2 * K * np.pi - n)
            term2 = coef2 * (2 * K * np.pi + n)
            erf_term = special.erf(term1) + special.erf(term2)
            phase = np.exp(-1j * (n**2 / (4 * K) + 3 * np.pi / 4))
            d_vect[idx] = coef1 * phase * erf_term

        I_mat = np.eye(self.N)
        Q_inv = LA.inv(Q + self.epsilon * I_mat)
        self.h = Q_inv @ d_vect
        self.K = K

    def forward(self, x: np.ndarray) -> np.ndarray:
        return signal.convolve(x, self.h)
