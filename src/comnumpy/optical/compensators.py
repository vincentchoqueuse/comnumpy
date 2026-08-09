import numpy as np
import numpy.linalg as LA
from comnumpy.core.generics import Processor
from scipy import signal
from scipy.linalg import toeplitz
from scipy import special
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional
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
    h: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    K: Optional[float] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
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


def _cd_gram_matrix(N: int, Omega_1: float, Omega_2: float) -> np.ndarray:
    r"""Hermitian Toeplitz Gram matrix of the delays over a design band.

    Returns the :math:`N \times N` matrix :math:`Q_{m,l} = q(l - m)` with

    .. math::

        q(p) = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2} e^{-j p \Omega} d\Omega
             = \frac{e^{-j p \Omega_1} - e^{-j p \Omega_2}}{2 j \pi p},
        \qquad q(0) = \frac{\Omega_2 - \Omega_1}{2\pi}

    Only :math:`p \ge 0` is evaluated; the negative lags follow from
    :math:`q(-p) = q(p)^*`, so the first column handed to
    :func:`scipy.linalg.toeplitz` is the conjugate of the first row and
    the result satisfies :math:`Q = Q^H`.

    On the full band :math:`[-\pi, \pi]` one has :math:`q(p) =
    \mathrm{sinc}(p) = 0` for :math:`p \neq 0`, hence :math:`Q = I`. On a
    narrower band :math:`Q` is a prolate matrix: about :math:`N -
    N(\Omega_2 - \Omega_1) / (2\pi)` of its eigenvalues are numerically
    zero, because the taps have degrees of freedom that the cost function
    does not see outside the band.

    Parameters
    ----------
    N : int
        Filter length in taps.
    Omega_1, Omega_2 : float
        Design band bounds in rad/sample, with ``Omega_1 < Omega_2``.

    Returns
    -------
    np.ndarray
        Hermitian Toeplitz matrix of shape ``(N, N)``.
    """
    q_row = np.zeros(N, dtype=complex)
    q_row[0] = (Omega_2 - Omega_1) / (2 * np.pi)
    p_vect = np.arange(1, N)
    q_row[1:] = (np.exp(-1j * p_vect * Omega_1)
                 - np.exp(-1j * p_vect * Omega_2)) / (2j * np.pi * p_vect)
    return toeplitz(np.conj(q_row), q_row)


def _cd_cross_correlation(K: float, N: int, Omega_1: float,
                          Omega_2: float) -> np.ndarray:
    r"""Cross-correlation between the target all-pass and the delays.

    Returns the vector, for :math:`|n| \le (N-1)/2`,

    .. math::

        d[n] = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2}
        e^{j K \Omega^2} e^{j n \Omega} d\Omega

    evaluated in closed form. Completing the square,
    :math:`K \Omega^2 + n \Omega = K (\Omega + n / 2K)^2 - n^2 / 4K`, and
    substituting :math:`t = \sqrt{K} e^{-j\pi/4} (\Omega + n / 2K)` turns
    the integrand into :math:`e^{-t^2}`, whose primitive is the complex
    error function. The band enters only through the two bounds:

    .. math::

        d[n] = \frac{e^{j \pi / 4}}{4 \sqrt{\pi K}} e^{-j \frac{n^2}{4K}}
        \left[ \mathrm{erf}\!\left(\gamma \left(\Omega_2 + \frac{n}{2K}\right)\right)
             - \mathrm{erf}\!\left(\gamma \left(\Omega_1 + \frac{n}{2K}\right)\right) \right],
        \qquad \gamma = \sqrt{K} \, e^{-j\pi/4}

    Parameters
    ----------
    K : float
        Dimensionless dispersion :math:`K = -\beta_2 z f_s^2 / 2`.
    N : int
        Filter length in taps (odd).
    Omega_1, Omega_2 : float
        Design band bounds in rad/sample, with ``Omega_1 < Omega_2``.

    Returns
    -------
    np.ndarray
        Complex vector of length ``N``.
    """
    bound = N // 2
    n_vect = np.arange(-bound, bound + 1)
    gamma = np.sqrt(K) * np.exp(-1j * np.pi / 4)
    shift = n_vect / (2 * K)
    erf_term = (special.erf(gamma * (Omega_2 + shift))
                - special.erf(gamma * (Omega_1 + shift)))
    coef = np.exp(1j * np.pi / 4) / (4 * np.sqrt(np.pi * K))
    return coef * np.exp(-1j * (n_vect**2) / (4 * K)) * erf_term


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

        Q_{m,l} = q(l - m), \quad
        q(p) = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2}
        e^{-j p \Omega} d\Omega, \qquad
        d[n] = \frac{1}{2\pi} \int_{\Omega_1}^{\Omega_2}
        e^{j K \Omega^2} e^{j n \Omega} d\Omega

    Only :math:`q(p)` for :math:`p \ge 0` is evaluated, the negative lags
    following from the Hermitian symmetry :math:`q(-p) = q(p)^*` that
    makes :math:`Q = Q^H`. Completing the square in the integrand of
    :math:`d[n]` and mapping :math:`j K \Omega^2` onto :math:`-t^2` gives
    the closed form on an *arbitrary* band, in terms of the complex error
    function:

    .. math::

        d[n] = \frac{e^{j \pi / 4}}{4 \sqrt{\pi K}}
        e^{-j \frac{n^2}{4 K}}
        \left[ \mathrm{erf}\!\left(\gamma \left(\Omega_2 + \frac{n}{2K}\right)\right)
             - \mathrm{erf}\!\left(\gamma \left(\Omega_1 + \frac{n}{2K}\right)\right) \right],
        \qquad \gamma = \sqrt{K} \, e^{-j \pi / 4}

    (the band enters only through the two integration bounds; at
    :math:`[\Omega_1, \Omega_2] = [-\pi, \pi]` this collapses, using the
    oddness of :math:`\mathrm{erf}`, onto the usual full-band expression
    :math:`\mathrm{erf}(\kappa (2\pi K - n)) + \mathrm{erf}(\kappa (2\pi K + n))`
    with :math:`\kappa = e^{j 3 \pi / 4} / (2 \sqrt{K})`).

    The ridge-regularized system :math:`(Q + \epsilon I) h = d` is then
    *solved* rather than inverted. This matters away from the full band:
    :math:`Q` is a prolate matrix whose eigenvalues collapse to zero
    beyond the :math:`\approx N (\Omega_2 - \Omega_1) / (2\pi)` degrees of
    freedom the band can constrain, so the explicit inverse carries
    entries of order :math:`1/\epsilon` that cancel catastrophically
    against :math:`d`. The taps in that numerical null space are not
    individually determined -- they do not affect the response inside the
    band, which is the quantity the design pins down.

    On the full band :math:`[-\pi, \pi]` the degeneracy disappears:
    :math:`q(p) = \mathrm{sinc}(p) = 0` for :math:`p \neq 0` makes
    :math:`Q = I` exactly and :math:`h = d`. For taps well inside the
    group-delay span, :math:`|n| \ll 2 \pi K`, both erf arguments
    saturate and :math:`d[n] \to \sqrt{j / (4 \pi K)} e^{-j n^2 / (4K)}`:
    there the design falls back on the truncated (Savory) filter. The two
    keep differing near :math:`|n| \approx 2 \pi K`, where the Fresnel
    integral is only half-completed and the least-squares design tapers
    the taps that the truncation keeps at full amplitude. The block then
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
        Design band :math:`[\Omega_1, \Omega_2]` in rad/sample, with
        :math:`\Omega_1 < \Omega_2`. Default is ``[-np.pi, np.pi]``
        (full band). A narrower band trades accuracy outside the band
        for accuracy inside it, which is what one wants when the signal
        is oversampled.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"optimal"``.

    Attributes
    ----------
    h : np.ndarray
        Least-squares taps :math:`h[n]`, solution of :math:`Q h = d`.
    K : float
        Dimensionless dispersion :math:`K = -\beta_2 z f_s^2 / 2`.
    epsilon : float
        Class-level ridge term :math:`\epsilon` added to the diagonal of
        :math:`Q` before solving. Default is ``1e-14``.
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
        If the filter length ``N`` is even, or if the design band
        ``w_vect`` is not a strictly increasing pair.

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
    h: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    K: Optional[float] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.N % 2 == 0:
            raise ValueError(f"The value of N must be odd (current={self.N})")
        if len(self.w_vect) != 2:
            raise ValueError(
                f"design band w_vect holds {len(self.w_vect)} entries; "
                f"expected exactly 2 (Omega_1, Omega_2) in rad/sample; "
                f"pass a pair such as [-np.pi, np.pi]"
            )
        Omega_1, Omega_2 = self.w_vect
        if not Omega_1 < Omega_2:
            raise ValueError(
                f"design band w_vect is [{Omega_1}, {Omega_2}]; "
                f"expected Omega_1 < Omega_2 (a band of positive width in rad/sample); "
                f"swap the two entries"
            )

        beta2_ps2_per_km = compute_beta2(self.lamb, self.D, self.c)
        beta2 = ((10**-12)**2)*beta2_ps2_per_km  # convert into s^2/km
        K = -beta2 * self.z * (self.fs**2) / 2

        Q = _cd_gram_matrix(self.N, Omega_1, Omega_2)
        d_vect = _cd_cross_correlation(K, self.N, Omega_1, Omega_2)

        # Solve (Q + eps I) h = d. Never form the inverse: on a narrow band
        # Q is rank-deficient by construction and the explicit inverse holds
        # entries of order 1/eps, which cancel catastrophically against d.
        self.h = LA.solve(Q + self.epsilon * np.eye(self.N), d_vect)
        self.K = K

    def forward(self, x: np.ndarray) -> np.ndarray:
        return signal.convolve(x, self.h)
