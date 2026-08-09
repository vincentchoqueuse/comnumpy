import numpy as np
import numpy.linalg as LA
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from scipy import signal
from scipy.linalg import toeplitz
from scipy.optimize import least_squares
from comnumpy.core import Processor
from comnumpy.exceptions import NotFittedError
from .utils import hard_projector, zf_estimator, mmse_estimator
from .processors import Amplifier, DataExtractor
from .validators import validate_data, validate_single_path


@dataclass(slots=True)
class DataAidedMixin():
    def __post_init__(self):
        validate_data(self.reference)

    def validate_paths(self, X: np.ndarray) -> None:
        """Refuse a multi-path signal (D49).

        A data-aided estimator is fitted against one reference sequence,
        so it measures one record. Which of the two families it belongs
        to -- shared over the paths, or one per path -- depends on the
        quantity and on where in the receiver it sits, and the caller is
        the one who knows: this says so instead of broadcasting by
        accident.
        """
        validate_single_path(np.asarray(X), type(self).__name__,
                             self.estimand)

    estimand = "a quantity"

    def get_reference(self):
        """
        Retrieve the reference signal associated with the model.

        The reference is a plain array. When it is known in advance
        (preamble, training sequence) it is passed directly as
        ``reference=...``; when it is produced by the chain itself, the
        edge is declared with ``Sequential(wiring={"block.reference":
        "source"})`` so that each Monte-Carlo run uses its own reference.

        Returns
        -------
        np.ndarray
            The reference signal array.
        """
        return np.asarray(self.reference)


@dataclass(slots=True)
class DCCorrector(Processor):
    r"""Remove the DC component of a signal and set it to a target value.

    Signal Model
    ------------
    The empirical mean :math:`\mu_x` of the input is subtracted and
    replaced by the target value :math:`\alpha`:

    .. math::

       y[n] = x[n] - \mu_x + \alpha, \qquad
       \mu_x = \frac{1}{N} \sum_{n=0}^{N-1} x[n]

    so that the output satisfies the DC constraint
    :math:`\frac{1}{N}\sum_n y[n] = \alpha`. This is the digital
    counterpart of the DC-offset cancellation performed at the output of
    a direct-conversion (zero-IF) receiver, where the self-mixing of the
    local oscillator adds a constant term to the baseband signal.

    Axes: *declared axis* -- the mean :math:`\mu_x` is computed along
    ``axis`` (default 0) and broadcast over the remaining axes; use
    ``axis=-1`` for the canonical serial layout ``(..., N)``. The
    estimand is **per path** (D49): a DC offset belongs to a converter,
    so ``axis=-1`` on a ``(..., P, N)`` signal gives each path its own.

    Parameters
    ----------
    value : float, optional
        Target mean value :math:`\alpha` of the output. Default is 0.0,
        i.e. a zero-mean output.
    axis : int, optional, keyword-only
        Axis along which :math:`\mu_x` is computed. Default is 0.
    name : str, optional, keyword-only
        Name of the corrector instance. Default is ``"mean_corrector"``.

    References
    ----------
    B. Razavi, "Design considerations for direct-conversion receivers,"
    IEEE Transactions on Circuits and Systems II, vol. 44, no. 6,
    pp. 428-435, 1997 (DC offset in zero-IF receivers).

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 6.0])   # mean 3.0
    >>> print(DCCorrector()(x))
    [-2. -1.  0.  3.]
    >>> print(DCCorrector(2.0)(x))
    [0. 1. 2. 5.]
    """
    value: float = 0.0
    axis: int = field(default=0, kw_only=True)
    name: str = field(default="mean_corrector", kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # keepdims: without it the mean of a (P, N) signal along axis=-1
        # came back as (P,) and the subtraction raised a broadcast error,
        # so the block's own documented axis did not work above 1-D
        x_mean = np.mean(x, axis=self.axis, keepdims=True)
        y = x - x_mean + self.value
        return y


@dataclass(slots=True)
class Normalizer(Amplifier):
    r"""Scale a signal by a gain derived from one of its own statistics.

    Signal Model
    ------------
    The block is an ``Amplifier`` whose gain :math:`\alpha` is measured on
    the input instead of being configured:

    .. math::

       y[n] = \alpha \, x[n]

    The gain depends on the selected ``method`` and on the target value
    :math:`v`:

    - **'amp'** -- fixed gain, no measurement:

      .. math::

         \alpha = v

    - **'abs'** -- the largest modulus becomes :math:`v`:

      .. math::

         \alpha = \frac{v}{\max_n |x[n]|}

    - **'var'** -- the output variance becomes :math:`v`, with
      :math:`\sigma_x^2` the empirical variance of the input:

      .. math::

         \alpha = \sqrt{\frac{v}{\sigma_x^2}}

    - **'max'** -- the largest rectangular excursion becomes :math:`v`
      (the useful one before a DAC or a clipping stage):

      .. math::

         \alpha = \frac{v}{\max\left(\max_n |\Re e(x[n])|,\;
                                     \max_n |\Im m(x[n])|\right)}

    Axes: *element-wise* -- the gain :math:`\alpha` is a single scalar
    measured over the whole array in ``prepare()``, then applied
    pointwise.

    Unlike ``Amplifier``, the gain is *not* a parameter here: it is
    measured, so it is exposed as ``gain_`` (decision D23) and the
    inherited ``gain`` argument is removed from the constructor. The
    first positional argument is therefore ``method``, which makes
    ``Normalizer('max')`` mean what it reads.

    Parameters
    ----------
    method : {'amp', 'abs', 'var', 'max'}, optional
        Statistic used to derive the gain :math:`\alpha`. Default is
        ``'amp'`` (constant gain). It is the first positional argument.
    value : float, optional, keyword-only
        Target value :math:`v` of the selected statistic. Must be
        strictly positive. Default is 1.0.
    name : str, optional, keyword-only
        Name of the instance. Default is ``"signal_amplifier"``.

    Attributes
    ----------
    gain_ : float
        Gain :math:`\alpha` actually applied (data-dependent, hence the
        trailing underscore, decision D23). Measured from the input in
        ``prepare()`` for every method except ``'amp'``, and overwritten
        at each call.

    Raises
    ------
    ValueError
        If ``value`` is not strictly positive, or if ``method`` is not
        one of ``'amp'``, ``'abs'``, ``'var'``, ``'max'``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 3 (average energy and power of a
    constellation, and their normalization).

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> print(Normalizer('max', value=2.0)(x))
    [0.5 1.  1.5 2. ]
    >>> y = Normalizer(method='var')(x)          # unit-variance output
    >>> print(np.round(y, 4), round(float(np.var(y)), 6))
    [0.8944 1.7889 2.6833 3.5777] 1.0
    """
    METHODS = ('amp', 'abs', 'var', 'max')

    method: Literal['amp', 'abs', 'var', 'max'] = "amp"
    value: float = field(default=1., kw_only=True)
    # the gain of Amplifier is configured, the one of Normalizer is
    # measured: drop the inherited parameter (D23) so that the first
    # positional argument is `method`
    gain: float = field(init=False, repr=False, default_factory=lambda: 1.0)
    # estimated quantity (D23), declared for slots (D40a)
    gain_: float = field(init=False, repr=False, default_factory=lambda: 1.0)

    def __post_init__(self):
        if self.method not in self.METHODS:
            raise ValueError(
                f"Normalizer: method={self.method!r}, expected one of "
                f"{self.METHODS} -- pass the statistic as the first "
                "positional argument, e.g. Normalizer('max', value=2.0); "
                "the gain is measured, not configured.")
        if self.value <= 0:
            raise ValueError(
                f"Normalizer: value={self.value!r}, expected a strictly "
                "positive target -- set value to the wanted amplitude, "
                "power or variance.")
        self.gain_ = 1.0

    def prepare(self, X):
        match self.method:
            case "amp":
                gain = self.value
            case "abs":
                gain = self.value / np.max(np.abs(X))
            case "var":
                gain = np.sqrt(self.value / np.var(X))
            case "max":
                max_value = np.max([np.max(np.abs(np.real(X))), np.max(np.abs(np.imag(X)))])
                gain = self.value / max_value
            case _:
                # unreachable through __init__ (validated in __post_init__),
                # reachable if `method` is reassigned after construction
                raise ValueError(
                    f"Normalizer: method={self.method!r}, expected one of "
                    f"{self.METHODS} -- restore a valid statistic name.")

        self.gain_ = gain

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Amplifier.forward reads the configured `gain`; the Normalizer
        # applies the gain it measured in prepare() instead (D23).
        return self.gain_ * X


@dataclass(slots=True)
class BlindIQCompensator(Processor):

    r"""Blind IQ-imbalance compensator by whitening of the real covariance matrix.

    Signal Model
    ------------
    A receiver with gain and quadrature imbalance delivers an
    *improper* (non-circular) signal: the in-phase and quadrature
    components no longer have equal power, nor are they uncorrelated.
    The compensator applies the widely-linear transformation

    .. math::

        y[n] = c \left(\alpha \, \Re e(x[n]) + \beta \, \Im m(x[n])\right),
        \qquad c = \sqrt{\frac{P}{2}}

    where the complex coefficients :math:`\alpha` and :math:`\beta` are
    read column-wise from the :math:`2 \times 2` whitening matrix
    :math:`\mathbf{M}`, i.e.
    :math:`\alpha = M_{00} + j M_{10}` and
    :math:`\beta = M_{01} + j M_{11}`. Writing
    :math:`\mathbf{x}[n] = [\Re e(x[n]),\, \Im m(x[n])]^T`, the estimator
    eigen-decomposes the empirical covariance matrix

    .. math::

        \widehat{\mathbf{R}} = \frac{1}{N} \sum_{n=0}^{N-1}
        \mathbf{x}[n] \mathbf{x}^T[n]
        = \mathbf{U} \boldsymbol\Lambda \mathbf{U}^T,
        \qquad
        \mathbf{M} = \boldsymbol\Lambda^{-1/2} \mathbf{U}^T

    so that :math:`\mathbf{M} \widehat{\mathbf{R}} \mathbf{M}^T =
    \mathbf{I}_2`. The compensated signal is therefore second-order
    circular with the requested mean power :math:`P`:

    .. math::

        \mathbb{E}\left[\Re e^2(y[n])\right] =
        \mathbb{E}\left[\Im m^2(y[n])\right] = \frac{P}{2},
        \qquad
        \mathbb{E}\left[\Re e(y[n]) \Im m(y[n])\right] = 0

    Whitening only constrains the second-order statistics: the residual
    :math:`2 \times 2` rotation/reflection (and hence a possible I/Q swap
    or sign flip) is left to a downstream phase compensator. The
    estimator is therefore *not* the Gram-Schmidt orthogonalization
    procedure (GSOP) of Fatadin *et al.*, which whitens by successive
    projection and keeps the in-phase axis fixed.

    Axes: *declared axis* -- accepts ``(N,)`` or ``(..., P, N)``. The
    estimand is **per path** (D49): an IQ imbalance belongs to a
    receiver, so each path gets its own :math:`2 \times 2` covariance
    and its own whitening. Pooling them, which is what stacking the real
    and imaginary parts of a multi-path array used to do, mixes the
    paths and returns a signal worse than the one it was given.

    Parameters
    ----------
    should_fit : bool, optional
        If True (default), :math:`\alpha` and :math:`\beta` are
        re-estimated from the input at every call (per-block regime,
        D22). If False, the coefficients of the last ``fit`` are reused.
    coef : float, optional, keyword-only
        Target mean power :math:`P = \mathbb{E}[|y[n]|^2]` of the
        compensated signal. Default is 1.
    name : str, optional, keyword-only
        Name of the compensator instance. Default is
        ``"iq_compensator"``.

    Attributes
    ----------
    alpha_ : complex
        Estimated coefficient :math:`\alpha` applied to the in-phase
        component (data-dependent, hence the trailing underscore,
        decision D23; initialized to 1).
    beta_ : complex
        Estimated coefficient :math:`\beta` applied to the quadrature
        component (data-dependent, hence the trailing underscore,
        decision D23; initialized to 0).

    References
    ----------
    * I. Fatadin, S. J. Savory, D. Ives, "Compensation of quadrature
      imbalance in an optical QPSK coherent receiver," IEEE Photonics
      Technology Letters, vol. 20, no. 20, pp. 1733-1735, 2008,
      doi: 10.1109/LPT.2008.2004630.
    * P. J. Schreier, L. L. Scharf, *Statistical Signal Processing of
      Complex-Valued Data*, Cambridge University Press, 2010, Chapter 2
      (properness, circularity and widely-linear transformations).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> s = (rng.integers(0, 2, 2000) * 2 - 1) + 1j * (rng.integers(0, 2, 2000) * 2 - 1)
    >>> x = 2 * s.real + 1j * (0.5 * s.real + s.imag)   # gain + quadrature imbalance
    >>> y = BlindIQCompensator(coef=1.0)(x)
    >>> print(round(float(np.var(y.real)), 3), round(float(np.var(y.imag)), 3))
    0.499 0.5
    >>> print(round(float(abs(np.mean(y.real * y.imag))), 3), round(float(np.mean(np.abs(y)**2)), 3))
    0.0 1.0
    """
    should_fit: bool = True
    coef: float = field(default=1, kw_only=True)
    name: str = field(default="iq_compensator", kw_only=True)
    # estimated quantities (D23), declared for slots (D40a)
    alpha_: Union[complex, np.ndarray] = field(init=False, repr=False,
                                               default_factory=lambda: 1 + 0j)
    beta_: Union[complex, np.ndarray] = field(init=False, repr=False,
                                              default_factory=lambda: 0 + 0j)

    def __post_init__(self):
        self.alpha_ = 1 + 0j
        self.beta_ = 0 + 0j

    def fit(self, x):
        signal = np.asarray(x)
        N = signal.shape[-1]
        paths = signal.reshape(-1, N)
        alpha = np.empty(paths.shape[0], dtype=complex)
        beta = np.empty(paths.shape[0], dtype=complex)

        for index, path in enumerate(paths):
            X = np.vstack([path.real, path.imag])

            # compute covariance matrix
            R = (1/N) * np.matmul(X, np.transpose(X))

            # perform eigenvalue decomposition
            V, U = LA.eig(R)

            # perform whitening
            D = np.diag(1/np.sqrt(V))
            M = np.matmul(D, np.transpose(U))

            alpha[index] = M[0, 0] + 1j * M[1, 0]
            beta[index] = M[0, 1] + 1j * M[1, 1]

        # one coefficient per path, shaped to broadcast against the
        # signal -- and a plain scalar when there is a single path, the
        # same "scalar in, scalar out" rule the rest of the library uses
        if signal.ndim == 1:
            self.alpha_, self.beta_ = complex(alpha[0]), complex(beta[0])
        else:
            shape = signal.shape[:-1] + (1,)
            self.alpha_ = alpha.reshape(shape)
            self.beta_ = beta.reshape(shape)

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.should_fit:
            self.fit(x)

        coef = np.sqrt(self.coef/2)
        y = coef*(self.alpha_ * x.real + self.beta_ * x.imag)
        return y


@dataclass(slots=True)
class BlindCFOCompensator(Processor):
    r"""Blind carrier frequency offset compensator (fourth-power periodogram).

    Signal Model
    ------------
    The input carries a residual carrier frequency offset
    :math:`\omega_0` (in rad/sample), which the block removes:

    .. math::

        x[n] = s[n] \, e^{j \omega_0 n}, \qquad
        y[n] = x[n] \, e^{-j \widehat{\omega}_0 n}

    For a QPSK-like constellation, raising the signal to the fourth power
    wipes out the modulation (:math:`s^4[n]` is constant up to a sign)
    and leaves a pure tone at :math:`4\omega_0`. The offset is therefore
    obtained by maximizing the periodogram of :math:`x^4[n]`:

    .. math::

        \widehat{\omega}_0 = \frac{1}{4} \arg \max_{\omega}
        \frac{1}{N} \left| \sum_{n=0}^{N-1} x^4[n] \, e^{-j\omega n}
        \right|^2

    The maximization is performed in two stages: an optional coarse grid
    search over :math:`\omega_0`, followed by ``N_iter`` refinements of
    the criterion by Newton's method (or by plain gradient ascent). The
    Newton step uses the exact first and second derivatives of the
    periodogram with respect to :math:`\omega`.

    Axes: *declared axis* -- accepts ``(N,)`` or ``(..., P, N)``; the
    correction :math:`e^{-j\widehat{\omega}_0 n}` uses the sample index
    :math:`n` along the last axis. The estimand is **shared** (D49): a
    frequency offset is one laser beating against one local oscillator,
    so the periodogram is accumulated over every path and a single
    :math:`\widehat{\omega}_0` is applied to all of them. That is not
    only tidier than one estimate per path -- it is the better
    estimator, since it sees all the data.

    Parameters
    ----------
    w0_init : float, optional
        Initial value of :math:`\omega_0` in rad/sample, used as the
        starting point when ``grid_search`` is False. Default is 0.0.
    N_iter : int, optional, keyword-only
        Number :math:`N_{iter}` of local refinement steps. Default is 3.
    should_fit : bool, optional, keyword-only
        If True (default), :math:`\widehat{\omega}_0` is re-estimated at
        every call (per-block regime, D22); otherwise the value of the
        last ``fit`` is reused.
    grid_search : bool, optional, keyword-only
        If True (default), a coarse grid search initializes the local
        refinement.
    save_history : bool, optional, keyword-only
        If True, every intermediate value of :math:`\omega_0` is appended
        to ``history``. Default is False.
    method : {"grad", "newton"}, optional, keyword-only
        Local optimizer. Default is ``"newton"``.
    step_size : float, optional, keyword-only
        Gradient step used when ``method="grad"``. Default is 1e-8.
    grid_search_tuple : tuple, optional, keyword-only
        ``(start, stop, step)`` of the grid over :math:`\omega_0`, in
        rad/sample. Default is ``(-0.1, 0.1, 0.0001)``.
    name : str, optional, keyword-only
        Name of the compensator instance. Default is
        ``"cfo_compensator"``.

    Attributes
    ----------
    w0_ : float
        Estimated frequency offset :math:`\widehat{\omega}_0` in
        rad/sample (data-dependent, hence the trailing underscore,
        decision D23; None before the first ``fit``).
    history : list
        Successive iterates of :math:`\omega_0` when
        ``save_history=True``.

    Raises
    ------
    NotFittedError
        If ``forward`` is called with ``should_fit=False`` before any
        ``fit`` (decision D23).

    References
    ----------
    * A. J. Viterbi, A. M. Viterbi, "Nonlinear estimation of
      PSK-modulated carrier phase with application to burst digital
      transmission," IEEE Transactions on Information Theory, vol. 29,
      no. 4, pp. 543-551, 1983 (the fourth-power nonlinearity).
    * D. C. Rife, R. R. Boorstyn, "Single-tone parameter estimation from
      discrete-time observations," IEEE Transactions on Information
      Theory, vol. 20, no. 5, pp. 591-598, 1974 (periodogram maximization
      and its local refinement).
    * J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
      McGraw-Hill, 2008, Chapter 5 (carrier and symbol synchronization).

    Examples
    --------
    >>> alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    >>> rng = np.random.default_rng(2)
    >>> s = alphabet[rng.integers(0, 4, 500)]
    >>> x = s * np.exp(1j * 0.01 * np.arange(500))   # omega_0 = 0.01 rad/sample
    >>> compensator = BlindCFOCompensator()
    >>> y = compensator(x)
    >>> print(round(float(compensator.w0_), 5))
    0.01
    """
    w0_init: float = 0.0
    N_iter: int = field(default=3, kw_only=True)
    should_fit: bool = field(default=True, kw_only=True)
    grid_search: bool = field(default=True, kw_only=True)
    save_history: bool = field(default=False, kw_only=True)
    method: Literal["grad", "newton"] = field(default="newton", kw_only=True)
    step_size: float = field(default=1e-8, kw_only=True)
    grid_search_tuple: tuple = field(default=(-0.1, 0.1, 0.0001), kw_only=True)
    name: str = field(default="cfo_compensator", kw_only=True)
    # internal state (declared for slots, D40a)
    grid_search_array: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    history: list = field(init=False, repr=False, default_factory=list)
    # estimated quantity (D23), declared for slots (D40a)
    w0_: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        self.grid_search_array = np.arange(self.grid_search_tuple[0], self.grid_search_tuple[1], self.grid_search_tuple[2])
        self.history = []

    def loss(self, x, w):
        N = x.shape[-1]
        x4 = x**4
        dtft = self.compute_dtft(x4, w)
        return (np.abs(dtft)**2)/N

    def compute_dtft(self, x, w):
        # the sum runs over *every* axis, so several paths contribute to
        # one periodogram: the joint estimate of the shared offset (D49)
        N = x.shape[-1]
        N_vect = np.arange(N)
        dtft = np.sum(x*np.exp(-1j*w*N_vect))
        return dtft

    def callback(self, intermediate_result):
        if self.save_history:
            self.history.append(intermediate_result)

    def fit(self, x, w0):
        w = 4*w0
        N = x.shape[-1]
        x4 = x**4
        N_vect = np.arange(N)
        step_size = self.step_size

        if self.grid_search:
            w_vect = 4*self.grid_search_array
            cost_vect = np.zeros(len(w_vect))

            for index, w in enumerate(w_vect):
                cost_vect[index] = self.loss(x, w)

            index_max = np.argmax(cost_vect)
            w = w_vect[index_max]
            self.callback(4*w)

        for _ in range(self.N_iter):

            if self.method == "grad":
                dtft = self.compute_dtft(x4, w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                h = step_size * grad.real

            if self.method == "newton":
                dtft = self.compute_dtft(x4, w)
                dtft_diff = self.compute_dtft(-1j*N_vect*x4, w)
                dtft_diff2 = self.compute_dtft(-(N_vect**2)*x4, w)
                grad = (1/N) * (dtft_diff*np.conj(dtft) + dtft*np.conj(dtft_diff))
                J = (2/N) * (np.real(dtft_diff2*np.conj(dtft)) + np.abs(dtft_diff)**2)
                h = -grad.real/J.real

            w = w + h
            self.callback(4*w)

        self.w0_ = np.real(w)/4

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = x.shape[-1]
        N_vect = np.arange(N)

        if self.should_fit:
            self.fit(x, self.w0_init)
        elif self.w0_ is None:
            raise NotFittedError(
                "BlindCFOCompensator: forward called with should_fit=False "
                "but fit() was never called -- call fit(x, w0_init) first.")

        x = x*np.exp(-1j*self.w0_*N_vect)
        return x


@dataclass(slots=True)
class BlindPhaseCompensation(Processor):
    r"""Blind phase compensator minimizing the error vector magnitude (EVM).

    Signal Model
    ------------
    The input carries an unknown constant phase offset, removed by

    .. math::

        y[n] = x[n] \, e^{j \widehat{\theta}}

    No reference is available (blind regime): the estimator uses the
    constellation itself as a soft reference and minimizes the squared
    distance to the nearest alphabet point, i.e. the EVM,

    .. math::

        \widehat{\theta} = \arg \min_{\theta} \sum_{n=0}^{N-1}
        \left| x[n] e^{j\theta}
        - \mathcal{P}_{\mathcal{A}}\!\left(x[n] e^{j\theta}\right)
        \right|^2

    where :math:`\mathcal{P}_{\mathcal{A}}` is the hard projection onto
    the alphabet :math:`\mathcal{A}`. This is the decision-directed
    carrier phase estimator; the criterion is solved by a nonlinear
    least-squares search (``scipy.optimize.least_squares``) started at
    :math:`\theta_0`. Being decision-directed, it inherits the
    :math:`2\pi/M` phase ambiguity of the constellation, so
    :math:`\theta_0` must lie in the correct basin.

    Axes: *declared axis* -- accepts ``(N,)`` or ``(..., P, N)``. The
    estimand is **per path** (D49): after a butterfly equalizer each
    output carries its own residual rotation, so one angle is fitted per
    path and ``theta_`` holds one value each. A phase *common* to the
    paths, laser noise before any equalizer, is the shared case -- fit
    it on the flattened signal.

    Parameters
    ----------
    alphabet : np.ndarray
        Modulation alphabet :math:`\mathcal{A}`, a 1D array of
        constellation symbols.
    theta0 : float, optional, keyword-only
        Starting point :math:`\theta_0` of the optimizer, in radians.
        Default is 0.
    should_fit : bool, optional, keyword-only
        If True (default), :math:`\widehat{\theta}` is re-estimated at
        every call (per-block regime, D22); otherwise the value of the
        last ``fit`` is reused.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"phase correction"``.

    Attributes
    ----------
    theta_ : float
        Estimated phase :math:`\widehat{\theta}` in radians
        (data-dependent, hence the trailing underscore, decision D23).

    Raises
    ------
    NotFittedError
        If ``forward`` is called with ``should_fit=False`` before any
        ``fit`` (decision D23).

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 5 (decision-directed carrier phase
    estimation).

    Examples
    --------
    >>> alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    >>> rng = np.random.default_rng(1)
    >>> s = alphabet[rng.integers(0, 4, 200)]
    >>> compensator = BlindPhaseCompensation(alphabet)
    >>> y = compensator(s * np.exp(-1j * 0.3))    # a -0.3 rad rotation
    >>> print(round(float(compensator.theta_), 4), bool(np.allclose(y, s)))
    0.3 True
    """
    alphabet: np.ndarray
    theta0: float = field(default=0.0, kw_only=True)
    should_fit: bool = field(default=True, kw_only=True)
    name: str = field(default="phase correction", kw_only=True)
    # estimated quantity (D23), declared for slots (D40a)
    theta_: Optional[Union[float, np.ndarray]] = field(init=False, repr=False, default_factory=lambda: None)

    def cost(self, theta: float, x: np.ndarray) -> np.ndarray:
        y = x * np.exp(1j * theta)
        s, y_est = hard_projector(y, self.alphabet)
        error = np.ravel(y - y_est)
        error_real = np.hstack([np.real(error), np.imag(error)])
        return error_real

    def fit(self, X: np.ndarray, y=None):
        signal = np.asarray(X)
        paths = signal.reshape(-1, signal.shape[-1])
        angles = np.array([least_squares(self.cost, self.theta0,
                                         args=(path,)).x[0]
                           for path in paths])
        # one angle per path, shaped to broadcast against the signal --
        # and a plain float when there is a single path
        self.theta_ = (float(angles[0]) if signal.ndim == 1
                       else angles.reshape(signal.shape[:-1] + (1,)))
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.should_fit:
            self.fit(X)
        elif self.theta_ is None:
            raise NotFittedError(
                "BlindPhaseCompensation: forward called with should_fit=False "
                "but fit() was never called -- call fit(x) first.")
        Y = X * np.exp(1j * self.theta_)
        return Y


@dataclass(slots=True)
class LinearEqualizer(Processor):
    r"""Block linear equalizer (ZF or MMSE) for a known FIR channel.

    Signal Model
    ------------
    The block input is the received signal of a channel with inter-symbol
    interference (ISI), of known impulse response :math:`h[l]` of length
    :math:`L`:

    .. math::

        x[n] = \sum_{l=0}^{L-1} h[l] \, s[n-l] + b[n], \qquad
        b[n] \sim \mathcal{CN}\left(0, \sigma^2\right)

    Stacking the :math:`N` received samples gives
    :math:`\mathbf{x} = \mathbf{H}\mathbf{s} + \mathbf{b}`, where
    :math:`\mathbf{H}` is the :math:`N \times (N-L+1)` banded Toeplitz
    convolution matrix built from :math:`h[l]`. The block returns the
    estimate :math:`\mathbf{y} = \widehat{\mathbf{s}}` of the
    :math:`N-L+1` transmitted symbols, by one of two closed forms:

    .. math::

        \widehat{\mathbf{s}}_{\mathrm{ZF}} &=
        \mathbf{H}^{\dagger} \mathbf{x}
        = \left(\mathbf{H}^H \mathbf{H}\right)^{-1} \mathbf{H}^H \mathbf{x} \\
        \widehat{\mathbf{s}}_{\mathrm{MMSE}} &=
        \left(\mathbf{H}^H \mathbf{H}
        + \sigma^2 \mathbf{I}\right)^{-1} \mathbf{H}^H \mathbf{x}

    The zero-forcing solution cancels the ISI exactly but amplifies the
    noise wherever :math:`\mathbf{H}` is ill-conditioned; the MMSE
    solution trades residual ISI against noise enhancement through the
    loading term :math:`\sigma^2` (written for unit-power symbols,
    :math:`\mathbb{E}[|s[n]|^2] = 1`). The two coincide when
    :math:`\sigma^2 = 0`.

    Axes: *declared axis* -- expects a 1D serial signal ``(N,)`` and
    returns ``(N-L+1,)``.

    Parameters
    ----------
    h : np.ndarray
        Channel impulse response :math:`h[l]`, a 1D array of length
        :math:`L`. It is *configured*, not estimated: use
        ``DataAidedFIRCompensator`` when the channel is unknown.
    method : {"zf", "mmse"}, optional, keyword-only
        Closed form used. Default is ``"zf"``.
    sigma2 : float, optional, keyword-only
        Noise variance :math:`\sigma^2`, used by ``"mmse"`` only.
        Default is 0.0.
    name : str, optional, keyword-only
        Name of the equalizer instance. Default is ``"equalizer"``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 9 (linear equalization of band-limited
    channels: peak-distortion / zero-forcing and MMSE criteria).

    Examples
    --------
    >>> h = np.array([1.0, 0.5])
    >>> s = np.array([1.0, -1.0, 1.0, 1.0])
    >>> x = np.convolve(s, h)                       # noiseless ISI channel
    >>> print(np.round(LinearEqualizer(h)(x).real, 6))
    [ 1. -1.  1.  1.]
    >>> print(np.round(LinearEqualizer(h, method="mmse", sigma2=0.1)(x).real, 3))
    [ 0.862 -0.827  0.872  0.973]
    """
    h: np.ndarray
    method: Literal["zf", "mmse"] = field(default="zf", kw_only=True)
    sigma2: float = field(default=0.0, kw_only=True)
    name: str = field(default="equalizer", kw_only=True)


    def get_H(self, N):
        Nx = N - len(self.h)
        c = np.r_[self.h, np.zeros(Nx)]
        r = np.r_[self.h[0], np.zeros(Nx)]
        H = toeplitz(c, r)
        return H

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).astype(complex)
        H = self.get_H(X.shape[-1])

        match self.method:
            case "zf":
                Y = zf_estimator(X, H)
            case "mmse":
                Y = mmse_estimator(X, H, self.sigma2)
        return Y


@dataclass(slots=True)
class DataAidedFIRCompensator(DataAidedMixin, Processor):
    r"""Data-aided FIR compensator: least-squares channel estimation, then deconvolution.

    Signal Model
    ------------
    The block observes the output of an unknown FIR channel driven by a
    known reference :math:`d[n]` of length :math:`N` (preamble, training
    sequence):

    .. math::

        x[n] = \sum_{l} h[l] \, d[n-l]

    The impulse response is estimated by least squares (zero-forcing
    criterion) from the :math:`N \times N` lower-triangular Toeplitz
    convolution matrix :math:`\mathbf{D}` built from :math:`d[n]`:

    .. math::

        \widehat{\mathbf{h}} = \arg \min_{\mathbf{h}}
        \left\| \mathbf{D} \mathbf{h} - \mathbf{x} \right\|^2
        = \mathbf{D}^{\dagger} \mathbf{x}

    The correction is the deconvolution of the input by that estimate,
    so that :math:`y[n] \simeq d[n]`:

    .. math::

        y[n] = \left(x * \widehat{h}^{-1}\right)[n]

    Because :math:`\mathbf{D}` is square here, the input and the
    reference must have the same length :math:`N`.

    Axes: *declared axis* -- expects a 1D serial signal ``(N,)`` of the
    same length as ``reference``.

    Parameters
    ----------
    reference : np.ndarray
        Known reference :math:`d[n]`. When the reference is produced by
        the chain itself, declare the edge with
        ``Sequential(wiring={"data_aided_fir.reference": "source"})``
        instead of freezing an array.
    should_fit : bool, optional, keyword-only
        If True (default), the channel is re-estimated at every call
        (per-block regime, D22); otherwise the estimate of the last
        ``fit`` is reused.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"data_aided_fir"``.

    Attributes
    ----------
    h_ : np.ndarray
        Estimated impulse response :math:`\widehat{\mathbf{h}}`
        (data-dependent, hence the trailing underscore, decision D23).
        The response is never configured here: use
        ``LinearEqualizer(h)`` when it is already known.

    Raises
    ------
    NotFittedError
        If ``forward`` is called with ``should_fit=False`` before any
        ``fit`` (decision D23).

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 9 (training-sequence channel estimation
    and zero-forcing equalization).

    Examples
    --------
    >>> x_ref = np.array([1.0, -1.0, 1.0, 1.0])
    >>> received = np.convolve(x_ref, [1.0, 0.5])[:4]
    >>> compensator = DataAidedFIRCompensator(reference=x_ref)
    >>> print(np.round(compensator(received), 6))
    [ 1. -1.  1.  1.]
    >>> print(np.round(compensator.h_, 6) + 0.0)
    [1.  0.5 0.  0. ]
    """

    reference: np.ndarray
    should_fit: bool = field(default=True, kw_only=True)
    name: str = field(default="data_aided_fir", kw_only=True)
    # estimated quantity (D23), declared for slots (D40a)
    h_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    estimand = "one FIR response against a reference"

    def fit(self, x, y=None):
        if y is None:
            y = self.get_reference()
        L = len(x)
        H = toeplitz(y, np.zeros(L))
        self.h_ = np.matmul(LA.pinv(H), x)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_paths(x)
        if self.should_fit:
            self.fit(x)
        elif self.h_ is None:
            raise NotFittedError(
                "DataAidedFIRCompensator: forward called with "
                "should_fit=False but fit() was never called -- call "
                "fit(x) first.")

        L = len(self.h_)
        y, _ = signal.deconvolve(np.hstack([x, np.zeros(L-1)]), self.h_)
        return y


@dataclass(slots=True)
class DataAidedPhaseCompensator(DataAidedMixin, Processor):
    r"""Data-aided phase compensator (maximum-likelihood phase from a known reference).

    Signal Model
    ------------
    The input carries an unknown constant phase offset with respect to a
    known reference :math:`d[n]`:

    .. math::

        x[n] = d[n] \, e^{-j\theta} + b[n], \qquad
        y[n] = x[n] \, e^{j\widehat{\theta}}

    In additive white Gaussian noise, the maximum-likelihood estimate of
    :math:`\theta` is the argument of the cross-correlation between the
    input and the reference at lag zero:

    .. math::

        \widehat{\theta} = \arg \left(
        \sum_{n=0}^{N-1} x^*[n] \, d[n] \right)

    Unlike the blind estimator, this one is free of the :math:`2\pi/M`
    constellation ambiguity, since the reference fixes the absolute
    phase.

    Axes: *declared axis* -- estimation expects a 1D serial signal
    ``(N,)`` aligned with ``reference``; the correction is element-wise.

    Parameters
    ----------
    reference : np.ndarray
        Known reference :math:`d[n]`. When the reference is produced by
        the chain itself, declare the edge with
        ``Sequential(wiring={"data_aided_phase.reference": "source"})``
        instead of freezing an array.
    name : str, optional, keyword-only
        Name of the processor. Default is ``"data_aided_phase"``.

    Attributes
    ----------
    theta_ : float
        Estimated phase :math:`\widehat{\theta}` in radians
        (data-dependent, hence the trailing underscore, decision D23).
        Re-estimated at every call (per-block regime, D22).

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 5 (data-aided maximum-likelihood carrier
    phase estimation).

    Examples
    --------
    >>> d = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    >>> x = d * np.exp(-1j * np.pi / 4)          # a -pi/4 rotation
    >>> compensator = DataAidedPhaseCompensator(reference=d)
    >>> y = compensator(x)
    >>> print(round(float(compensator.theta_), 6), round(float(np.pi / 4), 6))
    0.785398 0.785398
    >>> print(np.round(y, 6))
    [ 0.707107+0.707107j  0.707107-0.707107j -0.707107+0.707107j
     -0.707107-0.707107j]
    """
    reference: np.ndarray
    name: str = field(default="data_aided_phase", kw_only=True)
    # estimated quantity (D23), declared for slots (D40a)
    theta_: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)

    estimand = "one phase against a reference"

    def __post_init__(self):
        # explicit parent call: zero-arg super() breaks with slots=True
        # (the dataclass decorator recreates the class)
        DataAidedMixin.__post_init__(self)
        self.theta_ = None

    def fit(self, x, y=None):
        if y is None:
            y = self.get_reference()
        self.theta_ = np.angle(np.sum(np.conj(x)*y))
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_paths(x)
        self.fit(x)
        return x*np.exp(1j*self.theta_)


@dataclass(slots=True)
class DataAidedComplexGainCompensator(DataAidedMixin, Processor):
    r"""Data-aided compensator of a flat complex gain (amplitude and phase).

    Signal Model
    ------------
    The channel is assumed frequency-flat over the record, so it reduces
    to a single complex coefficient. A preamble is extracted from the
    input and compared to the known reference :math:`d[n]` of length
    :math:`N_d`:

    .. math::

        \widehat{g} = \arg \min_{g} \sum_{n=0}^{N_d-1}
        \left| g \, x[n] - d[n] \right|^2
        = \frac{\sum_{n} x^*[n] \, d[n]}{\sum_{n} |x[n]|^2}

    The estimate is the *compensation* gain (an estimate of the inverse
    channel gain), applied to the whole record:

    .. math::

        y[n] = \widehat{g} \, x[n]

    It corrects amplitude and phase at once, and generalizes
    ``DataAidedPhaseCompensator``, which keeps only
    :math:`\arg(\widehat{g})`.

    Axes: *declared axis* -- estimation expects a 1D preamble ``(N_d,)``
    (after ``extractor``) aligned with ``reference``; the correction is
    element-wise on the full record.

    Parameters
    ----------
    reference : np.ndarray
        Known preamble :math:`d[n]`. When it is produced by the chain
        itself, declare the edge with
        ``Sequential(wiring={"complex_gain_compensator.reference":
        "source"})`` instead of freezing an array.
    extractor : DataExtractor, optional, keyword-only
        Selects the preamble samples inside the received record. Default
        is a pass-through (the whole record is the preamble).
    should_fit : bool, optional, keyword-only
        If True (default), :math:`\widehat{g}` is re-estimated at every
        call (per-block regime, D22); if False, the gain of the last
        ``fit`` is reused (reused-preamble regime).
    name : str, optional, keyword-only
        Name of the processor. Default is
        ``"complex_gain_compensator"``.

    Attributes
    ----------
    gain_ : complex
        Estimated complex gain :math:`\widehat{g}` (data-dependent, hence
        the trailing underscore, decision D23).

    Raises
    ------
    NotFittedError
        If ``forward`` is called with ``should_fit=False`` before any
        ``fit`` (decision D23).

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 5 (data-aided estimation of the carrier
    amplitude and phase).

    Examples
    --------
    >>> d = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    >>> x = 2 * np.exp(1j * np.pi / 3) * d           # channel gain 2 exp(j pi/3)
    >>> compensator = DataAidedComplexGainCompensator(reference=d)
    >>> y = compensator(x)
    >>> print(np.round(compensator.gain_, 6))        # 0.5 exp(-j pi/3)
    (0.25-0.433013j)
    >>> print(np.round(y, 6))
    [ 1.+1.j  1.-1.j -1.+1.j -1.-1.j]
    """
    reference: np.ndarray
    extractor: DataExtractor = field(default_factory=lambda: DataExtractor(selector=None), kw_only=True)
    should_fit: bool = field(default=True, kw_only=True)
    name: str = field(default="complex_gain_compensator", kw_only=True)
    # estimated quantity (D23), declared for slots (D40a)
    gain_: Optional[complex] = field(init=False, repr=False, default_factory=lambda: None)

    estimand = "one complex gain against a reference"

    def fit(self, x, y=None):
        if y is None:
            y = self.get_reference()
        x_resized = np.resize(x, (len(x), 1))
        x_pinv = np.linalg.pinv(x_resized)
        self.gain_ = np.dot(x_pinv, y).item()
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_paths(x)
        if self.should_fit:
            x_preamble = self.extractor(x)
            self.fit(x_preamble)
        elif self.gain_ is None:
            raise NotFittedError(
                "DataAidedComplexGainCompensator: forward called with "
                "should_fit=False but fit() was never called -- call "
                "fit(x_preamble) first.")

        y = x * self.gain_
        return y


@dataclass(slots=True)
class DataAidedSimpleSynchronizer(DataAidedMixin, Processor):
    r"""Data-aided frame synchronizer at one-sample resolution (correlation peak).

    Signal Model
    ------------
    The received record is a delayed and scaled copy of a known preamble
    :math:`d[n]` of length :math:`N_d`, followed by the payload:

    .. math::

        x[n] = a \, d[n - m_0] + b[n]

    The delay is obtained from the peak of the cross-correlation between
    the input and the preamble, normalized by the preamble energy:

    .. math::

        c[m] = \frac{\sum_{n} x[n+m] \, d^*[n]}{\sum_{n} |d[n]|^2},
        \qquad
        \widehat{m}_0 = \arg \max_{m} \left| c[m] \right|^2

    That normalization makes the peak a least-squares estimate of the
    channel gain itself, :math:`\widehat{a} = c[\widehat{m}_0] \simeq a`.
    The output realigns the record on the preamble and, when
    ``scale_correction`` is enabled, *divides* it by that gain, so that
    the block restores the transmitted signal instead of returning a
    doubly-distorted copy of it:

    .. math::

        y[n] = \frac{x[n + \widehat{m}_0]}{\widehat{a}}

    For a unit channel gain (:math:`a = 1`), :math:`\widehat{a} = 1` and
    the block is a pure realignment, whatever the preamble power.

    Axes: *declared axis* -- expects a 1D serial signal ``(N,)``; the
    output length depends on the estimated delay and on ``signal_len``.

    Parameters
    ----------
    reference : np.ndarray
        Known preamble :math:`d[n]`. When it is produced by the chain
        itself, declare the edge with
        ``Sequential(wiring={"synchronizer.reference": "source"})``
        instead of freezing an array.
    scale_correction : bool, optional
        If True (default), the output is divided by the complex peak
        :math:`\widehat{a}`; otherwise only the delay is corrected.
    save_cross_correlation : bool, optional, keyword-only
        If True (default), stores :math:`c[m]` and its lag axis for
        inspection or plotting (``plot``).
    signal_len : int, optional, keyword-only
        Truncates the realigned signal to that length. Default is None
        (keep everything after the peak).
    name : str, optional, keyword-only
        Name of the synchronizer instance. Default is ``"synchronizer"``.

    Attributes
    ----------
    delay_ : int
        Estimated delay :math:`\widehat{m}_0` in samples (data-dependent,
        hence the trailing underscore, decision D23; re-estimated at
        every call).
    scale_ : complex
        Compensation gain :math:`1/\widehat{a}` actually applied, or 1
        when ``scale_correction`` is False.
    cross_corr_ : np.ndarray
        Full normalized cross-correlation :math:`c[m]` (when saved).
    n_vect_ : np.ndarray
        Lag axis :math:`m` associated with ``cross_corr_``.

    Raises
    ------
    ValueError
        If the reference has zero energy, which makes the normalized
        correlation undefined.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 5 (frame and symbol synchronization by
    correlation with a known preamble).

    Examples
    --------
    >>> d = np.array([1.0, -1.0, 1.0, 1.0])
    >>> x = np.hstack([np.zeros(2), d, [1.0, -1.0]])   # delayed by 2 samples
    >>> synchronizer = DataAidedSimpleSynchronizer(d)
    >>> y = synchronizer(x)
    >>> print(synchronizer.delay_, round(float(synchronizer.scale_), 6))
    2 1.0
    >>> print(y)
    [ 1. -1.  1.  1.  1. -1.]
    >>> print(np.round(DataAidedSimpleSynchronizer(d)(0.5 * x), 6))   # 6 dB attenuation
    [ 1. -1.  1.  1.  1. -1.]
    """
    reference: np.ndarray
    scale_correction: bool = True
    save_cross_correlation: bool = field(default=True, kw_only=True)
    signal_len: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="synchronizer", kw_only=True)
    # estimated quantities (D23), declared for slots (D40a)
    delay_: Optional[int] = field(init=False, repr=False, default_factory=lambda: None)
    scale_: complex = field(init=False, repr=False, default_factory=lambda: 1)
    cross_corr_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    n_vect_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    estimand = "one delay against a reference"

    def __post_init__(self):
        # explicit parent call: zero-arg super() breaks with slots=True
        DataAidedMixin.__post_init__(self)
        self.delay_ = None
        self.scale_ = 1
        self.cross_corr_ = None
        self.n_vect_ = None

    def fit(self, x, y=None):
        if y is None:
            y = self.get_reference()
        x_preamble = y
        N = len(x)
        N_preamble = len(x_preamble)

        x_preamble_padded = np.zeros(N, dtype=x.dtype)
        x_preamble_padded[:N_preamble] = x_preamble

        # compute the cross correlation, normalized by the preamble
        # energy so that its peak estimates the channel gain a itself
        energy = float(np.sum(np.abs(x_preamble)**2))
        if energy == 0:
            raise ValueError(
                "DataAidedSimpleSynchronizer: reference energy is 0, "
                "expected a non-zero preamble -- pass the transmitted "
                "preamble as reference.")
        cross_corr = np.correlate(x, x_preamble_padded, mode='full')
        cross_corr = cross_corr / energy
        n_vect = np.arange(len(cross_corr)) - (N - 1)

        # Find the time delay: the index of the maximum cross-correlation
        # minus the length of x minus 1
        index_max = np.argmax(np.abs(cross_corr)**2)
        value_max = cross_corr[index_max]

        self.delay_ = n_vect[index_max]
        if self.scale_correction:
            # compensation gain: the inverse of the estimated channel gain
            self.scale_ = 1/value_max

        # save correlation if needed
        if self.save_cross_correlation:
            self.cross_corr_ = cross_corr
            self.n_vect_ = n_vect

    def plot(self, ax=None):
        import matplotlib.pyplot as plt  # local import (D36)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.n_vect_, np.abs(self.cross_corr_))
        ax.set_title('Cross-correlation magnitude')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        return ax

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_paths(x)
        self.fit(x)
        if self.signal_len:
            y = self.scale_*x[self.delay_:self.delay_+self.signal_len]
        else:
            y = self.scale_*x[self.delay_:]
        return y


@dataclass(slots=True)
class DataAidedFineSynchronizer(DataAidedMixin, Processor):

    r"""Data-aided frame synchronizer at fractional-sample resolution.

    Signal Model
    ------------
    Same correlation-based model as ``DataAidedSimpleSynchronizer``, but
    the delay :math:`\tau_0` is no longer assumed to be an integer number
    of samples:

    .. math::

        x[n] = a \, d\!\left(nT - \tau_0\right) + b[n]

    The input and the preamble are first interpolated by a polyphase
    resampler of factor :math:`U` (``scipy.signal.resample_poly``), which
    brings the search grid down to :math:`T/U`:

    .. math::

        c[m] = \frac{\sum_{n} x_{\uparrow}[n+m] \,
        d_{\uparrow}^*[n]}{\sum_{n} |d_{\uparrow}[n]|^2}, \qquad
        \widehat{m}_0 = \max\left(0,\;
        \arg \max_{m} \left| c[m] \right|^2 \right)

    As in ``DataAidedSimpleSynchronizer``, the normalization by the
    preamble energy makes the peak an estimate of the channel gain,
    :math:`\widehat{a} = c[\widehat{m}_0] \simeq a`. The upsampled record
    is realigned on that grid, decimated back by :math:`U`, and
    optionally *divided* by that gain:

    .. math::

        \widehat{\tau}_0 = \frac{\widehat{m}_0}{U} T, \qquad
        y[n] = \frac{x_{\uparrow}\!\left[U n + \widehat{m}_0\right]}
                    {\widehat{a}}

    Negative lags are clamped to zero, so the block only corrects
    non-negative delays.

    Axes: *declared axis* -- expects a 1D serial signal ``(N,)``; the
    output length depends on the estimated delay and on ``signal_len``.

    Parameters
    ----------
    reference : np.ndarray
        Known preamble :math:`d[n]`. When it is produced by the chain
        itself, declare the edge with
        ``Sequential(wiring={"synchronizer.reference": "source"})``
        instead of freezing an array.
    scale_correction : bool, optional
        If True (default), the output is multiplied by the complex peak
        :math:`\widehat{a}`; otherwise only the delay is corrected.
    up_factor : int, optional, keyword-only
        Interpolation factor :math:`U` applied before the correlation;
        the timing resolution is :math:`T/U`. Default is 2.
    save_cross_correlation : bool, optional, keyword-only
        If True (default), stores :math:`c[m]` and its lag axis for
        inspection or plotting (``plot``).
    signal_len : int, optional, keyword-only
        Truncates the realigned signal to that length. Default is None.
    d_max : int, optional, keyword-only
        Maximum expected delay, in samples of :math:`x`. Restricts the
        correlation window and therefore the search range. Default is
        None (search over the whole record).
    name : str, optional, keyword-only
        Name of the synchronizer instance. Default is ``"synchronizer"``.

    Attributes
    ----------
    delay_ : int
        Estimated delay :math:`\widehat{m}_0`, expressed in *upsampled*
        samples (data-dependent, hence the trailing underscore, decision
        D23; divide by ``up_factor`` to get it in symbol periods).
    scale_ : complex
        Compensation gain :math:`1/\widehat{a}` actually applied, or 1
        when ``scale_correction`` is False.
    cross_corr_ : np.ndarray
        Full normalized cross-correlation :math:`c[m]` on the upsampled
        grid.
    n_vect_ : np.ndarray
        Lag axis :math:`m` associated with ``cross_corr_``.

    Raises
    ------
    ValueError
        If the reference has zero energy, which makes the normalized
        correlation undefined.

    References
    ----------
    * J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
      McGraw-Hill, 2008, Chapter 5 (frame and symbol synchronization).
    * F. M. Gardner, "Interpolation in digital modems -- Part I:
      Fundamentals," IEEE Transactions on Communications, vol. 41, no. 3,
      pp. 501-507, 1993 (interpolation-based fractional timing recovery).

    Examples
    --------
    >>> d = np.array([1.0, -1.0, 1.0, 1.0])
    >>> x = np.hstack([np.zeros(3), d, np.zeros(3)])   # delayed by 3 samples
    >>> synchronizer = DataAidedFineSynchronizer(d, up_factor=2, signal_len=4)
    >>> y = synchronizer(x)
    >>> print(synchronizer.delay_, synchronizer.delay_ / synchronizer.up_factor, y.shape)
    6 3.0 (4,)
    """
    reference: np.ndarray
    scale_correction: bool = True
    up_factor: int = field(default=2, kw_only=True)
    save_cross_correlation: bool = field(default=True, kw_only=True)
    signal_len: Optional[int] = field(default=None, kw_only=True)
    d_max: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="synchronizer", kw_only=True)
    # estimated quantities (D23), declared for slots (D40a)
    delay_: Optional[int] = field(init=False, repr=False, default_factory=lambda: None)
    scale_: complex = field(init=False, repr=False, default_factory=lambda: 1)
    cross_corr_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    n_vect_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    estimand = "one fractional delay against a reference"

    def __post_init__(self):
        # explicit parent call: zero-arg super() breaks with slots=True
        DataAidedMixin.__post_init__(self)
        self.delay_ = None
        self.scale_ = 1
        self.cross_corr_ = None
        self.n_vect_ = None

    def fit(self, x, y=None):
        if y is None:
            y = self.get_reference()
        x_preamble = y
        N = len(x)
        N_preamble = len(x_preamble)
        x_preamble_padded = np.zeros(N, dtype=x.dtype)
        x_preamble_padded[:N_preamble] = x_preamble

        # compute the cross correlation, normalized by the preamble
        # energy so that its peak estimates the channel gain a itself
        energy = float(np.sum(np.abs(x_preamble)**2))
        if energy == 0:
            raise ValueError(
                "DataAidedFineSynchronizer: reference energy is 0, "
                "expected a non-zero preamble -- pass the transmitted "
                "preamble as reference.")
        cross_corr = np.correlate(x,  x_preamble_padded, mode='full')
        cross_corr = cross_corr / energy
        n_vect = np.arange(len(cross_corr)) - (N - 1)

        # Find the time delay: the index of the maximum cross-correlation
        # minus the length of x minus 1
        index_max = np.argmax(np.abs(cross_corr)**2)
        value_max = cross_corr[index_max]

        self.delay_ = np.max([n_vect[index_max], 0])
        if self.scale_correction:
            # compensation gain: the inverse of the estimated channel gain
            self.scale_ = 1/value_max

        # save correlation if needed
        if self.save_cross_correlation:
            self.cross_corr_ = cross_corr
            self.n_vect_ = n_vect

    def plot(self, ax=None):
        import matplotlib.pyplot as plt  # local import (D36)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.n_vect_, np.abs(self.cross_corr_))
        ax.set_title('Cross-correlation magnitude')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        return ax

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.validate_paths(x)
        x_preamble = self.get_reference()

        # upsampling
        x_up = signal.resample_poly(x, self.up_factor, 1)
        x_preamble_up = signal.resample_poly(x_preamble, self.up_factor, 1)

        if not self.d_max:
            Nmax = len(x_up)
        else:
            Nmax = min(len(x_up), len(x_preamble_up)+int((self.d_max+1)*self.up_factor))

        self.fit(x_up[:Nmax], x_preamble_up)

        y_up = x_up[self.delay_:]
        # downsampling
        y = signal.resample_poly(y_up, 1, self.up_factor)

        y = self.scale_*y
        if self.signal_len:
            y = y[:self.signal_len]

        return y
