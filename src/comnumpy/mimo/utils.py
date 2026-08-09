import numpy as np
from typing import Optional, Sequence


def rayleigh_channel(N_r: int, N_t: int,
                L: int = 1,
                scale_per_tap: Optional[Sequence[float]] = None,
                seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    Draw an i.i.d. Rayleigh MIMO channel.

    Signal Model
    ------------
    The channel taps feed the MIMO observation model

    .. math::

        \mathbf{y}[n] = \sum_{l=0}^{L-1} \mathbf{H}[l]\,
        \mathbf{x}[n-l] + \mathbf{b}[n]

    Each coefficient of the :math:`N_r \times N_t` matrix
    :math:`\mathbf{H}[l]` is drawn independently from a circularly
    symmetric complex normal law

    .. math::

        H_{ij}[l] \sim \mathcal{CN}\left(0, \gamma_l\right),
        \qquad \gamma_l = \texttt{scale\_per\_tap}[l]

    i.e. real and imaginary parts are independent
    :math:`\mathcal{N}(0, \gamma_l/2)`, so that
    :math:`\mathbb{E}\left[|H_{ij}[l]|^2\right] = \gamma_l`. The modulus
    :math:`|H_{ij}[l]|` is then Rayleigh-distributed and the instantaneous
    power is exponential -- the classical model for a rich scattering
    environment with no dominant path.

    Normalization: with the default :math:`\gamma_l = 1` every antenna
    pair has unit average gain (no :math:`1/\sqrt{N_t}` transmit
    normalization is applied, hence
    :math:`\mathbb{E}\|\mathbf{H}[l]\|_F^2 = N_r N_t`), and the total
    multipath energy is :math:`L`. Feeding
    :func:`pdp_to_scales` gives instead :math:`\sum_l \gamma_l = 1`, i.e.
    a delay spread that does not change the received power.

    Layout: the returned array follows ``(L, N_r, N_t)`` -- taps first,
    then the MIMO layout of CONVENTIONS.md (receive antennas on axis -2)
    -- and is squeezed to ``(N_r, N_t)`` when ``L == 1``, which is
    exactly what :class:`~comnumpy.mimo.channels.FlatMIMOChannel` and
    :class:`~comnumpy.mimo.channels.SelectiveMIMOChannel` expect.

    Parameters
    ----------
    N_r : int
        Number of receive antennas :math:`N_r`.
    N_t : int
        Number of transmit antennas :math:`N_t`.
    L : int, optional
        Number of discrete-time channel taps :math:`L`. Default is 1
        (flat channel).
    scale_per_tap : sequence of float, optional
        Per-tap variances :math:`\gamma_l` in linear scale (length
        :math:`L`). If None, uses ones.
    seed : int, optional
        Random seed used if `rng` is not provided.
    rng : numpy.random.Generator, optional
        Pre-initialized random number generator. If provided, `seed` is ignored.

    Returns
    -------
    numpy.ndarray
        Array of shape (L, N_r, N_t) containing the Rayleigh channel taps.

    Raises
    ------
    AssertionError
        If `scale_per_tap` is provided and its length is not equal to `L`.

    References
    ----------
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 2 (Rayleigh fading) and
      Chapter 7 (MIMO channel models).
    * A. Goldsmith, *Wireless Communications*, Cambridge University
      Press, 2005, Chapter 3.

    Examples
    --------
    >>> H = rayleigh_channel(N_r=2, N_t=2, L=3, seed=0)
    >>> H.shape
    (3, 2, 2)
    >>> H.dtype
    dtype('complex128')
    >>> H1 = rayleigh_channel(N_r=4, N_t=2, seed=0)
    >>> H1.shape  # with the default L=1, the tap axis is squeezed
    (4, 2)
    >>> H2 = rayleigh_channel(N_r=32, N_t=32, seed=1)
    >>> print(round(float(np.mean(np.abs(H2)**2)), 2))  # unit average gain
    1.01
    """
    if not rng:
        rng = np.random.default_rng(seed)

    if scale_per_tap is None:
        scales = np.ones(L, dtype=float)
    else:
        scales = np.asarray(scale_per_tap, dtype=float)
        assert scales.shape == (L,)
    H = np.empty((L, N_r, N_t), dtype=complex)
    for l in range(L):
        std = np.sqrt(scales[l])
        H[l] = rng.normal(0, std/np.sqrt(2), (N_r, N_t)) + 1j * rng.normal(0, std/np.sqrt(2), (N_r, N_t))

    if L == 1:
        H = H[0]
    return H


def rician_channel(N_r: int, N_t: int, K: float,
        L: int = 1,
        H_los: Optional[np.ndarray] = None,
        scale_per_tap: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    Draw a Rician MIMO channel with an optional line-of-sight (LoS) component.

    Signal Model
    ------------
    A Rician channel splits each tap into a deterministic (specular)
    part and a diffuse part:

    .. math::

        \mathbf{H}[l] = \sqrt{\frac{K}{K+1}}\, \mathbf{H}_{\mathrm{los}}[l]
        + \sqrt{\frac{1}{K+1}}\, \mathbf{W}[l],
        \qquad W_{ij}[l] \sim \mathcal{CN}\left(0, \gamma_l\right)

    with :math:`\gamma_l =` ``scale_per_tap[l]``. The Rician factor
    :math:`K` is the ratio of the specular power to the diffuse power;
    the weights :math:`\sqrt{K/(K+1)}` and :math:`\sqrt{1/(K+1)}` are
    chosen so that the total power is preserved: if the entries of
    :math:`\mathbf{H}_{\mathrm{los}}[l]` have unit modulus and
    :math:`\gamma_l = 1`, then
    :math:`\mathbb{E}\left[|H_{ij}[l]|^2\right] = K/(K+1) + 1/(K+1) = 1`
    for every :math:`K`. Two limits: :math:`K = 0` gives back the
    Rayleigh channel of :func:`rayleigh_channel`, and
    :math:`K \to \infty` gives the deterministic LoS channel. The modulus
    of each coefficient is then Rice-distributed.

    Layout: the returned array follows ``(L, N_r, N_t)``, squeezed to
    ``(N_r, N_t)`` when ``L == 1``.

    Parameters
    ----------
    N_r : int
        Number of receive antennas :math:`N_r`.
    N_t : int
        Number of transmit antennas :math:`N_t`.
    K : float
        Rician K-factor :math:`K` (specular-to-diffuse power ratio,
        linear scale).
    L : int, optional
        Number of discrete-time channel taps :math:`L`. Default is 1.
    H_los : numpy.ndarray, optional
        Deterministic LoS component :math:`\mathbf{H}_{\mathrm{los}}` of
        shape (L, N_r, N_t) or (N_r, N_t). If None, a zero LoS is
        assumed. Must be complex-valued (see Notes).
    scale_per_tap : sequence of float, optional
        Per-tap variances :math:`\gamma_l` of the diffuse component
        (length :math:`L`). If None, uses ones.
    seed : int, optional
        Random seed used if `rng` is not provided.
    rng : numpy.random.Generator, optional
        Pre-initialized random number generator. If provided, `seed` is ignored.

    Returns
    -------
    numpy.ndarray
        Array of shape (L, N_r, N_t) containing the Rician channel taps.

    Notes
    -----
    - K = 0 reduces to a Rayleigh channel with variance `scale_per_tap[l]`.
    - As K → ∞, the channel approaches the deterministic LoS component `H_los`.
    - If `H_los` is provided as (N_r, N_t), it is broadcast to (1, N_r, N_t).
    - The output array inherits the dtype of `H_los`: a real-valued
      `H_los` silently discards the imaginary part of the diffuse
      component (a ``ComplexWarning`` is emitted). Pass
      ``np.ones((N_r, N_t), dtype=complex)`` rather than
      ``np.ones((N_r, N_t))``.

    Raises
    ------
    AssertionError
        If `H_los` (after normalization) does not match (L, N_r, N_t),
        or if `scale_per_tap` is provided and its length is not equal to `L`.

    References
    ----------
    * A. Goldsmith, *Wireless Communications*, Cambridge University
      Press, 2005, Chapter 3 (Rician fading and the K-factor).
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 2.

    Examples
    --------
    >>> H = rician_channel(N_r=2, N_t=2, K=6.0, seed=0)
    >>> H.shape  # with the default L=1, the tap axis is squeezed
    (2, 2)
    >>> H.dtype
    dtype('complex128')
    >>> H_los = np.ones((2, 2), dtype=complex)  # unit-modulus LoS
    >>> print(np.round(rician_channel(N_r=2, N_t=2, K=9.0, H_los=H_los, seed=0), 4))
    [[0.9768-0.1198j 0.9191+0.0809j]
     [1.0919+0.2916j 0.9721+0.2118j]]
    """
    if not rng:
        rng = np.random.default_rng(seed)

    if H_los is None:
        H_los = np.zeros((L, N_r, N_t), dtype=complex)
    H_los = np.asarray(H_los)
    if H_los.ndim == 2:
        H_los = H_los[None, ...]
    assert H_los.shape == (L, N_r, N_t)
    if scale_per_tap is None:
        scales = np.ones(L, dtype=float)
    else:
        scales = np.asarray(scale_per_tap, dtype=float)
        assert scales.shape == (L,)

    alpha = np.sqrt(K/(K+1))
    beta  = np.sqrt(1/(K+1))
    H = np.empty(np.shape(H_los), dtype=complex)  # never inherit a real dtype (annex A.5)
    for l in range(L):
        std = np.sqrt(scales[l])
        W = rng.normal(0, std/np.sqrt(2), (N_r, N_t)) + 1j * rng.normal(0, std/np.sqrt(2), (N_r, N_t))
        H[l] = alpha * H_los[l] + beta * W

    if L == 1:
        H = H[0]
    return H


def kronecker_rayleigh_channel(N_r: int, N_t: int,
                    L: int = 1,
                    R_rx: Optional[np.ndarray] = None,
                    R_tx: Optional[np.ndarray] = None,
                    scale_per_tap: Optional[Sequence[float]] = None,
                    seed: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    r"""
    Draw a spatially correlated Rayleigh MIMO channel (Kronecker model).

    Signal Model
    ------------
    The Kronecker (separable) model assumes that the correlation seen at
    the transmitter is independent of the one seen at the receiver, so
    each tap factorizes as

    .. math::

        \mathbf{H}[l] = \mathbf{R}_r^{1/2}\, \mathbf{W}[l]\,
        \left(\mathbf{R}_t^{1/2}\right)^{T},
        \qquad W_{ij}[l] \sim \mathcal{CN}\left(0, \gamma_l\right)

    where :math:`\gamma_l =` ``scale_per_tap[l]`` and the square roots
    are the lower-triangular Cholesky factors
    :math:`\mathbf{R}_r = \mathbf{L}_r\mathbf{L}_r^H`,
    :math:`\mathbf{R}_t = \mathbf{L}_t\mathbf{L}_t^H`. Stacking the
    columns of a tap then gives the separable covariance that defines the
    model:

    .. math::

        \mathbb{E}\left[\mathrm{vec}(\mathbf{H}[l])\,
        \mathrm{vec}(\mathbf{H}[l])^H\right]
        = \gamma_l \, \mathbf{R}_t \otimes \mathbf{R}_r

    Setting a correlation matrix to None amounts to using the identity on
    that side; with both set to None the i.i.d. channel of
    :func:`rayleigh_channel` is recovered. Correlation does not change
    the average power of a coefficient when the correlation matrices have
    unit diagonal, but it lowers the rank of a typical realization --
    which is exactly what costs spatial multiplexing gain.

    Layout: the returned array follows ``(L, N_r, N_t)``, squeezed to
    ``(N_r, N_t)`` when ``L == 1``.

    Parameters
    ----------
    N_r : int
        Number of receive antennas :math:`N_r`.
    N_t : int
        Number of transmit antennas :math:`N_t`.
    L : int, optional
        Number of discrete-time channel taps :math:`L`. Default is 1.
    R_rx : numpy.ndarray, optional
        Receive-side correlation matrix :math:`\mathbf{R}_r` of shape
        (N_r, N_r). If None, identity is used.
    R_tx : numpy.ndarray, optional
        Transmit-side correlation matrix :math:`\mathbf{R}_t` of shape
        (N_t, N_t). If None, identity is used.
    scale_per_tap : sequence of float, optional
        Per-tap variances :math:`\gamma_l` in linear scale (length
        :math:`L`). If None, uses ones.
    seed : int, optional
        Random seed used if `rng` is not provided.
    rng : numpy.random.Generator, optional
        Pre-initialized random number generator. If provided, `seed` is ignored.

    Returns
    -------
    numpy.ndarray
        Array of shape (L, N_r, N_t) containing the correlated Rayleigh channel taps.

    Notes
    -----
    Both correlation matrices must be Hermitian positive definite, as
    required by ``numpy.linalg.cholesky``. The right factor is
    transposed, not conjugate-transposed: this is what makes the
    transmit-side factor of the covariance :math:`\mathbf{R}_t` itself
    (and not its conjugate) with the column-stacking ``vec`` convention.

    Raises
    ------
    AssertionError
        If `scale_per_tap` is provided and its length is not equal to `L`.

    References
    ----------
    * J. P. Kermoal, L. Schumacher, K. I. Pedersen, P. E. Mogensen and
      F. Frederiksen, "A stochastic MIMO radio channel model with
      experimental validation," IEEE J. Sel. Areas Commun., vol. 20,
      no. 6, pp. 1211-1226, 2002.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 7.

    Examples
    --------
    >>> Rr = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> Rt = np.eye(2)
    >>> H = kronecker_rayleigh_channel(N_r=2, N_t=2, L=2, R_rx=Rr, R_tx=Rt, seed=0)
    >>> H.shape
    (2, 2, 2)
    >>> H.dtype
    dtype('complex128')
    """
    if not rng:
        rng = np.random.default_rng(seed)

    if scale_per_tap is None:
        scales = np.ones(L, dtype=float)
    else:
        scales = np.asarray(scale_per_tap, dtype=float)
        assert scales.shape == (L,)

    Lr = np.linalg.cholesky(R_rx) if R_rx is not None else None
    Lt = np.linalg.cholesky(R_tx) if R_tx is not None else None

    H = np.empty((L, N_r, N_t), dtype=complex)
    for l in range(L):
        std=np.sqrt(scales[l])
        W = rng.normal(0, std/np.sqrt(2), (N_r, N_t)) + 1j * rng.normal(0, std/np.sqrt(2), (N_r, N_t))
        if Lr is not None:
            W = np.matmul(Lr, W)
        if Lt is not None:
            W = np.matmul(W, Lt.T)
        H[l] = W

    if L == 1:
        H = H[0]
    return H


def pdp_to_scales(pdp_db: Sequence[float]) -> np.ndarray:
    r"""
    Convert a power delay profile (PDP) in dB into normalized per-tap variances.

    Signal Model
    ------------
    A power delay profile gives the average received power
    :math:`P_l` (in dB) of the path arriving with delay :math:`l T_s`.
    It is converted to the linear per-tap variances
    :math:`\gamma_l = \mathbb{E}\left[|H_{ij}[l]|^2\right]` used by the
    channel generators, and normalized to unit total energy:

    .. math::

        \gamma_l = \frac{10^{P_l/10}}
        {\displaystyle\sum_{j=0}^{L-1} 10^{P_j/10}},
        \qquad \sum_{l=0}^{L-1} \gamma_l = 1

    The normalization is what makes a comparison across delay profiles
    meaningful: the delay spread changes the frequency selectivity of the
    channel without changing the total received power, hence without
    changing the operating SNR.

    Layout: 1-D array of length :math:`L`, to be passed as
    ``scale_per_tap`` to :func:`rayleigh_channel`,
    :func:`rician_channel` or :func:`kronecker_rayleigh_channel`.

    Parameters
    ----------
    pdp_db : sequence of float
        Per-tap powers :math:`P_l` in dB (length :math:`L`), the first
        entry being the first arriving path.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length L with non-negative entries summing to 1.

    Notes
    -----
    If the PDP contains all -inf values or sums to zero after conversion,
    the result would be undefined; ensure the PDP has finite values.

    References
    ----------
    A. Goldsmith, *Wireless Communications*, Cambridge University Press,
    2005, Chapter 3 (multipath channel, power delay profile and delay
    spread).

    Examples
    --------
    >>> scales = pdp_to_scales([0.0, -3.0, -6.0])
    >>> print(np.round(scales, 4))
    [0.5707 0.286  0.1433]
    >>> print(float(round(scales.sum(), 6)))
    1.0
    """
    p_lin = 10.0 ** (np.asarray(pdp_db, dtype=float) / 10.0)
    p_lin /= p_lin.sum()
    return p_lin
