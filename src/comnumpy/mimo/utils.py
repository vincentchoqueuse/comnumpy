import numpy as np
from typing import Optional, Literal, Sequence


def apply_correlation(H, Rx=None, Ry=None):
    if Rx:
        Rx_sqrt = np.linalg.cholesky(Rx)
        H = np.matmul(H, Rx_sqrt)

    if Ry:
        Ry_sqrt = np.linalg.cholesky(Ry)
        H = np.matmul(H, Ry_sqrt)
    return H


def rayleigh_channel(N_r: int, N_t: int, 
                L: Optional[int] = 1,
                scale_per_tap: Optional[Sequence[float]] = None,
                seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generates an i.i.d. Rayleigh MIMO channel.

    The function draws a set of complex Gaussian taps where each element is
    independently distributed as CN(0, scale_per_tap[l]). The output has shape
    (L, N_r, N_t), with L the number of taps, N_r the number of receive
    antennas, and N_t the number of transmit antennas.

    Attributes
    ----------
    N_r : int
        Number of receive antennas.
    N_t : int
        Number of transmit antennas.
    L : int, optional
        Number of discrete-time channel taps.
    scale_per_tap : sequence of float, optional
        Per-tap variances in linear scale (length L). If None, uses ones.
    seed : int, optional
        Random seed used if `rng` is not provided.
    rng : numpy.random.Generator, optional
        Pre-initialized random number generator. If provided, `seed` is ignored.

    Returns
    -------
    numpy.ndarray
        Array of shape (L, N_r, N_t) containing the Rayleigh channel taps.

    Notes
    -----
    Each entry H[l, i, j] is drawn i.i.d. from a circularly symmetric complex
    normal distribution with zero mean and variance `scale_per_tap[l]`.

    Raises
    ------
    AssertionError
        If `scale_per_tap` is provided and its length is not equal to `L`.

    Examples
    --------
    >>> H = rayleigh_iid(L=3, N_r=2, N_t=2, seed=0)
    >>> H.shape
    (3, 2, 2)
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
        L: Optional[int] = 1,
        H_los: Optional[np.ndarray] = None,
        scale_per_tap: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generates a Rician MIMO channel with optional line-of-sight (LoS) component.

    The channel is modeled as H = α * H_los + β * W, where W is i.i.d. Rayleigh
    (CN(0, scale_per_tap[l])), α = sqrt(K/(K+1)) and β = sqrt(1/(K+1)).
    The output has shape (L, N_r, N_t).

    Attributes
    ----------
    N_r : int
        Number of receive antennas.
    N_t : int
        Number of transmit antennas.
    K : float
        Rician K-factor (ratio of deterministic to diffuse power).
    L : int, optional
        Number of discrete-time channel taps.
    H_los : numpy.ndarray, optional
        Deterministic LoS component of shape (L, N_r, N_t) or (N_r, N_t).
        If None, a zero LoS is assumed.
    scale_per_tap : sequence of float, optional
        Per-tap variances for the diffuse component (length L). If None, uses ones.
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

    Raises
    ------
    AssertionError
        If `H_los` (after normalization) does not match (L, N_r, N_t),
        or if `scale_per_tap` is provided and its length is not equal to `L`.

    Examples
    --------
    >>> H = rician(L=1, N_r=2, N_t=2, K=6.0, seed=0)
    >>> H.shape
    (1, 2, 2)
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
    H = np.empty_like(H_los)
    for l in range(L):
        std = np.sqrt(scales[l])
        W = rng.normal(0, std/np.sqrt(2), shape) + 1j * rng.normal(0, std/np.sqrt(2), shape) 
        H[l] = alpha * H_los[l] + beta * W

    if L == 1:
        H = H[0]
    return H


def kronecker_rayleigh_channel(N_r: int, N_t: int,
                    L: Optional[int] = 1,
                    R_rx: Optional[np.ndarray] = None,
                    R_tx: Optional[np.ndarray] = None,
                    scale_per_tap: Optional[Sequence[float]] = None,
                    seed: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generates a correlated Rayleigh MIMO channel using the Kronecker model.

    For each tap l, the channel is constructed as H[l] = L_r @ W @ L_t^T, where
    W has i.i.d. CN(0, scale_per_tap[l]) entries, and L_r/L_t are Cholesky
    factors of the receive/transmit correlation matrices R_rx and R_tx.
    The output has shape (L, N_r, N_t).

    Attributes
    ----------
    N_r : int
        Number of receive antennas.
    N_t : int
        Number of transmit antennas.
    L : int, optional
        Number of discrete-time channel taps.
    R_rx : numpy.ndarray, optional
        Receive-side correlation matrix of shape (N_r, N_r). If None, identity is used.
    R_tx : numpy.ndarray, optional
        Transmit-side correlation matrix of shape (N_t, N_t). If None, identity is used.
    scale_per_tap : sequence of float, optional
        Per-tap variances in linear scale (length L). If None, uses ones.
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
    Under the Kronecker model, vec(H[l]) has covariance R_tx ⊗ R_rx. This
    implementation uses the Cholesky factors of R_rx and R_tx to induce the
    desired spatial correlation. If a correlation matrix is not provided, an
    identity matrix is effectively assumed on that side.

    Raises
    ------
    AssertionError
        If `scale_per_tap` is provided and its length is not equal to `L`.

    Examples
    --------
    >>> Rr = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> Rt = np.eye(2)
    >>> H = kronecker_rayleigh(L=2, N_r=2, N_t=2, R_rx=Rr, R_tx=Rt, seed=0)
    >>> H.shape
    (2, 2, 2)
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
        W = rng.normal(0, std/np.sqrt(2), shape) + 1j * rng.normal(0, std/np.sqrt(2), shape) 
        if Lr is not None:
            W = np.matmul(Lr, W)
        if Lt is not None:
            W = np.matmul(W, Lt.T)
        H[l] = W
    
    if L == 1:
        H = H[0]
    return H


def pdp_to_scales(pdp_db: Sequence[float]) -> np.ndarray:
    """
    Converts a power delay profile (PDP) in dB to per-tap linear variances.

    The input PDP in decibels is converted to linear scale and normalized so
    that the variances across taps sum to one. The output can be used as
    `scale_per_tap` for channel generators.

    Attributes
    ----------
    pdp_db : sequence of float
        Per-tap powers in dB (length L).

    Returns
    -------
    numpy.ndarray
        One-dimensional array of length L with non-negative entries summing to 1.

    Notes
    -----
    If the PDP contains all -inf values or sums to zero after conversion,
    the result would be undefined; ensure the PDP has finite values.

    Examples
    --------
    >>> pdp_db = [0.0, -3.0, -6.0]
    >>> scales = pdp_to_scales(pdp_db)
    >>> scales.sum()
    1.0
    """
    p_lin = 10.0 ** (np.asarray(pdp_db, dtype=float) / 10.0)
    p_lin /= p_lin.sum()
    return p_lin
