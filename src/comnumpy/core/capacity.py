"""Channel capacity: the reference axis every other metric is read against.

A code is not judged in the abstract but by its distance to capacity, and
a modulation not by its SER alone but by the rate it can carry. This
module provides the closed forms where they exist (Gaussian input over
AWGN, Rayleigh ergodic capacity) and the numerical ones where they do not
(constellation-constrained and BICM capacity, MIMO), so that a simulated
curve can be placed against the limit it is chasing.

Every function returns bits per channel use.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "awgn_capacity", "constellation_capacity", "bicm_capacity",
    "rayleigh_ergodic_capacity", "mimo_ergodic_capacity",
    "outage_capacity", "waterfilling",
]

# Gauss-Hermite is the natural quadrature for an integral against a
# Gaussian weight. Convergence is not uniform in the constellation size:
# measured against a 100-node reference over rho in [0.01, 1000], the
# error at 40 nodes is 1.6e-7 bit for QPSK, 1.2e-6 for 16-QAM and 2.1e-5
# for 64-QAM -- a dense constellation makes the integrand sharper. 40 is
# the default because it is accurate enough to read a coded curve against
# and still cheap; pass more through ``n_nodes`` if a figure needs it.
_GH_NODES = 40

# The likelihood tensor is (M, M, Q): for 256-QAM with 40 nodes that would
# be 1.7 GB in one block, so the quadrature is consumed in chunks. 4096
# points per chunk keeps the peak under ~100 MB for any constellation the
# library supports.
_QUAD_CHUNK = 4096

# Truncation of the classical rule. The Gaussian weight at 8 standard
# deviations is 5e-15 of its peak and the tail beyond it carries 1.2e-15
# of the mass -- below the accuracy any rule reaches on this integrand,
# so the span is fixed and the accuracy knob stays ``n_nodes`` alone.
_SIMPSON_SPAN = 8.0

# Measured on 16-QAM at rho = 10, against a 200-node Gauss-Hermite
# reference: Simpson is 1.9e-2 bit off at 20 nodes, 1.2e-4 at 40 and
# 1.1e-9 at 80, where Gauss-Hermite is already at 1.1e-5, 7.9e-7 and
# 3.8e-10. Roughly a factor two in nodes, i.e. four in cost, for the
# same accuracy -- which is why the default is what it is, and why the
# classical rule is still usable enough to be worth keeping as a check
# rather than as a museum piece.
_METHODS = ("gauss-hermite", "simpson")


def _noise_quadrature(method: str, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Nodes and weights of a standard normal expectation, per dimension.

    Both capacity functions integrate a smooth function against a
    Gaussian weight, so both reduce to the same one-dimensional rule
    :math:`\mathbb{E}\left[f(X)\right] \approx \sum_i w_i f(x_i)` with
    :math:`X \sim \mathcal{N}(0,1)`, applied as a tensor product on the
    two dimensions of the complex noise. Only the rule differs between
    methods; nothing downstream of this function knows which one it got.

    ``"gauss-hermite"`` places :math:`n` nodes at the roots of the
    :math:`n`-th Hermite polynomial and solves for the weights, which
    makes it exact for every polynomial of degree up to :math:`2n - 1`
    -- twice what a fixed grid of the same size can reach -- and needs no
    truncation, since the rule is built on the infinite domain.

    ``"simpson"`` is the textbook composite rule on a truncated grid.
    It is not a strawman here: the integrand decays to zero with all its
    derivatives, so Euler-Maclaurin's endpoint corrections vanish and the
    rule converges far faster than its nominal :math:`O(h^4)`. What
    Gauss-Hermite buys is node *placement* -- it spends its evaluations
    where the weight is -- and no truncation to choose.

    Parameters
    ----------
    method : str
        ``"gauss-hermite"`` or ``"simpson"``.
    n_nodes : int
        Nodes per dimension. Simpson needs an odd count, so an even one
        is raised by one.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Nodes and weights, both of shape ``(n,)``. The weights of an
        exact rule sum to one.

    Raises
    ------
    ValueError
        If ``method`` is not one of the two, or ``n_nodes`` is below 2.
    """
    if n_nodes < 2:
        raise ValueError(f"a quadrature needs at least 2 nodes, got {n_nodes}")
    if method == "gauss-hermite":
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        return nodes, weights / np.sqrt(2 * np.pi)   # normalized N(0, 1)
    if method == "simpson":
        count = n_nodes if n_nodes % 2 else n_nodes + 1
        nodes = np.linspace(-_SIMPSON_SPAN, _SIMPSON_SPAN, count)
        step = float(nodes[1] - nodes[0])
        coefficients = np.ones(count)
        coefficients[1:-1:2] = 4.0                   # 1, 4, 2, 4, ..., 4, 1
        coefficients[2:-1:2] = 2.0
        density = np.exp(-nodes ** 2 / 2) / np.sqrt(2 * np.pi)
        return nodes, (step / 3) * coefficients * density
    raise ValueError(
        f"unknown integration method {method!r}, expected one of "
        f"{_METHODS} -- 'gauss-hermite' is the default and the accurate "
        f"one, 'simpson' is the classical rule kept as an independent "
        f"check of it.")


def awgn_capacity(snr: np.ndarray | float) -> np.ndarray:
    r"""Shannon capacity of the complex AWGN channel.

    Signal Model
    ------------
    For :math:`y = x + b` with :math:`b \sim \mathcal{CN}(0, \sigma^2)`
    and an average power constraint
    :math:`\mathbb{E}\left[|x|^2\right] \leq P`, the capacity is reached
    by a Gaussian input and equals

    .. math::

        C = \log_2\left(1 + \rho\right), \qquad \rho = P / \sigma^2

    Parameters
    ----------
    snr : float or np.ndarray
        Signal-to-noise ratio :math:`\rho`, in linear scale.

    Returns
    -------
    np.ndarray
        Capacity in bits per channel use.

    References
    ----------
    C. E. Shannon, "A mathematical theory of communication", Bell System
    Technical Journal 27, 1948; T. M. Cover, J. A. Thomas, *Elements of
    Information Theory*, 2nd ed., Wiley, 2006, Chapter 9.

    Examples
    --------
    >>> print(np.round(awgn_capacity(np.array([1.0, 10.0, 100.0])), 4))
    [1.     3.4594 6.6582]
    """
    return np.log2(1.0 + np.asarray(snr, dtype=float))


def constellation_capacity(alphabet: np.ndarray, snr: np.ndarray | float, *,
                           px: Optional[np.ndarray] = None,
                           n_nodes: int = _GH_NODES,
                           method: str = "gauss-hermite") -> np.ndarray:
    r"""Mutual information of a discrete constellation over AWGN.

    Signal Model
    ------------
    A real system does not transmit a Gaussian: it transmits one of
    :math:`M` constellation points, equiprobably by default. The
    achievable rate is then the mutual information

    .. math::

        I(X; Y) = \log_2 M - \frac{1}{M} \sum_{m=1}^{M}
        \mathbb{E}_{b}\left[ \log_2 \sum_{m'=1}^{M}
        \exp\left(-\frac{|x_m - x_{m'} + b|^2 - |b|^2}{\sigma^2}\right)
        \right]

    where the expectation is over the noise :math:`b \sim
    \mathcal{CN}(0, \sigma^2)`. This is the *constrained* capacity: it
    saturates at :math:`\log_2 M` however high the SNR, and it is the
    ceiling a coded system using that constellation can approach --
    not the Shannon capacity, which assumes a Gaussian input.

    Probabilistic shaping makes the input non-uniform, and then the
    :math:`1/M` factors are no longer constants to pull out of the sum:

    .. math::

        I(X; Y) = - \sum_{m} P_X(x_m) \;
        \mathbb{E}_{b}\left[ \log_2 \sum_{m'} P_X(x_{m'})
        \exp\left(-\frac{|x_m - x_{m'} + b|^2 - |b|^2}{\sigma^2}\right)
        \right]

    which is the same expression with ``px`` in both places and reduces
    to the one above when it is uniform. Passing ``px`` is how the rate
    of a shaped constellation is read exactly rather than estimated from
    samples with :func:`~comnumpy.core.information.compute_mi`.

    The expectation is a two-dimensional integral against a Gaussian
    weight, evaluated by quadrature -- no random draws. Which rule does
    it is left open by ``method``; see :func:`_noise_quadrature` for what
    separates them.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation points, of unit average energy (as returned by
        :func:`~comnumpy.core.utils.get_alphabet`).
    snr : float or np.ndarray
        Signal-to-noise ratio :math:`\rho = 1/\sigma^2`, linear.
    px : np.ndarray, optional, keyword-only
        Input distribution :math:`P_X`, of length :math:`M`. Default
        (None) is uniform. Note that the SNR is still read from the
        alphabet's own scale, so a shaped input of lower energy is
        compared at *lower* power unless the alphabet is rescaled --
        see :func:`~comnumpy.core.shaping.maxwell_boltzmann`.
    n_nodes : int, optional, keyword-only
        Quadrature nodes per noise dimension, so the cost grows as
        :math:`n^2`. Default 40.
    method : str, optional, keyword-only
        ``"gauss-hermite"`` (default), the rule matched to the Gaussian
        weight, or ``"simpson"``, the classical composite rule on a
        truncated grid. Same quantity, different numerics: the second is
        there to be able to check the first without leaving the library.

    Returns
    -------
    np.ndarray
        Mutual information in bits per channel use, in
        :math:`[0, \log_2 M]` -- at most :math:`H(P_X)` when ``px`` is
        given.

    Raises
    ------
    ValueError
        If ``method`` is not one of the two rules above, or if ``px``
        does not have one non-negative probability per constellation
        point.

    References
    ----------
    G. Ungerboeck, "Channel coding with multilevel/phase signals", IEEE
    Trans. Inf. Theory 28(1), 1982; Cover & Thomas, 2nd ed., Chapter 9.

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> qpsk = get_alphabet("PSK", 4)
    >>> print(np.round(constellation_capacity(qpsk, np.array([1.0, 1e4])), 4))
    [0.9719 2.    ]

    The classical rule lands on the same number, which is the point of
    keeping it -- it checks the default rule without leaving the library:

    >>> qam16 = get_alphabet("QAM", 16)
    >>> gauss = constellation_capacity(qam16, 10.0)
    >>> simpson = constellation_capacity(qam16, 10.0, method="simpson")
    >>> print(np.round([float(gauss), float(simpson)], 3))
    [3.164 3.164]

    A non-uniform input carries less than :math:`\log_2 M` however clean
    the channel, because it is its entropy that saturates:

    >>> law = np.array([0.4, 0.1, 0.1, 0.4])
    >>> print(round(float(constellation_capacity(
    ...     get_alphabet("PSK", 4), 1e6, px=law)), 4))
    1.7219
    """
    alphabet = np.asarray(alphabet, dtype=complex).ravel()
    order = alphabet.size
    if px is None:
        weight = np.full(order, 1.0 / order)
    else:
        weight = np.asarray(px, dtype=float).ravel()
        if weight.size != order or np.any(weight < 0) or not np.isclose(
                float(np.sum(weight)), 1.0, atol=1e-9):
            raise ValueError(
                f"px must hold one non-negative probability per "
                f"constellation point and sum to one, got {weight.size} "
                f"values summing to {float(np.sum(weight))} for an "
                f"alphabet of {order} points.")
    scalar_input = np.ndim(snr) == 0
    snr = np.atleast_1d(np.asarray(snr, dtype=float))
    sigma2 = 1.0 / snr

    nodes, weights = _noise_quadrature(method, n_nodes)
    # complex noise: independent N(0, sigma2/2) on each dimension
    grid_r, grid_i = np.meshgrid(nodes, nodes, indexing="ij")
    w2 = np.outer(weights, weights)
    unit_noise = (grid_r + 1j * grid_i).ravel()     # unit variance per dim
    w2 = w2.ravel()

    with np.errstate(divide="ignore"):
        log_weight = np.log(weight)      # -inf on a point that is never sent
    diff = alphabet[:, None] - alphabet[None, :]    # (M, M)
    diff2 = np.abs(diff) ** 2
    diff_re, diff_im = np.real(diff), np.imag(diff)
    out = np.empty(snr.size)
    for index, s2 in enumerate(sigma2):
        noise = np.sqrt(s2 / 2) * unit_noise        # (Q,)
        total = 0.0
        for start in range(0, noise.size, _QUAD_CHUNK):
            block = noise[start:start + _QUAD_CHUNK]
            # |d + b|^2 - |b|^2 = |d|^2 + 2 Re(d* b): the standard pairwise
            # step. Only the projection of the noise on the difference
            # vector matters, and no complex (M, M, Q) tensor is built.
            arg = -(diff2[:, :, None] + 2 * (
                diff_re[:, :, None] * np.real(block)
                + diff_im[:, :, None] * np.imag(block))) / s2
            # The inner sum is weighted by P_X, the outer one too: with a
            # uniform input both weights are 1/M and the two log2(M) they
            # produce collapse into the closed form written above.
            #
            # The shift is not optional once P_X is inside. With a uniform
            # input the j = i term of the sum is exp(0) = 1, so the sum can
            # never underflow; weighted, that term is P_X(x_i), which a
            # strongly shaped law drives to zero -- and then every term of
            # the sum can underflow at once, log2 returns -inf, and the
            # outer 0 * (-inf) makes the whole rate NaN. Shifting by the
            # largest exponent keeps a term equal to one in every sum.
            shifted = log_weight[None, :, None] + arg
            top = np.max(shifted, axis=1)                     # (M, q)
            inner = (top + np.log(np.sum(np.exp(shifted - top[:, None, :]),
                                         axis=1))) / np.log(2)
            total += float(weight @ (inner @ w2[start:start + _QUAD_CHUNK]))
        out[index] = -total
    return out[0] if scalar_input else out


def bicm_capacity(alphabet: np.ndarray, snr: np.ndarray | float, *,
                  px: Optional[np.ndarray] = None,
                  n_nodes: int = _GH_NODES,
                  method: str = "gauss-hermite") -> np.ndarray:
    r"""BICM capacity: the bound a bit-interleaved coded system chases.

    Signal Model
    ------------
    A chain that maps bits to symbols, demaps to per-bit LLRs and then
    decodes does **not** achieve the constellation capacity: the parallel
    binary channels seen by the decoder ignore the dependence between the
    bits of a symbol. The achievable rate is the sum of the per-bit
    mutual informations

    .. math::

        C_{\mathrm{BICM}} = \sum_{i=1}^{m} I(B_i; Y),
        \qquad m = \log_2 M

    which is the right reference for the ``SymbolDemapper(soft=True)`` +
    FEC chain of this library. The gap to
    :func:`constellation_capacity` is the price of the bit-wise
    interface, and it depends on the labelling -- Gray mapping minimizes
    it, which is why it is the default.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation points of unit average energy, in the labelling
        order used by the mapper (index = symbol value).
    snr : float or np.ndarray
        Signal-to-noise ratio, linear.
    n_nodes : int, optional, keyword-only
        Quadrature nodes per noise dimension. Default 40.
    method : str, optional, keyword-only
        ``"gauss-hermite"`` (default) or ``"simpson"``, as in
        :func:`constellation_capacity`.

    Returns
    -------
    np.ndarray
        BICM capacity in bits per channel use.

    Raises
    ------
    ValueError
        If the constellation size is not a power of two, or ``method``
        is not one of the two rules.

    References
    ----------
    G. Caire, G. Taricco, E. Biglieri, "Bit-interleaved coded
    modulation", IEEE Trans. Inf. Theory 44(3), 1998.

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> qam16 = get_alphabet("QAM", 16)
    >>> snr = np.array([1.0, 10.0])
    >>> print(np.round(bicm_capacity(qam16, snr), 4))       # bit-wise interface
    [0.8993 3.1636]
    >>> print(np.round(constellation_capacity(qam16, snr), 4))  # the ceiling above it
    [0.9897 3.1639]

    A shaped input is passed the same way as to
    :func:`constellation_capacity`, and the ceiling moves with it: the
    sum of the per-bit entropies, not :math:`\log_2 M`.

    >>> from comnumpy.core.shaping import maxwell_boltzmann
    >>> law = maxwell_boltzmann(qam16, lam=0.5)
    >>> print(np.round(bicm_capacity(qam16, snr, px=law), 4))
    [0.7928 3.0653]
    """
    alphabet = np.asarray(alphabet, dtype=complex).ravel()
    order = alphabet.size
    n_bits = int(np.log2(order))
    if 2 ** n_bits != order:
        raise ValueError(
            f"BICM capacity needs a power-of-two constellation, got "
            f"{order} points -- the labelling maps {n_bits} bits.")
    if px is None:
        weight = np.full(order, 1.0 / order)
    else:
        weight = np.asarray(px, dtype=float).ravel()
        if weight.size != order or np.any(weight < 0) or not np.isclose(
                float(np.sum(weight)), 1.0, atol=1e-9):
            raise ValueError(
                f"px must hold one non-negative probability per "
                f"constellation point and sum to one, got {weight.size} "
                f"values summing to {float(np.sum(weight))} for an "
                f"alphabet of {order} points.")
    scalar_input = np.ndim(snr) == 0
    snr = np.atleast_1d(np.asarray(snr, dtype=float))
    sigma2 = 1.0 / snr

    labels = np.arange(order)
    bit_value = ((labels[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1)

    # H(B_i) under px: one bit of a shaped constellation no longer carries
    # a full bit, and the sum of these is the ceiling the rate is measured
    # down from -- n_bits only when px is uniform.
    bit_entropy = np.zeros(n_bits)
    for bit in range(n_bits):
        one = float(np.sum(weight[bit_value[:, bit] == 1]))
        if 0.0 < one < 1.0:
            bit_entropy[bit] = -(one * np.log2(one)
                                 + (1 - one) * np.log2(1 - one))

    nodes, weights = _noise_quadrature(method, n_nodes)
    grid_r, grid_i = np.meshgrid(nodes, nodes, indexing="ij")
    unit_noise = (grid_r + 1j * grid_i).ravel()
    w2 = np.outer(weights, weights).ravel()

    diff = alphabet[:, None] - alphabet[None, :]
    diff2 = np.abs(diff) ** 2
    diff_re, diff_im = np.real(diff), np.imag(diff)
    out = np.zeros(snr.size)
    for index, s2 in enumerate(sigma2):
        noise = np.sqrt(s2 / 2) * unit_noise
        capacity = float(np.sum(bit_entropy))
        for start in range(0, noise.size, _QUAD_CHUNK):
            block = noise[start:start + _QUAD_CHUNK]
            weight_block = w2[start:start + _QUAD_CHUNK]
            arg = -(diff2[:, :, None] + 2 * (
                diff_re[:, :, None] * np.real(block)
                + diff_im[:, :, None] * np.imag(block))) / s2
            # a point that is never sent contributes to no sum, and the
            # ratio below is invariant to the common factor f(y|x_m), so
            # the weights go in exactly where the probabilities do
            likelihood = weight[None, :, None] * np.exp(arg)   # (M, M', q)
            total = np.sum(likelihood, axis=1)                 # (M, q)
            for bit in range(n_bits):
                if bit_entropy[bit] == 0.0:
                    continue            # that bit is constant: it carries nothing
                same = bit_value[:, bit][None, :] == bit_value[:, bit][:, None]
                partial = np.sum(likelihood * same[:, :, None], axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    term = np.log2(total / partial)
                # rows of probability zero are never transmitted; their
                # ratio can be 0/0 and must not poison the average
                term = np.where(weight[:, None] > 0, np.nan_to_num(term), 0.0)
                capacity -= float(weight @ (term @ weight_block))
        out[index] = capacity
    return out[0] if scalar_input else out


def rayleigh_ergodic_capacity(snr: np.ndarray | float) -> np.ndarray:
    r"""Ergodic capacity of a flat Rayleigh fading channel.

    Signal Model
    ------------
    With :math:`y = h x + b`, :math:`h \sim \mathcal{CN}(0,1)` known at
    the receiver only, the ergodic capacity averages the instantaneous
    one over the fading:

    .. math::

        C = \mathbb{E}_{h}\left[\log_2\left(1 + \rho |h|^2\right)\right]
          = \frac{1}{\ln 2} \, e^{1/\rho} \, E_1\!\left(1/\rho\right)

    since :math:`|h|^2 \sim \mathrm{Exp}(1)`, with :math:`E_1` the
    exponential integral. Fading always costs: this is strictly below
    :func:`awgn_capacity` at the same :math:`\rho`, by Jensen's
    inequality.

    Parameters
    ----------
    snr : float or np.ndarray
        Average signal-to-noise ratio :math:`\rho`, linear.

    Returns
    -------
    np.ndarray
        Ergodic capacity in bits per channel use.

    References
    ----------
    D. Tse, P. Viswanath, *Fundamentals of Wireless Communication*,
    Cambridge University Press, 2005, Section 5.4.

    Examples
    --------
    >>> capacity = rayleigh_ergodic_capacity(np.array([1.0, 10.0, 100.0]))
    >>> print(np.round(capacity, 4))
    [0.8603 2.9065 5.884 ]
    >>> bool(np.all(capacity < awgn_capacity(np.array([1.0, 10.0, 100.0]))))
    True
    """
    from scipy.special import exp1                   # local import (D36)
    snr = np.asarray(snr, dtype=float)
    inverse = 1.0 / snr
    return np.exp(inverse) * exp1(inverse) / np.log(2)


def mimo_ergodic_capacity(n_tx: int, n_rx: int, snr: np.ndarray | float, *,
                          n_realizations: int = 2000,
                          rng: Optional[np.random.Generator] = None
                          ) -> np.ndarray:
    r"""Ergodic capacity of an i.i.d. Rayleigh MIMO channel.

    Signal Model
    ------------
    With equal power split across the transmit antennas and channel
    knowledge at the receiver only,

    .. math::

        C = \mathbb{E}_{\mathbf{H}}\left[ \log_2 \det\left(
        \mathbf{I}_{N_r} + \frac{\rho}{N_t}
        \mathbf{H} \mathbf{H}^H \right) \right]

    with :math:`\mathbf{H}` of i.i.d. :math:`\mathcal{CN}(0,1)` entries.
    The famous consequence is the multiplexing gain: at high SNR the
    capacity grows as :math:`\min(N_t, N_r) \log_2 \rho`, i.e. spatial
    dimensions buy rate, not just reliability.

    No closed form exists for finite dimensions, so the expectation is
    estimated by Monte Carlo.

    Parameters
    ----------
    n_tx, n_rx : int
        Numbers of transmit and receive antennas :math:`N_t`, :math:`N_r`.
    snr : float or np.ndarray
        Total signal-to-noise ratio :math:`\rho`, linear.
    n_realizations : int, optional, keyword-only
        Channel draws averaged. Default 2000.
    rng : numpy.random.Generator, optional, keyword-only
        Random generator.

    Returns
    -------
    np.ndarray
        Ergodic capacity in bits per channel use.

    References
    ----------
    E. Telatar, "Capacity of multi-antenna Gaussian channels", European
    Trans. Telecommunications 10(6), 1999; Tse & Viswanath, Chapter 8.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> siso = mimo_ergodic_capacity(1, 1, 10.0, n_realizations=20000, rng=rng)
    >>> bool(abs(siso - rayleigh_ergodic_capacity(10.0)) < 0.05)   # 1x1 = SISO
    True
    """
    rng = np.random.default_rng() if rng is None else rng
    scalar_input = np.ndim(snr) == 0
    snr = np.atleast_1d(np.asarray(snr, dtype=float))
    shape = (n_realizations, n_rx, n_tx)
    h = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    gram = h @ np.conj(np.swapaxes(h, -1, -2))       # (R, N_r, N_r)
    eye = np.eye(n_rx)
    out = np.empty(snr.size)
    for index, rho in enumerate(snr):
        _, logdet = np.linalg.slogdet(eye + (rho / n_tx) * gram)
        out[index] = float(np.mean(logdet)) / np.log(2)
    return out[0] if scalar_input else out


def outage_capacity(snr: float, outage: float = 0.01) -> float:
    r"""Outage capacity of a flat Rayleigh channel.

    Signal Model
    ------------
    A slowly fading channel has no ergodic behaviour over one codeword:
    the rate either is or is not supported. The
    :math:`\varepsilon`-outage capacity is the largest rate supported
    with probability :math:`1 - \varepsilon`. With
    :math:`|h|^2 \sim \mathrm{Exp}(1)`,
    :math:`\mathbb{P}\left[|h|^2 < t\right] = 1 - e^{-t}`, hence

    .. math::

        C_{\varepsilon} = \log_2\left(1 - \rho
        \ln\left(1 - \varepsilon\right)\right)

    It is far below the ergodic capacity at small
    :math:`\varepsilon` -- the price of not being able to average over
    the fading.

    Parameters
    ----------
    snr : float
        Average signal-to-noise ratio :math:`\rho`, linear.
    outage : float, optional
        Outage probability :math:`\varepsilon` in :math:`(0, 1)`.
        Default 0.01.

    Returns
    -------
    float
        Outage capacity in bits per channel use.

    References
    ----------
    Tse & Viswanath, *Fundamentals of Wireless Communication*, 2005,
    Section 5.4.1.

    Examples
    --------
    >>> print(round(outage_capacity(100.0, outage=0.01), 4))
    1.0036
    >>> print(round(rayleigh_ergodic_capacity(100.0).item(), 4))
    5.884
    """
    if not 0 < outage < 1:
        raise ValueError(f"outage must lie in (0, 1), got {outage}")
    return float(np.log2(1 - snr * np.log(1 - outage)))


def waterfilling(gains: np.ndarray, snr: float) -> tuple[np.ndarray, float]:
    r"""Optimal power allocation over parallel Gaussian channels.

    Signal Model
    ------------
    Given :math:`K` parallel channels of gain :math:`g_k` -- the
    subcarriers of an OFDM system, the eigenmodes of a MIMO channel --
    the capacity-achieving allocation solves

    .. math::

        \max_{p_k \geq 0} \sum_k \log_2\left(1 + \rho g_k p_k\right)
        \quad \text{subject to} \quad \sum_k p_k = K

    whose solution pours power onto the good channels up to a common
    water level :math:`\mu`:

    .. math::

        p_k = \left(\mu - \frac{1}{\rho g_k}\right)^{+}

    Channels below the water level get nothing: it is not worth spending
    power where the noise dominates.

    Parameters
    ----------
    gains : np.ndarray
        Channel power gains :math:`g_k`, non-negative.
    snr : float
        Reference signal-to-noise ratio :math:`\rho` at unit power.

    Returns
    -------
    tuple of (np.ndarray, float)
        The allocation :math:`p_k` (summing to :math:`K`) and the
        resulting capacity in bits per channel use, *per channel*.

    Raises
    ------
    ValueError
        If a gain is negative or all gains are zero.

    References
    ----------
    T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd
    ed., Wiley, 2006, Section 9.4 (parallel Gaussian channels).

    Examples
    --------
    >>> power, capacity = waterfilling(np.array([1.0, 1.0, 1.0]), snr=10.0)
    >>> print(np.round(power, 6), round(capacity, 4))
    [1. 1. 1.] 3.4594
    >>> power, capacity = waterfilling(np.array([4.0, 1.0, 0.01]), snr=10.0)
    >>> print(np.round(power, 4), round(capacity, 4) + 0.0)
    [1.5375 1.4625 0.    ] 3.3105
    """
    gains = np.asarray(gains, dtype=float)
    if np.any(gains < 0):
        raise ValueError("channel gains must be non-negative")
    n_channels = gains.size
    if not np.any(gains > 0):
        raise ValueError("at least one channel must have a non-zero gain")

    budget = float(n_channels)
    noise = np.where(gains > 0, 1.0 / (snr * np.maximum(gains, 1e-300)), np.inf)
    order = np.argsort(noise)
    sorted_noise = noise[order]
    # activate channels one by one, cheapest noise first: with k channels
    # active the water level is (budget + sum of their noise) / k
    power = np.zeros(n_channels)
    for k in range(1, n_channels + 1):
        if not np.isfinite(sorted_noise[k - 1]):
            break
        level = (budget + sorted_noise[:k].sum()) / k
        if k < n_channels and np.isfinite(sorted_noise[k]) and level > sorted_noise[k]:
            continue                                  # one more channel fits
        power[order[:k]] = np.maximum(level - sorted_noise[:k], 0.0)
        break
    capacity = float(np.sum(np.log2(1 + snr * gains * power)) / n_channels)
    return power, capacity
