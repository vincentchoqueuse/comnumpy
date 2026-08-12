r"""Probabilistic shaping: choosing *how often* each symbol is sent.

A uniform constellation is not the best a channel can do. The capacity
of the Gaussian channel is reached by a Gaussian input, and a uniform
distribution over a square QAM grid loses up to **1.53 dB** of it -- the
*shaping gap*, :math:`10 \log_{10}(\pi e / 6)`. Probabilistic shaping
closes that gap by sending inner constellation points more often than
outer ones, at the same average power.

Two things are needed for that, and this module provides both.

**The target distribution.** For an average-power constraint the
entropy-maximizing distribution over a discrete constellation is the
Maxwell-Boltzmann family :func:`maxwell_boltzmann`, and nothing else:
the shape is forced by the constraint, only its parameter is free.

**A way to produce it from data.** Data is uniform, so a *distribution
matcher* is an invertible map from uniform bit strings onto sequences
with the wanted empirical distribution. Two classical constructions are
implemented, and they differ in what they hold fixed:

* :class:`ConstantCompositionMatcher` (CCDM) fixes the **composition** --
  every output block contains exactly :math:`n_i` copies of symbol
  :math:`i` -- and enumerates the permutations of that multiset;
* :class:`SphereShaper` (ESS) fixes an **energy budget** -- every output
  block satisfies :math:`\sum_j e(a_j) \leq E_{\max}` -- and enumerates
  the sequences inside that sphere. It is the better of the two at short
  blocklengths, because it does not force each block to reproduce the
  target distribution exactly.

Both are *exactly* invertible: they are enumerative codes, ranking and
unranking a finite set with integer arithmetic, so nothing is
approximated and ``decode(encode(bits)) == bits`` holds by construction
rather than by tolerance.

**Where this sits in a transmitter (PAS).** The standard architecture is
probabilistic amplitude shaping: the matcher shapes the **amplitudes**,
a systematic FEC encoder produces the parity bits, and those parity bits
-- which are uniform -- become the **signs**. A sign is equiprobable, so
it costs nothing in shaping and the composite constellation keeps the
symmetric Maxwell-Boltzmann distribution. That is why the matchers here
work on non-negative amplitudes and why their alphabet is half a PAM
constellation.

References
----------
G. Böcherer, F. Steiner, P. Schulte, "Bandwidth efficient and
rate-matched low-density parity-check coded modulation", IEEE Trans.
Commun., vol. 63, no. 12, pp. 4651-4665, Dec. 2015 (PAS);
P. Schulte, G. Böcherer, "Constant composition distribution matching",
IEEE Trans. Inf. Theory, vol. 62, no. 1, pp. 430-434, Jan. 2016 (CCDM);
F. M. J. Willems, J. J. Wuijts, "A pragmatic approach to shaped coded
modulation", IEEE Symp. Commun. Veh. Technol., 1993 (enumerative shell
mapping); Y. C. Gültekin, W. J. van Houtum, A. G. C. Koonen, F. M. J.
Willems, "Enumerative sphere shaping for wireless communications with
short packets", IEEE Trans. Wireless Commun., vol. 19, no. 2,
pp. 1098-1112, Feb. 2020; G. D. Forney, L.-F. Wei, "Multidimensional
constellations -- Part I", IEEE J. Sel. Areas Commun., vol. 7, no. 6,
pp. 877-892, 1989 (the 1.53 dB shaping gain).

Examples
--------
>>> import numpy as np
>>> amplitudes = np.array([1.0, 3.0, 5.0, 7.0])       # half of 8-PAM
>>> target = maxwell_boltzmann(amplitudes, entropy=1.5)
>>> print(np.round(target, 4))
[0.5376 0.322  0.1156 0.0248]
>>> matcher = ConstantCompositionMatcher(amplitudes, distribution=target,
...                                      length=32)
>>> matcher.n_bits, round(matcher.rate, 4)
(42, 1.3125)
>>> bits = np.zeros(42, dtype=int); bits[::3] = 1
>>> block = matcher.encode(bits)
>>> bool(np.array_equal(matcher.decode(block), bits))    # exactly invertible
True
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

logger = logging.getLogger(__name__)

__all__ = [
    "maxwell_boltzmann", "blahut_arimoto", "distribution_entropy",
    "composition_from_distribution",
    "shaping_gain_dB", "ConstantCompositionMatcher", "SphereShaper",
    "DistributionMatcher", "DistributionDematcher",
    "AmplitudeMapper", "AmplitudeDemapper",
]

# The bisection on the Maxwell-Boltzmann parameter runs on lambda in
# [0, _LAMBDA_MAX / min spacing^2]. The entropy is monotone in lambda, so
# a plain bisection converges; 200 iterations take it to the last bit of
# a double whatever the bracket.
_LAMBDA_MAX = 1e4
_BISECTION_STEPS = 200


def distribution_entropy(distribution: np.ndarray) -> float:
    r"""Entropy of a symbol distribution, in bits.

    Signal Model
    ------------
    .. math::

        H(P) = -\sum_{i} p_i \log_2 p_i

    It is the number of bits a symbol of that distribution carries, hence
    the ceiling any distribution matcher can approach and the quantity
    its *rate loss* is measured against.

    Axes: *element-wise* -- the distribution is a 1-D array of
    probabilities.

    Parameters
    ----------
    distribution : np.ndarray
        Probabilities :math:`p_i`, non-negative and summing to one.

    Returns
    -------
    float
        Entropy in bits per symbol, in :math:`[0, \log_2 M]`.

    Raises
    ------
    ValueError
        If the probabilities are negative or do not sum to one.

    References
    ----------
    T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed.,
    Wiley, 2006, Section 2.1.

    Examples
    --------
    >>> print(round(distribution_entropy(np.full(8, 1 / 8)), 4))
    3.0
    >>> print(round(distribution_entropy(np.array([0.5, 0.25, 0.25])), 4))
    1.5
    """
    p = np.asarray(distribution, dtype=float)
    if np.any(p < 0):
        raise ValueError(
            f"a distribution cannot have a negative probability, got a "
            f"minimum of {float(np.min(p))}.")
    if not np.isclose(float(np.sum(p)), 1.0, atol=1e-9):
        raise ValueError(
            f"a distribution must sum to one, got {float(np.sum(p))}. "
            f"Divide by the sum if these are unnormalized weights.")
    positive = p[p > 0]
    return float(-np.sum(positive * np.log2(positive)))


def maxwell_boltzmann(alphabet: np.ndarray, *, lam: Optional[float] = None,
                      entropy: Optional[float] = None) -> np.ndarray:
    r"""The distribution shaping uses, and the only one it can use.

    Signal Model
    ------------
    Among all distributions on a fixed constellation with a given average
    energy, the one of maximum entropy is

    .. math::

        p_i = \frac{e^{-\lambda \left|a_i\right|^2}}
                   {\sum_j e^{-\lambda \left|a_j\right|^2}},
        \qquad \lambda \geq 0

    the Maxwell-Boltzmann family. This is not a modelling choice: it is
    what maximizing :math:`H(P)` subject to
    :math:`\sum_i p_i |a_i|^2 \leq E` gives, :math:`\lambda` being the
    Lagrange multiplier of the power constraint. :math:`\lambda = 0`
    returns the uniform distribution; growing :math:`\lambda` moves mass
    onto the inner points, lowering both the energy and the entropy.

    Since the entropy is strictly decreasing in :math:`\lambda`,
    specifying a target entropy determines :math:`\lambda` uniquely; that
    is the useful parameterization in practice, because the entropy *is*
    the rate the constellation carries.

    Axes: *element-wise* -- the alphabet is a 1-D array of constellation
    points, and the result is a distribution over it.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation points :math:`a_i`, real or complex.
    lam : float, optional, keyword-only
        The parameter :math:`\lambda` itself, non-negative.
    entropy : float, optional, keyword-only
        Target entropy :math:`H(P)` in bits per symbol, in
        :math:`(0, \log_2 M]`. Exactly one of ``lam`` and ``entropy``
        must be given (D41).

    Returns
    -------
    np.ndarray
        The distribution :math:`p_i`, same length as ``alphabet``.

    Raises
    ------
    ValueError
        If both or neither parameterization is given, if ``lam`` is
        negative, or if ``entropy`` is outside :math:`(0, \log_2 M]`.

    References
    ----------
    F. R. Kschischang, S. Pasupathy, "Optimal nonuniform signaling for
    Gaussian channels", IEEE Trans. Inf. Theory, vol. 39, no. 3,
    pp. 913-929, 1993; Böcherer, Steiner, Schulte, IEEE Trans. Commun.
    63(12), 2015, Section III.

    Examples
    --------
    >>> amplitudes = np.array([1.0, 3.0, 5.0, 7.0])
    >>> print(np.round(maxwell_boltzmann(amplitudes, lam=0.0), 4))
    [0.25 0.25 0.25 0.25]
    >>> shaped = maxwell_boltzmann(amplitudes, entropy=1.5)
    >>> print(round(distribution_entropy(shaped), 6))
    1.5
    >>> float(np.sum(shaped * amplitudes ** 2)) < 21.0    # uniform: 21
    True
    """
    points = np.asarray(alphabet).ravel()
    exclusive = (
        "maxwell_boltzmann takes exactly one of lam= and entropy=, got "
        f"lam={lam} and entropy={entropy}. Pass lam= for the parameter "
        "itself, entropy= for the rate you want the constellation to "
        "carry (D41).")
    if lam is not None and entropy is not None:
        raise ValueError(exclusive)
    energies = np.abs(points) ** 2

    if lam is not None:
        if lam < 0:
            raise ValueError(
                f"the Maxwell-Boltzmann parameter is non-negative, got "
                f"lam={lam}. A negative value would favour the *outer* "
                f"points, which costs energy and entropy at once.")
        return _mb_from_lambda(energies, lam)
    if entropy is None:
        raise ValueError(exclusive)

    target = float(entropy)
    ceiling = math.log2(points.size)
    # The family cannot go below the entropy of its own minimum-energy
    # set: points of equal energy always keep equal probability, so a
    # symmetric PAM constellation floors at 1 bit, whatever lambda does.
    degeneracy = int(np.sum(np.isclose(energies, np.min(energies))))
    floor = math.log2(degeneracy)
    if not floor <= target <= ceiling:
        raise ValueError(
            f"got entropy={target}, expected a value in "
            f"[{floor}, {ceiling}] for this alphabet. The upper end is "
            f"log2(M), reached by the uniform distribution. The lower end "
            f"is log2 of the number of points sharing the smallest "
            f"energy ({degeneracy} here): Maxwell-Boltzmann gives equal "
            f"probability to points of equal energy, so no lambda can "
            f"separate them -- a symmetric constellation cannot be "
            f"shaped below 1 bit per symbol.")

    # H(lambda) is strictly decreasing between those two ends: bisect
    spread = float(np.ptp(energies))
    high = _LAMBDA_MAX / spread if spread > 0 else 1.0
    low = 0.0
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if distribution_entropy(_mb_from_lambda(energies, middle)) > target:
            low = middle
        else:
            high = middle
    return _mb_from_lambda(energies, 0.5 * (low + high))


def _mb_from_lambda(energies: np.ndarray, lam: float) -> np.ndarray:
    """Softmax of ``-lam * energies``, shifted so it cannot overflow."""
    weights = np.exp(-lam * (energies - np.min(energies)))
    return weights / np.sum(weights)


def composition_from_distribution(distribution: np.ndarray,
                                  length: int) -> tuple[int, ...]:
    r"""Round a distribution to an integer composition of a block.

    Signal Model
    ------------
    A constant-composition matcher emits blocks holding exactly
    :math:`n_i` copies of symbol :math:`i`, so it needs integers, not
    probabilities. The composition closest to :math:`n p_i` under the
    constraint :math:`\sum_i n_i = n` is obtained by the largest-remainder
    rule: take :math:`\lfloor n p_i \rfloor` for every symbol, then give
    the :math:`n - \sum_i \lfloor n p_i \rfloor` remaining slots to the
    symbols with the largest fractional parts.

    The composition is not the distribution: it is a quantization of it,
    and the difference is one reason a finite block loses rate. It
    vanishes as :math:`n` grows.

    Axes: *element-wise* -- a 1-D distribution in, a tuple of counts out.

    Parameters
    ----------
    distribution : np.ndarray
        Target probabilities, summing to one.
    length : int
        Block length :math:`n`, the total number of symbols.

    Returns
    -------
    tuple of int
        The counts :math:`n_i`, summing to ``length``.

    Raises
    ------
    ValueError
        If ``length`` is not positive, or the distribution is invalid.

    References
    ----------
    P. Schulte, G. Böcherer, IEEE Trans. Inf. Theory 62(1), 2016,
    Section IV (choice of the composition).

    Examples
    --------
    >>> composition_from_distribution(np.array([0.5, 0.25, 0.25]), 8)
    (4, 2, 2)
    >>> composition_from_distribution(np.array([0.4, 0.35, 0.25]), 10)
    (4, 4, 2)
    """
    p = np.asarray(distribution, dtype=float)
    distribution_entropy(p)                      # validates it (D20)
    if length <= 0:
        raise ValueError(
            f"a block holds at least one symbol, got length={length}.")
    scaled = length * p
    counts = np.floor(scaled).astype(int)
    remainder = length - int(np.sum(counts))
    if remainder:
        # largest fractional parts first; ties go to the lower index, so
        # the result is a function of the input and not of the sort used
        order = np.argsort(-(scaled - counts), kind="stable")
        counts[order[:remainder]] += 1
    return tuple(int(value) for value in counts)


def shaping_gain_dB(alphabet: np.ndarray,
                    distribution: np.ndarray) -> float:
    r"""Energy saved by a shaped distribution, at equal rate.

    Signal Model
    ------------
    Shaping is only worth something if it lowers the energy *without*
    lowering the rate, so the comparison has to be made at equal entropy.
    On a one-dimensional grid of spacing :math:`\Delta`, a source of
    entropy :math:`H` spread uniformly would occupy an interval of width
    :math:`2^{H}\Delta` and cost

    .. math::

        E_{\mathrm{unif}} = \frac{\left(2^{H} \Delta\right)^2}{12}

    against the :math:`E_P = \sum_i p_i a_i^2` the shaped distribution
    actually costs. The gain is their ratio,

    .. math::

        G_s = 10 \log_{10} \frac{E_{\mathrm{unif}}}{E_P}

    A uniform distribution over :math:`M` points gives
    :math:`10\log_{10}(M^2/(M^2-1))`, i.e. essentially 0 dB, which is the
    calibration this definition needs to be worth anything. The
    supremum over all one-dimensional distributions is
    :math:`10\log_{10}(\pi e/6) = 1.53` dB, reached in the limit of a
    Gaussian over an infinitely fine, infinitely wide grid.

    Axes: *element-wise* -- a 1-D real alphabet and its distribution.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation points, **real** and on a regular grid. A complex
        constellation is two independent dimensions; pass one of them.
    distribution : np.ndarray
        Probability of each point.

    Returns
    -------
    float
        Shaping gain in dB, at most 1.53.

    Raises
    ------
    ValueError
        If the alphabet is complex, or has fewer than two points.

    References
    ----------
    G. D. Forney, L.-F. Wei, "Multidimensional constellations -- Part I:
    Introduction, figures of merit, and generalized cross
    constellations", IEEE J. Sel. Areas Commun., vol. 7, no. 6,
    pp. 877-892, 1989, Section III (shaping gain and the 1.53 dB
    ultimate gain).

    Examples
    --------
    >>> pam8 = np.arange(-7, 8, 2).astype(float)
    >>> uniform = np.full(8, 1 / 8)
    >>> print(round(shaping_gain_dB(pam8, uniform), 4))     # the calibration
    0.0684
    >>> shaped = maxwell_boltzmann(pam8, entropy=2.5)
    >>> print(round(shaping_gain_dB(pam8, shaped), 4))
    1.5055
    """
    points = np.asarray(alphabet).ravel()
    if np.iscomplexobj(points) and np.any(np.imag(points) != 0):
        raise ValueError(
            "shaping_gain_dB is defined on a one-dimensional grid, got a "
            "complex alphabet. A QAM constellation is the product of two "
            "PAM constellations: pass np.unique(np.real(alphabet)).")
    points = np.real(points).astype(float)
    if points.size < 2:
        raise ValueError(
            f"a grid spacing needs at least two points, got {points.size}.")
    p = np.asarray(distribution, dtype=float)
    distribution_entropy(p)                      # validates it
    spacing = float(np.min(np.diff(np.sort(np.unique(points)))))
    entropy = distribution_entropy(p)
    uniform_energy = (2 ** entropy * spacing) ** 2 / 12
    shaped_energy = float(np.sum(p * points ** 2))
    return float(10 * np.log10(uniform_energy / shaped_energy))


def blahut_arimoto(alphabet: np.ndarray, *, sigma2: float,
                   lam: Optional[float] = None,
                   energy: Optional[float] = None,
                   n_nodes: int = 40, tol: float = 1e-9,
                   max_iter: int = 5000) -> np.ndarray:
    r"""The distribution that maximizes the mutual information, computed.

    Signal Model
    ------------
    :func:`maxwell_boltzmann` answers "which distribution has the largest
    *entropy* at a given energy?", and has a closed form. The question an
    engineer actually cares about is a different one -- "which
    distribution carries the most *bits over this channel* at a given
    energy?" -- and it has no closed form at all:

    .. math::

        \max_{P_X} \; I(X;Y)
        \quad \text{subject to} \quad \sum_i p_i \left|a_i\right|^2
        \leq E

    over the real AWGN channel :math:`Y = X + Z`,
    :math:`Z \sim \mathcal{N}(0, \sigma^2)`, with :math:`X` drawn from
    the fixed constellation. The problem is concave in :math:`P_X`, and
    the Blahut-Arimoto algorithm solves it by alternating maximization:
    with the Lagrange multiplier :math:`\lambda` of the energy
    constraint,

    .. math::

        q(i \mid y) = \frac{p_i \, f(y \mid a_i)}
                           {\sum_j p_j \, f(y \mid a_j)},
        \qquad
        p_i \;\leftarrow\; \frac{e^{D_i - \lambda |a_i|^2}}
                                {\sum_j e^{D_j - \lambda |a_j|^2}},

    where :math:`D_i = \int f(y \mid a_i) \log q(i \mid y)\,\mathrm{d}y`
    is evaluated by Gauss-Hermite quadrature: the substitution
    :math:`y = a_i + \sqrt{2}\,\sigma t` turns the Gaussian integral into
    the quadrature's own weight function exactly, so :math:`D_i` is a
    weighted sum over ``n_nodes`` points and nothing is sampled.

    :math:`\lambda` plays the same role as in the Maxwell-Boltzmann
    family and is measured in the same units, nats per unit energy, so
    the two can be compared directly -- and should be, because the
    result is the reference this module's closed form is judged against:
    Maxwell-Boltzmann is *not* the maximizer of the mutual information,
    only of the entropy, and how little that costs is a number worth
    measuring rather than assuming (Kschischang and Pasupathy, 1993).

    .. warning::

       The energy constraint is what makes the answer bell-shaped. Left
       unconstrained (``lam=0``) the maximizer may spend **more** energy
       than the uniform distribution, pushing mass onto the outer points
       to separate them further -- at low SNR it does exactly that. A
       distribution that concentrates on the inner points is the answer
       to a constrained problem, never to an unconstrained one.

    Axes: *element-wise* -- the alphabet is a 1-D array of constellation
    points, and the result is a distribution over it.

    Parameters
    ----------
    alphabet : np.ndarray
        Constellation points :math:`a_i`, **real** (a complex alphabet
        with a zero imaginary part is accepted). A square QAM is the
        product of two independent PAM axes; pass one of them.
    sigma2 : float
        Noise variance :math:`\sigma^2` of the real AWGN channel,
        positive. Only the ratio of the alphabet's energy to
        :math:`\sigma^2` matters, so this is where the SNR enters.
    lam : float, optional, keyword-only
        The Lagrange multiplier :math:`\lambda` itself, non-negative.
        Useful when the multiplier is the object of interest; note that
        the energy it lands on is an output, not a choice.
    energy : float, optional, keyword-only
        Average energy budget :math:`\sum_i p_i a_i^2` the answer must
        spend, reached by a root-find on :math:`\lambda`. This is the
        parameterization with an operational meaning -- a transmitter
        has a power budget, not a Lagrange multiplier -- and the one to
        use when comparing against :func:`maxwell_boltzmann`, which only
        makes sense at equal energy. Exactly one of ``lam`` and
        ``energy`` must be given (D41).
    n_nodes : int, optional, keyword-only
        Gauss-Hermite nodes for :math:`D_i`. Default 40. Measured on
        16-PAM at 10 dB against a 200-node reference, the largest
        probability moves by 3.3e-7 at 20 nodes, 7.7e-10 at 40 and
        3.2e-13 at 80; a cleaner channel converges faster still, so 40
        is well inside what the result is read to.
    tol : float, optional, keyword-only
        Stop when Blahut's bound puts the objective within this many
        nats of its maximum. Default 1e-9.
    max_iter : int, optional, keyword-only
        Iteration ceiling. Default 5000. Reaching it is not silent: the
        remaining bound is logged, so the answer always comes with the
        distance it may still be from the maximum.

    Returns
    -------
    np.ndarray
        The maximizing distribution :math:`p_i`, same length as
        ``alphabet``.

    Raises
    ------
    ValueError
        If both or neither parameterization is given, if ``sigma2`` is
        not positive, if ``lam`` is negative, if the alphabet is
        complex, or if ``energy`` is outside the range the constraint
        can reach.

    References
    ----------
    R. E. Blahut, "Computation of channel capacity and rate-distortion
    functions", IEEE Trans. Inf. Theory, vol. 18, no. 4, pp. 460-473,
    July 1972; S. Arimoto, "An algorithm for computing the capacity of
    arbitrary discrete memoryless channels", IEEE Trans. Inf. Theory,
    vol. 18, no. 1, pp. 14-20, Jan. 1972; F. R. Kschischang,
    S. Pasupathy, "Optimal nonuniform signaling for Gaussian channels",
    IEEE Trans. Inf. Theory, vol. 39, no. 3, pp. 913-929, May 1993
    (that Maxwell-Boltzmann is within a hundredth of a bit of this).

    Examples
    --------
    The uniform distribution is optimal when the constraint is loose and
    the channel is clean -- a 4-PAM at 20 dB has nothing to gain:

    >>> pam4 = np.array([-3.0, -1.0, 1.0, 3.0])
    >>> clean = blahut_arimoto(pam4, sigma2=5.0 / 100, lam=0.0)
    >>> print(np.round(clean, 4))
    [0.25 0.25 0.25 0.25]

    Impose a budget and the answer becomes the bell shape shaping is
    named after. The uniform law spends 5 on this alphabet, so asking
    for 3 is asking for 2.2 dB less power:

    >>> best = blahut_arimoto(pam4, sigma2=5.0 / 100, energy=3.0)
    >>> print(np.round(best, 4))
    [0.125 0.375 0.375 0.125]
    >>> print(round(float(np.sum(best * pam4 ** 2)), 6))
    3.0

    At that energy the closed form is the same law, which is the whole
    reason the closed form is used everywhere else:

    >>> print(np.round(maxwell_boltzmann(pam4, lam=0.1373), 4))
    [0.125 0.375 0.375 0.125]
    """
    points = np.asarray(alphabet).ravel()
    if np.iscomplexobj(points) and np.any(np.imag(points) != 0):
        raise ValueError(
            "blahut_arimoto integrates over a real observation, got a "
            "complex alphabet. A square QAM constellation is the product "
            "of two independent PAM axes, each shaped on its own: pass "
            "np.unique(np.real(alphabet)).")
    points = np.real(points).astype(float)
    if not sigma2 > 0:
        raise ValueError(
            f"the noise variance is positive, got sigma2={sigma2}. A "
            f"noiseless channel makes every distribution carry its own "
            f"entropy, so the maximizer is the uniform one.")
    exclusive = (
        "blahut_arimoto takes exactly one of lam= and energy=, got "
        f"lam={lam} and energy={energy}. Pass energy= for the power "
        "budget the transmitter actually has -- that is the parameter "
        "with an operational meaning -- and lam= only when the "
        "multiplier itself is what you want (D41).")
    if (lam is None) == (energy is None):
        raise ValueError(exclusive)
    if lam is not None and lam < 0:
        raise ValueError(
            f"the Lagrange multiplier of an energy constraint is "
            f"non-negative, got lam={lam}. A negative value would pay "
            f"the input to be expensive.")

    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    observation = points[:, None] + np.sqrt(2.0 * sigma2) * nodes
    # (M, K, M): squared distance from every node to every alphabet point.
    # Node k of the integral attached to point i sits at
    # y = a_i + sqrt(2) sigma t_k, so the *self* term of that integral is
    # exactly t_k^2 and only the cross terms depend on the alphabet.
    exponent = (observation[..., None] - points) ** 2 / (2 * sigma2)
    self_exponent = nodes ** 2
    quadrature = weights / np.sqrt(np.pi)
    energies = points ** 2

    def solve(multiplier: float) -> np.ndarray:
        """Alternating maximization at a fixed multiplier.

        Always from the uniform law, never from a previous answer: a
        zero probability is an *absorbing* state of the iteration, since
        it makes its own posterior zero, so a warm start from a converged
        answer at a larger multiplier would be stuck there for good.
        """
        p = np.full(points.size, 1.0 / points.size)
        remaining = np.inf
        for _ in range(max_iter):
            log_p = np.log(np.maximum(p, 1e-300))
            shifted = log_p - exponent
            largest = np.max(shifted, axis=-1)
            evidence = largest + np.log(np.sum(
                np.exp(shifted - largest[..., None]), axis=-1))
            # log q(i | y) on the grid of the i-th integral
            posterior = log_p[:, None] - self_exponent - evidence
            gain = posterior @ quadrature - multiplier * energies
            # Blahut's stopping rule. The per-point divergence is
            # D'_i = D_i - log p_i -- the Kullback-Leibler distance from
            # f(.|a_i) to the current output law -- and the objective is
            # bracketed by its mean and its maximum:
            #   sum_i p_i (D'_i - lam E_i) <= max <= max_i (D'_i - lam E_i)
            # so the difference is a *certificate*: the answer is within
            # it of the maximum, in nats. Waiting for the probabilities
            # to settle instead is far stricter and much slower, because
            # the law can creep towards a zero long after the rate has
            # stopped moving.
            divergence = gain - log_p
            remaining = float(np.max(divergence) - p @ divergence)
            p = np.exp(gain - np.max(gain))
            p /= np.sum(p)
            if remaining < tol:
                break
        else:
            logger.warning(
                "blahut_arimoto stopped at max_iter=%d with %.2e nat "
                "still to gain (asked for %.0e). The alternating "
                "maximization converges slowly when the answer puts an "
                "exact zero on some constellation point, which is what "
                "it does at low SNR: the returned law is within that "
                "bound of the maximum, so read it as such or raise "
                "max_iter.", max_iter, remaining, tol)
        return p

    if lam is not None:
        return solve(float(lam))

    # internal invariant: the exclusivity check above rejected the case
    # where both are None, so energy is set on this branch
    assert energy is not None
    target = float(energy)
    # The reachable range. The lower end is the energy of the cheapest
    # points; the upper end is what the *unconstrained* maximizer spends,
    # and it can sit above the uniform law's energy -- see the warning
    # above -- so it has to be computed rather than assumed.
    ceiling = float(solve(0.0) @ energies)
    floor = float(np.min(energies))
    if not floor <= target <= ceiling:
        raise ValueError(
            f"got energy={target}, expected a value in "
            f"[{floor}, {ceiling}] for this alphabet and sigma2={sigma2}. "
            f"The upper end is what the maximizer spends when nothing "
            f"constrains it, which on a noisy channel is more than the "
            f"uniform law's {float(np.mean(energies))}: asking for more "
            f"than that is asking for a constraint that does not bind.")
    if target >= ceiling * (1 - 1e-12):
        return solve(0.0)

    # The energy is strictly decreasing in the multiplier, so this is a
    # root-find. Every evaluation is a full alternating maximization, so
    # the method matters: the Illinois variant of regula falsi converges
    # superlinearly and lands in about a dozen solves, where a plain
    # bisection to the same accuracy takes sixty.
    high = 1.0 / float(np.ptp(energies))
    while float(solve(high) @ energies) > target:
        high *= 4.0
        if high > _LAMBDA_MAX:
            raise ValueError(
                f"no multiplier below {_LAMBDA_MAX} brings the energy "
                f"down to {target} on this alphabet.")
    low, f_low = 0.0, ceiling - target
    f_high = float(solve(high) @ energies) - target
    atol = 1e-10 * max(target, 1.0)
    side = 0
    guess = high
    for _ in range(_BISECTION_STEPS):
        guess = (low * f_high - high * f_low) / (f_high - f_low)
        f_guess = float(solve(guess) @ energies) - target
        if abs(f_guess) < atol:
            break
        if f_guess > 0:                      # still spending too much
            low, f_low = guess, f_guess
            if side == 1:
                f_high *= 0.5
            side = 1
        else:
            high, f_high = guess, f_guess
            if side == -1:
                f_low *= 0.5
            side = -1
    return solve(guess)


@dataclass(slots=True)
class ConstantCompositionMatcher:
    r"""Constant composition distribution matching (CCDM).

    Signal Model
    ------------
    The matcher is an enumerative code on the permutations of a fixed
    multiset. Given a composition :math:`(n_1, \ldots, n_M)` with
    :math:`\sum_i n_i = n`, the set of output blocks is every sequence
    containing exactly :math:`n_i` copies of symbol :math:`i`, and there
    are

    .. math::

        N = \binom{n}{n_1, \ldots, n_M}
          = \frac{n!}{n_1! \, n_2! \cdots n_M!}

    of them. The matcher maps :math:`k = \lfloor \log_2 N \rfloor` input
    bits onto them by *unranking*: the integer the bits spell out is the
    lexicographic index of the block. Reading the index back off the
    block -- *ranking* -- inverts it exactly, which is why the pair is a
    bijection and not an approximation.

    Every block has the same empirical distribution :math:`n_i / n` by
    construction, so the shaped statistics are exact at every block
    length. What a finite block costs is **rate**: the matcher carries
    :math:`k/n` bits per symbol where the composition itself is worth
    :math:`H(n_i/n)`, and

    .. math::

        R_{\mathrm{loss}} = H\left(\frac{n_i}{n}\right) - \frac{k}{n}
        \;\xrightarrow[n \to \infty]{}\; 0

    Axes: *element-wise* -- a block is a 1-D sequence of ``length``
    symbol indices; batches are handled by :class:`DistributionMatcher`.

    Parameters
    ----------
    alphabet : np.ndarray
        The symbols the matcher emits, usually the non-negative
        amplitudes of a PAM constellation (see the module docstring on
        PAS). Only its length is used by the enumeration.
    composition : tuple of int, optional, keyword-only
        The counts :math:`n_i` directly.
    distribution : np.ndarray, optional, keyword-only
        A target distribution, quantized to a composition by
        :func:`composition_from_distribution`. Requires ``length``.
        Exactly one of ``composition`` and ``distribution`` is given
        (D41).
    length : int, optional, keyword-only
        Block length :math:`n`, required with ``distribution`` and
        forbidden with ``composition``, which already carries it.

    Attributes
    ----------
    n_bits : int
        :math:`k`, the number of input bits per block.
    length : int
        :math:`n`, the number of output symbols per block.

    Raises
    ------
    ValueError
        If the parameterization is ambiguous, or if the composition
        supports fewer than one bit.

    References
    ----------
    P. Schulte, G. Böcherer, "Constant composition distribution
    matching", IEEE Trans. Inf. Theory, vol. 62, no. 1, pp. 430-434,
    Jan. 2016.

    Examples
    --------
    >>> matcher = ConstantCompositionMatcher(np.array([1.0, 3.0]),
    ...                                      composition=(6, 2))
    >>> matcher.n_bits, matcher.length
    (4, 8)
    >>> block = matcher.encode(np.array([1, 0, 1, 1]))
    >>> print(block)
    [0 0 1 0 0 0 1 0]
    >>> print(np.bincount(block))            # the composition, exactly
    [6 2]
    >>> print(matcher.decode(block))
    [1 0 1 1]
    """

    alphabet: np.ndarray
    composition: tuple[int, ...] = field(default=(), kw_only=True)
    distribution: Optional[np.ndarray] = field(default=None, kw_only=True)
    length: int = field(default=0, kw_only=True)
    n_bits: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.alphabet = np.asarray(self.alphabet).ravel()
        given_composition = bool(self.composition)
        given_distribution = self.distribution is not None
        if given_composition == given_distribution:
            raise ValueError(
                "ConstantCompositionMatcher takes exactly one of "
                "composition= and distribution=, got "
                f"composition={self.composition!r} and "
                f"distribution={self.distribution!r}. Pass the counts "
                "themselves, or a target distribution together with the "
                "block length to quantize it (D41).")
        if given_distribution:
            if self.length <= 0:
                raise ValueError(
                    "distribution= needs the block length it is "
                    f"quantized onto, got length={self.length}. Pass "
                    "length=n, or give the composition directly.")
            self.composition = composition_from_distribution(
                np.asarray(self.distribution), self.length)
        elif self.length:
            raise ValueError(
                f"length={self.length} contradicts composition="
                f"{self.composition!r}, which already sums to "
                f"{sum(self.composition)}. Drop one of the two (D41).")

        if len(self.composition) != self.alphabet.size:
            raise ShapeError(
                f"the composition has {len(self.composition)} counts but "
                f"the alphabet has {self.alphabet.size} symbols -- one "
                f"count per symbol is expected, zeros included.")
        if any(count < 0 for count in self.composition):
            raise ValueError(
                f"a composition counts occurrences, got {self.composition!r}.")
        self.length = sum(self.composition)
        self.n_bits = _bits_in(_multinomial(self.composition))
        if self.n_bits < 1:
            raise ValueError(
                f"the composition {self.composition!r} admits "
                f"{_multinomial(self.composition)} sequence(s), which "
                f"carries no bit. Lengthen the block or spread the "
                f"counts over more symbols.")

    @property
    def rate(self) -> float:
        """Bits per output symbol, :math:`k/n`."""
        return self.n_bits / self.length

    @property
    def rate_loss(self) -> float:
        """What the finite block costs, in bits per symbol."""
        counts = np.asarray(self.composition, dtype=float) / self.length
        return distribution_entropy(counts) - self.rate

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Map ``n_bits`` bits onto one block of symbol indices."""
        index = _bits_to_int(bits, self.n_bits, type(self).__name__)
        remaining = list(self.composition)
        output = np.empty(self.length, dtype=int)
        for position in range(self.length):
            for symbol in range(len(remaining)):
                if remaining[symbol] == 0:
                    continue
                remaining[symbol] -= 1
                count = _multinomial(remaining)
                if index < count:
                    output[position] = symbol
                    break
                index -= count
                remaining[symbol] += 1
        return output

    def decode(self, block: np.ndarray) -> np.ndarray:
        """Read the ``n_bits`` bits back off one block of indices."""
        symbols = np.asarray(block).ravel()
        if symbols.size != self.length:
            raise ShapeError(
                f"{type(self).__name__} decodes blocks of {self.length} "
                f"symbols, got {symbols.size}.")
        counts = np.bincount(symbols, minlength=self.alphabet.size)
        if tuple(int(value) for value in counts) != self.composition:
            raise ValueError(
                f"the block has composition {tuple(int(v) for v in counts)}"
                f" but this matcher enumerates {self.composition!r}. A "
                f"detector error can produce such a block: it is not in "
                f"the code, so there is no index to read.")
        remaining = list(self.composition)
        index = 0
        for position in range(self.length):
            for symbol in range(len(remaining)):
                if remaining[symbol] == 0:
                    continue
                remaining[symbol] -= 1
                if symbol == symbols[position]:
                    break
                index += _multinomial(remaining)
                remaining[symbol] += 1
        return _int_to_bits(index, self.n_bits)


@dataclass(slots=True)
class SphereShaper:
    r"""Enumerative sphere shaping (ESS).

    Signal Model
    ------------
    Instead of forcing every block to have the same composition, ESS
    keeps every block inside an energy sphere: with an integer energy
    :math:`e_i` attached to each amplitude, the code is

    .. math::

        \mathcal{C} = \left\{ (s_1, \ldots, s_n) :
        \sum_{j=1}^{n} e_{s_j} \leq E_{\max} \right\}

    and the matcher enumerates it. Counting is a one-dimensional
    recursion over the remaining budget,

    .. math::

        N_t(E) = \sum_{i} N_{t-1}\left(E - e_i\right), \qquad
        N_0(E) = 1 \;\; \text{for } E \geq 0

    from which the lexicographic unranking follows exactly as for CCDM.

    Compared with a constant composition, the sphere is a **larger** set
    at the same average energy, because it lets a block spend more on one
    symbol and less on another. That is worth up to a few tenths of a dB
    at the block lengths where shaping is actually deployed: CCDM needs
    thousands of symbols to be efficient, ESS is already good at a few
    dozen. What it gives up is the exact per-block distribution, which is
    only reproduced on average.

    The energies are derived from the alphabet: :math:`|a_i|^2` shifted
    so the cheapest amplitude costs zero, then divided by the greatest
    common divisor. Both steps are exact on the odd-integer grids used in
    practice (amplitudes 1, 3, 5, 7 give energies 0, 1, 3, 6) and only
    change the *labelling* of the budget, not the code.

    Axes: *element-wise* -- a block is a 1-D sequence of ``length``
    symbol indices.

    Parameters
    ----------
    alphabet : np.ndarray
        Amplitudes, on a grid whose squared values are integers after
        shifting (odd integers, as produced by ``get_alphabet("PAM", M,
        norm=False)`` restricted to its positive half).
    length : int, keyword-only
        Block length :math:`n`.
    max_energy : int, optional, keyword-only
        The budget :math:`E_{\max}`, in the reduced integer unit.
    n_bits : int, optional, keyword-only
        Target number of input bits; the smallest budget carrying that
        many bits is used. Exactly one of the two is given (D41).

    Attributes
    ----------
    n_bits : int
        :math:`k = \lfloor \log_2 |\mathcal{C}| \rfloor`.
    max_energy : int
        The budget actually used.
    energies : tuple of int
        The reduced integer energy of each amplitude. The ``rate``
        property gives :math:`k/n`.

    Raises
    ------
    ValueError
        If the parameterization is ambiguous, if the squared amplitudes
        are not integers after shifting, or if the budget carries no bit.

    References
    ----------
    F. M. J. Willems, J. J. Wuijts, "A pragmatic approach to shaped coded
    modulation", IEEE Symp. Commun. Veh. Technol., 1993; Y. C. Gültekin
    et al., "Enumerative sphere shaping for wireless communications with
    short packets", IEEE Trans. Wireless Commun., vol. 19, no. 2,
    pp. 1098-1112, Feb. 2020.

    Examples
    --------
    >>> shaper = SphereShaper(np.array([1.0, 3.0]), length=8, max_energy=3)
    >>> shaper.energies, shaper.n_bits
    ((0, 1), 6)
    >>> block = shaper.encode(np.array([1, 0, 1, 1, 0, 1]))
    >>> print(block)
    [0 1 0 0 0 0 1 1]
    >>> int(np.sum(block))                   # at most max_energy = 3
    3
    >>> print(shaper.decode(block))
    [1 0 1 1 0 1]
    """

    alphabet: np.ndarray
    length: int = field(kw_only=True)
    max_energy: int = field(default=-1, kw_only=True)
    n_bits: int = field(default=-1, kw_only=True)
    energies: tuple[int, ...] = field(init=False, default=())
    # cumulative counts: _reachable[t][E] = sequences of length t of
    # energy at most E (declared for slots, D40a)
    _reachable: list[list[int]] = field(init=False, repr=False,
                                        default_factory=list)

    def __post_init__(self) -> None:
        self.alphabet = np.asarray(self.alphabet).ravel()
        if self.length <= 0:
            raise ValueError(
                f"a block holds at least one symbol, got length="
                f"{self.length}.")
        if (self.max_energy >= 0) == (self.n_bits >= 0):
            raise ValueError(
                "SphereShaper takes exactly one of max_energy= and "
                f"n_bits=, got max_energy={self.max_energy} and "
                f"n_bits={self.n_bits}. Pass the energy budget, or the "
                "number of bits a block must carry and let the budget "
                "follow (D41).")
        self.energies = _reduced_energies(self.alphabet)

        if self.n_bits >= 0:
            wanted = self.n_bits
            budget = 0
            ceiling = max(self.energies) * self.length
            while budget <= ceiling:
                self._build_table(budget)
                if _bits_in(self._reachable[self.length][budget]) >= wanted:
                    break
                budget += 1
            else:
                raise ValueError(
                    f"{self.length} symbols over {self.alphabet.size} "
                    f"amplitudes carry at most "
                    f"{_bits_in(self._reachable[self.length][ceiling])} "
                    f"bits, and {wanted} were asked for. Lengthen the "
                    f"block or widen the alphabet.")
            self.max_energy = budget
        else:
            self._build_table(self.max_energy)

        total = self._reachable[self.length][self.max_energy]
        self.n_bits = _bits_in(total)
        if self.n_bits < 1:
            raise ValueError(
                f"an energy budget of {self.max_energy} over "
                f"{self.length} symbols admits {total} sequence(s), which "
                f"carries no bit. Raise max_energy.")

    def _build_table(self, budget: int) -> None:
        """Cumulative counts up to ``budget``, rebuilt when it grows."""
        exact = [[0] * (budget + 1) for _ in range(self.length + 1)]
        exact[0][0] = 1
        for position in range(1, self.length + 1):
            row, previous = exact[position], exact[position - 1]
            for energy in range(budget + 1):
                row[energy] = sum(previous[energy - cost]
                                  for cost in self.energies
                                  if cost <= energy)
        self._reachable = []
        for row in exact:
            running, cumulative = 0, []
            for value in row:
                running += value
                cumulative.append(running)
            self._reachable.append(cumulative)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Map ``n_bits`` bits onto one block of symbol indices."""
        index = _bits_to_int(bits, self.n_bits, type(self).__name__)
        budget = self.max_energy
        output = np.empty(self.length, dtype=int)
        for position in range(self.length):
            left = self.length - position - 1
            for symbol, cost in enumerate(self.energies):
                if cost > budget:
                    continue
                count = self._reachable[left][budget - cost]
                if index < count:
                    output[position] = symbol
                    budget -= cost
                    break
                index -= count
        return output

    def decode(self, block: np.ndarray) -> np.ndarray:
        """Read the ``n_bits`` bits back off one block of indices."""
        symbols = np.asarray(block).ravel()
        if symbols.size != self.length:
            raise ShapeError(
                f"{type(self).__name__} decodes blocks of {self.length} "
                f"symbols, got {symbols.size}.")
        spent = sum(self.energies[int(symbol)] for symbol in symbols)
        if spent > self.max_energy:
            raise ValueError(
                f"the block costs {spent} energy units and the code "
                f"holds {self.max_energy}: it is outside the sphere, so "
                f"there is no index to read. A detector error can "
                f"produce such a block.")
        budget = self.max_energy
        index = 0
        for position in range(self.length):
            left = self.length - position - 1
            for symbol, cost in enumerate(self.energies):
                if cost > budget:
                    continue
                if symbol == symbols[position]:
                    budget -= cost
                    break
                index += self._reachable[left][budget - cost]
        return _int_to_bits(index, self.n_bits)

    @property
    def rate(self) -> float:
        """Bits per output symbol, :math:`k/n`."""
        return self.n_bits / self.length


@dataclass(slots=True)
class DistributionMatcher(Processor):
    r"""Shape a bit stream into symbol indices, block by block.

    Signal Model
    ------------
    The chain-level wrapper of :class:`ConstantCompositionMatcher` and
    :class:`SphereShaper`: it cuts the input into blocks of :math:`k`
    bits, maps each one onto :math:`n` symbol indices, and concatenates.
    Its output is what ``SymbolMapper`` consumes, so a shaped
    transmitter is a chain like any other.

    Axes: *element-wise* -- the bit stream is read along the last axis,
    and leading axes are carried through unchanged.

    Parameters
    ----------
    shaper : ConstantCompositionMatcher or SphereShaper
        The enumerative code doing the work.
    name : str, optional, keyword-only
        Block name.

    Raises
    ------
    ShapeError
        If the input length is not a multiple of ``shaper.n_bits``.

    References
    ----------
    G. Böcherer, F. Steiner, P. Schulte, IEEE Trans. Commun. 63(12),
    2015 (the PAS transmitter this block sits in).

    Examples
    --------
    >>> shaper = ConstantCompositionMatcher(np.array([1.0, 3.0]),
    ...                                     composition=(6, 2))
    >>> bits = np.array([1, 0, 1, 1, 0, 0, 0, 1])
    >>> indices = DistributionMatcher(shaper)(bits)
    >>> indices.size, int(np.sum(indices == 1))
    (16, 4)
    >>> print(DistributionDematcher(shaper)(indices))
    [1 0 1 1 0 0 0 1]
    """

    shaper: ConstantCompositionMatcher | SphereShaper
    name: str = field(default="distribution matcher", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        bits = np.asarray(X)
        step = self.shaper.n_bits
        if bits.shape[-1] % step:
            raise ShapeError(
                f"{type(self).__name__} consumes {step} bits per block "
                f"and got {bits.shape[-1]} along the last axis, which is "
                f"not a multiple of it. Pad the stream, or choose a "
                f"shaper whose n_bits divides it.")
        flat = bits.reshape(-1, step)
        blocks = np.stack([self.shaper.encode(row) for row in flat])
        return blocks.reshape(bits.shape[:-1] + (-1,))


@dataclass(slots=True)
class DistributionDematcher(Processor):
    r"""Read the bit stream back off shaped symbol indices.

    Signal Model
    ------------
    The exact inverse of :class:`DistributionMatcher`: blocks of
    :math:`n` indices become blocks of :math:`k` bits. Since the
    enumerative code is a bijection onto its block set, this is an
    identity on any sequence the matcher produced -- and an error on any
    other, which is the honest answer after a detection mistake.

    Axes: *element-wise* -- indices are read along the last axis.

    Parameters
    ----------
    shaper : ConstantCompositionMatcher or SphereShaper
        The same code used to shape.
    name : str, optional, keyword-only
        Block name.

    Raises
    ------
    ShapeError
        If the input length is not a multiple of ``shaper.length``.

    References
    ----------
    P. Schulte, G. Böcherer, IEEE Trans. Inf. Theory 62(1), 2016.

    Examples
    --------
    >>> shaper = SphereShaper(np.array([1.0, 3.0]), length=8, max_energy=3)
    >>> bits = np.array([1, 0, 1, 1, 0, 1])
    >>> indices = DistributionMatcher(shaper)(bits)
    >>> print(DistributionDematcher(shaper)(indices))
    [1 0 1 1 0 1]
    """

    shaper: ConstantCompositionMatcher | SphereShaper
    name: str = field(default="distribution dematcher", kw_only=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        indices = np.asarray(X)
        step = self.shaper.length
        if indices.shape[-1] % step:
            raise ShapeError(
                f"{type(self).__name__} consumes {step} symbols per block "
                f"and got {indices.shape[-1]} along the last axis, which "
                f"is not a multiple of it.")
        flat = indices.reshape(-1, step)
        blocks = np.stack([self.shaper.decode(row) for row in flat])
        return blocks.reshape(indices.shape[:-1] + (-1,))


@dataclass(slots=True)
class AmplitudeMapper(Processor):
    r"""Amplitude indices to signed PAM symbols: the sign of PAS.

    Signal Model
    ------------
    A distribution matcher emits amplitudes; a PAM constellation is
    signed. The missing half is one bit per symbol,

    .. math::

        y[n] = \varepsilon[n] \, a_{x[n]}, \qquad
        \Pr(\varepsilon[n] = \pm 1) = \tfrac{1}{2}

    and it is free: an equiprobable sign leaves the amplitude
    distribution untouched, so the composite constellation carries the
    symmetric Maxwell-Boltzmann law

    .. math::

        P_Y(\pm a_i) = \tfrac{1}{2} P_A(a_i)

    at the same energy, and adds exactly one bit per symbol to the rate.
    That is the whole reason probabilistic amplitude shaping works: the
    systematic FEC encoder's parity bits, which are uniform, are spent on
    the signs, and only the amplitudes go through the matcher.

    This block draws the signs itself, from its own generator, which is
    the same distribution a parity stream has. It is a stochastic block,
    so :meth:`~comnumpy.core.generics.Sequential.seed` reaches it (D6).

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    alphabet : np.ndarray
        Non-negative amplitudes :math:`a_i`, indexed by the input.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the block. Default ``"amplitude mapper"``.

    Raises
    ------
    ValueError
        If an amplitude is negative -- the sign is this block's business,
        and a signed input would make the constellation asymmetric.

    References
    ----------
    G. Böcherer, F. Steiner, P. Schulte, "Bandwidth efficient and
    rate-matched low-density parity-check coded modulation", IEEE Trans.
    Commun., vol. 63, no. 12, pp. 4651-4665, Dec. 2015, Section IV
    (the sign bits of PAS).

    Examples
    --------
    >>> mapper = AmplitudeMapper(np.array([1.0, 3.0, 5.0, 7.0]), seed=0)
    >>> print(mapper(np.array([0, 1, 0, 3, 2])))
    [-1. -3. -1.  7.  5.]
    >>> print(AmplitudeDemapper(mapper.alphabet)(mapper(np.array([0, 3]))))
    [0 3]
    """

    alphabet: np.ndarray
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="amplitude mapper", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.alphabet = np.real(np.asarray(self.alphabet)).ravel().astype(float)
        if np.any(self.alphabet < 0):
            raise ValueError(
                f"AmplitudeMapper signs the amplitudes itself, so it "
                f"expects non-negative ones, got a minimum of "
                f"{float(np.min(self.alphabet))}. Pass the positive half "
                f"of the PAM constellation.")
        self.rng = np.random.default_rng(self.seed)

    def forward(self, X: np.ndarray) -> np.ndarray:
        indices = np.asarray(X)
        signs = 1.0 - 2.0 * self.rng.integers(0, 2, size=indices.shape)
        return signs * self.alphabet[indices]


@dataclass(slots=True)
class AmplitudeDemapper(Processor):
    r"""Signed PAM samples back to amplitude indices.

    Signal Model
    ------------
    The inverse of :class:`AmplitudeMapper` on the half that carries the
    shaped data:

    .. math::

        \hat{x}[n] = \arg\min_i \left| \left|y[n]\right| - a_i \right|

    Taking the magnitude first is not a shortcut, it is the maximum
    likelihood decision: on a constellation symmetric about the origin,
    the nearest point to :math:`y` always has the amplitude nearest to
    :math:`|y|`, so deciding on :math:`|y|` and deciding on :math:`y`
    give the same amplitude. The sign is dropped because in PAS it
    carries parity, not data -- a real receiver feeds it to the FEC
    decoder instead.

    Axes: *element-wise* -- applied pointwise, shape-agnostic.

    Parameters
    ----------
    alphabet : np.ndarray
        The same non-negative amplitudes the mapper was given.
    name : str, optional, keyword-only
        Name of the block. Default ``"amplitude demapper"``.

    Raises
    ------
    ValueError
        If an amplitude is negative.

    References
    ----------
    G. Böcherer, F. Steiner, P. Schulte, IEEE Trans. Commun. 63(12),
    2015, Section IV.

    Examples
    --------
    >>> demapper = AmplitudeDemapper(np.array([1.0, 3.0, 5.0, 7.0]))
    >>> print(demapper(np.array([-0.8, 3.4, -5.2, 6.2])))
    [0 1 2 3]
    """

    alphabet: np.ndarray
    name: str = field(default="amplitude demapper", kw_only=True)

    def __post_init__(self) -> None:
        self.alphabet = np.real(np.asarray(self.alphabet)).ravel().astype(float)
        if np.any(self.alphabet < 0):
            raise ValueError(
                f"AmplitudeDemapper decides on |y|, so its alphabet holds "
                f"amplitudes, got a minimum of "
                f"{float(np.min(self.alphabet))}.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        magnitude = np.abs(np.real(np.asarray(X)))
        distance = np.abs(magnitude[..., None] - self.alphabet)
        return np.argmin(distance, axis=-1)


def _multinomial(counts: Sequence[int]) -> int:
    """``n! / prod(n_i!)`` in exact integer arithmetic."""
    total = sum(counts)
    value = math.factorial(total)
    for count in counts:
        value //= math.factorial(count)
    return value


def _bits_in(count: int) -> int:
    """``floor(log2(count))``, exactly, for an arbitrarily large count."""
    return count.bit_length() - 1 if count > 0 else 0


def _bits_to_int(bits: np.ndarray, width: int, block: str) -> int:
    values = np.asarray(bits).ravel()
    if values.size != width:
        raise ShapeError(
            f"{block} encodes {width} bits at a time, got {values.size}.")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(
            f"{block} encodes bits, got values outside {{0, 1}}.")
    index = 0
    for bit in values:
        index = (index << 1) | int(bit)
    return index


def _int_to_bits(index: int, width: int) -> np.ndarray:
    return np.array([(index >> shift) & 1
                     for shift in range(width - 1, -1, -1)], dtype=int)


def _reduced_energies(alphabet: np.ndarray) -> tuple[int, ...]:
    """Integer energies, shifted to start at zero and divided by their gcd."""
    squared = np.abs(np.asarray(alphabet, dtype=complex).ravel()) ** 2
    shifted = squared - np.min(squared)
    rounded = np.rint(shifted)
    if not np.allclose(shifted, rounded, atol=1e-9):
        raise ValueError(
            f"enumerative sphere shaping counts sequences by their "
            f"energy, so the squared amplitudes must be integers once "
            f"the cheapest one is subtracted. Got {shifted} -- pass the "
            f"raw odd-integer grid (get_alphabet(..., norm=False)) "
            f"rather than a normalized alphabet.")
    values = [int(value) for value in rounded]
    divisor = 0
    for value in values:
        divisor = math.gcd(divisor, value)
    return tuple(value // divisor for value in values) if divisor else \
        tuple(values)
