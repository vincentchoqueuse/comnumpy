r"""Achievable information rates measured on data: MI, GMI, normalized GMI.

A pre-FEC bit error rate does not say what a code will recover from a
channel: two channels with the same error rate can carry very different
amounts of information, and the "soft FEC limit" -- a fixed pre-FEC BER
threshold -- therefore mispredicts post-FEC performance when the
modulation format or the channel changes. That is the subject of the
reference below, whose conclusion is that the **generalized mutual
information** does predict it, consistently, across formats and across
linear and nonlinear optical channels.

The three rates here are lower bounds on the number of bits per symbol
a decoder of the matching structure can be driven to:

* :func:`compute_mi` -- the mutual information, for a **symbol-wise**
  (memoryless) decoder;
* :func:`compute_gmi` -- the generalized mutual information, for a
  **bit-wise** decoder: the structure of every practical soft-decision
  system, where a binary code is wrapped around a bit-metric demapper;
* :func:`compute_ngmi` -- the GMI divided by the bits per symbol, on the
  scale of a code rate.

Sign convention
---------------
The paper writes its L-values with the bit-1 hypothesis in the
numerator (its eq. (6)), so a positive L-value favours the bit **1**.
This module keeps comnumpy's own convention instead -- the one
:class:`~comnumpy.core.mappers.SymbolDemapper` produces, where a
positive LLR favours the bit **0** -- because two conventions inside
one library is a defect waiting to happen. The estimators below are
therefore written with :math:`(1 - 2 b)` where the paper writes
:math:`(-1)^{b}` on the negated L-value; the two are the same
expression.

References
----------
A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel, "Replacing
the soft-decision FEC limit paradigm in the design of optical
communication systems", J. Lightwave Technol., vol. 33, no. 20,
pp. 4338-4352, Oct. 2015, doi:10.1109/JLT.2015.2450537. Equations (6),
(7), (16), (17), (21)-(26) and (27)-(32) are cited individually below.

Examples
--------
>>> import numpy as np
>>> from comnumpy.core.utils import get_alphabet
>>> alphabet = get_alphabet("QAM", 4)
>>> rng = np.random.default_rng(0)
>>> symbols = rng.integers(0, 4, size=20000)
>>> received = alphabet[symbols] + np.sqrt(0.2 / 2) * (
...     rng.normal(size=20000) + 1j * rng.normal(size=20000))
>>> print(round(compute_mi(received, symbols, alphabet), 3))
1.893
>>> print(round(compute_gmi(received, symbols, alphabet), 3))
1.893
"""
import numpy as np

__all__ = ["compute_llr", "compute_mi", "compute_gmi", "compute_ngmi"]


def _bit_table(order: int) -> np.ndarray:
    """(M, m) bit labels of the alphabet indices, MSB first.

    Same convention as :func:`~comnumpy.core.utils.sym_2_bin`: the label
    of a constellation point is the binary expansion of its index, so a
    Gray mapping is carried by the *order* of the alphabet and there is
    no second labelling table to disagree with it.
    """
    bits_per_symbol = int(np.log2(order))
    if 2 ** bits_per_symbol != order:
        raise ValueError(
            f"bit-wise rates need an alphabet whose size is a power of two, "
            f"got M={order}; use compute_mi, which does not label the "
            f"symbols with bits")
    indices = np.arange(order)
    return (indices[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1


def _auxiliary_snr(X_received: np.ndarray, X_sent: np.ndarray) -> float:
    r"""The :math:`\rho` of the paper's eq. (29), matched to the data.

    The estimators assume a memoryless AWGN law
    :math:`f_{Y|X}(y|x) \propto e^{-\rho |y - x|^2}` with
    :math:`\rho = \mathbb{E}[|X|^2] / \mathbb{E}[|Z|^2]` the SNR
    (Section III-D). When the true channel is not Gaussian the rates
    become *mismatched* ones -- still achievable by a decoder built for
    that law, hence still lower bounds. Estimating :math:`\rho` from the
    data is the standard choice: it is the value that maximizes the
    Gaussian bound.
    """
    noise = float(np.mean(np.abs(X_received - X_sent) ** 2))
    if noise <= 0:
        raise ValueError(
            "the observation matches the transmitted symbols exactly, so no "
            "auxiliary SNR can be estimated; pass snr= explicitly")
    return float(np.mean(np.abs(X_sent) ** 2)) / noise


def _log_sum_exp(distance: np.ndarray, snr: float) -> np.ndarray:
    r"""\(\log \sum e^{-\rho d}\) over the last axis, without underflow."""
    smallest = np.min(distance, axis=-1)
    return (-snr * smallest
            + np.log(np.sum(np.exp(-snr * (distance - smallest[..., None])),
                            axis=-1)))


def compute_llr(X_received: np.ndarray, alphabet: np.ndarray, snr: float, *,
                max_log: bool = False) -> np.ndarray:
    r"""Bit-wise log-likelihood ratios of a memoryless AWGN channel.

    Signal Model
    ------------
    A symbol :math:`x[n] \in \mathcal{X}` carries :math:`m = \log_2 M`
    label bits :math:`b_1[n], \ldots, b_m[n]`, its index written in
    binary, MSB first. Writing :math:`\mathcal{X}_k^{\beta}` for the
    symbols whose :math:`k`-th bit is :math:`\beta`, and with a uniform
    input, eq. (29) of the reference gives

    .. math::

        L_k[n] = \log
            \frac{\sum_{x \in \mathcal{X}_k^{0}} e^{-\rho |y[n] - x|^2}}
                 {\sum_{x \in \mathcal{X}_k^{1}} e^{-\rho |y[n] - x|^2}}

    -- the paper's (29) with the two sums exchanged, so that a
    **positive** LLR favours the bit 0, comnumpy's convention. With
    ``max_log=True``, eq. (31):

    .. math::

        L_k[n] \simeq \rho \left[
            \min_{x \in \mathcal{X}_k^{1}} |y[n] - x|^2
          - \min_{x \in \mathcal{X}_k^{0}} |y[n] - x|^2 \right]

    which is what a practical demapper computes and what
    :class:`~comnumpy.core.mappers.SymbolDemapper` produces. The
    approximation is not free: eq. (32) shows the GMI it supports must
    then be maximized over a scaling of the L-values, which
    :func:`compute_gmi` does.

    Axes: *element-wise* -- the alphabet is searched on an axis appended
    internally and the bit index becomes a new last axis, so the output
    has the shape of ``X_received`` plus ``(m,)``.

    Parameters
    ----------
    X_received : np.ndarray
        Observation :math:`y[n]`, of any shape.
    alphabet : np.ndarray
        1-D constellation :math:`\mathcal{X}` of size :math:`M`, a power
        of two.
    snr : float
        Signal-to-noise ratio :math:`\rho` of the assumed AWGN law, in
        natural units (not dB).
    max_log : bool, optional, keyword-only
        Use the max-log approximation, eq. (7)/(31). Default False, the
        exact L-values of eq. (6)/(29).

    Returns
    -------
    np.ndarray
        L-values :math:`L_k[n]`, of shape ``X_received.shape + (m,)``,
        positive for the bit 0.

    References
    ----------
    A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel,
    J. Lightwave Technol., vol. 33, no. 20, pp. 4338-4352, 2015,
    eqs. (6), (7), (29) and (31).

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> alphabet = get_alphabet("QAM", 4)
    >>> llr = compute_llr(alphabet[[0, 3]], alphabet, 10.0)
    >>> print(np.sign(llr).astype(int))   # symbol 0 is "00", symbol 3 is "11"
    [[ 1  1]
     [-1 -1]]
    """
    bits = _bit_table(len(alphabet))
    distance = np.abs(X_received[..., None] - alphabet) ** 2
    llr = np.empty(X_received.shape + (bits.shape[1],))
    for index in range(bits.shape[1]):
        ones = bits[:, index] == 1
        if max_log:
            llr[..., index] = snr * (np.min(distance[..., ones], axis=-1)
                                     - np.min(distance[..., ~ones], axis=-1))
        else:
            # log-sum-exp with a shift *per hypothesis*, not a shared one:
            # with a single shift the weaker sum underflows to zero at high
            # SNR and the ratio returns an infinity. Shifted this way each
            # sum contains an exp(0) = 1 term, so it can never vanish, and
            # the expression degrades continuously to the max-log form.
            llr[..., index] = (_log_sum_exp(distance[..., ~ones], snr)
                               - _log_sum_exp(distance[..., ones], snr))
    return llr


def compute_mi(X_received: np.ndarray, symbols: np.ndarray,
               alphabet: np.ndarray, *, snr: float | None = None,
               px: np.ndarray | None = None) -> float:
    r"""Mutual information, in bit/symbol: the rate of a symbol-wise decoder.

    Signal Model
    ------------
    For a discrete constellation :math:`\mathcal{X}` with input
    distribution :math:`P_X`, eq. (16) of the reference gives

    .. math::

        I(X;Y) = \sum_{x \in \mathcal{X}} P_X(x) \int_{\mathbb{C}}
            f_{Y|X}(y|x) \log_2 \frac{f_{Y|X}(y|x)}{f_Y(y)}\, \mathrm{d}y

    and its Monte-Carlo estimate, eq. (17), replaces the integral by a
    sample mean over the received sequence:

    .. math::

        I(X;Y) \approx \frac{1}{n_s} \sum_{n=1}^{n_s}
            \log_2 \frac{f_{Y|X}(y[n] \mid x[n])}
                        {\sum_{a \in \mathcal{X}} P_X(a) f_{Y|X}(y[n]|a)}

    with the memoryless AWGN law
    :math:`f_{Y|X}(y|x) \propto e^{-\rho|y-x|^2}` of Section III-D. This
    is the rate a decoder operating on *symbols* can reach; Shannon's
    theorem makes it the limit of the coded-modulation rate
    :math:`R_\mathrm{cm}`, and it upper-bounds the bit-wise rate of
    :func:`compute_gmi` (paper's eq. (24) and the discussion after it).

    The estimator is evaluated on the transmitted sequence rather than
    on all :math:`M` symbols times :math:`n_s` noise realizations as the
    paper's (27) writes it; for an i.i.d. input the two have the same
    expectation, and one pass over the measured data is what a
    laboratory or a simulation actually produces.

    Axes: *reduces everything* -- the estimator is a mean over all the
    samples given, whatever their shape.

    Parameters
    ----------
    X_received : np.ndarray
        Observation :math:`y[n]`, of any shape.
    symbols : np.ndarray
        Indices of the transmitted symbols, same shape as
        ``X_received``.
    alphabet : np.ndarray
        1-D constellation :math:`\mathcal{X}` of size :math:`M`.
    snr : float, optional, keyword-only
        SNR :math:`\rho` of the assumed AWGN law, natural units. Default
        (None) estimates it from the data.
    px : np.ndarray, optional, keyword-only
        Input distribution :math:`P_X`, of length :math:`M`. Default
        (None) is uniform.

    Returns
    -------
    float
        Estimated mutual information, in bit per symbol, between 0 and
        :math:`\log_2 M`.

    References
    ----------
    A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel,
    "Replacing the soft-decision FEC limit paradigm in the design of
    optical communication systems", J. Lightwave Technol., vol. 33,
    no. 20, pp. 4338-4352, 2015, eqs. (16), (17) and (27).

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> alphabet = get_alphabet("PSK", 2)          # BPSK
    >>> rng = np.random.default_rng(2)
    >>> symbols = rng.integers(0, 2, size=50000)
    >>> received = alphabet[symbols] + 0.3 * rng.normal(size=50000)
    >>> print(round(compute_mi(received, symbols, alphabet), 3))
    0.997
    """
    received = np.asarray(X_received)
    sent = alphabet[np.asarray(symbols)]
    if snr is None:
        snr = _auxiliary_snr(received, sent)
    if px is None:
        px = np.full(len(alphabet), 1.0 / len(alphabet))
    px = np.asarray(px, dtype=float)

    distance = np.abs(received[..., None] - alphabet) ** 2
    shift = np.min(distance, axis=-1, keepdims=True)     # keeps exp() in range
    numerator = -snr * (np.abs(received - sent) ** 2 - shift[..., 0])
    denominator = np.log(np.sum(px * np.exp(-snr * (distance - shift)),
                                axis=-1))
    return float(np.mean(numerator - denominator) / np.log(2))


def compute_gmi(X_received: np.ndarray, symbols: np.ndarray,
                alphabet: np.ndarray, *, snr: float | None = None,
                max_log: bool = False) -> float:
    r"""Generalized mutual information: the rate of a bit-wise decoder.

    Signal Model
    ------------
    A soft-decision system wraps a binary code around a demapper that
    emits one L-value per bit, so the decoder sees :math:`m` parallel
    binary channels and not the symbol channel. The rate it can be
    driven to is the generalized mutual information, defined in eq. (21)
    of the reference as a maximization over a scaling :math:`s \geq 0`
    of the bit metric, and equal to the sum of the bit-wise mutual
    informations, eq. (24):

    .. math::

        \mathrm{GMI} = \sum_{k=1}^{m} I(B_k; Y)

    For **exact** L-values the optimum is :math:`s = 1` and eq. (30)
    gives the estimator, written here with comnumpy's LLR sign:

    .. math::

        \mathrm{GMI} \approx m - \frac{1}{n_s} \sum_{k=1}^{m}
            \sum_{n=1}^{n_s}
            \log_2\!\left(1 + e^{-(1 - 2 b_k[n])\, L_k[n]}\right)

    where :math:`b_k[n]` is the :math:`k`-th label bit of the
    transmitted symbol. The sum is the bit-metric penalty: it vanishes
    when every L-value is confident and correct, leaving the full
    :math:`m` bits.

    For **max-log** L-values that is not valid. The paper is explicit:
    "the minimization over :math:`s` in (32) is a mandatory step for
    approximated L-values", because using (30) on them returns a rate
    *lower* than the true one. With ``max_log=True`` this function
    therefore evaluates eq. (32),

    .. math::

        \mathrm{GMI} \approx m - \frac{1}{n_s} \min_{s \geq 0}
            \sum_{k=1}^{m} \sum_{n=1}^{n_s}
            \log_2\!\left(1 + e^{-s (1 - 2 b_k[n])\, L_k[n]}\right)

    solving the one-dimensional minimization numerically -- the penalty
    is convex in :math:`s`.

    The GMI never exceeds the mutual information of :func:`compute_mi`;
    the gap is the price of the bit-wise structure, small for a Gray
    labelling and large for a bad one.

    Axes: *reduces everything* -- the estimator is a mean over all the
    samples given, whatever their shape.

    Parameters
    ----------
    X_received : np.ndarray
        Observation :math:`y[n]`, of any shape.
    symbols : np.ndarray
        Indices of the transmitted symbols, same shape as
        ``X_received``. Their binary expansion is the label, MSB first.
    alphabet : np.ndarray
        1-D constellation of size :math:`M`, a power of two. The
        labelling is carried by the order of the alphabet, so
        ``get_alphabet("QAM", M)`` is Gray-labelled by construction.
    snr : float, optional, keyword-only
        SNR :math:`\rho` of the assumed AWGN law, natural units. Default
        (None) estimates it from the data.
    max_log : bool, optional, keyword-only
        Use max-log L-values, eq. (31), and the mandatory minimization
        over :math:`s` of eq. (32). Default False.

    Returns
    -------
    float
        Estimated GMI, in bit per symbol. Uniform input only: a shaped
        constellation needs the a-priori terms of eq. (26), which are
        not implemented.

    References
    ----------
    A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel,
    "Replacing the soft-decision FEC limit paradigm in the design of
    optical communication systems", J. Lightwave Technol., vol. 33,
    no. 20, pp. 4338-4352, 2015, eqs. (21)-(26), (30) and (32).

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> alphabet = get_alphabet("QAM", 16)
    >>> rng = np.random.default_rng(3)
    >>> symbols = rng.integers(0, 16, size=40000)
    >>> received = alphabet[symbols] + np.sqrt(0.2) * (
    ...     rng.normal(size=40000) + 1j * rng.normal(size=40000))
    >>> print(round(compute_gmi(received, symbols, alphabet), 3))
    1.686

    The bit-wise structure costs something -- the symbol-wise rate of
    :func:`compute_mi` is higher -- and the max-log demapper costs a
    little more, which eq. (32) is what measures honestly:

    >>> print(round(compute_mi(received, symbols, alphabet), 3))
    1.748
    >>> print(round(compute_gmi(received, symbols, alphabet,
    ...                         max_log=True), 3))
    1.675
    """
    received = np.asarray(X_received)
    indices = np.asarray(symbols)
    if snr is None:
        snr = _auxiliary_snr(received, alphabet[indices])
    bits_per_symbol = _bit_table(len(alphabet)).shape[1]

    llr = compute_llr(received, alphabet, snr, max_log=max_log)
    bits = (indices[..., None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1
    signed = (1 - 2 * bits) * llr

    def penalty(scale: float) -> float:
        # log2(1 + exp(z)) through logaddexp: exp() never overflows
        return float(np.mean(np.sum(
            np.logaddexp(0.0, -scale * signed), axis=-1)) / np.log(2))

    if not max_log:
        return bits_per_symbol - penalty(1.0)

    from scipy.optimize import minimize_scalar      # local import (D36)

    # eq. (32): mandatory for approximated L-values. The penalty is convex
    # in s, so a bounded scalar search is enough; s = 1 is the exact-LLR
    # optimum and the max-log optimum sits near it.
    solution = minimize_scalar(penalty, bounds=(1e-3, 10.0), method="bounded")
    return bits_per_symbol - float(min(solution.fun, penalty(1.0)))


def compute_ngmi(X_received: np.ndarray, symbols: np.ndarray,
                 alphabet: np.ndarray, *, snr: float | None = None,
                 max_log: bool = False) -> float:
    r"""Normalized GMI, on the scale of a code rate.

    Signal Model
    ------------
    Section IV of the reference compares metrics by their *normalized*
    value, dividing by the number of bits a symbol carries:

    .. math::

        \mathrm{NGMI} = \frac{\mathrm{GMI}}{m}

    A binary soft-decision code of rate :math:`R_c` decodes when
    :math:`\mathrm{NGMI} \geq R_c`, and -- this is the paper's result --
    that threshold is a property of the *code*, transferable from one
    channel and one modulation format to another, which a pre-FEC bit
    error rate threshold is not. Fig. 4 of the reference makes the point
    with two 8-QAM demappers whose normalized GMIs coincide at 0.55 and
    whose post-FEC BERs then coincide too.

    For a shaped input the general form normalizes the gap to the source
    entropy, and it reduces to the expression above when
    :math:`H(X) = m`. Shaping is not implemented.

    Axes: *reduces everything*.

    Parameters
    ----------
    X_received : np.ndarray
        Observation :math:`y[n]`, of any shape.
    symbols : np.ndarray
        Indices of the transmitted symbols.
    alphabet : np.ndarray
        1-D constellation of size :math:`M`, a power of two.
    snr : float, optional, keyword-only
        SNR :math:`\rho` of the assumed AWGN law, natural units. Default
        (None) estimates it from the data.
    max_log : bool, optional, keyword-only
        Use max-log L-values and the minimization of eq. (32). Default
        False.

    Returns
    -------
    float
        Normalized GMI, between 0 and 1.

    References
    ----------
    A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel,
    "Replacing the soft-decision FEC limit paradigm in the design of
    optical communication systems", J. Lightwave Technol., vol. 33,
    no. 20, pp. 4338-4352, 2015, Section IV and Figs. 3(c), 4.

    Examples
    --------
    >>> from comnumpy.core.utils import get_alphabet
    >>> alphabet = get_alphabet("QAM", 16)
    >>> rng = np.random.default_rng(4)
    >>> symbols = rng.integers(0, 16, size=40000)
    >>> received = alphabet[symbols] + np.sqrt(0.02) * (
    ...     rng.normal(size=40000) + 1j * rng.normal(size=40000))
    >>> ngmi = compute_ngmi(received, symbols, alphabet)
    >>> print(round(ngmi, 3), "-> a rate 0.8 code decodes:", bool(ngmi >= 0.8))
    0.963 -> a rate 0.8 code decodes: True
    """
    bits_per_symbol = _bit_table(len(alphabet)).shape[1]
    return compute_gmi(X_received, symbols, alphabet, snr=snr,
                       max_log=max_log) / bits_per_symbol
