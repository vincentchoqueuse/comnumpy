"""Synchronization sequence catalog (decision D29).

Free functions returning the classic synchronization sequences, each
with its reference and its defining property (tested in
``tests/core/test_sequences.py``): Zadoff-Chu (LTE/NR PRACH),
Schmidl & Cox (OFDM timing + CFO), Barker (802.11b), Golay pairs
(802.11ad) and maximum-length sequences.
"""
from __future__ import annotations

import numpy as np

__all__ = ["zadoff_chu", "schmidl_cox_preamble", "barker", "golay_pair",
           "m_sequence"]


def zadoff_chu(u, N):
    r"""Zadoff-Chu (CAZAC) sequence of root :math:`u` and length :math:`N`.

    .. math::

        x_u[n] = \exp\left(-j \pi u \frac{n(n+1)}{N}\right), \qquad
        n = 0, \dots, N-1, \quad N \text{ odd}

    Constant amplitude, ideal (zero) periodic autocorrelation sidelobes
    when :math:`\gcd(u, N) = 1`. Used by the LTE and NR PRACH.

    References
    ----------
    3GPP TS 36.211, section 5.7.2; D. Chu, IEEE Trans. Inf. Theory 18(4), 1972.

    Examples
    --------
    >>> x = zadoff_chu(25, 139)
    >>> print(round(float(np.max(np.abs(np.abs(x) - 1))), 12))
    0.0
    """
    if N % 2 == 0:
        raise ValueError(f"Zadoff-Chu length N must be odd, got {N}")
    if np.gcd(u, N) != 1:
        raise ValueError(f"root u={u} must be coprime with N={N}")
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def schmidl_cox_preamble(N_fft, seed=None):
    r"""Schmidl & Cox OFDM preamble (two identical time-domain halves).

    QPSK symbols are placed on the even subcarriers only; the resulting
    time-domain OFDM symbol satisfies :math:`x[n + N/2] = x[n]`, the
    property the Schmidl & Cox timing/CFO metric relies on.

    Parameters
    ----------
    N_fft : int
        FFT size (even).
    seed : int, optional
        Seed for the QPSK symbols.

    Returns
    -------
    np.ndarray
        Time-domain preamble of length ``N_fft`` (unit average power).

    References
    ----------
    T. Schmidl, D. Cox, "Robust frequency and timing synchronization for
    OFDM," IEEE Trans. Commun. 45(12), 1997.

    Examples
    --------
    >>> x = schmidl_cox_preamble(64, seed=1)
    >>> print(round(float(np.max(np.abs(x[32:] - x[:32]))), 12))
    0.0
    """
    if N_fft % 2 != 0:
        raise ValueError(f"N_fft must be even, got {N_fft}")
    rng = np.random.default_rng(seed)
    X = np.zeros(N_fft, dtype=complex)
    qpsk = np.exp(1j * np.pi / 4 * (2 * rng.integers(0, 4, N_fft // 2) + 1))
    X[0::2] = np.sqrt(2) * qpsk
    x = np.fft.ifft(X) * np.sqrt(N_fft)
    return x


_BARKER = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def barker(length):
    r"""Barker code of the given length.

    Aperiodic autocorrelation sidelobes bounded by 1 in magnitude. The
    length-11 code is the 802.11b DSSS spreading sequence.

    References
    ----------
    IEEE 802.11-2020, clause 15.4.4.4 (length 11); R. H. Barker, 1953.

    Examples
    --------
    >>> print(barker(7))
    [ 1  1  1 -1 -1  1 -1]
    """
    if length not in _BARKER:
        raise ValueError(f"no Barker code of length {length}; "
                         f"known lengths: {sorted(_BARKER)}")
    return np.array(_BARKER[length])


def golay_pair(length):
    r"""Complementary Golay pair :math:`(a, b)` of the given length.

    The pair satisfies :math:`R_a[k] + R_b[k] = 2N \delta[k]` (the
    aperiodic autocorrelations cancel outside zero), the property used
    by the 802.11ad channel-estimation fields. Built by the standard
    recursive doubling ``a' = [a, b]``, ``b' = [a, -b]``.

    References
    ----------
    IEEE 802.11ad-2012, clause 21.11; M. Golay, IRE Trans. Inf. Theory, 1961.

    Examples
    --------
    >>> a, b = golay_pair(8)
    >>> Ra = np.correlate(a, a, "full"); Rb = np.correlate(b, b, "full")
    >>> print((np.abs(Ra + Rb) > 1e-12).sum())  # single nonzero lag
    1
    """
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError(f"length must be a power of two, got {length}")
    a, b = np.array([1.0]), np.array([1.0])
    while len(a) < length:
        a, b = np.concatenate([a, b]), np.concatenate([a, -b])
    return a, b


# primitive polynomials (taps fed back into stage 0), degree -> taps
_PRIMITIVE_TAPS = {
    2: (2, 1), 3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5),
    7: (7, 6), 8: (8, 6, 5, 4), 9: (9, 5), 10: (10, 7),
}


def m_sequence(degree, seed_state=None):
    r"""Maximum-length sequence from a degree-:math:`m` LFSR.

    Period :math:`2^m - 1`, two-valued periodic autocorrelation
    (:math:`N` at lag 0, :math:`-1` elsewhere), balance property.
    Output is bipolar (:math:`\pm 1`).

    Parameters
    ----------
    degree : int
        LFSR degree :math:`m` (2 to 10; primitive polynomial built in).
    seed_state : iterable of int, optional
        Initial register state (nonzero). Default: all ones.

    References
    ----------
    S. W. Golomb, *Shift Register Sequences*, Holden-Day, 1967.

    Examples
    --------
    >>> x = m_sequence(4)
    >>> len(x), int(np.sum(x == -1)), int(np.sum(x == 1))
    (15, 8, 7)
    """
    if degree not in _PRIMITIVE_TAPS:
        raise ValueError(f"degree must be in {sorted(_PRIMITIVE_TAPS)}, got {degree}")
    taps = _PRIMITIVE_TAPS[degree]
    state = np.ones(degree, dtype=int) if seed_state is None \
        else np.asarray(list(seed_state), dtype=int)
    if state.shape != (degree,) or not np.any(state):
        raise ValueError(f"seed_state must be a nonzero length-{degree} register")
    N = 2**degree - 1
    bits = np.empty(N, dtype=int)
    for i in range(N):
        bits[i] = state[-1]
        feedback = 0
        for tap in taps:
            feedback ^= state[tap - 1]
        state = np.roll(state, 1)
        state[0] = feedback
    return 1 - 2 * bits  # bipolar
