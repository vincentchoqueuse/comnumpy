"""Synchronization sequence catalog (decision D29).

Free functions returning the classic synchronization sequences, each
with its reference and its defining property (tested in
``tests/core/test_frames_sequences.py``): Zadoff-Chu (LTE/NR PRACH),
Schmidl & Cox (OFDM timing + CFO), Barker (802.11b), Golay pairs
(802.11ad) and maximum-length sequences.

These sequences carry no information: they are chosen for a
*correlation* property, which is what makes them detectable in noise
after a single correlation with a locally stored replica. Each
``Signal Model`` below therefore states that property -- not only the
construction rule -- and the doctest that follows *computes* it.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["zadoff_chu", "schmidl_cox_preamble", "barker", "golay_pair",
           "m_sequence"]


def zadoff_chu(u: int, N: int) -> np.ndarray:
    r"""Zadoff-Chu (CAZAC) sequence of root :math:`u` and length :math:`N`.

    Signal Model
    ------------
    For an **odd** length :math:`N`, the sequence of root :math:`u` is

    .. math::

        x_u[n] = \exp\left(-j \pi u \frac{n(n+1)}{N}\right), \qquad
        n = 0, \dots, N-1

    This is the odd-length convention; the even-length variant replaces
    :math:`n(n+1)` by :math:`n^2` and is *not* accepted here (odd
    :math:`N` is enforced, see ``Raises``).

    CAZAC stands for *constant amplitude, zero autocorrelation*, the two
    properties that make the sequence a preamble. Writing the periodic
    (cyclic) correlation of two roots as

    .. math::

        \theta_{uv}[k] = \sum_{n=0}^{N-1} x_u[n] \, x_v^*[(n + k) \bmod N],

    the sequence satisfies, whenever :math:`\gcd(u, N) = 1`,

    .. math::

        \left|x_u[n]\right| = 1 \quad \forall n,
        \qquad
        \theta_{uu}[k] = N \, \delta[k]

    -- unit modulus (so the peak-to-average power ratio is 1 and no
    amplifier back-off is wasted on the preamble), and a periodic
    autocorrelation that is *exactly* zero at every nonzero lag, not
    merely small. Two distinct roots with :math:`\gcd(|u - v|, N) = 1`
    are in addition maximally separated: their cross-correlation has
    constant magnitude

    .. math::

        \left|\theta_{uv}[k]\right| = \sqrt{N} \quad \forall k,
        \qquad \text{i.e.} \quad
        \frac{\left|\theta_{uv}[k]\right|}{N} = \frac{1}{\sqrt{N}}

    once normalized by the sequence energy :math:`N`. A single
    correlation therefore both locates the sequence and identifies its
    root, which is why cell search in LTE (PSS) and 5G-NR, and the
    random-access preambles (PRACH) of both, are built from Zadoff-Chu
    sequences.

    Axes: *element-wise* -- generator: each sample is a closed-form
    function of its index; the output is 1-D of length ``N``.

    Parameters
    ----------
    u : int
        Root index :math:`u`, coprime with :math:`N`. Distinct roots give
        the low, flat cross-correlation floor above.
    N : int
        Sequence length :math:`N`, odd.

    Returns
    -------
    np.ndarray
        Complex sequence :math:`x_u[n]` of length :math:`N`, unit modulus.

    Raises
    ------
    ValueError
        If :math:`N` is even, or if :math:`\gcd(u, N) \neq 1` (the
        autocorrelation property does not hold otherwise).

    References
    ----------
    D. C. Chu, "Polyphase codes with good periodic correlation properties,"
    IEEE Trans. Inf. Theory 18(4), pp. 531-532, 1972; usage in
    3GPP TS 36.211 (LTE) and 3GPP TS 38.211 (5G-NR).

    Examples
    --------
    >>> x = zadoff_chu(25, 139)
    >>> print(round(float(np.max(np.abs(np.abs(x) - 1))), 12))  # constant amplitude
    0.0
    >>> theta = np.array([abs(np.vdot(x, np.roll(x, k))) for k in range(139)])
    >>> print(round(float(theta[0]), 9), round(float(np.max(theta[1:])), 9))
    139.0 0.0
    >>> y = zadoff_chu(34, 139)  # a second root, |u - v| = 9 coprime with 139
    >>> c = np.array([abs(np.vdot(x, np.roll(y, k))) for k in range(139)])
    >>> print(round(float(np.max(np.abs(c - np.sqrt(139)))), 9))  # flat, = sqrt(N)
    0.0
    """
    if N % 2 == 0:
        raise ValueError(f"Zadoff-Chu length N must be odd, got {N}")
    if np.gcd(u, N) != 1:
        raise ValueError(f"root u={u} must be coprime with N={N}")
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def schmidl_cox_preamble(N_fft: int, seed: int | None = None) -> np.ndarray:
    r"""Schmidl & Cox OFDM preamble (two identical time-domain halves).

    Signal Model
    ------------
    Unit-power QPSK symbols :math:`c[k]` are placed on the **even**
    subcarriers of an :math:`N`-point OFDM symbol, the odd ones being
    left empty:

    .. math::

        X[k] = \begin{cases}
            \sqrt{2}\, c\!\left[k/2\right], & k \text{ even} \\
            0, & k \text{ odd}
        \end{cases}
        \qquad
        x[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k] \,
               e^{\,j 2 \pi k n / N}

    the :math:`\sqrt{2}` restoring unit average power
    :math:`\mathbb{E}\left[|x[n]|^2\right] = 1` on the half-filled grid.
    Because only even :math:`k` contribute, every exponential is
    :math:`N/2`-periodic and the preamble consists of **two identical
    halves**:

    .. math::

        x\!\left[n + N/2\right] = x[n], \qquad n = 0, \dots, N/2 - 1

    This is the structure the Schmidl & Cox synchronizer exploits. On the
    received signal :math:`y[n]` the metric

    .. math::

        P(d) = \sum_{m=0}^{N/2-1} y^*[d+m] \, y\!\left[d+m+N/2\right],
        \qquad
        M(d) = \frac{\left|P(d)\right|^2}
                    {\left(\sum_{m=0}^{N/2-1}
                     \left|y\!\left[d+m+N/2\right]\right|^2\right)^2}

    reaches its plateau :math:`M(d) = 1` when the window :math:`d` is
    aligned with the preamble -- that gives the **timing**. A carrier
    frequency offset of :math:`\varepsilon` subcarrier spacings turns the
    repetition into :math:`y[n + N/2] = y[n] e^{\,j \pi \varepsilon}`, so
    the same quantity also gives the **CFO**:

    .. math::

        \hat{\varepsilon} = \frac{\angle P(\hat{d})}{\pi},
        \qquad \left|\varepsilon\right| < 1

    one correlation delivering both estimates.

    Axes: *element-wise* -- generator: returns a 1-D preamble of length
    ``N_fft``.

    Parameters
    ----------
    N_fft : int
        FFT size :math:`N` (even, so that the two halves have equal length).
    seed : int, optional
        Seed of the RNG drawing the QPSK symbols :math:`c[k]`. Fix it to
        make the preamble reproducible between transmitter and receiver.

    Returns
    -------
    np.ndarray
        Time-domain preamble :math:`x[n]` of length :math:`N`, of unit
        average power, with :math:`x[n + N/2] = x[n]`.

    Raises
    ------
    ValueError
        If ``N_fft`` is odd.

    References
    ----------
    T. M. Schmidl, D. C. Cox, "Robust frequency and timing synchronization
    for OFDM," IEEE Trans. Commun. 45(12), pp. 1613-1621, 1997;
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 4.

    Examples
    --------
    >>> x = schmidl_cox_preamble(64, seed=1)
    >>> print(round(float(np.max(np.abs(x[32:] - x[:32]))), 12))  # x[n + N/2] = x[n]
    0.0
    >>> eps = 0.25  # CFO of a quarter subcarrier spacing
    >>> y = x * np.exp(2j * np.pi * eps * np.arange(64) / 64)
    >>> P = np.vdot(y[:32], y[32:])  # Schmidl & Cox metric, window aligned (d = 0)
    >>> M = abs(P) ** 2 / float(np.sum(np.abs(y[32:]) ** 2)) ** 2
    >>> print(round(M, 12), round(float(np.angle(P) / np.pi), 12))  # timing, then CFO
    1.0 0.25
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


def barker(length: int) -> np.ndarray:
    r"""Barker code of the given length.

    Signal Model
    ------------
    A Barker code is a binary sequence :math:`x[n] \in \{-1, +1\}` of
    length :math:`N` whose **aperiodic** autocorrelation

    .. math::

        R_x[k] = \sum_{n} x[n] \, x[n + k], \qquad
        k = -(N-1), \dots, N-1

    (the sum running over the overlapping samples, so no cyclic
    extension) has all its sidelobes bounded by one:

    .. math::

        R_x[0] = N, \qquad \left|R_x[k]\right| \leq 1 \quad \forall k \neq 0

    This is the best possible sidelobe level for a binary sequence, and
    the reason the aperiodic -- rather than periodic -- autocorrelation
    is the relevant one: a packet preamble is transmitted **once**, so
    the receiver correlates over a partial overlap at every trial lag.
    A merit factor :math:`N^2 / (2 \sum_{k \neq 0} R_x[k]^2)` that grows
    with :math:`N` is what makes the peak detectable in noise.

    Only a finite set of lengths is known -- this implementation provides
    :math:`N \in \{2, 3, 4, 5, 7, 11, 13\}`. No longer Barker code
    exists for odd :math:`N > 13`, and none is conjectured to exist for
    any :math:`N > 13`. The length-11 code is the DSSS spreading
    sequence of 802.11b; the length-13 code is the classic radar pulse
    compression code.

    Axes: *element-wise* -- generator: returns a 1-D bipolar sequence of
    length ``length``.

    Parameters
    ----------
    length : int
        Code length :math:`N`, one of ``{2, 3, 4, 5, 7, 11, 13}``.

    Returns
    -------
    np.ndarray
        Integer array :math:`x[n] \in \{-1, +1\}` of length :math:`N`.

    Raises
    ------
    ValueError
        If no Barker code of the requested length is tabulated.

    References
    ----------
    R. H. Barker, "Group synchronizing of binary digital systems," in
    *Communication Theory*, W. Jackson (ed.), Academic Press, 1953,
    pp. 273-287; J. G. Proakis, M. Salehi, *Digital Communications*,
    5th ed., McGraw-Hill, 2008 (spread-spectrum signalling);
    IEEE 802.11-2020, clause 15.4.4.4 (length 11).

    Examples
    --------
    >>> code = barker(7)
    >>> print(code)
    [ 1  1  1 -1 -1  1 -1]
    >>> R = np.correlate(code, code, "full")  # aperiodic autocorrelation
    >>> print(R)
    [-1  0 -1  0 -1  0  7  0 -1  0 -1  0 -1]
    >>> print(int(np.max(np.abs(np.delete(R, 6)))))  # every sidelobe bounded by 1
    1
    """
    if length not in _BARKER:
        raise ValueError(f"no Barker code of length {length}; "
                         f"known lengths: {sorted(_BARKER)}")
    return np.array(_BARKER[length])


def golay_pair(length: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Complementary Golay pair :math:`(a, b)` of the given length.

    Signal Model
    ------------
    Two bipolar sequences :math:`a[n], b[n] \in \{-1, +1\}` of length
    :math:`N` form a *complementary pair* when their **aperiodic**
    autocorrelations

    .. math::

        R_a[k] = \sum_{n} a[n] \, a[n + k], \qquad
        R_b[k] = \sum_{n} b[n] \, b[n + k]

    cancel each other everywhere except at the origin:

    .. math::

        R_a[k] + R_b[k] = 2 N \, \delta[k]

    Neither sequence has small sidelobes on its own -- it is their
    **sum** that is a perfect impulse. Correlating the received signal
    with both replicas and adding the two outputs therefore removes the
    sidelobes *exactly*, at any length, where a Barker code only bounds
    them by one and stops at :math:`N = 13`. This is what makes Golay
    pairs the channel-estimation fields of 802.11ad/ay and of many radar
    waveforms.

    The pair is built by the standard recursive doubling, starting from
    :math:`a_0 = b_0 = [1]`:

    .. math::

        a_{m+1} = \left[a_m, \; b_m\right], \qquad
        b_{m+1} = \left[a_m, \; -b_m\right]

    which preserves complementarity and doubles the length at each step,
    hence the restriction to powers of two.

    Axes: *element-wise* -- generator: returns two 1-D sequences of
    length ``length``.

    Parameters
    ----------
    length : int
        Sequence length :math:`N`, a power of two.

    Returns
    -------
    tuple of np.ndarray
        The pair :math:`(a[n], b[n])`, each of length :math:`N` with
        values in :math:`\{-1, +1\}`.

    Raises
    ------
    ValueError
        If ``length`` is not a positive power of two.

    References
    ----------
    M. J. E. Golay, "Complementary series," IRE Trans. Inf. Theory 7(2),
    pp. 82-87, 1961; IEEE 802.11ad-2012, clause 21.11.

    Examples
    --------
    >>> a, b = golay_pair(8)
    >>> print(np.correlate(a, a, "full"))  # sidelobes are large on their own
    [ 1.  0.  1.  0.  3.  0. -1.  8. -1.  0.  3.  0.  1.  0.  1.]
    >>> print(np.correlate(b, b, "full"))  # and opposite in the partner
    [-1.  0. -1.  0. -3.  0.  1.  8.  1.  0. -3.  0. -1.  0. -1.]
    >>> print(np.correlate(a, a, "full") + np.correlate(b, b, "full"))  # 2N delta[k]
    [ 0.  0.  0.  0.  0.  0.  0. 16.  0.  0.  0.  0.  0.  0.  0.]
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


def m_sequence(degree: int, seed_state: Iterable[int] | None = None) -> np.ndarray:
    r"""Maximum-length sequence from a degree-:math:`m` LFSR.

    Signal Model
    ------------
    A linear-feedback shift register of degree :math:`m` produces the
    binary recursion

    .. math::

        b[n] = \bigoplus_{i \in \mathcal{T}} b[n - i]
        \qquad (\text{modulo } 2)

    where the tap set :math:`\mathcal{T}` is that of a **primitive**
    polynomial :math:`g(D) = 1 + \sum_{i \in \mathcal{T}} D^{i}` of
    degree :math:`m` (tabulated here for :math:`m = 2, \dots, 10`). The
    output is mapped to a bipolar sequence

    .. math::

        x[n] = 1 - 2 \, b[n] \in \{-1, +1\}

    Primitivity is what makes the register visit **every** nonzero state
    exactly once before repeating, so the period is maximal:

    .. math::

        N = 2^m - 1

    Two properties follow, and they are the reason m-sequences are used
    as spreading and ranging codes. First the *balance* property: one
    period contains :math:`2^{m-1}` ones and :math:`2^{m-1} - 1` zeros,
    hence :math:`\sum_{n=0}^{N-1} x[n] = -1` -- a nearly zero-mean,
    noise-like waveform. Second, the periodic autocorrelation takes only
    **two values**:

    .. math::

        \theta_x[k] = \sum_{n=0}^{N-1} x[n] \, x[(n + k) \bmod N]
        = \begin{cases}
            N, & k \equiv 0 \pmod N \\
            -1, & \text{otherwise}
        \end{cases}

    a single sharp peak over a floor at :math:`-1/N` of the peak, which
    is why a receiver can acquire code phase by sliding correlation and
    why the sequence behaves, spectrally, almost like white noise.

    Axes: *element-wise* -- generator: returns one 1-D period of length
    :math:`2^m - 1`.

    Parameters
    ----------
    degree : int
        LFSR degree :math:`m`, from 2 to 10; the primitive polynomial
        :math:`g(D)` (hence the tap set :math:`\mathcal{T}`) is built in.
    seed_state : iterable of int, optional
        Initial register state, a nonzero binary word of length
        :math:`m`. Only the *phase* of the sequence depends on it, not
        its properties. Default: all ones.

    Returns
    -------
    np.ndarray
        One full period :math:`x[n] \in \{-1, +1\}` of length
        :math:`N = 2^m - 1`.

    Raises
    ------
    ValueError
        If ``degree`` has no tabulated primitive polynomial, or if
        ``seed_state`` is not a nonzero word of length :math:`m`.

    References
    ----------
    S. W. Golomb, *Shift Register Sequences*, Holden-Day, 1967
    (Chapters 3-4: randomness postulates and the two-valued
    autocorrelation); J. G. Proakis, M. Salehi, *Digital Communications*,
    5th ed., McGraw-Hill, 2008 (PN sequences in spread spectrum).

    Examples
    --------
    >>> x = m_sequence(4)
    >>> print(x)
    [-1 -1 -1 -1  1  1  1 -1  1  1 -1 -1  1 -1  1]
    >>> theta = [int(np.dot(x, np.roll(x, k))) for k in range(15)]
    >>> print(theta)  # two-valued: N at lag 0, -1 at every other lag
    [15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    >>> int(np.sum(x == 1)), int(np.sum(x == -1)), int(np.sum(x))  # balance
    (7, 8, -1)
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
