import numpy as np

from comnumpy.core.utils import sym_2_bin  # single definition (annex A.5)

__all__ = [
    "compute_ser_awgn_psk", "compute_ser_awgn_qam", "compute_metric_awgn_theo",
    "compute_ser", "compute_ber", "compute_evm", "compute_effective_snr",
    "compute_power", "compute_ccdf", "compute_acpr", "signal_report",
]


def compute_ser_awgn_psk(order, snr_per_bit):
    r"""
    Compute the theoretical Symbol Error Rate (SER) for PSK modulation over an AWGN channel.

    Signal Model
    ------------
    The receiver observes :math:`y[n] = x[n] + b[n]` with
    :math:`b[n] \sim \mathcal{CN}(0, \sigma^2)` and detects the closest
    :math:`M`-PSK symbol. With :math:`k = \log_2 M` bits per symbol, the
    **signal-to-noise ratio per bit** :math:`\gamma_b = E_b / N_0` and the
    Gaussian tail function

    .. math::

        Q(u) = \frac{1}{\sqrt{2\pi}} \int_u^{\infty} e^{-t^2/2} \, dt,

    the closed-form symbol error probabilities are

    * :math:`M = 2` (BPSK), exact:

    .. math::

        P_s = Q\left(\sqrt{2 \gamma_b}\right)

    * :math:`M = 4` (QPSK), exact:

    .. math::

        P_s = 2 Q\left(\sqrt{2 \gamma_b}\right)
              \left[1 - \frac{1}{2} Q\left(\sqrt{2 \gamma_b}\right)\right]

    * :math:`M > 4`, the standard union-bound approximation, tight at high SNR:

    .. math::

        P_s \simeq 2 Q\left(\sqrt{2 k \gamma_b} \, \sin\frac{\pi}{M}\right)

    Note that the argument is the SNR **per bit** :math:`\gamma_b`, not the
    SNR per symbol :math:`\gamma_s = k \gamma_b`: pass
    ``snr_per_bit = 10**(snr_dB/10) / np.log2(order)`` when the sweep is
    parameterized by the symbol SNR (as ``AWGN(snr_dB=...)`` is).

    Parameters
    ----------
    order : int
        Modulation order :math:`M` (2, 4, 8, 16, ...).
    snr_per_bit : float or np.ndarray
        Signal-to-noise ratio per bit :math:`\gamma_b = E_b / N_0` in linear scale.

    Returns
    -------
    float or np.ndarray
        Theoretical SER value(s) :math:`P_s`, with the same shape as ``snr_per_bit``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.3 (error probability of M-ary PSK);
    the same result appears in Proakis, *Digital Communications*,
    McGraw-Hill, 2001, pp. 271-272.

    Examples
    --------
    >>> print(f"{compute_ser_awgn_psk(2, 10.0):.4e}")   # BPSK, Eb/N0 = 10 dB
    3.8721e-06
    >>> print(f"{compute_ser_awgn_psk(4, 10.0):.4e}")   # QPSK, Eb/N0 = 10 dB
    7.7442e-06
    >>> print(f"{compute_ser_awgn_psk(8, 10.0):.4e}")   # 8-PSK, Eb/N0 = 10 dB
    3.0342e-03
    """
    from scipy.stats import norm  # local import (D36)

    gamma_b = snr_per_bit
    if order < 2 or (order & (order - 1)) != 0:
        raise ValueError(
            f"compute_ser_awgn_psk: expected a power-of-two PSK order "
            f"(2, 4, 8, ...), got order={order} -- the closed forms below "
            f"are derived for M-PSK with M a power of two.")
    k = int(np.log2(order))

    if order == 2:
        # see book Proakis "Digital communication", p 271
        argument = np.sqrt(2*gamma_b)
        value = norm.sf(argument)

    if order == 4:
        # see book Proakis "Digital communication", p 272
        argument = np.sqrt(2*gamma_b)
        term = norm.sf(argument)
        value = 2*term*(1-0.5*term)

    if order > 4:
        M = order
        argument = np.sqrt(2*k*gamma_b)*np.sin(np.pi/M)
        value = 2*norm.sf(argument)

    return value


def compute_ser_awgn_qam(order, snr_per_bit):
    r"""
    Compute the theoretical Symbol Error Rate (SER) for square QAM modulation over an AWGN channel.

    Signal Model
    ------------
    A square :math:`M`-QAM constellation (:math:`M = 4, 16, 64, 256, \dots`)
    is the product of two independent :math:`\sqrt{M}`-PAM constellations.
    With :math:`k = \log_2 M` bits per symbol, the **signal-to-noise ratio
    per bit** :math:`\gamma_b = E_b / N_0` and the Gaussian tail function
    :math:`Q(\cdot)`, the error probability of one PAM component is

    .. math::

        P_{\sqrt{M}} = 2 \left(1 - \frac{1}{\sqrt{M}}\right)
                       Q\left(\sqrt{\frac{3 k \gamma_b}{M-1}}\right)

    and a symbol is correct only if both components are, hence

    .. math::

        P_s = 1 - \left(1 - P_{\sqrt{M}}\right)^2

    The argument is the SNR **per bit** :math:`\gamma_b`; the average SNR
    per symbol is :math:`\gamma_s = k \gamma_b`, so the argument of
    :math:`Q(\cdot)` also reads :math:`\sqrt{3 \gamma_s / (M-1)}`.

    Parameters
    ----------
    order : int
        Modulation order :math:`M` (4, 16, 64, 256, ...). Only square
        constellations, i.e. even :math:`k = \log_2 M`, are covered by
        this expression.
    snr_per_bit : float or np.ndarray
        Signal-to-noise ratio per bit :math:`\gamma_b = E_b / N_0` in linear scale.

    Returns
    -------
    float or np.ndarray
        Theoretical SER value(s) :math:`P_s`, with the same shape as ``snr_per_bit``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.3 (error probability of QAM);
    the same result appears in Proakis, *Digital Communications*,
    McGraw-Hill, 2001, p. 280.

    Examples
    --------
    >>> print(f"{compute_ser_awgn_qam(4, 10.0):.4e}")    # QPSK, Eb/N0 = 10 dB
    7.7442e-06
    >>> print(f"{compute_ser_awgn_qam(16, 10.0):.4e}")   # 16-QAM, Eb/N0 = 10 dB
    7.0043e-03
    >>> gamma_b = 10 ** (np.array([10.0, 20.0]) / 10) / 4   # 16-QAM at 10 and 20 dB symbol SNR
    >>> print(np.array2string(compute_ser_awgn_qam(16, gamma_b), precision=4))
    [2.2203e-01 1.1616e-05]
    """
    from scipy.stats import norm  # local import (D36)

    gamma_b = snr_per_bit

    # see book Proakis "Digital communication", p 280
    M = order
    k = np.log2(order)
    argument = np.sqrt(3*k*gamma_b/(M-1))
    P_sqrt_M = 2*(1-1/np.sqrt(M))*norm.sf(argument)

    value = 1-(1-P_sqrt_M)**2

    return value


def compute_metric_awgn_theo(modulation, order, snr_per_bit, type="ser"):
    r"""
    Compute the theoretical error rate of a given modulation over an AWGN channel.

    Signal Model
    ------------
    Front-end to the closed-form expressions of
    :func:`compute_ser_awgn_psk` and :func:`compute_ser_awgn_qam`, both
    parameterized by the signal-to-noise ratio **per bit**
    :math:`\gamma_b = E_b / N_0`. With ``type="bin"`` the symbol error
    rate is converted into a bit error rate through the usual Gray-mapping
    approximation, in which a symbol error corrupts a single bit out of
    :math:`k = \log_2 M`:

    .. math::

        P_b \simeq \frac{P_s}{k}

    This conversion is an upper-SNR approximation, not an identity: it is
    accurate when :math:`P_s \ll 1` and the constellation is Gray-mapped.

    Parameters
    ----------
    modulation : str
        Modulation family: ``"PSK"`` or ``"QAM"``.
    order : int
        Modulation order :math:`M` (4, 16, 64, ...).
    snr_per_bit : float or np.ndarray
        Signal-to-noise ratio per bit :math:`\gamma_b = E_b / N_0` in linear scale.
    type : str, optional
        Error metric: ``"ser"`` for the symbol error rate :math:`P_s`, or
        ``"bin"`` for the bit error rate :math:`P_b = P_s / k`.
        Default is ``"ser"``.

    Returns
    -------
    float or np.ndarray
        Theoretical error rate value(s), with the same shape as ``snr_per_bit``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.3 (symbol and bit error probabilities
    with Gray mapping).

    Examples
    --------
    >>> print(f"{compute_metric_awgn_theo('QAM', 16, 10.0):.4e}")
    7.0043e-03
    >>> print(f"{compute_metric_awgn_theo('QAM', 16, 10.0, type='bin'):.4e}")
    1.7511e-03
    >>> print(f"{compute_metric_awgn_theo('PSK', 8, 10.0):.4e}")
    3.0342e-03
    """
    if modulation == "PSK":
        value = compute_ser_awgn_psk(order, snr_per_bit)

    elif modulation == "QAM":
        value = compute_ser_awgn_qam(order, snr_per_bit)
    else:
        raise ValueError(
            f"compute_metric_awgn_theo: unknown modulation "
            f"{modulation!r}; expected 'PSK' or 'QAM' -- these are the two "
            f"families with a closed-form SER implemented here.")

    if type == "bin":
        k = int(np.log2(order))
        value = value/k

    return value

def compute_ser(X_target, X_detected, axis=None):
    r"""
    Compute the Symbol Error Rate (SER) between the target and detected symbols.

    Signal Model
    ------------
    The transmitted symbol indices :math:`x[n]` are compared with the
    detected ones :math:`\hat{x}[n]` over :math:`N` symbols; the SER is
    the normalized count of mismatches:

    .. math::

        \mathrm{SER} = \frac{1}{N} \sum_{n=0}^{N-1}
                       \mathbb{1}\left\{x[n] \neq \hat{x}[n]\right\}

    Comparison is exact equality, so the two arrays must carry the same
    kind of quantity -- symbol *indices* on both sides (the output of a
    ``SymbolGenerator`` and of a ``SymbolDemapper``), not complex
    constellation points.

    Axes: *declared axis* -- errors are counted along ``axis``, which must
    then hold the symbols; with ``axis=None`` (default) both arrays are
    raveled and truncated to their common length
    :math:`N = \min(N_{\text{target}}, N_{\text{detected}})`.

    Parameters
    ----------
    X_target : ndarray
        Reference symbol indices :math:`x[n]` (what was transmitted).
    X_detected : ndarray
        Detected symbol indices :math:`\hat{x}[n]` (what the receiver decided).
    axis : int or None, optional
        Axis along which to count the errors. If None, the arrays are
        raveled into 1-D arrays. Default is None.

    Returns
    -------
    float or ndarray
        The SER. A float when ``axis`` is None, an array of SER values
        otherwise (one per position of the remaining axes).

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.2 (symbol error probability).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 0])
    >>> x_hat = np.array([0, 1, 2, 0, 0])
    >>> print(compute_ser(x, x_hat))
    0.2
    >>> X = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    >>> X_hat = np.array([[0, 1, 2, 0], [0, 0, 2, 3]])
    >>> print(compute_ser(X, X_hat, axis=1))
    [0.25 0.25]
    """
    if axis is None:
        x_target = np.ravel(X_target)
        x_detected = np.ravel(X_detected)
        N = min(len(x_detected), len(x_target))
        nb_errors = np.count_nonzero(x_target[:N] - x_detected[:N])
        return nb_errors / N
    else:
        N = X_target.shape[axis]
        nb_errors = np.count_nonzero(X_target - X_detected, axis=axis)
        return nb_errors / N


def compute_ber(X_target, X_detected, width, axis=None):
    r"""
    Compute the Bit Error Rate (BER) between target and detected symbols.

    Signal Model
    ------------
    Both symbol streams are first expanded into bits by :func:`sym_2_bin`,
    which writes each symbol on :math:`w` bits (natural binary, MSB first).
    The BER is the normalized count of differing bits:

    .. math::

        \mathrm{BER} = \frac{N_e}{N_b}, \qquad
        N_e = \sum_{m=0}^{N_b-1} \mathbb{1}\left\{b[m] \neq \hat{b}[m]\right\},
        \qquad N_b = N w

    where :math:`b[m]` and :math:`\hat{b}[m]` are the transmitted and
    detected bits and :math:`N` is the number of symbols. Because the bits
    come from the natural binary code, this BER equals the true one only if
    the constellation is labelled the same way; with a Gray-mapped
    constellation the bit labelling must match ``sym_2_bin`` for the count
    to be meaningful.

    Axes: *element-wise* -- with ``axis=None`` (default) both arrays are
    raveled and the errors are counted over the whole bit stream.

    Parameters
    ----------
    X_target : ndarray
        Reference symbol indices :math:`x[n]`. Should be convertible to a 1-D array.
    X_detected : ndarray
        Detected symbol indices :math:`\hat{x}[n]` obtained from a
        transmission system or a decoder. Should be convertible to a 1-D array.
    width : int
        Number of bits per symbol :math:`w = \log_2 M`.
    axis : int or None, optional
        Axis along which to count the errors. Default is None. Note that
        :func:`sym_2_bin` only accepts a 1-D input, so values other than
        None are not currently usable.

    Returns
    -------
    float or ndarray
        The Bit Error Rate :math:`N_e / N_b`.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.3 (bit error probability).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])          # bits 00 01 10 11
    >>> x_hat = np.array([0, 1, 2, 0])      # bits 00 01 10 00 -> 2 wrong bits out of 8
    >>> print(compute_ber(x, x_hat, width=2))
    0.25
    """
    if axis is None:
        s_target = sym_2_bin(np.ravel(X_target), width)
        s_detected = sym_2_bin(np.ravel(X_detected), width)
        nb_errors = np.count_nonzero(s_target - s_detected)
        return nb_errors / len(s_detected)
    else:
        s_target = sym_2_bin(X_target, width)
        s_detected = sym_2_bin(X_detected, width)
        nb_errors = np.count_nonzero(s_target - s_detected, axis=axis)
        total_bits = s_target.shape[axis]
        return nb_errors / total_bits


def compute_evm(X_target, X_estimated, axis=None):
    r"""
    Compute the Error Vector Magnitude (EVM) between the target and estimated signals.

    Signal Model
    ------------
    The error vector is the difference between the estimated (measured)
    symbol :math:`\hat{x}[n]` and the ideal reference symbol :math:`x[n]`.
    The EVM returned here is the RMS error **normalized by the average
    power of the reference constellation**:

    .. math::

        \mathrm{EVM} = \sqrt{
            \frac{\frac{1}{N} \sum_{n=0}^{N-1} \left|x[n] - \hat{x}[n]\right|^2}
                 {\frac{1}{N} \sum_{n=0}^{N-1} \left|x[n]\right|^2}}

    This is the average-power normalization used by the cellular and WLAN
    standards, *not* the peak-amplitude (outermost constellation point)
    normalization also found in the literature: a given impairment yields a
    larger figure here than under the peak convention. The result is a
    linear ratio (multiply by 100 for a percentage, or take
    :math:`20 \log_{10}(\mathrm{EVM})` for dB).

    Axes: *declared axis* -- both mean powers are reduced along ``axis``
    (default ``None``: over the whole array).

    Parameters
    ----------
    X_target : ndarray
        Reference (ideal) constellation symbols :math:`x[n]`.
    X_estimated : ndarray
        Estimated (measured) symbols :math:`\hat{x}[n]`.
    axis : int or None, optional
        Axis along which the mean powers are computed. If None, the means
        are computed over all dimensions. Default is None.

    Returns
    -------
    float or ndarray
        The computed EVM value. If axis is None, returns a float. Otherwise, returns an array.

    References
    ----------
    3GPP TS 38.104, *NR; Base Station (BS) radio transmission and
    reception* -- error vector magnitude, normalized by the average power
    of the reference signal.

    Examples
    --------
    >>> X_target = np.array([1, 2, 3, 4])
    >>> X_estimated = np.array([1.1, 2.1, 3.1, 3.9])
    >>> print(float(compute_evm(X_target, X_estimated)))
    0.03651483716701111
    >>> print(round(float(compute_evm(X_target, X_estimated, axis=0)), 6))
    0.036515
    """
    # Compute the numerator and denominator
    error_vector = np.abs(X_target - X_estimated)**2
    target_power = np.abs(X_target)**2

    # Compute the mean along the specified axis
    if axis is None:
        num = np.mean(error_vector)
        den = np.mean(target_power)
    else:
        num = np.mean(error_vector, axis=axis)
        den = np.mean(target_power, axis=axis)

    return np.sqrt(num / den)



def compute_effective_snr(X_target, X_estimated, sigma2_s=1, unit="natural"):
    r"""
    Compute the effective Signal-to-Noise Ratio (SNR) between a target and an estimated signal.

    Signal Model
    ------------
    All the impairments accumulated along the link (noise, residual
    distortion, nonlinear interference) are lumped into an equivalent
    additive term of variance

    .. math::

        \sigma^2 = \frac{1}{N} \sum_{n=0}^{N-1}
                   \left|x[n] - \hat{x}[n]\right|^2

    where :math:`x[n]` is the reference symbol and :math:`\hat{x}[n]` the
    estimated one. The effective SNR is then

    .. math::

        \mathrm{SNR}_{\mathrm{eff}} = \frac{\sigma_s^2}{\sigma^2}

    with :math:`\sigma_s^2` the *nominal* signal power, supplied by the
    caller (``sigma2_s``, default 1 for a unit-power constellation) and
    **not** measured on ``X_target``. This is the reciprocal of the squared
    EVM when :math:`\sigma_s^2` equals the reference power:
    :math:`\mathrm{SNR}_{\mathrm{eff}} = \mathrm{EVM}^{-2}`.

    Axes: *element-wise* -- both arrays are raveled and the error power is
    reduced over all elements.

    Parameters
    ----------
    X_target : ndarray
        Reference signal :math:`x[n]`.
    X_estimated : ndarray
        Estimated signal :math:`\hat{x}[n]`.
    sigma2_s : float, optional
        Nominal signal power :math:`\sigma_s^2`. The default is 1.
    unit : {"natural", "dB", "dBm"}, optional
        Unit of the output SNR: linear scale, decibels, or
        decibel-milliwatts. The default is "natural".

    Returns
    -------
    float
        The effective SNR in the specified unit.

    Raises
    ------
    ValueError
        If the specified unit is not one of "natural", "dB", or "dBm".

    Notes
    -----
    - For ``"dB"``, the SNR is converted using :math:`\mathrm{SNR}_{dB} = 10 \log_{10}(\mathrm{SNR})`.
    - For ``"dBm"``, the SNR is converted using :math:`\mathrm{SNR}_{dBm} = 10 \log_{10}(\mathrm{SNR}) + 30`.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 4.2 (signal-to-noise ratio of a
    memoryless additive-noise link).

    Examples
    --------
    >>> x = np.ones(4)
    >>> x_hat = np.array([1.1, 0.9, 1.1, 0.9])   # error power 0.01
    >>> print(round(float(compute_effective_snr(x, x_hat)), 3))
    100.0
    >>> print(round(float(compute_effective_snr(x, x_hat, unit="dB")), 3))
    20.0
    """
    x_target = np.ravel(X_target)
    x_estimated = np.ravel(X_estimated)
    sigma2_b = np.mean(np.abs(x_target - x_estimated)**2)

    SNR = sigma2_s / sigma2_b

    match unit:
        case "natural":
            output = SNR
        case "dB":
            output = 10 * np.log10(SNR)
        case "dBm":
            output = 10 * np.log10(SNR) + 30
        case _:
            raise ValueError(f"Unknown unit: {unit}")

    return output


def compute_power(x, unit="natural"):
    r"""
    Compute the mean power of an input array.

    Signal Model
    ------------
    The mean power of the signal :math:`x[n]` of length :math:`N` is the
    time average of its squared modulus:

    .. math::

        P_x = \frac{1}{N} \sum_{n=0}^{N-1} \left|x[n]\right|^2

    expressed in the requested unit:

    .. math::

        P_{dB} = 10 \log_{10} \left(P_x\right), \qquad
        P_{dBm} = 10 \log_{10} \left(P_x\right) + 30

    The dBm conversion assumes :math:`P_x` is expressed in watts.

    Axes: *element-wise* -- the average is reduced over all elements of
    the array.

    Parameters
    ----------
    x : ndarray
        Input signal :math:`x[n]`.
    unit : {"natural", "dB", "dBm"}, optional
        Unit of the output power: watts, decibels, or decibel-milliwatts.
        The default is "natural".

    Returns
    -------
    float
        Mean power :math:`P_x` in the specified unit.

    Raises
    ------
    ValueError
        If the specified unit is not one of "natural", "dB", or "dBm".

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 2.2 (energy and power of a signal).

    Examples
    --------
    >>> x = np.array([1 + 1j, 1 - 1j])
    >>> print(round(float(compute_power(x)), 6))
    2.0
    >>> print(round(float(compute_power(x, unit="dB")), 4))
    3.0103
    """
    Px = np.mean(np.abs(x)**2)

    match unit:
        case "natural":
            output = Px
        case "dB":
            output = 10 * np.log10(Px)
        case "dBm":
            output = 10 * np.log10(Px) + 30
        case _:
            raise ValueError(f"Unknown unit: {unit}")

    return output


def compute_ccdf(data, axis=-1):
    r"""
    Compute the empirical Complementary Cumulative Distribution Function (CCDF) of a dataset.

    Signal Model
    ------------
    The CCDF of a random variable :math:`Z` is the probability of
    exceeding a threshold:

    .. math::

        \mathrm{CCDF}(z) = \Pr\left\{Z > z\right\} = 1 - F_Z(z)

    Given :math:`N` samples sorted in ascending order
    :math:`z_{(1)} \le \dots \le z_{(N)}`, the empirical estimate returned
    for the :math:`i`-th sorted value is

    .. math::

        \widehat{\mathrm{CCDF}}\left(z_{(i)}\right) = 1 - \frac{i}{N},
        \qquad i = 1, \dots, N

    i.e. the fraction of samples strictly above :math:`z_{(i)}` (the last
    entry is therefore exactly 0). This is the standard way of plotting
    the PAPR distribution of an OFDM signal: feed
    :func:`comnumpy.ofdm.metrics.compute_papr` values in, then plot
    ``ccdf`` against ``sorted_data`` on a semilog-y axis.

    Axes: *declared axis* -- sorting and the sample count :math:`N` use
    ``axis`` (default ``-1``).

    Parameters
    ----------
    data : array-like
        Input samples :math:`z` whose CCDF is estimated.
    axis : int, optional
        Axis along which to compute the CCDF. Default is the last axis.

    Returns
    -------
    sorted_data : np.ndarray
        The input data sorted in ascending order along the specified axis.
    ccdf : np.ndarray
        The CCDF values :math:`1 - i/N` matching the sorted data.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 6 (PAPR distribution).

    Examples
    --------
    >>> data = np.array([3.0, 1.0, 2.0, 4.0])
    >>> sorted_data, ccdf = compute_ccdf(data)
    >>> print(sorted_data)
    [1. 2. 3. 4.]
    >>> print(ccdf)
    [0.75 0.5  0.25 0.  ]
    """
    # Sort the data values in ascending order along the specified axis
    sorted_data = np.sort(data, axis=axis)

    # Compute the empirical CCDF along the specified axis
    n = data.shape[axis]
    indices = np.arange(1, n + 1)
    ccdf = 1.0 - indices / n

    # broadcast against sorted_data: n on `axis`, 1 elsewhere (annex A.5:
    # the previous expand_dims produced a shape that did not broadcast)
    shape = [1] * np.ndim(data)
    shape[axis] = n
    ccdf = ccdf.reshape(shape)

    return sorted_data, ccdf


def compute_acpr(signal, bandwidth, sampling_rate):
    r"""
    Calculate the Adjacent Channel Power Ratio (ACPR) of a given signal.

    Signal Model
    ------------
    ACPR measures spectral regrowth: the power a nonlinear transmitter
    leaks into the neighbouring channels, relative to the power it emits
    in its own channel. With :math:`X(f)` the DFT of the baseband signal
    :math:`x[n]` sampled at :math:`f_s`, the power in a band
    :math:`\mathcal{B}` is estimated by Parseval's relation as

    .. math::

        P(\mathcal{B}) = \frac{1}{N} \sum_{f_i \in \mathcal{B}}
                         \left|X(f_i)\right|^2

    The three bands are deduced from the channel bandwidth :math:`B`, the
    main channel being centred on DC:

    .. math::

        \mathcal{B}_{\mathrm{main}} = \left[-\tfrac{B}{2}, \tfrac{B}{2}\right],
        \quad
        \mathcal{B}_{\mathrm{right}} = \left[\tfrac{B}{2}, \tfrac{3B}{2}\right],
        \quad
        \mathcal{B}_{\mathrm{left}} = \left[-\tfrac{3B}{2}, -\tfrac{B}{2}\right]

    and each ratio is reported in decibels:

    .. math::

        \mathrm{ACPR}_{dB} = 10 \log_{10}
        \frac{P(\mathcal{B}_{\mathrm{adj}})}{P(\mathcal{B}_{\mathrm{main}})}

    A clean transmitter therefore gives a large *negative* value; the less
    negative the ACPR, the more spectral regrowth. The sampling rate must
    satisfy :math:`f_s \ge 3B` for the two adjacent bands to lie inside the
    represented spectrum.

    Axes: *element-wise* -- the DFT and the band powers are computed over
    the whole 1-D input; no axis argument.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain baseband signal :math:`x[n]`.
    bandwidth : float
        Bandwidth :math:`B` of the main channel, in Hz.
    sampling_rate : float
        Sampling rate :math:`f_s` of the signal, in Hz.

    Returns
    -------
    acpr_right : float
        ACPR (in dB) of the upper adjacent channel.
    acpr_left : float
        ACPR (in dB) of the lower adjacent channel.

    References
    ----------
    S. C. Cripps, *RF Power Amplifiers for Wireless Communications*,
    2nd ed., Artech House, 2006, Chapter 9 (spectral regrowth and
    adjacent channel power).

    Examples
    --------
    >>> f_s, B = 100.0, 20.0
    >>> n = np.arange(1000)
    >>> x = np.cos(2 * np.pi * 4 * n / f_s)     # single tone inside the channel
    >>> y = x - 0.2 * x ** 3                    # cubic nonlinearity -> IM3 at 12 Hz
    >>> print(np.round(compute_acpr(y, B, f_s), 4))
    [-27.6193 -27.6193]
    """

    # NOT TESTED

    def calculate_power(signal, lower_freq, upper_freq, sampling_rate):

        # Perform FFT
        fft_signal = np.fft.fft(signal)
        freq_axis = np.fft.fftfreq(len(signal), 1 / sampling_rate)

        # Select frequencies within the band
        mask = (freq_axis >= lower_freq) & (freq_axis <= upper_freq)
        band_power = np.sum(np.abs(fft_signal[mask])**2) / len(signal)

        return band_power

    # Define main and adjacent channel frequency bands
    half_bw = bandwidth / 2
    main_lower = -half_bw
    main_upper = half_bw

    adj_lower_right = main_upper
    adj_upper_right = main_upper + bandwidth

    adj_lower_left = main_lower - bandwidth
    adj_upper_left = main_lower

    # Calculate power in main and adjacent channels
    main_power = calculate_power(signal, main_lower, main_upper, sampling_rate)
    adj_power_right = calculate_power(signal, adj_lower_right, adj_upper_right, sampling_rate)
    adj_power_left = calculate_power(signal, adj_lower_left, adj_upper_left, sampling_rate)

    # Calculate ACPR in dB
    acpr_right = 10 * np.log10(adj_power_right / main_power)
    acpr_left = 10 * np.log10(adj_power_left / main_power)

    return acpr_right, acpr_left


def signal_report(x, compute_papr: bool = False,
                  papr_unit: str = "dB") -> "dict[str, float]":
    r"""
    Summary statistics of a signal, as plain data.

    Replaces the former in-chain monitor blocks: extract the signal with
    ``Sequential(taps=...)``, compute the report, and let the caller
    decide how to present it (``logging``, table, assertion, ...).

    Signal Model
    ------------
    All statistics are computed on the modulus :math:`|x[n]|` over every
    element; the average power is

    .. math::

        \widehat{P} = \frac{1}{N} \sum_{n=0}^{N-1} \left|x[n]\right|^2

    Axes: *element-wise* -- statistics are reduced over all elements.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    compute_papr : bool, optional
        Also include the PAPR. Default is False.
    papr_unit : {"dB", "natural"}, optional
        Unit of the PAPR entry. Default is "dB".

    Returns
    -------
    dict of str to float
        Keys: ``min``, ``max``, ``mean``, ``std``, ``rms``, ``energy``,
        ``avg_power`` and optionally ``papr``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Section 2.2 (signal energy and power).

    Examples
    --------
    >>> report = signal_report(np.array([1.0+0.0j, 0.0+1.0j]))
    >>> print(report["avg_power"], report["rms"])
    1.0 1.0
    """
    abs_x = np.abs(np.asarray(x))
    report = {
        "min": float(np.min(abs_x)),
        "max": float(np.max(abs_x)),
        "mean": float(np.mean(abs_x)),
        "std": float(np.std(abs_x)),
        "rms": float(np.sqrt(np.mean(abs_x ** 2))),
        "energy": float(np.sum(abs_x ** 2)),
        "avg_power": float(np.mean(abs_x ** 2)),
    }
    if compute_papr:
        # local import to avoid a circular dependency (ofdm imports core);
        # aliased because the boolean flag above already owns the name
        from comnumpy.ofdm.metrics import compute_papr as papr_of
        report["papr"] = float(papr_of(abs_x, unit=papr_unit))
    return report
