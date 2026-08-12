import logging

import numpy as np

__all__ = ["compute_papr", "compute_papr_ccdf_theo"]

logger = logging.getLogger(__name__)

# Oversampled samples are correlated, so the PAPR of an oversampled
# waveform is not that of alpha*N independent ones -- but the same
# expression fits the measurement with an *effective* count, and 2.8 is
# the value van Nee and Prasad report for an oversampling of 4 or more.
_ALPHA_OVERSAMPLED = 2.8


def compute_papr(x_data: object, unit: str = "natural",
                 axis: "int | None" = None,
                 reduction: str = "none") -> "np.ndarray | float":
    r"""
    Compute the Peak-to-Average Power Ratio (PAPR) of the input signal.

    Signal Model
    ------------
    With the peak amplitude :math:`x_{max} = \max_n |x[n]|` and the mean
    power :math:`P_x = \mathbb{E}\left[|x[n]|^2\right]` of the input signal
    :math:`x[n]`, the returned metric is:

    * for natural units (amplitude ratio):

    .. math::

        \mathrm{PAPR} = \frac{x_{max}}{\sqrt{P_x}}

    * for dB units (power ratio):

    .. math::

        \mathrm{PAPR_{dB}} = 10 \log_{10} \left( \frac{x_{max}^2}{P_x} \right)

    Axes: *declared axis* -- the peak :math:`x_{max}` and the mean power
    :math:`P_x` are reduced along ``axis`` (default ``None``: over the
    whole array). ``axis`` names the axis *one waveform* lies along, so
    an array of OFDM symbols ``(T, F)`` gives ``T`` values with
    ``axis=-1``; ``reduction`` then says what to do with those, which
    saves the caller a second call to reduce them by hand.

    Parameters
    ----------
    x_data : np.ndarray
        Input signal :math:`x[n]` for which the PAPR needs to be calculated.
    unit : {"natural", "dB"}, optional
        The unit for PAPR calculation. It can be either "natural" for natural units or
        "dB" for logarithmic units. Default is "natural".
    axis : int or None, optional
        The axis one waveform lies along. If None, the PAPR is computed
        over the entire array. Default is None.
    reduction : {"none", "mean", "max", "min"}, optional
        What to do with the several PAPR values an array of waveforms
        produces. ``"none"`` (default) returns them all; the others
        reduce them to a scalar, **in the unit asked for** -- the mean of
        a set of decibel values is not the decibel of their mean, and
        this returns the first.

    Returns
    -------
    float or np.ndarray
        The computed PAPR value(s). With ``axis`` given and
        ``reduction="none"``, one value per waveform.

    Raises
    ------
    TypeError
        If `x_data` is not a NumPy array.
    ValueError
        If `reduction` is not one of the values above.
    NotImplementedError
        If the specified unit is not supported.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 6.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4])
    >>> print(round(float(compute_papr(data, unit="natural")), 4))
    1.4606
    >>> print(round(float(compute_papr(data, unit="dB")), 4))
    3.2906
    >>> data_2d = np.array([[1, 2], [3, 4]])
    >>> print(np.round(compute_papr(data_2d, unit="natural", axis=0), 4))
    [1.3416 1.2649]

    One value per OFDM symbol, or their average, without reshaping
    anything:

    >>> blocks = np.array([[1, 2, 3, 4], [1, 1, 1, 4]])
    >>> print(np.round(compute_papr(blocks, unit="dB", axis=-1), 3))
    [3.291 5.274]
    >>> print(round(float(compute_papr(blocks, unit="dB", axis=-1,
    ...                                reduction="mean")), 3))
    4.282
    """
    if not isinstance(x_data, np.ndarray):
        raise TypeError("Input x_data must be a NumPy array.")
    if reduction not in ("none", "mean", "max", "min"):
        raise ValueError(
            f"reduction is 'none', 'mean', 'max' or 'min', got {reduction!r}.")

    x_abs_max = np.max(np.abs(x_data), axis=axis)
    P_moy = np.mean(np.abs(x_data)**2, axis=axis)

    if unit == "natural":
        papr = x_abs_max / np.sqrt(P_moy)
    elif unit == "dB":
        papr = 10 * np.log10(x_abs_max**2 / P_moy)
    else:
        raise NotImplementedError(f"PAPR with unit '{unit}' is not currently implemented")

    if reduction != "none":
        papr = getattr(np, reduction)(papr)
    return papr


def compute_papr_ccdf_theo(threshold: "np.ndarray | float", n_sub: int,
                           oversampling: int = 1,
                           unit: str = "natural",
                           method: str = "effective") -> "np.ndarray | float":
    r"""
    Probability that the PAPR of one OFDM symbol exceeds a threshold.

    Signal Model
    ------------
    After the IDFT each time sample is a sum of :math:`N_{sub}`
    independent terms, so for :math:`N_{sub}` large the central limit
    theorem makes it a circular complex Gaussian and its normalized power
    :math:`\gamma` exponential. If the :math:`N` samples of a symbol were
    independent, the peak would exceed :math:`\gamma` unless all of them
    stayed below it:

    .. math::

        \mathrm{CCDF}(\gamma) = \Pr\left\{\mathrm{PAPR} > \gamma\right\}
        = 1 - \left(1 - e^{-\gamma}\right)^{\alpha N_{sub}}

    Oversampled samples are *not* independent -- the waveform is band
    limited, so neighbouring samples are correlated -- and there are two
    usual ways to account for it.

    ``method="effective"`` keeps the expression and replaces the count by
    an **effective** one :math:`\alpha N_{sub}`, with :math:`\alpha = 1`
    at the Nyquist rate and :math:`\alpha = 2.8` for an oversampled
    waveform, the value van Nee and Prasad report for an oversampling of
    4 or more.

    ``method="level_crossing"`` replaces the fitted constant by a term
    derived from how often a Gaussian process crosses a level, which
    makes the effective count grow with the threshold:

    .. math::

        \mathrm{CCDF}(\gamma) \simeq 1 - \exp\left(-N_{sub}
        \sqrt{\frac{\pi}{3}}\, \sqrt{\gamma}\, e^{-\gamma}\right)

    That form has no fitted constant, but it comes with two conditions.
    It describes the peak of the **continuous-time** waveform, which any
    finite oversampling underestimates, so it reads high on sampled data
    and approaches the measurement as the oversampling grows. And it is
    a *large-threshold* approximation: near the top of the curve, where
    the probability approaches one, it is not even ordered against the
    other model. ``oversampling`` is not used by it.

    Axes: *element-wise* -- the result has the shape of ``threshold``.

    Parameters
    ----------
    threshold : np.ndarray or float
        PAPR threshold :math:`\gamma`, in the unit given by ``unit``.
    n_sub : int
        Number :math:`N_{sub}` of modulated subcarriers.
    oversampling : int, optional
        Oversampling factor of the waveform the PAPR is measured on.
        Default 1, i.e. Nyquist-rate sampling.
    unit : {"natural", "dB"}, optional
        Unit of ``threshold``, with the convention of
        :func:`compute_papr`: ``"natural"`` is the amplitude ratio,
        ``"dB"`` the power ratio in decibels. Default ``"natural"``.
    method : {"effective", "level_crossing"}, optional
        Which of the two models above. Default ``"effective"``, the one
        that matches a sampled waveform; ``"level_crossing"`` is the
        continuous-time approximation and ignores ``oversampling``.

    Returns
    -------
    np.ndarray or float
        The probability of exceeding each threshold.

    Raises
    ------
    ValueError
        If ``n_sub`` or ``oversampling`` is not positive, or if ``unit``
        or ``method`` is not one of the values above.

    References
    ----------
    R. van Nee, R. Prasad, *OFDM for Wireless Multimedia Communications*,
    Artech House, 2000, Chapter 6 (the :math:`\alpha = 2.8` fit);
    H. Ochiai and H. Imai, "On the distribution of the peak-to-average
    power ratio in OFDM signals", IEEE Trans. Commun. 49(2), 2001 (the
    level-crossing form); S. H. Han and J. H. Lee, "An overview of
    peak-to-average power ratio reduction techniques for multicarrier
    transmission", IEEE Wireless Communications 12(2), 2005.

    Examples
    --------
    >>> import numpy as np
    >>> print(round(float(compute_papr_ccdf_theo(8.0, 256, unit="dB")), 4))
    0.3725
    >>> print(np.round(compute_papr_ccdf_theo(
    ...     np.array([8.0, 11.0]), 256, oversampling=4, unit="dB"), 4))
    [0.7288 0.0024]

    The continuous-time form reads higher, as it should:

    >>> print(np.round(compute_papr_ccdf_theo(
    ...     11.0, 256, unit="dB", method="level_crossing"), 4))
    0.0032
    """
    if n_sub <= 0:
        raise ValueError(f"n_sub must be positive, got {n_sub}")
    if oversampling <= 0:
        raise ValueError(f"oversampling must be positive, got {oversampling}")
    if method not in ("effective", "level_crossing"):
        raise ValueError(
            f"method is 'effective' or 'level_crossing', got {method!r}")
    if unit == "natural":
        gamma = np.asarray(threshold, dtype=float) ** 2
    elif unit == "dB":
        gamma = 10 ** (np.asarray(threshold, dtype=float) / 10)
    else:
        raise ValueError(f"unit is 'natural' or 'dB', got {unit!r}")

    if method == "level_crossing":
        return 1 - np.exp(-n_sub * np.sqrt(np.pi / 3)
                          * np.sqrt(gamma) * np.exp(-gamma))

    if oversampling == 1:
        alpha = 1.0
    else:
        alpha = _ALPHA_OVERSAMPLED
        if oversampling < 4:
            # the fit is reported for an oversampling of 4 or more; below
            # that the samples are less correlated than it assumes, so
            # say so (D11) rather than return a number that looks exact
            logger.warning(
                "the effective-sample fit alpha = %.1f is reported for an "
                "oversampling of 4 or more, and this call uses %d: the "
                "returned CCDF is an extrapolation.",
                _ALPHA_OVERSAMPLED, oversampling)

    return 1 - (1 - np.exp(-gamma)) ** (alpha * n_sub)
