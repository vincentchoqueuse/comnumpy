import numpy as np

__all__ = ["compute_papr"]


def compute_papr(x_data: object, unit: str = "natural", axis: "int | None" = None) -> "np.ndarray | float":
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
    whole array).

    Parameters
    ----------
    x_data : np.ndarray
        Input signal :math:`x[n]` for which the PAPR needs to be calculated.
    unit : {"natural", "dB"}, optional
        The unit for PAPR calculation. It can be either "natural" for natural units or
        "dB" for logarithmic units. Default is "natural".
    axis : int or None, optional
        The axis along which to compute the PAPR. If None, the PAPR is computed over the
        entire array. Default is None.

    Returns
    -------
    float or np.ndarray
        The computed PAPR value(s). If axis is specified, returns an array of PAPR values.

    Raises
    ------
    TypeError
        If `x_data` is not a NumPy array.
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
    """
    if not isinstance(x_data, np.ndarray):
        raise TypeError("Input x_data must be a NumPy array.")

    x_abs_max = np.max(np.abs(x_data), axis=axis)
    P_moy = np.mean(np.abs(x_data)**2, axis=axis)

    if unit == "natural":
        papr = x_abs_max / np.sqrt(P_moy)
    elif unit == "dB":
        papr = 10 * np.log10(x_abs_max**2 / P_moy)
    else:
        raise NotImplementedError(f"PAPR with unit '{unit}' is not currently implemented")

    return papr
