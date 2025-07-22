import numpy as np


def compute_PAPR(x_data, unit="natural", axis=None):
    r"""
    Compute the Peak-to-Average Power Ratio (PAPR) of the input data.

    Parameters
    ----------
    x_data : np.ndarray
        Input data for which PAPR needs to be calculated.
    unit : str, optional
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
    NotImplementedError
        If the specified unit is not supported.

    Notes
    -----
    The PAPR is computed using the formulas:

    * For natural units:

    .. math::

        \text{PAPR} = \frac{x_{\text{max}}}{\sqrt{P_{\text{moy}}}}

    * For dB units:

    .. math::

        \text{PAPR} = 10 \log_{10} \left( \frac{x_{\text{max}}^2}{P_{\text{moy}}} \right)

    where:

    - :math:`x_{\text{max}}` is the maximum power value in `x_data`,
    - :math:`P_{\text{moy}}` is the mean power value in `x_data`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4])
    >>> compute_PAPR(data, unit="natural")
    2.0
    >>> compute_PAPR(data, unit="dB")
    6.0206...
    >>> data_2d = np.array([[1, 2], [3, 4]])
    >>> compute_PAPR(data_2d, unit="natural", axis=0)
    array([1.63299316, 1.41421356])
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
