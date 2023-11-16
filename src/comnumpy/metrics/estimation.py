import numpy as np


def compute_evm(X_target, X_estimated):
    """
    Compute the Error Vector Magnitude (EVM) between the target and estimated signals.

    Parameters
    ----------
    X_target : ndarray
        The target signal array.
    X_estimated : ndarray
        The estimated signal array.

    Returns
    -------
    float
        The computed EVM value.

    Notes
    -----
    The EVM is computed using the formula:

    .. math::

        \\text{EVM} = \\sqrt{\\frac{\\sum_{i} |x_{\\text{target},i} - x_{\\text{estimated},i}|^2}{\\sum_{i} |x_{\\text{target},i}|^2}}

    where:

    - \( x_{\\text{target},i} \) is the ith element of the target signal.
    - \( x_{\\text{estimated},i} \) is the ith element of the estimated signal.

    Both input arrays are first raveled into 1D arrays before computation.

    Examples
    --------
    >>> X_target = np.array([1, 2, 3, 4])
    >>> X_estimated = np.array([1.1, 2.1, 3.1, 3.9])
    >>> compute_evm(X_target, X_estimated)
    0.05057216690580728
    """
    x_target = np.ravel(X_target)
    x_estimated = np.ravel(X_estimated)

    num = np.sum(np.abs(x_target-x_estimated)**2)
    den = np.sum(np.abs(x_target)**2)
    return np.sqrt(num/den)

