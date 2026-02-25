import numpy as np


def validate_real(X, tol=1e-12):
    """
    Validate that the input array is purely real.

    Parameters
    ----------
    X : np.ndarray
        Input array to validate.
    tol : float, optional
        Tolerance for the imaginary part. Default is 1e-12.

    Raises
    ------
    ValueError
        If the imaginary part of any element exceeds ``tol``.
    """
    if not np.isclose(np.ravel(X).imag, 0, atol=tol).all():
        raise ValueError("the input data is not real since the imag part is non zero ")


def validate_data(data):
    """
    Validate that data is a numpy array or an object with a ``get_data`` method.

    Parameters
    ----------
    data : np.ndarray or object
        Data to validate.

    Raises
    ------
    TypeError
        If ``data`` is neither a numpy array nor has a ``get_data`` method.
    """
    if not (isinstance(data, np.ndarray) or hasattr(data, 'get_data')):
        raise TypeError("target_data must be a numpy array or an object with a get_data method.")