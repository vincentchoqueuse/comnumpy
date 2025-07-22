import numpy as np


def validate_real(X, tol=1e-12):
    if not np.isclose(np.ravel(X).imag, 0, atol=tol).all():
        raise ValueError("the input data is not real since the imag part is non zero ")


def validate_data(data):
    """
    Check if X is a numpy array or an object with get_data method
    """
    if not (isinstance(data, np.ndarray) or hasattr(data, 'get_data')):
        raise TypeError("target_data must be a numpy array or an object with a get_data method.")