import numpy as np


def validate_real(X: object, tol: float = 1e-12) -> None:
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
    if not np.isclose(np.imag(np.ravel(np.asarray(X))), 0, atol=tol).all():
        raise ValueError("the input data is not real since the imag part is non zero ")


def validate_data(data: object) -> None:
    """
    Validate that data is array-like (convertible to a numeric numpy array).

    Reference signals are plain arrays: extract them with
    ``Sequential(taps=...)`` before configuring a data-aided block.

    Parameters
    ----------
    data : np.ndarray
        Data to validate.

    Raises
    ------
    TypeError
        If ``data`` cannot be converted to a numeric numpy array.
    """
    try:
        arr = np.asarray(data)
    except Exception as exc:
        raise TypeError(f"reference must be array-like, got {type(data)!r}") from exc
    if arr.dtype == object:
        raise TypeError(
            f"reference must be a numeric array, got dtype=object from {type(data)!r} "
            "-- extract the reference signal with Sequential(taps=...) first.")
