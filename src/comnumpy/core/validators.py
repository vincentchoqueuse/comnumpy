import numpy as np

__all__ = ["validate_real", "validate_data", "validate_single_path"]


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


def validate_single_path(X: object, block: str, estimand: str) -> None:
    r"""Refuse a multi-path signal in an estimator that measures one record.

    Decision D49. Once a chain carries several paths on the antenna axis
    -- the two polarizations of a coherent link, the branches of a MIMO
    receiver -- every estimator has to answer a question it could ignore
    while signals were one-dimensional: **is what it measures one
    physical quantity shared by all the paths, or one per path?**

    * *Shared*: a carrier frequency offset comes from one laser beating
      against one local oscillator, so there is a single number and it
      should be estimated **jointly**, over all the paths at once.
      Estimating it per path throws away half the data and produces two
      numbers that then have to be reconciled.
    * *Per path*: an IQ imbalance belongs to a receiver, a DC offset to
      an analog-to-digital converter, a residual rotation to whichever
      output of a butterfly equalizer produced it. One estimate each.

    Letting numpy broadcast decides the question by accident, and it
    decides it wrong in both directions: it silently pools what should
    be separate, or crashes with a shape error that says nothing. This
    validator is for the blocks that have not been generalized: it
    states which of the two they are and what to do instead.

    Parameters
    ----------
    X : np.ndarray
        Signal about to be processed.
    block : str
        Name of the calling block, for the message.
    estimand : str
        What the block measures, e.g. ``"a delay"``, in a form that
        reads after "estimates".

    Raises
    ------
    ShapeError
        If ``X`` has more than one path on its antenna axis.

    Examples
    --------
    >>> import numpy as np
    >>> validate_single_path(np.zeros(8), "Block", "a delay")   # fine
    >>> try:
    ...     validate_single_path(np.zeros((2, 8)), "Block", "a delay")
    ... except Exception as error:
    ...     print(type(error).__name__)
    ShapeError
    """
    from comnumpy.exceptions import ShapeError      # local: avoids a cycle

    shape = getattr(X, "shape", ())
    if len(shape) < 2 or shape[-2] == 1:
        return
    raise ShapeError(
        f"{block} estimates {estimand} over one record, and got a signal "
        f"with {shape[-2]} paths on its antenna axis (shape {shape}). "
        f"Decide which it is: if the quantity is shared by the paths -- one "
        f"laser, one local oscillator -- estimate it jointly on the flattened "
        f"signal and apply the result to all of them; if it belongs to each "
        f"path -- one converter, one receiver, one equalizer output -- apply "
        f"one instance of this block per path. Letting it broadcast would "
        f"pick one of the two by accident (decision D49).")
