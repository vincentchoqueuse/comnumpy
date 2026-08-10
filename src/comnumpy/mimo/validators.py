import numpy as np


from comnumpy.exceptions import ShapeError

__all__ = ["validate_input"]


def validate_input(X: np.ndarray, N_t: int) -> None:
    """
    Validate that the antenna axis of the input matches the expected number of transmit antennas.

    MIMO blocks are *declared axis* blocks (see CONVENTIONS.md): they expect
    the layout ``(..., ant, N)`` with the antenna axis on -2.

    Parameters
    ----------
    X : np.ndarray
        Input signal array, expected layout ``(..., ant, N)``.
    N_t : int
        Expected number of transmit antennas.

    Raises
    ------
    ShapeError
        If the antenna axis of ``X`` does not match ``N_t``.
    """
    if X.ndim < 2 or X.shape[-2] != N_t:
        raise ShapeError(
            f"expected shape (..., ant={N_t}, N), got {X.shape} -- "
            "put the antenna axis on -2 (see CONVENTIONS.md) or fix the "
            "channel matrix size."
        )
