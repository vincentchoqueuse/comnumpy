
def validate_input(X, N_t):
    """
    Validate that the first dimension of the input matches the expected number of transmit antennas.

    Parameters
    ----------
    X : np.ndarray
        Input signal array.
    N_t : int
        Expected number of transmit antennas (first dimension of X).

    Raises
    ------
    ValueError
        If ``X.shape[0]`` does not match ``N_t``.
    """
    if X.shape[0] != N_t:
        X_shape = X.shape[0]
        raise ValueError(f"Dimension does not match (signal shape={X_shape}, expected={N_t})")
