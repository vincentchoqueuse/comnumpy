
def validate_input(X, N_t):
    if X.shape[0] != N_t:
        X_shape = X.shape[0]
        raise ValueError(f"Dimension does not match (signal shape={X_shape}, expected={N_t})")
