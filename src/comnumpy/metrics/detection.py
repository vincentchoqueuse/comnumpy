import numpy as np

def sym_2_bin(sym, width):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice],width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)



def compute_ser(X_target, X_detected):
    """
    Compute the Symbol Error Rate (SER) between the target and detected signals.

    Parameters
    ----------
    X_target : ndarray
        The target signal array.
    X_detected : ndarray
        The detected signal array after some transmission or processing.

    Returns
    -------
    float
        The computed SER value.

    Notes
    -----
    The SER is computed using the formula:

    .. math::

        \\text{SER} = \\frac{\|\mathbf{x}_{\\text{tar}} - \mathbf{x}_{\\text{est}}\|_0}{N}

    Both input arrays are first raveled into 1D arrays before computation.

    """
    x_target = np.ravel(X_target)
    x_detected = np.ravel(X_detected)
    N_1 = len(x_detected)
    N_2 = len(x_target)
    N = min(N_1, N_2)
    nb_errors = np.count_nonzero(x_target[:N]-x_detected[:N])
    return nb_errors / N


def compute_ber(X_target, X_detected, width):
    """
    Compute the Bit Error Rate (BER) between target and detected symbols.

    The function first converts the symbols to binary representations using the 
    `sym_2_bin` function. It then calculates the BER as the ratio of the number of 
    bit errors to the total number of bits.

    Parameters
    ----------
    X_target : ndarray
        The target symbols to be compared against. Should be convertible to a 1-D array.
    X_detected : ndarray
        The detected symbols obtained from a transmission system or a decoder.
        Should be convertible to a 1-D array.
    width : int
        The number of bits used to represent each symbol.

    Returns
    -------
    float
        The Bit Error Rate (BER), computed as the number of differing bits 
        between `s_target` and `s_detected` divided by the total number of bits.

    Notes
    -----
    The function assumes that `sym_2_bin` is a predefined function that converts
    symbols to their binary representation.

    The formula for BER is:

    .. math::
        \text{BER} = \frac{\text{Number of differing bits}}{\text{Total number of bits}}

    """
    s_target = sym_2_bin(np.ravel(X_target), width)
    s_detected = sym_2_bin(np.ravel(X_detected), width)
    nb_errors = np.count_nonzero(s_target-s_detected)
    return nb_errors / len(s_detected)
