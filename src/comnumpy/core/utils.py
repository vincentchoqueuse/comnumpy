import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import numpy.linalg as LA


def get_alphabet(modulation, order, type="gray", norm=True):
    """
    Retrieve the symbol alphabet for a given modulation scheme and order.

    This function loads the modulation alphabet from a predefined CSV file based on the specified modulation type, order, and symbol mapping (e.g., Gray coding).
    Optionally, it can normalize the alphabet to have unit average power.

    Parameters
    ----------
    modulation : str
        The type of modulation (e.g., ``"QAM"``, ``"PSK"``).
    order : int
        The order of modulation (e.g., 4, 16, 64 for QAM).
    type : str, optional
        The type of symbol mapping to be used (e.g., ``"gray"``). Default is ``"gray"``.
    norm : bool, optional
        If True, normalizes the alphabet to unit average power. Default is True.

    Returns
    -------
    np.ndarray
        The complex symbol alphabet for the specified modulation scheme.

    Notes
    -----
    The function reads from CSV files located in a ``data`` subdirectory relative to the module's directory.
    These files should be named following the pattern ``<modulation>_<order>_<type>.csv`` and contain symbol mappings as complex numbers.
    """
    # extract alphabet
    pathname = path.dirname(path.abspath(__file__))
    filename = "{}/data/{}_{}_{}.csv".format(pathname, modulation, order, type)
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    alphabet = data[:, 1] + 1j*data[:, 2]

    if norm:
        alphabet = alphabet/np.sqrt(np.mean(np.abs(alphabet)**2))

    return alphabet


def plot_alphabet(alphabet, num=None, label="alphabet", title="Constellation"):
    """
    Plot a constellation diagram of the given symbol alphabet.

    Parameters
    ----------
    alphabet : np.ndarray
        Complex-valued symbol alphabet to plot.
    num : int or None, optional
        Figure number. If None, a new figure is created.
    label : str, optional
        Label for the scatter plot. Default is ``"alphabet"``.
    title : str, optional
        Title of the plot. Default is ``"Constellation"``.
    """
    plt.figure(num)
    plt.plot(np.real(alphabet), np.imag(alphabet), "o", label=label)
    plt.xlabel("real part")
    plt.ylabel("imag part")
    plt.title(title)


def sym_2_bin(sym, width=4):
    """
    Convert an array of symbols to a binary representation.

    This function takes an array of symbols (as integers) and converts each symbol into its binary representation.
    The binary digits are then concatenated into a single string which is converted into an array of integers (0s and 1s).

    Parameters
    ----------
    sym : array-like
        Array of symbols to be converted. Each symbol should be an integer.
    width : int, optional
        The fixed width in bits for each symbol's binary representation. Default is 4.

    Returns
    -------
    np.ndarray
        An array of binary digits (0s and 1s) representing the input symbol array.

    Notes
    -----
    The function ensures that each symbol is represented by exactly ``width`` bits.
    If a symbol's binary representation is shorter than ``width``, it is left-padded with zeros.
    """

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice], width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)


def hard_projector(z, alphabet):
    """
    Project input symbols onto the nearest constellation point (hard decision).

    Parameters
    ----------
    z : np.ndarray
        Input symbols to be projected.
    alphabet : np.ndarray
        1-D array of constellation symbols.

    Returns
    -------
    s : np.ndarray
        Integer indices of the nearest constellation points.
    x : np.ndarray
        Nearest constellation symbols.
    """
    error = np.abs(z[..., np.newaxis] - alphabet)**2
    index = np.argmin(error, axis=-1)
    s = index.astype(int)
    x = alphabet[s]
    return s, x


def soft_projector(z: np.array, alphabet: np.array, sigma2: float, kernel: bool = None):
    r"""
    Compute the soft (MMSE) estimate of a symbol given a Gaussian noise model.

    The estimator computes a weighted mean of the constellation symbols,
    where the weights are proportional to the Gaussian likelihood.

    Parameters
    ----------
    z : np.ndarray
        Received symbols (1-D array).
    alphabet : np.ndarray
        1-D array of constellation symbols.
    sigma2 : float
        Noise variance.
    kernel : np.ndarray or None, optional
        Kernel values used in the numerator instead of ``alphabet``.
        If None, uses ``alphabet``.

    Returns
    -------
    np.ndarray
        Soft estimates for each input symbol.
    """
    alphabet = alphabet.reshape(1, -1)
    z = z.reshape(-1, 1)

    term1 = np.exp(-(1 / np.real(sigma2)) * np.abs(alphabet - z) ** 2)

    if kernel is None:
        kernel = alphabet

    num = np.sum(kernel * term1, axis=1)
    den = np.sum(term1, axis=1)
    return num / den


def compute_sigma2(value, input_unit, sigma2s=1):
    r"""
    Compute the noise variance :math:`\sigma^2` based on the input value and its unit.

    Parameters
    ----------
    value : float
        The input value representing the signal-to-noise ratio (SNR) or noise variance.
    input_unit : str
        The unit of the input value. Supported units are:

        - ``"sigma2"``: Natural noise variance :math:`\sigma^2`.
        - ``"snr"``: Natural SNR :math:`\sigma_s^2 / \sigma_n^2`.
        - ``"snr_dB"``: SNR in decibels (dB).
        - ``"snr_dBm"``: SNR in decibels-milliwatts (dBm).
    sigma2s : float, optional
        The signal variance :math:`\sigma_s^2`. Default is 1.

    Returns
    -------
    float
        The computed noise variance :math:`\sigma^2`.

    Raises
    ------
    ValueError
        If ``input_unit`` is not one of the supported units.

    Examples
    --------
    >>> compute_sigma2(10, "snr_dB")
    0.1
    >>> compute_sigma2(2, "sigma2")
    2
    """
    match input_unit:
        case "sigma2":
            output = value
        case "snr":
            # SNR : SNR = sigma_s^2 / sigma_n^2
            output = sigma2s / value
        case "snr_dB":
            # SNRdB : SNRdB = 10log10(sigma_s^2 / sigma_n^2) -> 10^(SNRdB/10) = sigma_s^2 / sigma_n^2
            output = sigma2s / (10 ** (value / 10))
        case "snr_dBm":
            # SNRdBm : SNRdBm = 10log10(sigma_s^2 / sigma_n^2) - 30 -> 10^((SNRdBm + 30)/10) = sigma_s^2 / sigma_n^2
            output = sigma2s / (10 ** ((value - 30) / 10))
        case _:
            raise ValueError(f"Unknown method: {input_unit}")

    return output


def zf_estimator(Y, H):
    r"""
    Perform Zero Forcing (ZF) linear equalization using the channel matrix pseudoinverse.

    .. math ::

        \mathbf{z}[n] = \mathbf{H}^{\dagger}\mathbf{y}[n]

    Parameters
    ----------
    Y : np.ndarray
        Received signal matrix.
    H : np.ndarray
        Channel matrix.

    Returns
    -------
    np.ndarray
        Estimated transmitted signal.
    """
    A = LA.pinv(H)
    Z_est = np.matmul(A, Y)
    return Z_est


def mmse_estimator(Y, H, sigma2):
    r"""
    Perform Minimum Mean Square Error (MMSE) linear equalization.

    .. math ::

        \mathbf{z}[n] = \left(\mathbf{H}^H\mathbf{H}+\sigma^2 \mathbf{I}_{N_t}\right)^{-1}\mathbf{H}^H\mathbf{y}[n]

    Parameters
    ----------
    Y : np.ndarray
        Received signal matrix.
    H : np.ndarray
        Channel matrix.
    sigma2 : float
        Noise variance.

    Returns
    -------
    np.ndarray
        Estimated transmitted signal.
    """
    _, N_t = H.shape
    H_H = np.conjugate(np.transpose(H))
    A = np.matmul(H_H, H) + sigma2 * np.eye(N_t)
    Z_est = LA.solve(A, np.matmul(H_H, Y))
    return Z_est
