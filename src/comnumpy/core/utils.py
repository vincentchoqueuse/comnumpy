import numpy as np
import os.path as path
import numpy.linalg as LA


def get_alphabet(modulation, order, type="gray", norm=True):
    """
    Retrieve the symbol alphabet for a given modulation scheme and order.

    This function loads the modulation alphabet from a predefined CSV file based on the specified modulation type, order,
    and symbol mapping (e.g., Gray coding).
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


def plot_alphabet(alphabet, ax=None, label="alphabet", title="Constellation", **kwargs):
    """
    Plot a constellation diagram of the given symbol alphabet.

    Parameters
    ----------
    alphabet : np.ndarray
        Complex-valued symbol alphabet to plot.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If None, a new figure and axis are created.
    label : str, optional
        Label for the scatter plot. Default is ``"alphabet"``.
    title : str, optional
        Title of the plot. Default is ``"Constellation"``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot (decision D25).
    """
    import matplotlib.pyplot as plt  # local import (D36)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.real(alphabet), np.imag(alphabet), "o", label=label, **kwargs)
    ax.set_xlabel("real part")
    ax.set_ylabel("imag part")
    ax.set_title(title)
    return ax


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


def esn0_to_snr_dB(esn0_dB, oversampling=1):
    r"""
    Convert a symbol-energy-to-noise ratio :math:`E_s/N_0` (dB) into the SNR (dB) expected by ``AWGN``.

    At one sample per symbol the two quantities coincide; with an
    oversampling factor :math:`L` the noise bandwidth is :math:`L` times
    larger, hence

    .. math::

        \mathrm{SNR_{dB}} = E_s/N_0\big|_{dB} - 10\log_{10}(L)

    Parameters
    ----------
    esn0_dB : float
        Symbol energy to noise spectral density ratio, in dB.
    oversampling : int, optional
        Number of samples per symbol :math:`L`. Default is 1.

    Returns
    -------
    float
        The SNR in dB, relative to the measured signal power.

    Examples
    --------
    >>> print(float(esn0_to_snr_dB(10)))
    10.0
    >>> print(round(float(esn0_to_snr_dB(10, oversampling=4)), 4))
    3.9794
    """
    return esn0_dB - 10 * np.log10(oversampling)


def ebn0_to_snr_dB(ebn0_dB, bits_per_symbol, code_rate=1.0, oversampling=1):
    r"""
    Convert a bit-energy-to-noise ratio :math:`E_b/N_0` (dB) into the SNR (dB) expected by ``AWGN``.

    The conversion requires chain-level knowledge (bits per symbol
    :math:`k`, code rate :math:`R`, oversampling :math:`L`), which is why
    it lives here and not inside the ``AWGN`` block (decision D41):

    .. math::

        \mathrm{SNR_{dB}} = E_b/N_0\big|_{dB} + 10\log_{10}(k R) - 10\log_{10}(L)

    Parameters
    ----------
    ebn0_dB : float
        Bit energy to noise spectral density ratio, in dB.
    bits_per_symbol : int
        Number of bits carried by one constellation symbol :math:`k`
        (e.g. 4 for 16-QAM).
    code_rate : float, optional
        FEC code rate :math:`R \in (0, 1]`. Default is 1.0 (uncoded).
    oversampling : int, optional
        Number of samples per symbol :math:`L`. Default is 1.

    Returns
    -------
    float
        The SNR in dB, relative to the measured signal power.

    Examples
    --------
    >>> print(round(float(ebn0_to_snr_dB(10, bits_per_symbol=4)), 4))
    16.0206
    >>> print(round(float(ebn0_to_snr_dB(10, bits_per_symbol=2, code_rate=0.5)), 4))
    10.0
    """
    return ebn0_dB + 10 * np.log10(bits_per_symbol * code_rate) - 10 * np.log10(oversampling)


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
