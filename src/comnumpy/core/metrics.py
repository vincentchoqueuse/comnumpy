import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Any
from scipy.stats import norm
from comnumpy.core.generics import Processor


def sym_2_bin(sym, width):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice], width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)


def compute_ser_awgn_psk(order, snr_per_bit):

    gamma_b = snr_per_bit
    k = int(np.log2(order))

    if order == 2:
        # see book Proakis "Digital communication", p 271
        argument = np.sqrt(2*gamma_b)
        value = norm.sf(argument)

    if order == 4:
        # see book Proakis "Digital communication", p 272
        argument = np.sqrt(2*gamma_b)
        term = norm.sf(argument)
        value = 2*term*(1-0.5*term)

    if order > 4:
        M = order
        argument = np.sqrt(2*k*gamma_b)*np.sin(np.pi/M)
        value = 2*norm.sf(argument)

    if type == "bin":
        value = value/k

    return value


def compute_ser_awgn_qam(order, snr_per_bit):

    gamma_b = snr_per_bit

    # see book Proakis "Digital communication", p 280
    M = order
    k = np.log2(order)
    argument = np.sqrt(3*k*gamma_b/(M-1))
    P_sqrt_M = 2*(1-1/np.sqrt(M))*norm.sf(argument)

    value = 1-(1-P_sqrt_M)**2

    return value


def compute_metric_awgn_theo(modulation, order, snr_per_bit, type="ser"):

    if modulation == "PSK":
        value = compute_ser_awgn_psk(order, snr_per_bit)

    if modulation == "QAM":
        value = compute_ser_awgn_qam(order, snr_per_bit)

    if type == "bin":
        k = int(np.log2(order))
        value = value/k

    return value

def compute_ser(X_target, X_detected, axis=None):
    """
    Compute the Symbol Error Rate (SER) between the target and detected signals along the specified axis.

    Parameters
    ----------
    X_target : ndarray
        The target signal array.
    X_detected : ndarray
        The detected signal array after some transmission or processing.
    axis : int or None, optional
        Axis along which to compute the SER. If None, the arrays are raveled into 1D arrays.

    Returns
    -------
    float or ndarray
        The computed SER value. If axis is None, returns a float. Otherwise, returns an array.

    Notes
    -----
    The SER is computed using the formula:

    .. math::

        \\text{SER} = \\frac{\\|\\mathbf{x}_{\\text{tar}} - \\mathbf{x}_{\\text{est}}\\|_0}{N}

    """
    if axis is None:
        x_target = np.ravel(X_target)
        x_detected = np.ravel(X_detected)
        N = min(len(x_detected), len(x_target))
        nb_errors = np.count_nonzero(x_target[:N] - x_detected[:N])
        return nb_errors / N
    else:
        N = X_target.shape[axis]
        nb_errors = np.count_nonzero(X_target - X_detected, axis=axis)
        return nb_errors / N


def compute_ber(X_target, X_detected, width, axis=None):
    """
    Compute the Bit Error Rate (BER) between target and detected symbols along the specified axis.

    Parameters
    ----------
    X_target : ndarray
        The target symbols to be compared against. Should be convertible to a 1-D array.
    X_detected : ndarray
        The detected symbols obtained from a transmission system or a decoder.
        Should be convertible to a 1-D array.
    width : int
        The number of bits used to represent each symbol.
    axis : int or None, optional
        Axis along which to compute the BER. If None, the arrays are raveled into 1D arrays.

    Returns
    -------
    float or ndarray
        The Bit Error Rate (BER), computed as the number of differing bits between `s_target` and `s_detected` divided by the total number of bits.

    Notes
    -----
    The function assumes that `sym_2_bin` is a predefined function that converts symbols to their binary representation. The formula for BER is:

    .. math ::

        \text{BER} = \\frac{N_e}{N_b}

    """
    if axis is None:
        s_target = sym_2_bin(np.ravel(X_target), width)
        s_detected = sym_2_bin(np.ravel(X_detected), width)
        nb_errors = np.count_nonzero(s_target - s_detected)
        return nb_errors / len(s_detected)
    else:
        s_target = sym_2_bin(X_target, width)
        s_detected = sym_2_bin(X_detected, width)
        nb_errors = np.count_nonzero(s_target - s_detected, axis=axis)
        total_bits = s_target.shape[axis]
        return nb_errors / total_bits


def compute_evm(X_target, X_estimated, axis=None):
    r"""
    Compute the Error Vector Magnitude (EVM) between the target and estimated signals along the given axis.

    Parameters
    ----------
    X_target : ndarray
        The target signal array.
    X_estimated : ndarray
        The estimated signal array.
    axis : int or None, optional
        Axis along which to compute the mean. If None, the mean is computed over all dimensions.

    Returns
    -------
    float or ndarray
        The computed EVM value. If axis is None, returns a float. Otherwise, returns an array.

    Notes
    -----
    The EVM is computed using the formula:

    .. math::

        \text{EVM} = \sqrt{\frac{\sum_{i} |x_{\text{target},i} - x_{\text{estimated},i}|^2}{\sum_{i} |x_{\text{target},i}|^2}}

    where:

    * :math:`x_{\text{target},i}` is the ith element of the target signal.
    * :math:`x_{\text{estimated},i}` is the ith element of the estimated signal.

    Examples
    --------
    >>> X_target = np.array([1, 2, 3, 4])
    >>> X_estimated = np.array([1.1, 2.1, 3.1, 3.9])
    >>> compute_evm(X_target, X_estimated)
    0.05057216690580728
    >>> compute_evm(X_target, X_estimated, axis=0)
    0.05057216690580728
    """
    # Compute the numerator and denominator
    error_vector = np.abs(X_target - X_estimated)**2
    target_power = np.abs(X_target)**2

    # Compute the mean along the specified axis
    if axis is None:
        num = np.mean(error_vector)
        den = np.mean(target_power)
    else:
        num = np.mean(error_vector, axis=axis)
        den = np.mean(target_power, axis=axis)

    return np.sqrt(num / den)



def compute_effective_SNR(X_target, X_estimated, sigma2_s=1, unit="natural"):
    r"""
    Compute the effective Signal-to-Noise Ratio (SNR) between a target and an estimated signal.

    Parameters
    ----------
    X_target : ndarray
        The target signal array.
    X_estimated : ndarray
        The estimated signal array.
    sigma2_s : float, optional
        The variance of the signal. The default is 1.
    unit : str, optional
        Unit of the output SNR. Can be "natural" (linear scale), "dB" (decibels),
        or "dBm" (decibel-milliwatts). The default is "natural".

    Returns
    -------
    float
        The effective SNR in the specified unit.

    Raises
    ------
    ValueError
        If the specified unit is not one of "natural", "dB", or "dBm".

    Notes
    -----
    - The effective SNR is calculated as the ratio of the signal variance to the mean squared error between
      the target and estimated signals.
    - For "dB" unit, the SNR is converted using the formula: \( \text{SNR}_{dB} = 10 \log_{10}(\text{SNR}) \).
    - For "dBm" unit, the SNR is converted using the formula: \( \text{SNR}_{dBm} = 10 \log_{10}(\text{SNR}) + 30 \).
    """
    x_target = np.ravel(X_target)
    x_estimated = np.ravel(X_estimated)
    sigma2_b = np.mean(np.abs(x_target - x_estimated)**2)

    SNR = sigma2_s / sigma2_b

    match unit:
        case "natural":
            output = SNR
        case "dB":
            output = 10 * np.log10(SNR)
        case "dBm":
            output = 10 * np.log10(SNR) + 30
        case _:
            raise ValueError(f"Unknown unit: {unit}")

    return output


def compute_power(x, unit="natural"):
    r"""
    Compute the mean power of an input array.

    Parameters
    ----------
    x : ndarray
        Input array containing the signal values.
    unit : str, optional
        Unit of the output power. Can be "natural" (Watts), "dB" (decibels),
        or "dBm" (decibel-milliwatts). The default is "natural".

    Returns
    -------
    float
        Mean power value in the specified unit.

    Raises
    ------
    ValueError
        If the specified unit is not one of "natural", "dB", or "dBm".

    Notes
    -----
    - The mean power is calculated as the average of the squared magnitudes of the input array.
    - For "dB" and "dBm" units, the power is converted using the formula: \( P_{dB} = 10 \log_{10}(P) \)
      and \( P_{dBm} = 10 \log_{10}(P) + 30 \), respectively.
    """
    Px = np.mean(np.abs(x)**2)

    match unit:
        case "natural":
            output = Px
        case "dB":
            output = 10 * np.log10(Px)
        case "dBm":
            output = 10 * np.log10(Px) + 30
        case _:
            raise ValueError(f"Unknown unit: {unit}")

    return output


def compute_ccdf(data, axis=-1):
    """
    Compute the Complementary Cumulative Distribution Function (CCDF) for a given dataset along a specified axis.

    Parameters
    ----------
    data : array-like
        Input data for which the CCDF needs to be calculated.
    axis : int, optional
        Axis along which to compute the CCDF. Default is the last axis.

    Returns
    -------
    sorted_data : np.ndarray
        The input data sorted in ascending order along the specified axis.
    ccdf : np.ndarray
        The computed CCDF values corresponding to the sorted data.

    Notes
    -----
    The CCDF is computed using the formula:

    .. math::

        \\text{CCDF}(x) = 1 - \\text{CDF}(x)

    where the CDF is the cumulative distribution function.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> sorted_data, ccdf = compute_ccdf(data, axis=1)
    >>> sorted_data
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> ccdf
    array([[0.66666667, 0.33333333, 0.        ],
           [0.66666667, 0.33333333, 0.        ]])
    """
    # Sort the data values in ascending order along the specified axis
    sorted_data = np.sort(data, axis=axis)

    # Compute the empirical CCDF along the specified axis
    n = data.shape[axis]
    indices = np.arange(1, n + 1)
    ccdf = 1.0 - indices / n

    # Expand ccdf to match the shape of sorted_data
    ccdf = np.expand_dims(ccdf, list(range(data.ndim))[::-1][:axis])

    return sorted_data, ccdf


def calculate_acpr(signal, bandwidth, sampling_rate):
    """
    Calculate the Adjacent Channel Power Ratio (ACPR) of a given signal.

    ACPR is a measure of spectral regrowth and quantifies the ratio between the power in the main 
    transmission band and the power leaked into adjacent frequency bands. This function computes 
    ACPR on both the left and right adjacent channels.

    Parameters
    ----------
    signal : np.ndarray
        The input time-domain signal.
    bandwidth : float
        The bandwidth of the main signal in Hz.
    sampling_rate : float
        The sampling rate of the signal in Hz.

    Returns
    -------
    acpr_right : float
        ACPR value (in dB) for the right adjacent channel.
    acpr_left : float
        ACPR value (in dB) for the left adjacent channel.
    """

    # NOT TESTED

    def calculate_power(signal, lower_freq, upper_freq, sampling_rate):

        # Perform FFT
        fft_signal = np.fft.fft(signal)
        freq_axis = np.fft.fftfreq(len(signal), 1 / sampling_rate)

        # Select frequencies within the band
        mask = (freq_axis >= lower_freq) & (freq_axis <= upper_freq)
        band_power = np.sum(np.abs(fft_signal[mask])**2) / len(signal)
        
        return band_power

    # Define main and adjacent channel frequency bands
    half_bw = bandwidth / 2
    main_lower = -half_bw
    main_upper = half_bw
    
    adj_lower_right = main_upper
    adj_upper_right = main_upper + bandwidth

    adj_lower_left = main_lower - bandwidth
    adj_upper_left = main_lower
    
    # Calculate power in main and adjacent channels
    main_power = calculate_power(signal, main_lower, main_upper, sampling_rate)
    adj_power_right = calculate_power(signal, adj_lower_right, adj_upper_right, sampling_rate)
    adj_power_left = calculate_power(signal, adj_lower_left, adj_upper_left, sampling_rate)
    
    # Calculate ACPR in dB
    acpr_right = 10 * np.log10(adj_power_right / main_power)
    acpr_left = 10 * np.log10(adj_power_left / main_power)

    return acpr_right, acpr_left


@dataclass
class MetricRecorder(Processor):
    """
    A class to compute and record the result of a metric function applied to input data.

    This class encapsulates a callable metric function with optional parameters.
    When the object is called with data, it evaluates the metric and stores the result
    in the `data` attribute, while returning the original input unchanged (for pipeline use).

    Attributes
    ----------
    metric_fn : Callable
        The metric function to be applied. Must take the input `X` as first argument.

    params : dict
        Optional keyword arguments to be passed to the metric function.

    data : Any
        The result of the metric function, stored after calling the object.

    Example
    -------
    >>> import numpy as np
    >>> def mean_std(X, axis=0):
    ...     return {"mean": X.mean(axis=axis), "std": X.std(axis=axis)}
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> recorder = MetricRecorder(metric_fn=mean_std, params={"axis": 1})
    >>> recorder(X)
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> recorder.data
    {'mean': array([2., 5.]), 'std': array([0.81649658, 0.81649658])}
    """

    metric_fn: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    name: str = "metric recorder"

    def forward(self, X):
        self.data = self.metric_fn(X, **self.params)
        return X

    def get_data(self):
        return self.data

    def __repr__(self):
        nom = self.metric_fn.__name__
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"MetricRecorder({nom}, {args})"
