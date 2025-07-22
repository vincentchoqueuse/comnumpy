import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift


def get_standard_carrier_allocation(config_name, os=1, custom=None, shift=False):
    """
    Allocate subcarriers based on a specified OFDM configuration.

    This function generates a subcarrier allocation array based on the given configuration name or custom parameters.
    It supports various OFDM configurations and allows for oversampling, Hermitian symmetry, and optional shifting.

    Parameters:
    ----------
    config_name : str
        The name of the OFDM configuration to use. If "Custom", the `custom` parameter must be provided.

    os : int, optional
        Oversampling factor. Default is 1 (no oversampling).

    custom : list, optional
        Custom configuration parameters in the form [N, N_nulled_DC, N_nulled_left, N_nulled_right, pilot_index].
        Required if `config_name` is "Custom".

    shift : bool, optional
        If False, apply FFT shift to the subcarrier allocation. Default is False.

    Returns:
    -------
    np.ndarray
        An array representing the subcarrier allocation, where:
        - 0 indicates a nulled subcarrier,
        - 1 indicates a data subcarrier,
        - 2 indicates a pilot subcarrier,
        - -1 indicates Hermitian symmetry (if applicable).

    Notes:
    ------
    - The function supports predefined configurations for various OFDM standards.
    - Hermitian symmetry, when applied, affects the allocation of data subcarriers.
    - Oversampling adds nulled subcarriers to the array.
    """
    ofdm_config_dict = {
        'IQtools_128': [128, 3, 6, 5, [16, 28, 40, 52, 76, 88, 100, 112]],
        '802.11ah_32': [32, 1, 3, 2, [9, 23]],
        '802.11ah_64': [64, 1, 4, 3, [11, 25, 39, 53]],
        '802.11ah_128': [128, 3, 6, 5, [11, 39, 53, 75, 89, 117]],
        '802.11ah_256': [256, 3, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231]],
        '802.11ah_512': [512, 11, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231, 281, 309, 345, 373, 395, 423, 459, 487]],
        'NoPilot_16': [16, 3, 6, 5, []],
        'NoPilot_32': [32, 3, 6, 5, []],
        'NoPilot_64': [64, 3, 6, 5, []],
        'NoPilot_128': [128, 3, 6, 5, []],
        'NoPilot_256': [256, 3, 6, 5, []],
        'NoPilot_512': [512, 3, 6, 5, []],
        'NoPilot_1024': [1024, 3, 6, 5, []],
        'NoPilot_2048': [2048, 3, 6, 5, []],
        'NoPilot_4096': [4096, 3, 6, 5, []],
        'NoPilot_8192': [8192, 3, 6, 5, []],
        'NoPilot_16384': [16384, 3, 6, 5, []],
        'NoPilot_Full_16': [16, 0, 0, 0, []],
        'NoPilot_Full_32': [32, 0, 0, 0, []],
        'NoPilot_Full_64': [64, 0, 0, 0, []],
        'NoPilot_Full_128': [128, 0, 0, 0, []],
        'NoPilot_Full_256': [256, 0, 0, 0, []],
        'NoPilot_Full_512': [512, 0, 0, 0, []],
        'NoPilot_Full_1024': [1024, 0, 0, 0, []],
        'NoPilot_Full_2048': [2048, 0, 0, 0, []],
        'NoPilot_Full_4096': [4096, 0, 0, 0, []],
        'NoPilot_Full_8192': [8192, 0, 0, 0, []],
        'NoPilot_Full_16384': [16384, 0, 0, 0, []]
    }

    if config_name == "Custom":
        N, N_nulled_DC, N_nulled_left, N_nulled_right, pilot_index = custom
    else:
        N, N_nulled_DC, N_nulled_left, N_nulled_right, pilot_index = ofdm_config_dict[config_name]

    N_pilot = len(pilot_index)

    N_data = N - N_nulled_DC - N_nulled_left - N_nulled_right - N_pilot
    oversampled_nulled_subcarriers = N * (os - 1)
    N_oversampled = N + oversampled_nulled_subcarriers
    carrier_type = np.zeros(N_oversampled)

    start_index = oversampled_nulled_subcarriers // 2
    end_index = start_index + N

    carrier_type[start_index:end_index] = 1
    carrier_type[start_index + np.array(pilot_index)] = 2
    carrier_type[start_index:start_index + N_nulled_left] = 0
    carrier_type[end_index - N_nulled_right:end_index] = 0

    middle = N // 2
    width = N_nulled_DC // 2
    carrier_type[start_index + middle - width: start_index + middle + width + 1] = 0

    if not shift:
        carrier_type = fftshift(carrier_type)

    return carrier_type


def plot_carrier_allocation(carrier_type, color_list = ["b", "g", "r"], label_list = ["null", "data", "pilots"], shift=False, num=None, title="Carrier allocation"):
    """
    Plot the allocation of subcarriers based on their types.

    This function visualizes the allocation of subcarriers in a carrier type array. It uses different colors and markers to represent different subcarrier types, such as Hermitian, null, data, and pilots. The plot can be shifted and customized with various parameters.

    Parameters:
    ----------
    carrier_type : np.ndarray
        An array representing the type of each subcarrier. The values in the array correspond to different subcarrier types:
        - 0: Null subcarrier
        - 1: Data subcarrier
        - 2: Pilot subcarrier

    color_list : list of str, optional
        A list of colors used to plot each subcarrier type. Default is ["g", "b", "r", "k"], which corresponds to green, blue, red, and black.

    label_list : list of str, optional
        A list of labels for each subcarrier type, used in the plot legend. Default is ["hermitian", "null", "data", "pilots"].

    shift : bool, optional
        If True, shift the x-axis by half the length of the carrier_type array. Default is False.

    num : int, optional
        The figure number to plot on. If None, a new figure is created. Default is None.

    title : str, optional
        The title of the plot. Default is "Carrier allocation".

    Notes:
    ------
    - The function uses `matplotlib.pyplot` to create the plot.
    - The `stem` plot is used to visualize the subcarrier types with vertical lines and markers.
    - Ensure that `color_list` and `label_list` have the correct length and order corresponding to the subcarrier types.

    Example:
    --------
    ```python
    import numpy as np
    carrier_type = np.array([1, 1, 0, 2, 1, 0, 0])
    plot_carrier_allocation(carrier_type)
    ```
    """
    if shift:
        offset = len(carrier_type)//2
    else:
        offset = 0

    plt.figure(num)
    for value in range(len(color_list)):
        color = color_list[value]
        index = np.where(carrier_type == value)[0]
        if len(index)>0:
            markerfmt = '{}o'.format(color)
            linefmt = '{}-'.format(color)
            label = label_list[value]
            plt.stem(index-offset, value*np.ones(len(index)), basefmt=" ", linefmt=linefmt, markerfmt=markerfmt, label=label)

    plt.xlabel("subcarrier index")
    plt.ylabel("subcarrier type")
    plt.title(title)
    plt.legend()