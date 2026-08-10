from __future__ import annotations

import warnings
from typing import (TYPE_CHECKING, Any, Dict, List, Optional, Sequence,
                    Tuple)

import numpy as np
from comnumpy._backend import fftshift  # cupy-compatible (D3)

__all__ = ["get_standard_carrier_allocation", "plot_carrier_allocation"]

if TYPE_CHECKING:  # matplotlib stays out of the import path (D36)
    from matplotlib.axes import Axes

# [N, N_nulled_DC, N_nulled_left, N_nulled_right, pilot_index]
_ConfigEntry = Tuple[int, int, int, int, List[int]]


def get_standard_carrier_allocation(config_name: str, os: int = 1,
                                    custom: Optional[Sequence[Any]] = None,
                                    shift: bool = False) -> np.ndarray:
    """
    Allocate subcarriers based on a specified OFDM configuration.

    .. deprecated:: 1.0
        Use :func:`comnumpy.ofdm.allocation.get_allocation`, which returns a
        :class:`~comnumpy.ofdm.allocation.CarrierAllocation` object carrying
        its metadata and self-checked against the standard's tables.

    This function generates a subcarrier allocation array based on the given configuration name or custom parameters.
    It supports various OFDM configurations and allows for oversampling, Hermitian symmetry, and optional shifting.

    Parameters
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

    Returns
    -------
    np.ndarray
        An array representing the subcarrier allocation, where:
        - 0 indicates a nulled subcarrier,
        - 1 indicates a data subcarrier,
        - 2 indicates a pilot subcarrier,
        - -1 indicates Hermitian symmetry (if applicable).

    Notes
    -----
    - The function supports predefined configurations for various OFDM standards.
    - Hermitian symmetry, when applied, affects the allocation of data subcarriers.
    - Oversampling adds nulled subcarriers to the array.
    """
    warnings.warn(
        "get_standard_carrier_allocation is deprecated; use "
        "comnumpy.ofdm.allocation.get_allocation instead",
        DeprecationWarning, stacklevel=2)

    ofdm_config_dict: Dict[str, _ConfigEntry] = {
        'IQtools_128': (128, 3, 6, 5, [16, 28, 40, 52, 76, 88, 100, 112]),
        '802.11ah_32': (32, 1, 3, 2, [9, 23]),
        '802.11ah_64': (64, 1, 4, 3, [11, 25, 39, 53]),
        '802.11ah_128': (128, 3, 6, 5, [11, 39, 53, 75, 89, 117]),
        '802.11ah_256': (256, 3, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231]),
        '802.11ah_512': (512, 11, 6, 5, [25, 53, 89, 117, 139, 167, 203, 231, 281, 309, 345, 373, 395, 423, 459, 487]),
        'NoPilot_16': (16, 3, 6, 5, []),
        'NoPilot_32': (32, 3, 6, 5, []),
        'NoPilot_64': (64, 3, 6, 5, []),
        'NoPilot_128': (128, 3, 6, 5, []),
        'NoPilot_256': (256, 3, 6, 5, []),
        'NoPilot_512': (512, 3, 6, 5, []),
        'NoPilot_1024': (1024, 3, 6, 5, []),
        'NoPilot_2048': (2048, 3, 6, 5, []),
        'NoPilot_4096': (4096, 3, 6, 5, []),
        'NoPilot_8192': (8192, 3, 6, 5, []),
        'NoPilot_16384': (16384, 3, 6, 5, []),
        'NoPilot_Full_16': (16, 0, 0, 0, []),
        'NoPilot_Full_32': (32, 0, 0, 0, []),
        'NoPilot_Full_64': (64, 0, 0, 0, []),
        'NoPilot_Full_128': (128, 0, 0, 0, []),
        'NoPilot_Full_256': (256, 0, 0, 0, []),
        'NoPilot_Full_512': (512, 0, 0, 0, []),
        'NoPilot_Full_1024': (1024, 0, 0, 0, []),
        'NoPilot_Full_2048': (2048, 0, 0, 0, []),
        'NoPilot_Full_4096': (4096, 0, 0, 0, []),
        'NoPilot_Full_8192': (8192, 0, 0, 0, []),
        'NoPilot_Full_16384': (16384, 0, 0, 0, [])
    }

    entry: _ConfigEntry
    if config_name == "Custom":
        # custom= defaults to None, and unpacking it used to raise a bare
        # "cannot unpack non-iterable NoneType" three frames deep (D38)
        if custom is None:
            raise ValueError(
                "config_name='Custom' requires custom=, got None; expected a "
                "sequence [N, N_nulled_DC, N_nulled_left, N_nulled_right, "
                "pilot_index]; either pass it or name a standard "
                f"configuration among {sorted(ofdm_config_dict)}")
        if len(custom) != 5:
            raise ValueError(
                f"custom= has {len(custom)} entries, expected 5 "
                f"[N, N_nulled_DC, N_nulled_left, N_nulled_right, "
                f"pilot_index]; pass an empty list as the last entry for a "
                f"configuration without pilots")
        entry = (int(custom[0]), int(custom[1]), int(custom[2]),
                 int(custom[3]), list(custom[4]))
    else:
        if config_name not in ofdm_config_dict:
            raise KeyError(
                f"unknown configuration {config_name!r}, expected one of "
                f"{sorted(ofdm_config_dict)} or 'Custom' with custom=")
        entry = ofdm_config_dict[config_name]

    N, N_nulled_DC, N_nulled_left, N_nulled_right, pilot_index = entry

    oversampled_nulled_subcarriers = N * (os - 1)
    N_oversampled = N + oversampled_nulled_subcarriers
    carrier_type = np.zeros(N_oversampled)

    start_index = oversampled_nulled_subcarriers // 2
    end_index = start_index + N

    carrier_type[start_index:end_index] = 1
    # dtype=int: an empty pilot list gives a float64 array, which numpy
    # refuses as an index -- that made every NoPilot_* configuration raise
    pilots = np.asarray(pilot_index, dtype=int)
    carrier_type[start_index + pilots] = 2
    carrier_type[start_index:start_index + N_nulled_left] = 0
    carrier_type[end_index - N_nulled_right:end_index] = 0

    # null exactly N_nulled_DC subcarriers centered on DC
    if N_nulled_DC > 0:
        middle = N // 2
        dc_start = start_index + middle - N_nulled_DC // 2
        carrier_type[dc_start: dc_start + N_nulled_DC] = 0

    if not shift:
        carrier_type = fftshift(carrier_type)

    return carrier_type


def plot_carrier_allocation(carrier_type: np.ndarray,
                            ax: Optional["Axes"] = None,
                            color_list: Optional[List[str]] = None,
                            label_list: Optional[List[str]] = None,
                            shift: bool = False,
                            title: str = "Carrier allocation") -> "Axes":
    """
    Plot the allocation of subcarriers based on their types.

    This function visualizes the allocation of subcarriers in a carrier type array. It uses different colors and markers to represent
    different subcarrier types, such as Hermitian, null, data, and pilots. The plot can be shifted and customized with various parameters.

    Parameters
    ----------
    carrier_type : np.ndarray
        An array representing the type of each subcarrier. The values in the array correspond to different subcarrier types:
        - 0: Null subcarrier
        - 1: Data subcarrier
        - 2: Pilot subcarrier

    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If None, a new figure and axis are created.

    color_list : list of str, optional
        Colors indexed by subcarrier type value. Defaults to the frozen
        ``CARRIER_STYLE`` palette of decision D27 (Okabe-Ito, safe for
        colour-vision deficiency), the same one the ASCII spectral map
        and :meth:`CarrierAllocation.plot` use.

    label_list : list of str, optional
        Legend labels indexed by subcarrier type value. Defaults to the
        ``CARRIER_STYLE`` labels.

    shift : bool, optional
        If True, shift the x-axis by half the length of the carrier_type array. Default is False.

    title : str, optional
        The title of the plot. Default is "Carrier allocation".

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot (decision D25).

    Notes
    -----
    - The function uses `matplotlib.pyplot` to create the plot.
    - The `stem` plot is used to visualize the subcarrier types with vertical lines and markers.
    - Ensure that `color_list` and `label_list` have the correct length and order corresponding to the subcarrier types.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> carrier_type = np.array([1, 1, 0, 2, 1, 0, 0])
    >>> ax = plot_carrier_allocation(carrier_type)
    >>> ax.get_xlabel()
    'subcarrier index'
    """
    import matplotlib.pyplot as plt  # local import (D36)

    from comnumpy.ofdm.allocation import CARRIER_STYLE, CarrierType  # local (D36)

    # D27: one palette for every carrier rendering, and the marker carries
    # the information too -- colour is never the only channel
    if color_list is None:
        color_list = [CARRIER_STYLE[t]["color"] for t in CarrierType]
    if label_list is None:
        label_list = [CARRIER_STYLE[t]["label"] for t in CarrierType]
    marker_list = ["o", "s", "D"]

    if shift:
        offset = len(carrier_type)//2
    else:
        offset = 0

    if ax is None:
        _, ax = plt.subplots()
    for value in range(len(color_list)):
        color = color_list[value]
        index = np.where(carrier_type == value)[0]
        if len(index)>0:
            label = label_list[value]
            marker = marker_list[value % len(marker_list)]
            markerline, stemlines, _ = ax.stem(
                index - offset, value * np.ones(len(index)),
                basefmt=" ", markerfmt=marker, label=label)
            markerline.set_color(color)
            stemlines.set_color(color)

    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("subcarrier type")
    ax.set_title(title)
    ax.legend()
    return ax
