import numpy as np


def compute_PAPR(x_data, unit="natural"):
    """
    Compute the Peak-to-Average Power Ratio (PAPR) of the input data.

    Parameters
    ----------
    x_data : ndarray
        Input data for which PAPR needs to be calculated.
    unit : str, optional
        The unit for PAPR calculation. It can be either "natural" for natural units or 
        "dB" for logarithmic units. Default is "natural".

    Returns
    -------
    float
        The computed PAPR value.

    Notes
    -----
    The PAPR is computed using the formulas:

    * For natural units:

    .. math::

        \\text{PAPR} = \\frac{x_{\\text{max}}}{\\sqrt{P_{\\text{moy}}}}

    * For dB units:

    .. math::

        \\text{PAPR} = 10 \\log_{10} \\left( \\frac{x_{\\text{max}}^2}{P_{\\text{moy}}} \\right)

    where:
    \( x_{\\text{max}} \) is the maximum power value in `x_data`,
    \( P_{\\text{moy}} \) is the mean power value in `x_data`.

    """
    x_max = np.max(np.abs(x_data))
    P_moy = np.mean(np.abs(x_data)**2)

    if unit == "natural":
        papr = x_max/np.sqrt(P_moy)
    if unit == "dB":
        papr = 10*np.log10(x_max**2/P_moy)
    
    return papr


def compute_ccdf(papr_dB_list, papr_dB_threshold=np.arange(6, 13, 0.5)):

    data = []
    for threshold in papr_dB_threshold:
        ccdf = np.mean((papr_dB_list>threshold))
        data.append(ccdf)

    return data