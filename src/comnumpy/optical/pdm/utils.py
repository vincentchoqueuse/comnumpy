from dataclasses import dataclass
import numpy as np
from scipy.stats import maxwell


def build_pmd_segments(L_km: float, D_pmd_ps_sqrt_km: float, N_segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Build PMD segment parameters with Maxwellian-distributed DGDs.
    
    Generates differential group delays (DGDs) drawn from a Maxwellian
    distribution, with total rms DGD matching tau_rms_link = D_pmd * sqrt(L_km).
    
    Parameters
    ----------
    L_km : float
        Link length in kilometers
    D_pmd_ps_sqrt_km : float
        PMD parameter in ps/sqrt(km)
    N_segments : int
        Number of fiber segments
    
    Returns
    -------
    tau_k : np.ndarray
        Array of DGDs per segment (ps)
    theta_k : np.ndarray
        Array of PSP angles per segment (rad)
    """
    
    # total rms DGD of the whole link
    tau_rms_link = D_pmd_ps_sqrt_km * np.sqrt(L_km)  # [ps]

    # rms DGD per segment
    tau_rms_seg = tau_rms_link / np.sqrt(N_segments)

    # Maxwell scale parameter a
    a = tau_rms_seg / np.sqrt(3.0)

    # draw DGDs from Maxwell(a)
    tau_k = maxwell.rvs(scale=a, size=N_segments)    # [ps]

    # PSP angles, uniform in [0, pi)
    theta_k = np.random.uniform(0.0, np.pi, size=N_segments)

    return tau_k, theta_k