import numpy as np


def compute_effective_SNR(X_target, X_estimated, sigma2_s=1, unit="natural"):
    x_target = np.ravel(X_target)
    x_estimated = np.ravel(X_estimated)
    sigma2_b = np.mean(np.abs(x_target-x_estimated)**2)

    SNR = sigma2_s / sigma2_b

    if unit == "dB":
        SNR = 10*np.log10(SNR)

    return SNR

