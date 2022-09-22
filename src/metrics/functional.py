import numpy as np


def sym_2_bin(sym, width):

    data = []
    for indice in range(len(sym)):
        data.append(np.binary_repr(sym[indice],width))

    string = ''.join(data)

    return np.array(list(string), dtype=int)


def compute_ser(X_target,X_detected):
    x_target = np.ravel(X_target)
    x_detected = np.ravel(X_detected)
    N = len(x_detected)
    nb_errors = np.count_nonzero(x_target-x_detected)
    return nb_errors / N


def compute_ber(X_target, X_detected, width):
    s_target = sym_2_bin(np.ravel(X_target), width)
    s_detected = sym_2_bin(np.ravel(X_detected), width)
    nb_errors = np.count_nonzero(s_target-s_detected)
    return nb_errors / len(s_detected)


def compute_effective_SNR(X_target, X_estimated, sigma2_s=1, unit="linear"):
    x_target = np.ravel(X_target)
    x_estimated = np.ravel(X_estimated)
    sigma2_b = np.mean(np.abs(x_target-x_estimated)**2)

    SNR = sigma2_s / sigma2_b

    if unit == "dB":
        SNR = 10*np.log10(SNR)

    return SNR