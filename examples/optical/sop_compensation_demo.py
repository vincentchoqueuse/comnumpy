import numpy as np
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser, compute_ber
from comnumpy.optical.pdm.channels import SOP
from comnumpy.optical.pdm.compensators import SOPCompensator


def add_awgn(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add complex AWGN to a 2×N signal at a given SNR per polarization."""
    M, N = y.shape
    out = np.zeros_like(y, dtype=complex)
    for m in range(M):
        p_sig = np.mean(np.abs(y[m, :]) ** 2)
        snr_lin = 10 ** (snr_db / 10.0)
        sigma2 = p_sig / snr_lin
        noise = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(sigma2 / 2)
        out[m, :] = y[m, :] + noise
    return out


def generate_symbols(alphabet: np.ndarray, N: int) -> np.ndarray:
    """Generate dual-pol symbols of length N from given alphabet."""
    idx1 = np.random.randint(0, len(alphabet), size=N)
    idx2 = np.random.randint(0, len(alphabet), size=N)
    x1 = alphabet[idx1]
    x2 = alphabet[idx2]
    return np.vstack([x1, x2])


def run_demo(M: int = 16, N: int = 200_000, snr_db: float = 20.0,
             Ts: float = 1/28e9, pol_linewidth_list=(1e4, 1e5, 1e6),
             block_size: int = 2048, alpha: float = 0.9):
    """
    Demonstrate SOP compensation using a decision-directed Jones tracker.

    Model assumptions:
    - SOP drift: Y[n] = J[n] X[n], with unitary Jones matrix J[n] evolving as
      small rotations; implemented via Pauli matrix exponential.
    - Compensation: Estimate J per block via LS using hard decisions, project to
      nearest unitary (polar decomposition), smooth over time, and apply inverse.
    """
    alphabet = get_alphabet("QAM", M)
    width = int(np.log2(M))

    print("SOP compensation demo: QAM{}, N={}, SNR={} dB".format(M, N, snr_db))
    print("block_size={}, alpha={}".format(block_size, alpha))

    # Ground-truth dual-pol symbols
    X = generate_symbols(alphabet, N)

    for lw in pol_linewidth_list:
        # Apply SOP drift at symbol rate
        sop = SOP(T_symb=Ts, linewidth=lw)
        Y = sop(X)

        # Add AWGN
        Y_noisy = add_awgn(Y, snr_db)

        # Baseline BER (no compensation)
        ser1_nc = compute_ser(X[0], Y_noisy[0])
        ber1_nc = compute_ber(X[0], Y_noisy[0], width=width)
        ser2_nc = compute_ser(X[1], Y_noisy[1])
        ber2_nc = compute_ber(X[1], Y_noisy[1], width=width)
        ber_nc = np.mean([ber1_nc, ber2_nc])

        # SOP compensation
        sopc = SOPCompensator(alphabet=alphabet, block_size=block_size, alpha=alpha)
        Y_comp = sopc(Y_noisy)

        # BER after compensation
        ser1_c = compute_ser(X[0], Y_comp[0])
        ber1_c = compute_ber(X[0], Y_comp[0], width=width)
        ser2_c = compute_ser(X[1], Y_comp[1])
        ber2_c = compute_ber(X[1], Y_comp[1], width=width)
        ber_c = np.mean([ber1_c, ber2_c])

        print(f"linewidth={lw:.2e} Hz -> BER no-comp={ber_nc:.3e}, BER comp={ber_c:.3e}")


if __name__ == "__main__":
    np.random.seed(0)
    run_demo()
