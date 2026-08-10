import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.mimo.coding import SpaceTimeDecoder, SpaceTimeEncoder, get_code

img_dir = "../../docs/examples/img/"

# Parameters
M = 4
alphabet = get_alphabet("PSK", M)
code = get_code("alamouti")
power = 1 / np.sqrt(code.n_tx)          # same total power as one antenna
encoder = SpaceTimeEncoder(code)
rng = np.random.default_rng(42)

# One channel realization, one block of codewords
N_codewords = 500
sigma2 = 0.2
H = (rng.normal(size=(1, 2)) + 1j * rng.normal(size=(1, 2))) / np.sqrt(2)
s_tx = rng.integers(0, M, size=code.n_symbols * N_codewords)
X = encoder(alphabet[s_tx] * power)
B = np.sqrt(sigma2 / 2) * (rng.normal(size=(1, X.shape[1]))
                           + 1j * rng.normal(size=(1, X.shape[1])))
Y = H @ X + B
Z = SpaceTimeDecoder(code, H=H)(Y) / power
s_rx, _ = hard_projector(Z, alphabet)
print(f"one-shot SER: {compute_ser(s_tx, s_rx):.4f}")

# Figure 1: what the single receive antenna sees, and what combining gives
fig1, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.2))
ax_left.plot(np.real(Y[0]), np.imag(Y[0]), ".", markersize=3)
ax_left.set_title("received on the single antenna")
ax_right.plot(np.real(Z), np.imag(Z), ".", markersize=3)
ax_right.plot(np.real(alphabet), np.imag(alphabet), "kx", markersize=9)
ax_right.set_title("after Alamouti combining")
for ax in (ax_left, ax_right):
    ax.set_xlabel("in phase")
    ax.set_ylabel("quadrature")
    ax.axis("equal")
    ax.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig1.png")


def alamouti_ser(snr_dB, n_channels=8000, per_channel=25, seed=1):
    """2 transmit antennas, 1 receive antenna, quasi-static fading."""
    generator = np.random.default_rng(seed)
    sigma2 = 10 ** (-snr_dB / 10)
    errors = total = 0
    for _ in range(n_channels):
        h = (generator.normal(size=(1, 2))
             + 1j * generator.normal(size=(1, 2))) / np.sqrt(2)
        sent = generator.integers(0, M, size=code.n_symbols * per_channel)
        word = encoder(alphabet[sent] * power)
        noise = np.sqrt(sigma2 / 2) * (
            generator.normal(size=(1, word.shape[1]))
            + 1j * generator.normal(size=(1, word.shape[1])))
        estimate = SpaceTimeDecoder(code, H=h)(h @ word + noise) / power
        detected, _ = hard_projector(estimate, alphabet)
        errors += int(np.sum(detected != sent))
        total += sent.size
    return errors / total


def siso_ser(snr_dB, n_symbols=400000, seed=2):
    """One transmit antenna, one receive antenna: no diversity."""
    generator = np.random.default_rng(seed)
    sent = generator.integers(0, M, size=n_symbols)
    h = (generator.normal(size=n_symbols)
         + 1j * generator.normal(size=n_symbols)) / np.sqrt(2)
    sigma2 = 10 ** (-snr_dB / 10)
    noise = np.sqrt(sigma2 / 2) * (generator.normal(size=n_symbols)
                                   + 1j * generator.normal(size=n_symbols))
    detected, _ = hard_projector((h * alphabet[sent] + noise) / h, alphabet)
    return float(np.mean(detected != sent))


def mrc_ser(snr_dB, n_symbols=400000, seed=3):
    """One transmit antenna, two receive antennas: maximum ratio combining."""
    generator = np.random.default_rng(seed)
    sent = generator.integers(0, M, size=n_symbols)
    h = (generator.normal(size=(2, n_symbols))
         + 1j * generator.normal(size=(2, n_symbols))) / np.sqrt(2)
    sigma2 = 10 ** (-snr_dB / 10)
    noise = np.sqrt(sigma2 / 2) * (generator.normal(size=(2, n_symbols))
                                   + 1j * generator.normal(size=(2, n_symbols)))
    received = h * alphabet[sent] + noise
    combined = np.sum(np.conj(h) * received, axis=0) / np.sum(np.abs(h) ** 2,
                                                              axis=0)
    detected, _ = hard_projector(combined, alphabet)
    return float(np.mean(detected != sent))


# Monte-Carlo comparison
snr_dB_list = np.arange(4, 29, 4)
curves = {
    "1 Tx, 1 Rx (no diversity)": [siso_ser(value) for value in snr_dB_list],
    "Alamouti, 2 Tx, 1 Rx": [alamouti_ser(value) for value in snr_dB_list],
    "MRC, 1 Tx, 2 Rx": [mrc_ser(value) for value in snr_dB_list],
}

# Figure 2: the diversity order is the slope
fig2, ax = plt.subplots(figsize=(6.5, 5))
for label, ser in curves.items():
    ax.semilogy(snr_dB_list, ser, "o-", label=label)
ax.set_xlabel("SNR [dB]")
ax.set_ylabel("symbol error rate")
ax.set_ylim(1e-5, 1)
ax.grid(True, which="both")
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_alamouti_fig2.png")

# The diversity order is the *asymptotic* slope, so the interesting
# thing is the local slope converging to it. Intervals whose upper end
# rests on a handful of errors are dropped: their slope is noise.
for label, ser in curves.items():
    values = np.array(ser)
    steps = [(np.log10(values[index + 1]) - np.log10(values[index]))
             / ((snr_dB_list[index + 1] - snr_dB_list[index]) / 10)
             for index in range(len(values) - 1) if values[index + 1] > 3e-4]
    print(f"{label:28s} local slope "
          + " ".join(f"{value:+.2f}" for value in steps))

# and the gap Alamouti pays for transmitting blind: about 3 dB
target = 1e-3
gaps = {label: np.interp(np.log10(target), np.log10(ser)[::-1],
                         snr_dB_list[::-1])
        for label, ser in curves.items()}
print(f"SNR needed for SER = {target:g}: "
      + ", ".join(f"{label} {value:.1f} dB" for label, value in gaps.items()))
