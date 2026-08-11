"""Regenerate the pyldpc reference decisions next to this file.

Committed for **provenance**, not for continuous integration: it needs
pyldpc installed (``pip install --no-build-isolation pyldpc`` -- its
setup.py imports numpy without declaring it as a build requirement) and
nothing in the test suite imports it.

The channel is regenerated from a seed rather than shipped sample by
sample, which keeps the reference small. That is only safe because the
stream is stable: ``default_rng(SEED).standard_normal`` was checked to
give bit-identical output under numpy 1.26.4 and 2.4.6, the floor and the
newest release this project supports. The manifest carries a hash of the
generated log-likelihood ratios so that a future change of stream fails
loudly instead of quietly comparing two different channels.

Both decoders are given *the same* parity-check matrix and *the same*
LLRs. That is the whole point: the usual obstacle to an external LDPC
reference is that published curves rarely come with the exact matrix
behind them, and sharing the matrix removes the obstacle entirely.
"""
import hashlib
import pathlib
import warnings

import numpy as np
import pyldpc

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from comnumpy.fec.ldpc import LDPCEncoder, make_gallager_parity_check  # noqa: E402

HERE = pathlib.Path(__file__).parent
SEED = 20260811
N_CODE, D_V, D_C, H_SEED = 96, 3, 6, 1
SNR_dB = (1.0, 2.0, 3.0, 4.0)
N_WORDS = 200
MAX_ITER = 50


def channel(rng, encoder, snr_dB):
    """One block of received words and their LLRs, for a given SNR."""
    variance = 10 ** (-snr_dB / 10)
    bits = rng.integers(0, 2, size=(N_WORDS, encoder.k))
    codewords = np.asarray(encoder(bits))
    received = (1.0 - 2.0 * codewords) + np.sqrt(variance) * rng.normal(
        size=codewords.shape)
    return codewords, received, 2 * received / variance


def main():
    warnings.filterwarnings("ignore")
    H = make_gallager_parity_check(N_CODE, d_v=D_V, d_c=D_C, seed=H_SEED)
    encoder = LDPCEncoder(H)

    rows = ["".join(str(int(bit)) for bit in row) for row in H]
    (HERE / "ldpc_parity_check.csv").write_text(
        "\n".join(["parity_check_row"] + rows) + "\n")

    rng = np.random.default_rng(SEED)
    digest = hashlib.sha256()
    lines = ["snr_dB,word,transmitted,pyldpc_decoded"]
    summary = []
    for snr_dB in SNR_dB:
        codewords, received, llr = channel(rng, encoder, snr_dB)
        digest.update(np.ascontiguousarray(llr, dtype=np.float64).tobytes())
        decoded = np.asarray(
            pyldpc.decode(H, received.T, snr_dB, maxiter=MAX_ITER)).T
        for index in range(N_WORDS):
            lines.append("%.1f,%d,%s,%s" % (
                snr_dB, index,
                "".join(str(int(b)) for b in codewords[index]),
                "".join(str(int(b)) for b in decoded[index])))
        ber = float(np.mean(decoded != codewords))
        summary.append((snr_dB, ber))
        print(f"snr {snr_dB:4.1f} dB   pyldpc BER {ber:.5f}")
    (HERE / "ldpc_reference.csv").write_text("\n".join(lines) + "\n")

    (HERE / "ldpc_manifest.csv").write_text(
        "key,value\n"
        f"seed,{SEED}\n"
        f"n_code,{N_CODE}\nd_v,{D_V}\nd_c,{D_C}\nh_seed,{H_SEED}\n"
        f"n_words,{N_WORDS}\nmax_iter,{MAX_ITER}\n"
        f"llr_sha256,{digest.hexdigest()}\n"
        f"pyldpc_version,{pyldpc.__version__ if hasattr(pyldpc, '__version__') else '0.7.9'}\n"
        + "".join(f"pyldpc_ber_{snr:.1f},{ber:.6f}\n" for snr, ber in summary))
    print("written:", sorted(p.name for p in HERE.glob("ldpc_*")))


if __name__ == "__main__":
    main()
