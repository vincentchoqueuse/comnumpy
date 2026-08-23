import numpy as np
import matplotlib.pyplot as plt

from comnumpy import monte_carlo, print_data
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.fec import (ConvolutionalEncoder, LDPCDecoder, LDPCEncoder,
                          ViterbiDecoder)
from comnumpy.fec.analysis import distance_spectrum, union_bound_ber
from comnumpy.fec.ldpc import make_gallager_parity_check
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

BPSK = np.array([1.0 + 0.0j, -1.0 + 0.0j])   # bit 0 -> +1, bit 1 -> -1
ebn0_dB = np.arange(0.0, 8.0, 1.0)
n_bits = 40000
rate = 0.5                                   # both codes below are rate 1/2


def snr_dB(ebn0, code_rate):
    """Es/N0 in dB for BPSK: one bit per symbol, code_rate of them useful."""
    return ebn0 + 10 * np.log10(code_rate)


encoder = ConvolutionalEncoder((0o133, 0o171))
generators = []
for value in encoder.g:
    generators.append(oct(value))
print(f"generators {tuple(generators)}  "
      f"K = {encoder.K}  states = {2 ** (encoder.K - 1)}  "
      f"rate = {encoder.rate}")
print("4 bits in ->", encoder(np.array([1, 0, 1, 1])), "(with the tail)")

spectrum = distance_spectrum((0o133, 0o171), n_terms=4)
print(f"free distance d_free = {spectrum.d_free}")
print_data({"x": spectrum.distances[:6],
            "curves": {"a_d": spectrum.a_d[:6],
                       "beta_d": spectrum.beta_d[:6]}},
           xlabel="d", ylabel="distance spectrum")


# --- three chains, three sweeps --------------------------------------
# One Sequential per scheme, written out; monte_carlo moves the noise.
uncoded = Sequential([
    SymbolGenerator(2, name="tx"),
    SymbolMapper(BPSK),
    AWGN(snr_dB=0.0, name="noise"),
    SymbolDemapper(BPSK),
], observations=["tx"], name="uncoded BPSK")

hard = Sequential([
    SymbolGenerator(2, name="tx"),
    ConvolutionalEncoder((0o133, 0o171)),
    SymbolMapper(BPSK),
    AWGN(snr_dB=0.0, name="noise"),
    SymbolDemapper(BPSK),
    ViterbiDecoder((0o133, 0o171)),
], observations=["tx"], name="K=7 convolutional, hard")

soft = Sequential([
    SymbolGenerator(2, name="tx"),
    ConvolutionalEncoder((0o133, 0o171)),
    SymbolMapper(BPSK),
    AWGN(snr_dB=0.0, name="noise"),
    SymbolDemapper(BPSK, soft=True),
    ViterbiDecoder((0o133, 0o171), soft=True),
], observations=["tx"], name="K=7 convolutional, soft")

curves = {}
curves["uncoded"] = monte_carlo(
    uncoded, "noise.snr_dB", snr_dB(ebn0_dB, 1.0), {"ber": compute_ser},
    n_bits, reference="tx", seed=4)["ber"]
curves["hard-decision Viterbi"] = monte_carlo(
    hard, "noise.snr_dB", snr_dB(ebn0_dB, rate), {"ber": compute_ser},
    n_bits, reference="tx", seed=4)["ber"]
curves["soft-decision Viterbi"] = monte_carlo(
    soft, "noise.snr_dB", snr_dB(ebn0_dB, rate), {"ber": compute_ser},
    n_bits, reference="tx", seed=4)["ber"]

# --- results: table ---------------------------------------------------
print()
print_data({"x": ebn0_dB, "curves": curves},
           xlabel="Eb/N0 [dB]", ylabel="BER")

# --- the union bound, against the soft measurement -------------------
bound = union_bound_ber(distance_spectrum((0o133, 0o171), n_terms=8), ebn0_dB)
print()
print_data({"x": ebn0_dB,
            "curves": {"soft-decision Viterbi": curves[
                           "soft-decision Viterbi"],
                       "union bound": bound}},
           xlabel="Eb/N0 [dB]", ylabel="BER")

ax = plot_error_rate(ebn0_dB, curves, theory={"soft-decision Viterbi": bound},
                     x_theory=ebn0_dB, xlabel="Eb/N0 [dB]", ylabel="BER",
                     title="Rate-1/2 convolutional code, K = 7")
ax.set_ylim(1e-6, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/channel_coding_fig1.png")

H = make_gallager_parity_check(2040, d_v=3, d_c=6, seed=1)
ldpc_encoder = LDPCEncoder(H)
print(f"\nLDPC: H is {H.shape[0]} x {H.shape[1]}, k = {ldpc_encoder.k} "
      f"information bits, rate = {ldpc_encoder.rate:.3f}, "
      f"column weight {H.sum(axis=0)[0]}, row weight {H.sum(axis=1)[0]}")

# One chain; the iteration count is a parameter, so the second sweep
# reconfigures the decoder with set_params instead of rebuilding.
n_frames = 40
ldpc = Sequential([
    SymbolGenerator(2, name="tx"),
    LDPCEncoder(H),
    SymbolMapper(BPSK),
    AWGN(snr_dB=0.0, name="noise"),
    SymbolDemapper(BPSK, soft=True),
    LDPCDecoder(H, n_iter=5, name="decoder"),
], observations=["tx"], name="LDPC")

ldpc_curves = {}
ldpc_curves[f"LDPC (2040, {ldpc_encoder.k}), 5 iterations"] = monte_carlo(
    ldpc, "noise.snr_dB", snr_dB(ebn0_dB, ldpc_encoder.rate),
    {"ber": compute_ser}, (n_frames, ldpc_encoder.k),
    reference="tx", seed=6)["ber"]
ldpc.set_params(decoder__n_iter=25)
ldpc_curves[f"LDPC (2040, {ldpc_encoder.k}), 25 iterations"] = monte_carlo(
    ldpc, "noise.snr_dB", snr_dB(ebn0_dB, ldpc_encoder.rate),
    {"ber": compute_ser}, (n_frames, ldpc_encoder.k),
    reference="tx", seed=6)["ber"]

print()
print_data({"x": ebn0_dB, "curves": ldpc_curves},
           xlabel="Eb/N0 [dB]", ylabel="BER")

comparison = {"uncoded": curves["uncoded"],
              "soft-decision Viterbi": curves["soft-decision Viterbi"]}
comparison.update(ldpc_curves)
ax = plot_error_rate(ebn0_dB, comparison, xlabel="Eb/N0 [dB]", ylabel="BER",
                     title="Two rate-1/2 codes, and none")
ax.set_ylim(1e-6, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/channel_coding_fig2.png")


