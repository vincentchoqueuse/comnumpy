import time

import numpy as np
import matplotlib.pyplot as plt

from comnumpy import monte_carlo
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
def row(label, values):
    """One line of the distance spectrum, six terms wide."""
    line = label
    for value in values[:6]:
        line += f"{value:7d}"
    return line


print(row("d      ", spectrum.distances))
print(row("a_d    ", spectrum.a_d))
print(row("beta_d ", spectrum.beta_d))


def get_uncoded_chain():
    return Sequential([
        SymbolGenerator(2, name="tx"),
        SymbolMapper(BPSK),
        AWGN(snr_dB=0.0, name="noise"),
        SymbolDemapper(BPSK),
    ], taps=["tx"], name="uncoded BPSK")


def get_coded_chain(soft):
    return Sequential([
        SymbolGenerator(2, name="tx"),
        ConvolutionalEncoder((0o133, 0o171)),
        SymbolMapper(BPSK),
        AWGN(snr_dB=0.0, name="noise"),
        SymbolDemapper(BPSK, soft=soft),
        ViterbiDecoder((0o133, 0o171), soft=soft),
    ], taps=["tx"], name=f"K=7 convolutional, {'soft' if soft else 'hard'}")


curves = {}
for label, chain, code_rate in (("uncoded", get_uncoded_chain(), 1.0),
                                ("hard-decision Viterbi", get_coded_chain(False), rate),
                                ("soft-decision Viterbi", get_coded_chain(True), rate)):
    start = time.perf_counter()
    results = monte_carlo(chain, "noise.snr_dB", snr_dB(ebn0_dB, code_rate),
                          {"ber": compute_ser}, n_bits, reference="tx", seed=4)
    curves[label] = results["ber"]
    line = f"{label:24s} "
    for value in results["ber"]:
        line += f"{value:.2e} "
    print(line + f"  ({time.perf_counter() - start:.1f} s)")

bound = union_bound_ber(distance_spectrum((0o133, 0o171), n_terms=8), ebn0_dB)
line = "union bound              "
for value in bound:
    line += f"{value:.2e} "
print(line)

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

n_frames = 40
ldpc_curves = {}
for n_iter in (5, 25):
    link = Sequential([
        SymbolGenerator(2, name="tx"),
        LDPCEncoder(H),
        SymbolMapper(BPSK),
        AWGN(snr_dB=0.0, name="noise"),
        SymbolDemapper(BPSK, soft=True),
        LDPCDecoder(H, n_iter=n_iter),
    ], taps=["tx"], name=f"LDPC, {n_iter} iterations")
    results = monte_carlo(link, "noise.snr_dB", snr_dB(ebn0_dB, ldpc_encoder.rate),
                          {"ber": compute_ser}, (n_frames, ldpc_encoder.k),
                          reference="tx", seed=6)
    ldpc_curves[f"LDPC (2040, {ldpc_encoder.k}), {n_iter} iterations"] = results["ber"]
    line = f"LDPC {n_iter:2d} iterations       "
    for value in results["ber"]:
        line += f"{value:.2e} "
    print(line)

comparison = {"uncoded": curves["uncoded"],
              "soft-decision Viterbi": curves["soft-decision Viterbi"]}
comparison.update(ldpc_curves)
ax = plot_error_rate(ebn0_dB, comparison, xlabel="Eb/N0 [dB]", ylabel="BER",
                     title="Two rate-1/2 codes, and none")
ax.set_ylim(1e-6, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/channel_coding_fig2.png")

mermaid_dir = "../../docs/tutorials/mermaid/"
for diagram_name, diagram_chain in [("channel_coding", get_coded_chain(True))]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
