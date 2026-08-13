"""Polarization demultiplexing of a PDM-QPSK link: CMA, then BPS.

Run from this directory: it writes the tutorial's figures into
../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import style
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.compensators import BlindPhaseSearchCompensator
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.processors import Upsampler
from comnumpy.core.utils import Constellation, hard_projector
from comnumpy.core.visualizers import plot_iq
from comnumpy.mimo.compensators import BlindDualMIMOCompensator
from comnumpy.optical.channels import PhaseNoise, PMDEmulator

style.use()

img_dir = "../../docs/tutorials/img/"

# parameters
constellation = Constellation("PSK", 4)
N = 2 ** 16               # symbols per polarization
oversampling = 2
rolloff = 0.1
R_s = 32e9                # baud rate
fs = R_s * oversampling
linewidth = 100e3         # laser linewidth, Hz
dgd = 10e-12              # total mean DGD, s -- a third of the symbol
snr_dB = 15
n_discard = 4000          # symbols spent converging the equalizer

# The channel: two polarizations shaped and sent through one laser's
# phase noise, the fibre's random rotation and first-order PMD, and the
# amplifier noise. The phase walk is common to the two rows -- one
# laser -- while the PMD mixes them into each other.
chain = Sequential([
    SymbolGenerator(constellation.order, name="data_tx"),
    SymbolMapper(constellation, name="signal_tx"),
    Upsampler(oversampling, scale=np.sqrt(oversampling)),
    SRRCFilter(rolloff, oversampling, method="fft"),
    PhaseNoise(2 * np.pi * linewidth / fs, name="laser"),
    PMDEmulator(dgd, n_sections=8, fs=fs, seed=11, name="fibre"),
    AWGN(snr_dB=snr_dB, name="noise"),
    ], taps=["data_tx", "signal_tx", "noise"])
chain.seed(0)
received = chain((2, N))
sent = chain.tap("signal_tx")
reference = chain.tap("data_tx")

# What the receiver faces, sampled naively at the symbol rate: the two
# polarizations are mixed by an unknown rotation, delayed against each
# other, and the constellation spins with the laser.
naive = received[:, ::oversampling]

# The 2x2 butterfly equalizer, blind (CMA): it inverts the rotation and
# the DGD without knowing either, from the constant modulus of QPSK
# alone. The first symbols are its convergence and are discarded.
cma = BlindDualMIMOCompensator(L=9, alphabet=constellation, mu=1e-3,
                               oversampling=oversampling, name="cma")
equalized = cma(received)[:, n_discard:]

# Feedforward carrier recovery (blind phase search): each polarization
# gets its own phase trajectory, estimated modulo pi/2.
bps = BlindPhaseSearchCompensator(constellation, name="bps")
recovered = bps(equalized)

# --- results: the ambiguities, then the error rate ---------------------
# A blind receiver leaves three things unresolved: which output is
# which polarization, a quadrant rotation per output, and the small
# group delay of the equalizer. A system resolves them with framing;
# here the known data plays that role, explicitly, over a short probe.
probe = 2000
detected_rows = []
truth_rows = []
for row in range(2):
    best = None
    for source in range(2):
        for quadrant in range(4):
            for lag in range(-8, 9):
                start = n_discard + lag
                if start < 0:
                    continue
                candidate = recovered[row, :probe] * (1j) ** quadrant
                target = sent[source, start:start + probe]
                error = float(np.mean(np.abs(candidate - target) ** 2))
                if best is None or error < best[0]:
                    best = (error, source, quadrant, lag)
    assert best is not None
    _, source, quadrant, lag = best
    print(f"output {row}: polarization {source}, rotated "
          f"{90 * quadrant} deg, delayed {lag} symbols")
    aligned = recovered[row] * (1j) ** quadrant
    decisions, _ = hard_projector(aligned, constellation)
    detected_rows.append(decisions)
    length = decisions.shape[-1]
    truth_rows.append(reference[source, n_discard + lag:
                                n_discard + lag + length])

detected = np.stack(detected_rows)
truth = np.stack(truth_rows)
ser = compute_ser(truth, detected)
print(f"SER after CMA + BPS: {ser:.2e} over {truth.size} symbols "
      f"({int(round(float(ser) * truth.size))} errors)")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
plot_iq(naive[0, n_discard:], title="received, polarization X", ax=axes[0])
plot_iq(equalized[0], title="after CMA", ax=axes[1])
plot_iq(recovered[0], reference=constellation,
        title="after BPS", ax=axes[2])
plt.tight_layout()
plt.savefig(f"{img_dir}/one_shot_pdm_fig1.png")
