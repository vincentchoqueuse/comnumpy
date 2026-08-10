import numpy as np
import matplotlib.pyplot as plt
import time
from comnumpy.core import Sequential
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser
from comnumpy.ofdm.chains import OFDMTransmitter, OFDMReceiver

img_dir = "../../docs/examples/img/"

# parameters
M = 16
N_h = 5
N = 1280
sigma2 = 0.015
alphabet = get_alphabet("QAM", M)

# A frequency-selective channel, drawn from an exponential power delay
# profile -- the standard multipath model -- and seeded, because the
# whole comparison below is about one realization. Its frequency
# response has a deep notch (|H| spans a factor 31), which is what
# "frequency selective" means and what the two receivers will disagree
# about.
rng = np.random.default_rng(18)
power = np.exp(-np.arange(N_h) / 2.0)
power = power / np.sum(power)
h = np.sqrt(power / 2) * (rng.standard_normal(N_h)
                          + 1j * rng.standard_normal(N_h))

# create a simple single carrier chain and simulate
simple_chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        FIRChannel(h),
        AWGN(sigma2=sigma2, name="data_rx"),
        LinearEqualizer(h, method="zf", name="data_rx_eq"),
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx", "data_rx_eq"])

simple_chain.seed(1)
start_time = time.time()
s_rx = simple_chain(N)
stop_time = time.time()

# extract signals, compute ser and elapsed time
s_tx = simple_chain.tap("data_tx")
ser = compute_ser(s_tx, s_rx)
elapsed_time = stop_time - start_time
print(f"SER: {ser}")
print(f"elapsed time: {elapsed_time} s")

# plot signal and save
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
for indice, processor_name in enumerate(["data_rx", "data_rx_eq"]):
    data_rx = simple_chain.tap(processor_name)
    axes[indice].plot(np.real(data_rx), np.imag(data_rx), ".")
    axes[indice].set_title(f"Received signal ({processor_name})")
    axes[indice].set_aspect("equal", adjustable="box")
    axes[indice].set_xlim([-2, 2])
    axes[indice].set_ylim([-2, 2])

plt.savefig(f"{img_dir}/one_shot_ofdm_fig1.png")

# create an OFDM chain and simulate
N_carrier = 128
N_cp = 10
ofdm_chain = Sequential([
        SymbolGenerator(M, name="data_tx"),
        SymbolMapper(alphabet),
        OFDMTransmitter(N_carrier, N_cp),   # <- add OFDM transmitter
        FIRChannel(h),
        AWGN(sigma2=sigma2),
        OFDMReceiver(N_carrier, N_cp, h=h, name="data_rx"), # <- add OFDM receiver
        SymbolDemapper(alphabet)
    ], taps=["data_tx", "data_rx"])

ofdm_chain.seed(1)
start_time = time.time()
s_rx = ofdm_chain(N)
stop_time = time.time()

# extract signals, compute ser and elapsed time
s_tx = ofdm_chain.tap("data_tx")
data_rx = ofdm_chain.tap("data_rx")
ser = compute_ser(s_tx, s_rx)
elapsed_time = stop_time - start_time
print(f"SER: {ser}")
print(f"elapsed time: {elapsed_time} s")

# plot signal and save
plt.figure()
plt.plot(np.real(data_rx), np.imag(data_rx), ".")
plt.title("OFDM Chain: received data")
plt.savefig(f"{img_dir}/one_shot_ofdm_fig2.png")

# The ranking is not a property of the two schemes, it is a property of
# the operating point -- so the two chains are run again over a range of
# noise variances, and the crossing is printed rather than asserted.
print("sigma2     single carrier      OFDM     |H| spans "
      f"{np.max(np.abs(np.fft.fft(h, N_carrier))) / np.min(np.abs(np.fft.fft(h, N_carrier))):.0f}")
for variance in [0.015, 0.008, 0.004, 0.002]:
    row = []
    for chain, block in ((simple_chain, "data_rx"), (ofdm_chain, "awgn")):
        chain.seed(1)
        chain.set_params(**{f"{block}.sigma2": variance})
        detected = chain(N)
        row.append(compute_ser(chain.tap("data_tx"), detected))
    print(f"{variance:6.3f}   {row[0]:16.4f} {row[1]:9.4f}")

# The chain diagrams this tutorial shows are exported from the chains
# themselves (D33c), so the picture cannot drift from the code -- the
# smoke test compares what a run writes with what the page displays.
mermaid_dir = "../../docs/examples/mermaid/"
for diagram_name, diagram_chain in [("ofdm_single_carrier", simple_chain),
        ("ofdm_chain", ofdm_chain),
        ("ofdm_transmitter", ofdm_chain[2].chain),
        ("ofdm_receiver", ofdm_chain[5].chain)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
