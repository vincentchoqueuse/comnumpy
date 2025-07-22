import numpy as np
import matplotlib.pyplot as plt
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.core.processors import Upsampler, Downsampler, Amplifier
from comnumpy.core.filters import SRRCFilter, BWFilter
from comnumpy.core.metrics import compute_ser
from comnumpy.optical.links import FiberLink
from comnumpy.optical.dbp import DBP

img_dir = "../../docs/examples/img/"

# parameters
M = 16
modulation = "QAM"
alphabet = get_alphabet(modulation, M)
N_s = 2**9  # number of symbols
oversampling_sim = 6  # number of samples/symbol for simulated channel
oversampling_dsp = 2  # number of samples/symbol for dsp processing
NF_dB = 5  # noise figure in dB
rolloff = 0.1
StPS = 500
StPS_DBP = 100
R_s = 10.7*(10**9)  # baud rate
L_span = 80  # in km
N_span = 25
dBm = -3

# compute simulation parameter
N = N_s*oversampling_sim   # signal length in number of samples
fs = R_s*oversampling_sim
oversampling_ratio = int(oversampling_sim/oversampling_dsp)
fs2 = R_s*oversampling_dsp
Po = (1e-3) * (10**(0.1*dBm))
amp = np.sqrt(Po)

# create a communication chain
chain = Sequential([
                SymbolGenerator(M),
                Recorder(name="data_tx"),
                SymbolMapper(alphabet),
                Recorder(name="signal_tx"),
                Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
                SRRCFilter(rolloff, oversampling_sim, method="fft"),
                Amplifier(amp),
                FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB, fs=fs, name="link"),
                BWFilter(1/oversampling_sim),
                Downsampler(oversampling_ratio)
                ])

# perform simulation
y_rx = chain(N)

# extract signal
s_tx = chain["data_tx"].get_data()
x_tx = chain["signal_tx"].get_data()

# plot signal
plt.figure()
plt.plot(np.real(y_rx), np.imag(y_rx), ".")
plt.title(f"Received Signal (oversampling={oversampling_dsp}, {dBm}dBm)")
plt.savefig(f"{img_dir}/one_shot_NLI_fig1.png")

# perform compensation
for num_compensator in range(2):

    if num_compensator == 0:
        use_only_linear = True
        technique_name = "linear equalization"
    else:
        use_only_linear = False
        technique_name = "nonlinear equalization"

    receiver = Sequential([
                    DBP(N_span, L_span, StPS_DBP, step_type="linear", use_only_linear=use_only_linear, fs=fs/oversampling_ratio, name="dbp"),
                    SRRCFilter(rolloff, oversampling_dsp, method="fft", scale=1/np.sqrt(oversampling_dsp)),
                    Downsampler(oversampling_dsp),
                    Amplifier(1/amp),
                    Recorder(name="signal_rx_compensated")
            ])

    # apply conventional chain
    x_rx = receiver(y_rx)

    # perform phase correction and evaluate metric
    theta_est = np.angle(np.sum(np.conj(x_rx)*x_tx))
    x_rx_phase_compensated = np.exp(1j*theta_est) * x_rx
    s_rx = hard_projector(x_rx_phase_compensated, alphabet)
    ser = compute_ser(s_tx, s_rx)

    # plot signal
    plt.figure()
    plt.plot(np.real(x_rx), np.imag(x_rx), ".", label="before phase correction")
    plt.plot(np.real(x_rx_phase_compensated), np.imag(x_rx_phase_compensated), ".", label="after phase correction")
    plt.legend()
    plt.title(f"{technique_name} (SER={ser:.3f})")
    plt.savefig(f"{img_dir}/one_shot_NLI_fig{num_compensator+2}.png")

plt.show()
