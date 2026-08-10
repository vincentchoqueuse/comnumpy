import time

import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.core.utils import get_alphabet
from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.gn_model import (gn_model_nli_power, gn_model_snr,
                                       optimal_launch_power)
from comnumpy.optical.links import FiberLink

img_dir = "../../docs/examples/img/"

# The link. Standard single-mode fibre, five 100 km spans, one erbium
# amplifier per span putting back exactly what the span took away.
SMF = FiberSpec(0.2, gamma=1.3, cd_coefficient=17.0, wavelength_nm=1550.0)
BAUD = 32e9
SPAN_KM = 100.0
N_SPANS = 5
NF_dB = 6.0

# The waveform. Two polarizations, 16QAM on each, root-raised-cosine
# pulses at four samples per symbol.
OS = 4
N_SYM = 4096
ROLLOFF = 0.1
FS = BAUD * OS
ALPHABET = get_alphabet("QAM", 16)


def transmitter(power_W):
    """Symbols in, waveform out, at the requested launch power.

    ``power_W`` is the power of the channel, i.e. the *sum* over the two
    polarizations -- the convention the GN model uses -- so each
    polarization carries half of it. Getting this wrong is a 3 dB error
    in the launch power and therefore a 9 dB error in the nonlinear
    interference, which is the single easiest way to make a correct
    model look broken.
    """
    return Sequential([
        Upsampler(OS, scale=np.sqrt(OS)),
        SRRCFilter(ROLLOFF, OS, N_h=40, method="fft"),
        Amplifier(np.sqrt(power_W / 2)),
    ], name="transmitter")


def receiver(power_W, name="receiver"):
    """Waveform in, symbols out. The only equalizer is linear."""
    return Sequential([
        DBP(N_SPANS, L_span=SPAN_KM, StPS=1, fs=FS, fiber=SMF,
            use_only_linear=True),
        SRRCFilter(ROLLOFF, OS, N_h=40, method="fft", scale=1 / np.sqrt(OS)),
        Downsampler(OS),
        Amplifier(1 / np.sqrt(power_W / 2)),
    ], name=name)


def link(*, kerr, noise, StPS=40):
    return FiberLink(N_SPANS, L_span=SPAN_KM, StPS=StPS, fs=FS, fiber=SMF,
                     use_only_linear=not kerr,
                     noise_scaling=1.0 if noise else 0.0, NF_dB=NF_dB,
                     seed=7, name="fibre")


def symbols(seed):
    """PM-16QAM: one independent stream per polarization."""
    rng = np.random.default_rng(seed)
    return ALPHABET[rng.integers(0, ALPHABET.size, (2, N_SYM))]


def noise_to_signal(received, reference):
    """Everything that is not a scaled copy of the reference.

    One complex scalar absorbs the gain and the mean nonlinear phase --
    which is what a carrier-phase estimator removes in a real receiver --
    and what is left is noise, whatever produced it.
    """
    gain = np.vdot(reference, received) / np.vdot(reference, reference)
    residual = received - gain * reference
    return float(np.mean(np.abs(residual) ** 2)
                 / (abs(gain) ** 2 * np.mean(np.abs(reference) ** 2)))


def run(power_W, *, kerr, noise, seed=0):
    sent = symbols(seed)
    waveform = transmitter(power_W)(sent)
    return receiver(power_W)(link(kerr=kerr, noise=noise)(waveform))


# 1. What the closed form says, before a single sample is propagated.
# The GN model turns the whole link into one number: eta, the
# coefficient of the cubic term in P_NLI = eta * P^3.
nli_at_1mW = gn_model_nli_power(SMF, span_length_km=SPAN_KM, n_spans=N_SPANS,
                                powers_W=np.array([1e-3]),
                                frequencies_Hz=np.array([SMF.carrier_frequency_Hz]),
                                baud_rates_Hz=np.array([BAUD]))[0]
eta = nli_at_1mW / (1e-3) ** 3
print(f"GN model: eta = {eta:.4g} /W^2 "
      f"({10 * np.log10(eta / 1e6):+.2f} dB in mW^-2)")

# 2. The amplifiers' share. The GN model predicts the fibre's noise; the
# amplifiers' is not its business, so we measure it once -- it does not
# depend on the launch power.
reference = run(1e-3, kerr=False, noise=False)
ase_ratio = noise_to_signal(run(1e-3, kerr=False, noise=True), reference)
ase_W = ase_ratio * 1e-3
print(f"measured ASE over {N_SPANS} spans at NF = {NF_dB:.0f} dB: "
      f"{10 * np.log10(ase_W / 1e-3):+.2f} dBm")

# 3. The design point, in closed form: the optimum is where the fibre's
# noise is half the amplifiers'.
best_power, best_snr = optimal_launch_power(ase_W, eta)
print(f"optimum: {10 * np.log10(best_power / 1e-3):+.2f} dBm, "
      f"SNR = {10 * np.log10(best_snr):.2f} dB")
print(f"check: eta*P^3 / (P_ASE/2) = {eta * best_power ** 3 / (ase_W / 2):.6f}")

# 4. And now the expensive way: propagate. Each launch power needs the
# fibre integrated once, which is why this is a `for` loop and not a
# `sweep` -- sweep drives one chain and reads one metric, while every
# point here is a comparison between a run and a noiseless reference.
powers_dBm = np.arange(-8.0, 5.1, 1.0)
measured_snr_dB = []
start = time.perf_counter()
for power_dBm in powers_dBm:
    power_W = 1e-3 * 10 ** (power_dBm / 10)
    ratio = noise_to_signal(run(power_W, kerr=True, noise=True), reference)
    measured_snr_dB.append(-10 * np.log10(ratio))
    print(f"  {power_dBm:+5.1f} dBm -> SNR {measured_snr_dB[-1]:5.2f} dB")
measured_snr_dB = np.array(measured_snr_dB)
print(f"({time.perf_counter() - start:.0f} s of split-step propagation)")

fine_powers = np.logspace(-1.0, 0.6, 400) * 1e-3
predicted_snr_dB = 10 * np.log10(gn_model_snr(ase_W, eta, fine_powers))
best_measured = powers_dBm[int(np.argmax(measured_snr_dB))]
print(f"optimum: {10 * np.log10(best_power / 1e-3):+.2f} dBm predicted, "
      f"{best_measured:+.1f} dBm measured; peak SNR "
      f"{10 * np.log10(best_snr):.2f} dB predicted, "
      f"{np.max(measured_snr_dB):.2f} dB measured")

_, ax = plt.subplots(figsize=(7, 5), layout="constrained")
ax.plot(10 * np.log10(fine_powers / 1e-3), predicted_snr_dB, "-",
        label="GN model (closed form, microseconds)")
ax.plot(powers_dBm, measured_snr_dB, "o",
        label="split-step simulation (minutes)")
ax.plot(10 * np.log10(fine_powers / 1e-3),
        10 * np.log10(fine_powers / ase_W), ":", color="0.5",
        label="amplifiers alone ($P/P_{ASE}$)")
ax.axvline(10 * np.log10(best_power / 1e-3), color="0.3", linestyle="--",
           linewidth=1)
ax.annotate(f"optimum {10 * np.log10(best_power / 1e-3):+.1f} dBm",
            (10 * np.log10(best_power / 1e-3), np.min(predicted_snr_dB) + 1),
            rotation=90, va="bottom", ha="right", color="0.3")
ax.set_xlabel("launch power per channel [dBm]")
ax.set_ylabel("SNR [dB]")
ax.set_title(f"PM-16QAM, {BAUD/1e9:.0f} GBd, {N_SPANS} x {SPAN_KM:.0f} km SMF")
ax.set_ylim(np.min(measured_snr_dB) - 2, np.max(predicted_snr_dB) + 2)
ax.legend()
ax.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/gn_model_fig1.png")

# 5. What the closed form buys: questions the simulation cannot afford.
# Filling the amplifier band costs far less than it looks -- the asinh
# again.
print("\nchannels   NLI at 0 dBm   optimum   peak SNR")
counts = [1, 3, 9, 27, 81]
curves = {}
for n_channels in counts:
    offsets = 50e9 * (np.arange(n_channels) - (n_channels - 1) / 2)
    comb_nli = gn_model_nli_power(
        SMF, span_length_km=SPAN_KM, n_spans=N_SPANS,
        powers_W=np.full(n_channels, 1e-3),
        frequencies_Hz=SMF.carrier_frequency_Hz + offsets,
        baud_rates_Hz=np.full(n_channels, BAUD))[n_channels // 2]
    comb_eta = comb_nli / (1e-3) ** 3
    power, snr = optimal_launch_power(ase_W, comb_eta)
    curves[n_channels] = 10 * np.log10(
        gn_model_snr(ase_W, comb_eta, fine_powers))
    print(f"{n_channels:8d}   {10*np.log10(comb_nli/1e-3):+8.2f} dBm   "
          f"{10*np.log10(power/1e-3):+6.2f} dBm   {10*np.log10(snr):6.2f} dB")

_, ax = plt.subplots(figsize=(7, 5), layout="constrained")
for n_channels in counts:
    ax.plot(10 * np.log10(fine_powers / 1e-3), curves[n_channels],
            label=f"{n_channels} channels")
ax.set_xlabel("launch power per channel [dBm]")
ax.set_ylabel("SNR [dB]")
ax.set_title("Filling the band, in closed form only")
ax.legend()
ax.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/gn_model_fig2.png")

# 6. The one thing the model cannot see. It assumes the signal is
# Gaussian; 16QAM is not, and QPSK is less so. Same link, same power,
# three stimuli, one prediction.
def a_nl_dB(draw, power_W=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    sent = draw(rng)
    waveform = transmitter(power_W)(sent)
    linear = receiver(power_W)(link(kerr=False, noise=False)(waveform))
    kerr = receiver(power_W)(link(kerr=True, noise=False)(waveform))
    ratio = noise_to_signal(kerr, linear)
    return 10 * np.log10(ratio / power_W ** 2 / 1e6)


def gaussian_draw(rng):
    return (rng.standard_normal((2, N_SYM))
            + 1j * rng.standard_normal((2, N_SYM))) / np.sqrt(2)


def qam_draw(order):
    alphabet = get_alphabet("QAM", order)
    return lambda rng: alphabet[rng.integers(0, alphabet.size, (2, N_SYM))]


model_dB = 10 * np.log10(eta / 1e6)
print(f"\nThe GN model predicts a_NL = {model_dB:+.2f} dB whatever is "
      f"modulated onto the carrier.")
print("stimulus    measured a_NL   vs the model   vs Gaussian")
measured_a_nl = {}
for label, draw in (("Gaussian", gaussian_draw), ("16QAM", qam_draw(16)),
                    ("QPSK", qam_draw(4))):
    measured_a_nl[label] = float(
        np.mean([a_nl_dB(draw, seed=seed) for seed in range(3)]))
    print(f"{label:10s}  {measured_a_nl[label]:+11.2f} dB   "
          f"{measured_a_nl[label] - model_dB:+9.2f} dB   "
          f"{measured_a_nl[label] - measured_a_nl['Gaussian']:+8.2f} dB")
print("A Gaussian stimulus lands on the model, as it must. Every real "
      "constellation lands")
print("below it, and the smaller the constellation the further below -- "
      "that is the EGN")
print("effect, and this library measures it rather than predicting it.")

# The chain diagram is exported from the chain itself (D33c), so the
# picture in the tutorial cannot drift from the code.
mermaid_dir = "../../docs/examples/mermaid/"
diagram = Sequential([*transmitter(1e-3).module_list,
                      link(kerr=True, noise=True),
                      *receiver(1e-3).module_list], name="GN model link")
with open(f"{mermaid_dir}/gn_model.mmd", "w") as stream:
    stream.write(diagram.to_mermaid())

plt.show()
