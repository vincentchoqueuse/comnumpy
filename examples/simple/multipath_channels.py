import numpy as np
import matplotlib.pyplot as plt

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN, TappedDelayLineChannel
from comnumpy.core.fading import (available_delay_profiles, get_delay_profile,
                                  validate_taps_fit)
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import get_alphabet
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.ofdm.chains import OFDMReceiver, OFDMTransmitter

img_dir = "../../docs/examples/img/"

# Parameters
fs = 15.36e6                  # LTE 15.36 MHz: 65 ns per sample
M = 16
alphabet = get_alphabet("QAM", M)
spread_ns = 300.0             # the scenario supplies the delay spread
profiles = ["TDL-A", "TDL-C", "TDL-D", "EVA", "ETU"]

# 1. The catalog. A TR 38.901 model is a *shape*: its delays are
# normalized, so the same table serves an indoor 30 ns scenario and an
# outdoor 1 us one. The LTE profiles are absolute instead, in ns.
print(f"{'profile':8s} {'taps':>5s} {'rms [ns]':>9s} {'max [ns]':>9s} "
      f"{'K [dB]':>7s}  taps at 15.36 MHz")
for name in profiles:
    kwargs = {"delay_spread_ns": spread_ns} if name.startswith("TDL") else {}
    profile = get_delay_profile(name, **kwargs)
    delays, _ = profile.to_taps(fs)
    rice = "-" if profile.rice_k_dB is None else f"{profile.rice_k_dB:.1f}"
    print(f"{name:8s} {profile.n_taps:5d} {profile.rms_delay_spread_ns:9.1f} "
          f"{profile.delays_ns[-1]:9.1f} {rice:>7s}  {delays.size:d} "
          f"(longest at sample {delays[-1]})")
print("catalog:", ", ".join(available_delay_profiles()))

fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(12, 3.6), sharey=True)
for ax, name in zip(axes1, ["TDL-A", "TDL-C", "TDL-D"], strict=True):
    profile = get_delay_profile(name, delay_spread_ns=spread_ns)
    ax.stem(profile.delays_ns * 1e-3, 10 * np.log10(profile.powers_lin),
            basefmt=" ")
    ax.set_title(f"{name}, {profile.n_taps} taps")
    ax.set_xlabel("delay [us]")
    ax.grid(True)
axes1[0].set_ylabel("relative power [dB]")
axes1[0].set_ylim(-40, 2)
plt.tight_layout()
plt.savefig(f"{img_dir}/multipath_fig1.png")

# 2. What a delay spread does to the frequency response. The channel is
# a chain block like any other: give it a profile and a sampling rate.
fig2, ax2 = plt.subplots(figsize=(7, 4))
print("\ndelay spread -> coherence bandwidth")
for spread in (100.0, 300.0, 1000.0):
    profile = get_delay_profile("TDL-C", delay_spread_ns=spread)
    channel = TappedDelayLineChannel(profile, fs=fs, seed=3)
    channel(np.ones(4096, dtype=complex))       # one block-fading draw
    taps = np.zeros(int(channel.delays_[-1]) + 1, dtype=complex)
    taps[channel.delays_] = channel.h_[:, 0]
    response = np.abs(np.fft.fftshift(np.fft.fft(taps, 1024))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(1024, 1 / fs)) * 1e-6
    ax2.plot(frequencies, 10 * np.log10(response),
             label=f"RMS spread {spread:.0f} ns")
    # The frequency correlation is the Fourier transform of the power
    # delay profile, R(df) = sum_l gamma_l exp(-j 2 pi df tau_l), so the
    # coherence bandwidth is a property of the profile and not of one
    # draw. Convention here: the width at which the envelope
    # correlation |R|^2 has fallen to one half.
    grid = np.linspace(0, fs / 2, 8192)
    correlation = np.abs(np.exp(-2j * np.pi * np.outer(
        grid, profile.delays_ns * 1e-9)) @ profile.powers_lin) ** 2
    crossing = np.flatnonzero(correlation < 0.5 * correlation[0])
    coherence = grid[crossing[0]]
    print(f"  {spread:6.0f} ns   B_c = {coherence * 1e-6:5.2f} MHz   "
          f"B_c x sigma = {coherence * spread * 1e-9:.3f}   "
          f"1/(5 sigma) = {1 / (5 * spread * 1e-9) * 1e-6:5.2f} MHz")
ax2.set_xlabel("frequency [MHz]")
ax2.set_ylabel("|H(f)|^2 [dB]")
ax2.set_title("TDL-C: the same shape, three delay spreads")
ax2.set_ylim(-40, 15)
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig(f"{img_dir}/multipath_fig2.png")

# 3. What it costs an OFDM link, and what the cyclic prefix is for.
# The prefix turns the linear convolution into a circular one -- but
# only if it is longer than the channel. Shorter, and the previous
# symbol leaks in: an error floor no SNR removes.
N_carrier = 128
N_symbols = 400
N = N_carrier * N_symbols
snr_dB_list = np.arange(0, 31, 5)
profile = get_delay_profile("TDL-C", delay_spread_ns=spread_ns)
longest = int(np.rint(profile.delays_ns[-1] * 1e-9 * fs))
print(f"\nTDL-C at {spread_ns:.0f} ns: longest path at sample {longest}")

curves = {}
for N_cp in (longest + 8, longest // 4):
    link = Sequential([
        SymbolGenerator(M, name="tx"),
        SymbolMapper(alphabet),
        OFDMTransmitter(N_carrier, N_cp),
        TappedDelayLineChannel(profile, fs=fs, name="fading"),
        AWGN(snr_dB=20.0, name="noise"),
    ], taps=["tx"], name=f"OFDM, CP = {N_cp}")
    validate_taps_fit(profile, fs, N)
    errors = []
    for snr_dB in snr_dB_list:
        link.seed(11)                     # same channel draw at every SNR
        link.set_params(**{"noise.snr_dB": float(snr_dB)})
        y = link(N)
        # the receiver has to know the channel: read what the fading
        # block actually realized (block fading, so column 0 is it)
        fading = link["fading"]
        taps = np.zeros(int(fading.delays_[-1]) + 1, dtype=complex)
        taps[fading.delays_] = fading.h_[:, 0]
        receiver = Sequential([
            OFDMReceiver(N_carrier, N_cp, h=taps),
            SymbolDemapper(alphabet),
        ])
        errors.append(compute_ser(link.tap("tx"), receiver(y)))
    curves[f"CP = {N_cp} samples"] = errors
    print(f"  CP = {N_cp:3d} samples ({N_cp / fs * 1e6:.2f} us): "
          + " ".join(f"{value:.3f}" for value in errors))

ax = plot_error_rate(snr_dB_list, curves, ylabel="SER",
                     title=f"{M}-QAM OFDM over TDL-C, {spread_ns:.0f} ns "
                           f"spread ({longest} samples)")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/multipath_fig3.png")

# The chain diagram this tutorial shows is exported from the chain
# itself (D33c), so the picture cannot drift from the code.
mermaid_dir = "../../docs/examples/mermaid/"
for diagram_name, diagram_chain in [("multipath", link)]:
    with open(f"{mermaid_dir}/{diagram_name}.mmd", "w") as stream:
        stream.write(diagram_chain.to_mermaid())

plt.show()
