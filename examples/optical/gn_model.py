import numpy as np
import matplotlib.pyplot as plt

from comnumpy import monte_carlo
from comnumpy.core import Sequential
from comnumpy.core.compensators import DataAidedComplexGainCompensator
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.generators import GaussianGenerator, SymbolGenerator
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_effective_snr
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.core.utils import Constellation
from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.gn_model import (gn_model_nli_power, gn_model_snr,
                                       optimal_launch_power)
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import (dbm_to_watt, launch_amplitude,
                                    watt_to_dbm)
from comnumpy.optical.wdm import WDMGrid
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

SMF = FiberSpec(0.2, gamma=1.3, cd_coefficient=17.0, wavelength_nm=1550.0)
BAUD = 32e9
SPAN_KM = 100.0
N_SPANS = 5
NF_dB = 6.0
SPACING = 50e9

OS = 4
N_SYM = 4096
ROLLOFF = 0.1
FS = BAUD * OS
STIMULUS = (2, N_SYM)


# ===========================================================================
#  Part 1 -- the prediction, with no signal propagated at all
# ===========================================================================

def comb(n_channels):
    """The channel layout, as an object rather than an arithmetic expression."""
    return WDMGrid.uniform(n_channels, spacing_Hz=SPACING,
                           bandwidth_Hz=BAUD * (1 + ROLLOFF),
                           center_Hz=SMF.carrier_frequency_Hz)


def comb_eta(n_channels):
    """The eta of P_NLI = eta P^3, for a comb of identical channels."""
    grid = comb(n_channels)
    nli = gn_model_nli_power(SMF, span_length_km=SPAN_KM, n_spans=N_SPANS,
                             powers_W=np.full(n_channels, 1e-3),
                             frequencies_Hz=np.asarray(grid.frequencies_Hz),
                             baud_rates_Hz=np.full(n_channels, BAUD))
    return nli[n_channels // 2] / (1e-3) ** 3


def analytic_ase(n_spans=N_SPANS, bandwidth_Hz=BAUD):
    """Amplifier noise in the channel, over both polarizations.

    The link budgets its own noise. ``polarizations=2`` is the term the
    channel-power convention makes easy to forget: the spectral density
    of an amplifier is **per polarization**, and a coherent link carries
    two independent copies of it.
    """
    link = FiberLink(n_spans, L_span=SPAN_KM, fs=FS, fiber=SMF, NF_dB=NF_dB)
    return link.budget(bandwidth_Hz, polarizations=2)["ase_power_W"]


grid = comb(9)
grid.plot(cut=grid.n_channels // 2)
plt.savefig(f"{img_dir}/gn_model_fig1.png")
print(f"comb: {grid.n_channels} channels, {grid.guard_Hz / 1e9:.1f} GHz of "
      f"guard, {grid.min_fs / 1e9:.0f} GHz to simulate")

eta = comb_eta(1)
ase_W = analytic_ase()
best_power, best_snr = optimal_launch_power(ase_W, eta)
print(f"\neta      = {eta:.4g} /W^2      (GN model, one channel)")
print(f"P_ASE    = {watt_to_dbm(ase_W):+.2f} dBm     "
      f"({N_SPANS} spans, NF = {NF_dB:.0f} dB, both polarizations)")
print(f"optimum  = {watt_to_dbm(best_power):+.2f} dBm     "
      f"SNR = {10 * np.log10(best_snr):.2f} dB")
print(f"check    : eta P^3 / (P_ASE/2) = "
      f"{eta * best_power ** 3 / (ase_W / 2):.6f}")

fine_powers = np.logspace(-1.0, 0.6, 400) * 1e-3
print("\nchannels   NLI at 0 dBm   optimum   peak SNR   fs to simulate it")
counts = [1, 3, 9, 27, 81]
curves = {}
for n_channels in counts:
    channel_eta = comb_eta(n_channels)
    power, snr = optimal_launch_power(ase_W, channel_eta)
    curves[n_channels] = 10 * np.log10(
        gn_model_snr(ase_W, channel_eta, fine_powers))
    print(f"{n_channels:8d}   "
          f"{watt_to_dbm(channel_eta * (1e-3) ** 3):+8.2f} dBm   "
          f"{watt_to_dbm(power):+6.2f} dBm   {10*np.log10(snr):6.2f} dB   "
          f"{comb(n_channels).min_fs/1e12:11.2f} THz")

_, ax = plt.subplots(figsize=(7, 5), layout="constrained")
for n_channels in counts:
    ax.plot(watt_to_dbm(fine_powers), curves[n_channels],
            label=f"{n_channels} channels")
ax.set_xlabel("launch power per channel [dBm]")
ax.set_ylabel("SNR [dB]")
ax.set_title("Filling the band, in closed form only")
ax.legend()
ax.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/gn_model_fig2.png")


# ===========================================================================
#  Part 2 -- the simulation, which is here to disagree
# ===========================================================================

def get_chain(order=16, power_W=1e-3):
    """The whole link, from symbols to equalized symbols, as one chain."""
    source = ([GaussianGenerator(1.0, name="tx")] if order is None else
              [SymbolGenerator(order, name="source"),
               SymbolMapper(Constellation("QAM", order), name="tx")])
    return Sequential([
        *source,
        Amplifier(launch_amplitude(power_W, polarizations=2), name="launch"),
        Upsampler(OS, scale=np.sqrt(OS)),
        SRRCFilter(ROLLOFF, OS, N_h=40, method="fft"),
        FiberLink(N_SPANS, L_span=SPAN_KM, StPS=40, fs=FS, fiber=SMF,
                  NF_dB=NF_dB, name="fibre"),
        DBP(N_SPANS, L_span=SPAN_KM, StPS=1, fs=FS, fiber=SMF,
            use_only_linear=True, name="dbp"),
        SRRCFilter(ROLLOFF, OS, N_h=40, method="fft", scale=1 / np.sqrt(OS)),
        Downsampler(OS),
    ], taps=["tx"], name=f"{N_SPANS} x {SPAN_KM:.0f} km SMF")


def effective_snr_dB(sent, received):
    """SNR once the complex gain a receiver would remove is removed."""
    compensator = DataAidedComplexGainCompensator(sent, shared=True)
    corrected = compensator(received)
    return compute_effective_snr(sent, corrected, unit="dB")


def measure(chain, powers_W, seed=3):
    return monte_carlo(chain, "launch.gain",
                       launch_amplitude(powers_W, polarizations=2),
                       {"snr_dB": effective_snr_dB}, STIMULUS,
                       reference="tx", seed=seed)["snr_dB"]


chain = get_chain()
chain.summary(STIMULUS)

chain.set_params(fibre__use_only_linear=True)
measured_ase_W = dbm_to_watt(-measure(chain, [dbm_to_watt(0.0)])[0])
gap_dB = watt_to_dbm(measured_ase_W) - watt_to_dbm(ase_W)
print(f"\nP_ASE  predicted {watt_to_dbm(ase_W):+.2f} dBm   "
      f"measured {watt_to_dbm(measured_ase_W):+.2f} dBm   "
      f"gap {gap_dB:+.2f} dB")
assert abs(gap_dB) < 0.3, gap_dB

chain.set_params(fibre__use_only_linear=False)
powers_dBm = np.arange(-8.0, 5.1, 1.0)
measured_snr_dB = measure(chain, dbm_to_watt(powers_dBm))
predicted_snr_dB = 10 * np.log10(gn_model_snr(ase_W, eta, fine_powers))
print(f"optimum {watt_to_dbm(best_power):+.2f} dBm predicted, "
      f"{powers_dBm[int(np.argmax(measured_snr_dB))]:+.1f} dBm measured; "
      f"peak SNR {10 * np.log10(best_snr):.2f} dB predicted, "
      f"{np.max(measured_snr_dB):.2f} dB measured")

_, ax = plt.subplots(figsize=(7, 5), layout="constrained")
ax.plot(watt_to_dbm(fine_powers), predicted_snr_dB, "-",
        label="GN model (closed form, microseconds)")
ax.plot(powers_dBm, measured_snr_dB, "o",
        label="split-step simulation, ~1 s per point")
ax.plot(watt_to_dbm(fine_powers),
        10 * np.log10(fine_powers / ase_W), ":", color="0.5",
        label="amplifiers alone ($P/P_{ASE}$)")
ax.axvline(watt_to_dbm(best_power), color="0.3", linestyle="--", linewidth=1)
ax.annotate(f"optimum {watt_to_dbm(best_power):+.1f} dBm",
            (watt_to_dbm(best_power), np.min(predicted_snr_dB) + 1),
            rotation=90, va="bottom", ha="right", color="0.3")
ax.set_xlabel("launch power per channel [dBm]")
ax.set_ylabel("SNR [dB]")
ax.set_title(f"PM-16QAM, {BAUD/1e9:.0f} GBd, {N_SPANS} x {SPAN_KM:.0f} km SMF")
ax.set_ylim(np.min(measured_snr_dB) - 2, np.max(predicted_snr_dB) + 2)
ax.legend()
ax.grid(True, alpha=0.4)
plt.savefig(f"{img_dir}/gn_model_fig3.png")

nli_only_dB = -10 * np.log10(eta * (1e-3) ** 2)
print(f"\nThe GN model predicts {nli_only_dB:.2f} dB of nonlinear SNR at "
      f"0 dBm, whatever is modulated.")
print("stimulus     nonlinear SNR   above the model")
for order, label in ((4, "QPSK"), (16, "16QAM"), (64, "64QAM"),
                     (256, "256QAM"), (None, "Gaussian")):
    noiseless = get_chain(order)
    noiseless.set_params(fibre__noise_scaling=0.0)
    value = measure(noiseless, [1e-3])[0]
    print(f"{label:12s} {value:10.2f} dB   {value - nli_only_dB:+10.2f} dB")

mermaid_dir = "../../docs/tutorials/mermaid/"
with open(f"{mermaid_dir}/gn_model.mmd", "w") as stream:
    stream.write(get_chain().to_mermaid())

plt.show()
