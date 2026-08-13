"""What back-propagation is worth, as a function of the number of steps.

The launch-power sweep of Hager and Pfister (arXiv:2010.14258), with one
curve per step count. Run from this directory: it writes the tutorial's
figures into ../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import print_data
from comnumpy.core import Sequential
from comnumpy.core.compensators import DataAidedPhaseCompensator
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import ErrorCounter, compute_effective_snr
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.core.utils import Constellation
from comnumpy.core.visualizers import plot_error_rate
from comnumpy.optical.dbp import DBP
from comnumpy.optical.fiber import FiberSpec
from comnumpy.optical.gn_model import (gn_model_nli_power, gn_model_snr,
                                       optimal_launch_power)
from comnumpy.optical.links import FiberLink
from comnumpy.optical.utils import (dbm_to_watt, launch_amplitude,
                                    watt_to_dbm)
from comnumpy import style

style.use()

img_dir = "../../docs/tutorials/img/"

constellation = Constellation("QAM", 16)
N_s = 2**11 * 6           # 12288 symbols per trial; over 4 trials
                          # the SER floor is one error in 49152, 2.0e-5
oversampling_sim = 6
oversampling_dsp = 2
NF_dB = 5
rolloff = 0.1
StPS = 50                 # forward: converged, see the tutorial
R_s = 32e9
L_span = 100
N_span = 10
dBm_list = np.arange(-6, 6, 1.5)
N_trial = 4
fiber = FiberSpec()           # the standard fibre, named so the model sees it

fs = R_s * oversampling_sim
oversampling_ratio = oversampling_sim // oversampling_dsp


# --- the chain, in two pieces -----------------------------------------
# The one-shot tutorial ran one chain, transmitter to decision. Here six
# receivers are compared over the *same* channel realization, at eight
# launch powers and four trials each: running the whole chain per
# receiver would re-do 192 split-step propagations instead of 32, and
# the propagation is where all the time goes. So the chain is cut where
# the physics ends and the DSP begins.
def get_channel():
    """Symbols to the field the receiver sees, launch power included."""
    return Sequential([
        SymbolGenerator(constellation.order, name="data_tx"),
        SymbolMapper(constellation, name="signal_tx"),
        Amplifier(1.0, name="launch"),
        Upsampler(oversampling_sim, scale=np.sqrt(oversampling_sim)),
        SRRCFilter(rolloff, oversampling_sim, method="fft"),
        FiberLink(N_spans=N_span, L_span=L_span, StPS=StPS, NF_dB=NF_dB,
                  fs=fs, fiber=fiber, name="link"),
        Downsampler(oversampling_ratio, use_filter=True),
        ], taps=["data_tx", "signal_tx"])


def get_receiver(steps, linear_only, *, gain, reference):
    """Back-propagation, matched filter and data-aided phase correction.

    ``reference`` is the transmitted symbol sequence of the run being
    received: the chain that produced it is a different object here, so
    the reference is passed rather than wired.
    """
    return Sequential([
        DBP(N_span, L_span=L_span, StPS=steps, step_type="linear",
            fs=fs / oversampling_ratio, fiber=fiber,
            use_only_linear=linear_only, name="dbp"),
        SRRCFilter(rolloff, oversampling_dsp, method="fft",
                   scale=1 / np.sqrt(oversampling_dsp)),
        Downsampler(oversampling_dsp),
        Amplifier(gain),
        DataAidedPhaseCompensator(reference, name="phase"),
        SymbolDemapper(constellation, name="data_rx"),
        ], taps=["phase"])


# --- what the closed form expects of this link -------------------------
# The previous tutorial's GN model is a prediction, so it can be made
# before anything is propagated. Two differences with the link it was
# printed for, and both matter: this link has ten spans rather than five,
# and it is *single*-polarization, so the scalar NLSE applies and the
# weights are the ones of `polarizations=1` -- 5.3 dB more interference
# at equal power. The ASE is over one polarization for the same reason.
ase_W = get_channel()["link"].budget(R_s)["ase_power_W"]
eta = gn_model_nli_power(fiber, span_length_km=L_span, n_spans=N_span,
                         powers_W=np.array([1e-3]),
                         frequencies_Hz=np.array([fiber.carrier_frequency_Hz]),
                         baud_rates_Hz=np.array([R_s]),
                         polarizations=1)[0] / (1e-3) ** 3
best_power, best_snr = optimal_launch_power(ase_W, eta)
print(f"GN model for this link: P_ASE = {watt_to_dbm(ase_W):+.2f} dBm, "
      f"optimum {watt_to_dbm(best_power):+.2f} dBm, "
      f"peak SNR {10 * np.log10(best_snr):.2f} dB")


# Every receiver is the same chain with one argument changed. "amplifier
# noise only" receives a link whose nonlinearity was switched off, which
# is the bound the others are trying to reach.
receivers = {
    "amplifier noise only": (1, True),
    "dispersion compensation": (1, True),
    "DBP, 1 step/span": (1, False),
    "DBP, 2 steps/span": (2, False),
    "DBP, 4 steps/span": (4, False),
    "DBP, 50 steps/span": (50, False),
}

channel = get_channel()
snr = {}
counters = {}
elapsed = {}
for name in receivers:
    snr[name] = np.zeros(len(dBm_list))
    # one counter per launch power: it accumulates errors and symbols
    # over the four trials rather than averaging four rates, so the
    # count that says whether a point means anything stays readable
    counters[name] = []
    for _ in dBm_list:
        counters[name].append(ErrorCounter())
    elapsed[name] = 0.0

for index, dBm in enumerate(dBm_list):
    amp = launch_amplitude(dbm_to_watt(dBm))
    for trial in range(N_trial):
        # the two fields differ only by the fibre's nonlinearity, and
        # the same seed gives them the same symbols and the same noise
        fields = {}
        for use_only_linear in (True, False):
            channel.seed(index * N_trial + trial)
            channel.set_params(launch__gain=amp,
                               link__use_only_linear=use_only_linear)
            fields[use_only_linear] = channel(N_s)
        symbols, reference = channel.tap("signal_tx"), channel.tap("data_tx")

        for name, (steps, linear_only) in receivers.items():
            receiver = get_receiver(steps, linear_only, gain=1 / amp,
                                    reference=symbols)
            bound = name == "amplifier noise only"
            detected = receiver(fields[bound])
            elapsed[name] += receiver.elapsed_
            estimate = receiver.tap("phase")
            snr[name][index] += compute_effective_snr(symbols, estimate) / N_trial
            counters[name][index].update(reference, detected)

snr_dB = {}
for name, values in snr.items():
    snr_dB[name] = 10 * np.log10(values)

ser = {}
for name, points in counters.items():
    ser[name] = np.zeros(len(points))
    for index, counter in enumerate(points):
        ser[name][index] = counter.rate

# The sweep result, written down once. It is printed here and drawn
# below from the same object; transposed because six receiver names as
# column headers would make the table 160 characters wide.
snr_data = {"x": dBm_list, "curves": snr_dB}
print_data(snr_data, xlabel="launch power [dBm]",
           ylabel="effective SNR [dB]", transpose=True)

print("\nreceiver                  best SNR   at power    total time")
for name, values in snr_dB.items():
    best = int(np.argmax(values))
    print(f"{name:24s} {values[best]:6.2f} dB {dBm_list[best]:6.1f} dBm "
          f"{elapsed[name]:9.1f} s")

reference = snr_dB["dispersion compensation"]
best = int(np.argmax(reference))
print(f"\nGN model {10 * np.log10(best_snr):.2f} dB at "
      f"{watt_to_dbm(best_power):+.2f} dBm, dispersion compensation "
      f"{reference[best]:.2f} dB at {dBm_list[best]:+.1f} dBm")

# What the error-rate figure can resolve, in counts rather than in a
# caveat. At its own best power each receiver is where its curve bottoms
# out, and a point that saw no error has not measured a rate: it has run
# out of symbols. The counters kept the numbers, so the page can say so.
print("\nreceiver                  errors at its best power")
for name, points in counters.items():
    counter = points[int(np.argmax(snr_dB[name]))]
    print(f"{name:24s} {counter.n_errors:7d} over {counter.n_symbols} symbols")

# The prediction, on the same axes as the measurement. It describes the
# receiver that only undoes the dispersion -- the GN model counts the
# nonlinear interference as noise, so it has nothing to say about a
# receiver that removes part of it.
fine_powers = dbm_to_watt(np.linspace(dBm_list[0], dBm_list[-1], 200))
# An effective SNR in dB is not an error rate -- a linear ordinate, six
# receivers and one closed form -- so it is drawn here rather than routed
# through plot_error_rate, whose name would then be describing something
# it is not.
fig1, ax = plt.subplots()
for name, values in snr_dB.items():
    ax.plot(dBm_list, values, "o-", fillstyle="none", label=name)
ax.plot(watt_to_dbm(fine_powers),
        10 * np.log10(gn_model_snr(ase_W, eta, fine_powers)), "k:",
        label="GN model, single polarization")
ax.set_xlabel("launch power [dBm]")
ax.set_ylabel("effective SNR [dB]")
ax.set_title(f"{N_span} x {L_span} km, {constellation.order}-"
             f"{constellation.family} at {R_s / 1e9:.0f} GBd")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig1.png")

ax = plot_error_rate(dBm_list, ser, xlabel="launch power [dBm]", ylabel="SER",
                     title="the same sweep, in symbol error rate")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig2.png")
