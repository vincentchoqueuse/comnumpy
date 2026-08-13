"""What back-propagation is worth, as a function of the number of steps.

The launch-power sweep of Hager and Pfister (arXiv:2010.14258), with one
curve per step count. Run from this directory: it writes the tutorial's
figures into ../../docs/tutorials/img/.
"""
import matplotlib.pyplot as plt
import numpy as np

from comnumpy import Experiment, plot_data, print_data
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


def simulate(config, seed):
    """One launch power: two fields per trial, every receiver on both.

    The two propagations of a trial share their seed, so they carry the
    same symbols and the same amplifier noise and differ only by the
    fibre's Kerr term. Each receiver then decides the field that matches
    its claim -- the bound receives the linear one -- and its errors are
    *counted* rather than averaged: the rate is a ratio of totals over
    the trials, and the count that says whether a point means anything
    survives to the tables below.
    """
    amp = launch_amplitude(dbm_to_watt(config["launch_dBm"]))
    snr = {}
    counters = {}
    times = {}
    for name in receivers:
        snr[name] = 0.0
        counters[name] = ErrorCounter()
        times[name] = 0.0
    for trial in range(config["n_trials"]):
        fields = {}
        for use_only_linear in (True, False):
            channel.seed(seed + trial)
            channel.set_params(launch__gain=amp,
                               link__use_only_linear=use_only_linear)
            fields[use_only_linear] = channel(config["n_symbols"])
        symbols = channel.tap("signal_tx")
        reference = channel.tap("data_tx")

        for name, (steps, linear_only) in receivers.items():
            receiver = get_receiver(steps, linear_only, gain=1 / amp,
                                    reference=symbols)
            detected = receiver(fields[name == "amplifier noise only"])
            snr[name] += compute_effective_snr(
                symbols, receiver.tap("phase")) / config["n_trials"]
            counters[name].update(reference, detected)
            times[name] += receiver.elapsed_

    # One flat record per point: the six SNRs under the receiver names,
    # everything else under a prefixed one, split again after the run.
    observed = {}
    for name in receivers:
        observed[name] = 10 * np.log10(snr[name])
        observed["ser " + name] = counters[name].rate
        observed["errors " + name] = counters[name].n_errors
        observed["time " + name] = times[name]
    return observed


experiment = Experiment({"n_symbols": N_s, "n_trials": N_trial},
                        parameter="launch_dBm", values=dBm_list, seed=0)
result = experiment.run(simulate)

snr_dB = {}
ser = {}
for name in receivers:
    snr_dB[name] = result.data[name]
    ser[name] = result.data["ser " + name]

# The sweep result, written down once. It is printed here and drawn
# below from the same object; transposed because six receiver names as
# column headers would make the table 160 characters wide.
snr_data = {"x": dBm_list, "curves": snr_dB}
print_data(snr_data, xlabel="launch power [dBm]",
           ylabel="effective SNR [dB]", transpose=True)

print("\nreceiver                  best SNR   at power    total time")
for name, values in snr_dB.items():
    best = int(np.argmax(values))
    total = float(np.sum(result.data["time " + name]))
    print(f"{name:24s} {values[best]:6.2f} dB {dBm_list[best]:6.1f} dBm "
          f"{total:9.1f} s")

reference = snr_dB["dispersion compensation"]
best = int(np.argmax(reference))
print(f"\nGN model {10 * np.log10(best_snr):.2f} dB at "
      f"{watt_to_dbm(best_power):+.2f} dBm, dispersion compensation "
      f"{reference[best]:.2f} dB at {dBm_list[best]:+.1f} dBm")

# What the error-rate figure can resolve, in counts rather than in a
# caveat. At its own best power each receiver is where its curve bottoms
# out, and a point that saw no error has not measured a rate: it has run
# out of symbols. The counters kept the numbers, so the page can say so.
symbols_per_point = N_s * N_trial
print("\nreceiver                  errors at its best power")
for name in receivers:
    best = int(np.argmax(snr_dB[name]))
    errors = int(result.data["errors " + name][best])
    print(f"{name:24s} {errors:7d} over {symbols_per_point} symbols")

# The prediction, on the same axes as the measurement. It describes the
# receiver that only undoes the dispersion -- the GN model counts the
# nonlinear interference as noise, so it has nothing to say about a
# receiver that removes part of it.
fine_powers = dbm_to_watt(np.linspace(dBm_list[0], dBm_list[-1], 200))
# The same object the table printed, drawn, with the closed form on top.
ax = plot_data(snr_data, xlabel="launch power [dBm]",
               ylabel="effective SNR [dB]", marker="o", fillstyle="none")
ax.plot(watt_to_dbm(fine_powers),
        10 * np.log10(gn_model_snr(ase_W, eta, fine_powers)), "k:",
        label="GN model, single polarization")
ax.set_title(f"{N_span} x {L_span} km, {constellation.order}-"
             f"{constellation.family} at {R_s / 1e9:.0f} GBd")
ax.legend()
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig1.png")

ax = plot_error_rate(dBm_list, ser, xlabel="launch power [dBm]", ylabel="SER",
                     title="the same sweep, in symbol error rate")
ax.set_ylim(1e-4, 1)
plt.tight_layout()
plt.savefig(f"{img_dir}/nli_simulation_fig2.png")
