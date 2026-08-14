"""Digital back-propagation against Häger and Pfister's published curves.

Decision D7 asks that a claim be confronted with something outside this
library. The claim here is the whole split-step + DBP stack, and the
confrontation is the effective-SNR curves of

    C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
    communication systems", IEEE JSAC 39(1), 2021 (arXiv:2010.14258),

whose reference implementation is
https://github.com/chaeger/LDBP/blob/master/ldbp/ldbp.py.

**The setup is theirs, not a teaching variant of it.** A *Gaussian*
stimulus, because the paper's curves are Gaussian-input effective SNR
and a real constellation would sit above them by the modulation-format
gap the GN model is blind to (measured in ``optical_gn_model.py``); 500
split steps per span forward, so the channel itself is not in question;
and the two links of the paper -- 25 x 80 km at 10.7 GBd, and 10 x 100 km
at 32 GBd. The low-step back-propagators use **logarithmic** steps, which
is what the paper uses and what makes one or two steps per span worth
anything: a logarithmic grid puts the short steps where the power, and
therefore the nonlinearity, is highest.

The tutorial ``examples/optical/NLI_simulation.py`` shares none of that
on purpose -- it carries 16-QAM so it can show a symbol error rate, and
fewer steps so it runs in minutes. It teaches; this reproduces.

**What is asserted, and what is only printed.** The paper's numbers are
read off figures, so this script asserts the structure those figures
show rather than values transcribed by eye:

* the linear reference climbs monotonically -- it is P/P_ASE, and
  nothing bends it;
* every other receiver peaks, because the nonlinearity eventually grows
  faster than the signal;
* more steps buy a higher peak at a higher launch power, in order;
* the best back-propagator stays within a few decibels of the
  reference -- 3.48 dB on the default run, the rest being the
  amplifier noise, which back-propagation amplifies rather than
  removes.

The table itself is printed in full so the curves can be laid over the
paper's figures, which is the check a reader actually wants.

**One departure from the original script, and it matters.** That script
band-limited with ``BWFilter(1 / oversampling_sim)`` before decimating by
``oversampling_sim // oversampling_dsp`` -- a brick wall at half the
width the decimation needed, cutting the roll-off shoulders off the
signal it was meant to pass. It put a deterministic distortion floor
under every curve, which is why the reference saturated instead of
climbing. ``Downsampler(use_filter=True)`` applies the mask the
decimation actually calls for.

The default run is the paper's second link at StPS = 60 with two trials
per point, and takes about ten seconds -- 512 Gaussian symbols per trial
is all an effective-SNR estimate needs. ``--full`` runs both links at the
paper's StPS = 500 with ten trials, which is of the order of an hour.

Measured on the default run, against the axes of the paper's own figure
for this link (its y range is 16.5 to 29.5 dB over -6 to +8 dBm):

    launch [dBm]     -6.0   -4.0   -2.0    0.0    2.0    4.0    6.0    8.0
    linear link     16.04  17.71  19.81  21.98  23.75  25.99  27.72  29.75
    linear          15.97  17.32  18.22  17.87  15.99  10.61   8.62   4.09
    DBP 1           15.96  17.50  18.83  19.06  17.35  12.70  10.40   5.52
    DBP 2 (log)     16.04  17.64  19.62  21.55  22.11  20.12  17.36  11.08
    DBP 4 (log)     16.03  17.67  19.77  21.87  23.58  24.75  24.68  19.68
    DBP full        16.03  17.68  19.78  21.89  23.68  25.47  26.27  25.71
"""
import sys
import time

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.compensators import DataAidedComplexGainCompensator
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.generators import GaussianGenerator
from comnumpy.core.metrics import compute_effective_snr
from comnumpy.core.processors import Amplifier, Downsampler, Upsampler
from comnumpy.optical.dbp import DBP
from comnumpy.optical.utils import dbm_to_watt, launch_amplitude
from comnumpy.optical.links import FiberLink

FULL = "--full" in sys.argv

# The two links of the paper. `dBm` is the launch-power axis of its figures.
LINKS = {
    "25 x 80 km, 10.7 GBd": dict(baud_Hz=10.7e9, span_km=80.0, n_spans=25,
                                 dBm=np.arange(-10.0, 6.0, 2.0)),
    "10 x 100 km, 32 GBd": dict(baud_Hz=32e9, span_km=100.0, n_spans=10,
                                dBm=np.arange(-6.0, 9.0, 2.0)),
}

# The receivers of the paper, in the order its legend lists them. The
# first is the reference: the same link with its Kerr term switched off,
# so it measures the amplifier noise alone.
RECEIVERS = [
    ("linear link", 1, True, "linear"),
    ("linear", 1, True, "linear"),
    ("DBP 1", 1, False, "linear"),
    ("DBP 2 (log)", 2, False, "logarithmic"),
    ("DBP 4 (log)", 4, False, "logarithmic"),
    ("DBP full", 500 if FULL else 60, False, "linear"),
]

N_SYMBOLS = 2 ** 9
OVERSAMPLING_SIM = 6
OVERSAMPLING_DSP = 2
NF_DB = 5.0
ROLLOFF = 0.1
STPS = 500 if FULL else 60
N_TRIALS = 10 if FULL else 2


def get_channel(link, *, linear_only):
    """Gaussian symbols to the field the receiver sees, launch included."""
    fs = link["baud_Hz"] * OVERSAMPLING_SIM
    ratio = OVERSAMPLING_SIM // OVERSAMPLING_DSP
    return Sequential([
        GaussianGenerator(1.0, name="tx"),
        Amplifier(1.0, name="launch"),
        Upsampler(OVERSAMPLING_SIM, scale=np.sqrt(OVERSAMPLING_SIM)),
        SRRCFilter(ROLLOFF, OVERSAMPLING_SIM, method="fft"),
        FiberLink(N_spans=link["n_spans"], L_span=link["span_km"], StPS=STPS,
                  NF_dB=NF_DB, fs=fs, use_only_linear=linear_only,
                  name="link"),
        Downsampler(ratio, use_filter=True),
    ], observations=["tx"])


def get_receiver(link, steps, linear_only, step_type, reference):
    """Back-propagation, matched filter, and the complex gain removed.

    The paper corrects amplitude and phase with a single complex scalar
    fitted on the transmitted signal; that is exactly
    ``DataAidedComplexGainCompensator(shared=True)``, so the estimate
    comes from the library rather than from four lines of algebra.
    """
    fs = link["baud_Hz"] * OVERSAMPLING_SIM
    ratio = OVERSAMPLING_SIM // OVERSAMPLING_DSP
    return Sequential([
        DBP(link["n_spans"], L_span=link["span_km"], StPS=steps,
            step_type=step_type, fs=fs / ratio, use_only_linear=linear_only,
            name="dbp"),
        SRRCFilter(ROLLOFF, OVERSAMPLING_DSP, method="fft",
                   scale=1 / np.sqrt(OVERSAMPLING_DSP)),
        Downsampler(OVERSAMPLING_DSP),
        DataAidedComplexGainCompensator(reference, shared=True, name="gain"),
    ])


def run(name, link):
    """Effective SNR against launch power, one column per receiver."""
    powers_dBm = link["dBm"]
    snr_dB = np.zeros((len(powers_dBm), len(RECEIVERS)))
    linear = get_channel(link, linear_only=True)
    nonlinear = get_channel(link, linear_only=False)

    for index, dBm in enumerate(powers_dBm):
        amplitude = launch_amplitude(dbm_to_watt(float(dBm)))
        for trial in range(N_TRIALS):
            fields = {}
            for is_linear, channel in ((True, linear), (False, nonlinear)):
                channel.seed(index * N_TRIALS + trial)
                channel.set_params(launch__gain=amplitude)
                fields[is_linear] = channel(N_SYMBOLS)
            sent = linear.observation("tx")

            for column, (_, steps, only_linear, step_type) in enumerate(RECEIVERS):
                receiver = get_receiver(link, steps, only_linear, step_type,
                                        sent)
                estimate = receiver(fields[column == 0])
                snr_dB[index, column] += (
                    compute_effective_snr(sent, estimate) / N_TRIALS)

    return powers_dBm, 10 * np.log10(snr_dB)


def report(name, powers_dBm, snr_dB):
    header = "launch [dBm]  "
    for value in powers_dBm:
        header += f"{value:7.1f}"
    print(f"\n{name}   ({STPS} steps/span forward, {N_TRIALS} trials)")
    print(header)
    for column, (label, _, _, _) in enumerate(RECEIVERS):
        line = f"{label:13s} "
        for value in snr_dB[:, column]:
            line += f"{value:7.2f}"
        print(line)


def check(name, powers_dBm, snr_dB):
    """The structure the paper's figures show, asserted."""
    reference = snr_dB[:, 0]
    increments = np.diff(reference)
    assert np.all(increments > 0), (
        f"{name}: the amplifier-noise reference must climb with the launch "
        f"power -- it is P/P_ASE and nothing bends it; got {reference}")

    peaks, best_dBm = [], []
    for column in range(1, len(RECEIVERS)):
        curve = snr_dB[:, column]
        index = int(np.argmax(curve))
        peaks.append(curve[index])
        best_dBm.append(powers_dBm[index])
        assert index < len(curve) - 1, (
            f"{name}: {RECEIVERS[column][0]} has not turned over inside the "
            f"swept range; the nonlinearity should eventually win")

    for earlier in range(len(peaks) - 1):
        assert peaks[earlier + 1] >= peaks[earlier] - 0.05, (
            f"{name}: more steps must not lower the peak -- "
            f"{RECEIVERS[earlier + 1][0]} {peaks[earlier]:.2f} dB then "
            f"{RECEIVERS[earlier + 2][0]} {peaks[earlier + 1]:.2f} dB")
    assert best_dBm[-1] >= best_dBm[0], (
        f"{name}: back-propagation must move the optimum to the right, "
        f"got {best_dBm[0]:+.1f} dBm then {best_dBm[-1]:+.1f} dBm")

    gap = float(np.max(reference) - peaks[-1])
    print(f"  best back-propagation is {gap:.2f} dB from the noise-only "
          f"reference, and moved the optimum by "
          f"{best_dBm[-1] - best_dBm[0]:+.1f} dB")
    assert gap < 6.0, f"{name}: {gap:.2f} dB left on the table"


def main():
    links = LINKS if FULL else {"10 x 100 km, 32 GBd": LINKS["10 x 100 km, 32 GBd"]}
    start = time.perf_counter()
    for name, link in links.items():
        powers_dBm, snr_dB = run(name, link)
        report(name, powers_dBm, snr_dB)
        check(name, powers_dBm, snr_dB)
    print(f"\ntotal {time.perf_counter() - start:.0f} s"
          f"{'' if FULL else '  (run with --full for both links at StPS=500)'}")


if __name__ == "__main__":
    main()
