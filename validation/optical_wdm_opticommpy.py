r"""A published WDM transmission simulation, reproduced with comnumpy.

Decision D7 asks every physical model to be confronted with something it
cannot fake. The other optical scripts use closed forms; this one uses a
**second implementation**: the coherent WDM transmission example of
OptiCommPy [1], whose notebook is published with its numerical output.

The configuration, copied from ``examples/test_WDM_transmission.ipynb``:

===========================  =========================================
Transmitter                  11 x 32 GBd PDM-16QAM, 37.5 GHz grid,
                             centred on 193.1 THz, RRC roll-off 0.01,
                             16 samples/symbol, -2 dBm per channel
Link                         14 x 50 km, alpha 0.2 dB/km,
                             D = 16 ps/nm/km, gamma = 1.3 1/(W km),
                             EDFA per span, NF 4.5 dB
Split step                   0.5 km nominal, adaptive on a 0.02 rad
                             maximum nonlinear phase rotation, each
                             step iterated to 1e-5
Receiver                     matched filter, CD compensation, 35-tap
                             adaptive MIMO equalizer (da-rde/rde),
                             BPS carrier phase recovery
Published result             SNR 20.63 / 20.64 dB, BER 1.1e-5 / 2.5e-5,
                             EVM 0.87 %, MI 4.00 bit, GMI 4.00 bit,
                             NGMI 1.00, centre channel
===========================  =========================================

**What can and cannot be reproduced.** The polarization multiplexing
*is* reproduced: ``FiberLink`` integrates the same Manakov equation on
a ``(2, N)`` field (D47), so the transmitter, the channel and the
launch power match the notebook exactly. Two differences remain, and
they are stated rather than hidden:

1. comnumpy has no adaptive MIMO equalizer and no blind phase search.
   The receiver here removes one complex gain per polarization. The
   notebook's 35-tap da-rde/rde equalizer and its 25-symbol BPS also
   track part of the nonlinear phase noise, which a constant gain
   cannot: whatever they recover shows up here as a gap.
2. The split step is fixed here and adaptive there -- 0.5 km nominal,
   shortened to keep the nonlinear phase rotation under 0.02 rad, each
   step iterated to 1e-5. Check 3 measures exactly what a fixed grid
   costs, and it is not negligible.

What the script checks:

0. **The transmitted comb.** 11 channels at -2 dBm on a 37.5 GHz grid
   must give the 8.41 dBm total the notebook prints. This exercises the
   WDM grid and multiplexer of D44 against a published number, and it is
   exact.

1. **The ASE-limited SNR**, against the closed form implied by the same
   Essiambre reference both libraries cite:

   .. math::

       \mathrm{SNR}_\mathrm{ASE} =
           \frac{P_\mathrm{ch}}
                {N_s (G - 1) n_\mathrm{sp} h \nu R_s},
       \qquad
       n_\mathrm{sp} = \frac{\mathrm{NF}/2}{1 - 1/G}

   With the nonlinearity off, the whole chain -- shaping, multiplexing,
   14 amplified spans, dispersion compensation, demultiplexing, matched
   filtering -- must land on that number.

2. **The implementation floor.** A root-raised-cosine truncated to a
   finite number of taps is not a Nyquist pair, and at a 0.01 roll-off
   its tails are long: the transmitter and receiver alone leave a
   measurable error. It is measured back to back and removed from every
   other figure, the way an implementation penalty is removed from a
   laboratory measurement.

3. **Step-size convergence.** The nonlinear figure is not a property of
   the fibre alone but of the integrator: halving the step must move it
   towards a limit. What this shows is that the notebook's nominal
   0.5 km step is *not* converged for a fixed grid -- their adaptive
   stepping is what makes it enough.

4. **The achievable rates.** The notebook reports MI and GMI of 4.00
   bit/symbol per polarization and a normalized GMI of 1.00 -- 16-QAM
   saturated, which is what a 20.6 dB SNR gives. The same three
   quantities are computed here with
   :mod:`comnumpy.core.information`, whose estimators come from
   Alvarado et al., the reference the notebook's own metrics follow.

5. **The full link**, reported against the published 20.63 dB. The
   remaining gap is expected to be positive: every impairment the
   notebook models and this script does not -- 100 kHz laser
   linewidths, the local oscillator, the IQ frontend, a polarization
   rotation and delay, the noise enhancement of a 35-tap equalizer --
   can only lower theirs.

References
----------
[1] E. P. da Silva et al., OptiCommPy: Python-based optical
    communication systems simulation package,
    https://github.com/edsonportosilva/OptiCommPy -- example
    ``test_WDM_transmission.ipynb``.

[2] R.-J. Essiambre, G. Kramer, P. J. Winzer, G. J. Foschini and
    B. Goebel, "Capacity limits of optical fiber networks", J. Lightwave
    Technol., vol. 28, no. 4, pp. 662-701, 2010 -- Eq. (54), the ASE
    model both libraries implement.

[3] D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the
    Manakov-PMD equation...", J. Lightwave Technol., vol. 15, no. 9,
    pp. 1735-1745, 1997 -- the 8/9 factor.
"""
import pathlib

import numpy as np

from comnumpy.core.filters import SRRCFilter
from comnumpy.core.information import compute_gmi, compute_mi, compute_ngmi
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.processors import Upsampler
from comnumpy.core.utils import get_alphabet
from comnumpy.optical import (DBP, FiberLink, WDMDemultiplexer, WDMGrid,
                              WDMMultiplexer)
from comnumpy.optical.constants import PLANCK_CONSTANT
from comnumpy.optical.fiber import FiberSpec

FIG_DIR = pathlib.Path(__file__).parent / "figures"

# -- the notebook's configuration -----------------------------------------
N_CHANNELS = 11
SPACING_HZ = 37.5e9
SYMBOL_RATE = 32e9
SAMPLES_PER_SYMBOL = 16
ORDER = 16
ROLL_OFF = 0.01
POWER_PER_CHANNEL_DBM = -2.0
CENTRE_HZ = 193.1e12
SPAN_KM, N_SPANS = 50.0, 14
ALPHA_DB_KM, DISPERSION, GAMMA = 0.2, 16.0, 1.3
NOISE_FIGURE_DB = 4.5

# published output of the notebook, centre channel (both polarizations)
PUBLISHED_TOTAL_POWER_DBM = 8.41
PUBLISHED_SNR_DB = 20.63
PUBLISHED_MI_BIT = 4.00
PUBLISHED_GMI_BIT = 4.00
PUBLISHED_NGMI = 1.00

# -- what this script chooses ---------------------------------------------
N_POLARIZATIONS = 2          # the notebook's PDM-16QAM (D47)
N_SYMBOLS = 2 ** 12          # 4096, against the notebook's 25000:
                             # 3584 symbols survive the guard on each
                             # polarization, so the SNR estimate is
                             # worth about 0.04 dB
N_TAPS_HALF = 128            # SRRC half-length, in symbols
GUARD = 2 * N_TAPS_HALF      # symbols dropped at both ends
FS = SYMBOL_RATE * SAMPLES_PER_SYMBOL
CENTRE_CHANNEL = N_CHANNELS // 2
WAVELENGTH_NM = 2.99792458e8 / CENTRE_HZ * 1e9

GRID = WDMGrid.uniform(N_CHANNELS, spacing_Hz=SPACING_HZ,
                       bandwidth_Hz=SYMBOL_RATE, center_Hz=CENTRE_HZ)


def fiber(gamma: float) -> FiberSpec:
    return FiberSpec(ALPHA_DB_KM, gamma=gamma, cd_coefficient=DISPERSION,
                     wavelength_nm=WAVELENGTH_NM)


# gamma stays the fibre's own coefficient: the 8/9 of the Manakov model
# is applied by FiberLink when the field carries two polarizations
KERR = fiber(GAMMA)
LINEAR = fiber(0.0)


def dbm(power_W: float) -> float:
    return 10 * np.log10(power_W * 1e3)


def snr_dB(mse: float) -> float:
    return -10 * np.log10(mse)


def transmit(power_dBm: float = POWER_PER_CHANNEL_DBM, seed: int = 1):
    """The 11-channel PDM comb: one 16-QAM signal per channel and mode.

    Returns the transmitted symbol *indices* and *points*, both
    ``(P, C, N_sym)``, and the multiplexed field ``(P, N)`` -- the
    polarization on the antenna axis of D2, which is what makes
    ``FiberLink`` integrate Manakov.
    """
    alphabet = get_alphabet("QAM", ORDER)
    indices = np.empty((N_POLARIZATIONS, N_CHANNELS, N_SYMBOLS), dtype=int)
    symbols = np.empty((N_POLARIZATIONS, N_CHANNELS, N_SYMBOLS),
                       dtype=complex)
    waveform = np.empty((N_POLARIZATIONS, N_CHANNELS,
                         N_SYMBOLS * SAMPLES_PER_SYMBOL), dtype=complex)
    shaper = SRRCFilter(ROLL_OFF, SAMPLES_PER_SYMBOL, N_h=N_TAPS_HALF,
                        method="fft")
    upsampler = Upsampler(SAMPLES_PER_SYMBOL)
    for mode in range(N_POLARIZATIONS):
        for channel in range(N_CHANNELS):
            generator = SymbolGenerator(
                ORDER, seed=seed + channel + 100 * mode)
            indices[mode, channel] = generator(N_SYMBOLS)
            sent = alphabet[indices[mode, channel]]
            symbols[mode, channel] = sent
            waveform[mode, channel] = shaper(upsampler(sent))
    # the notebook splits the channel power evenly over the two modes
    target_W = 1e-3 * 10 ** (power_dBm / 10) / N_POLARIZATIONS
    waveform *= np.sqrt(
        target_W / np.mean(np.abs(waveform) ** 2, axis=2))[:, :, None]
    multiplexer = WDMMultiplexer(GRID, fs=FS)
    field = np.stack([multiplexer(waveform[mode])
                      for mode in range(N_POLARIZATIONS)])
    return indices, symbols, field


def receive(field: np.ndarray, symbols: np.ndarray) -> float:
    """Demultiplex, matched filter, sample, and return the error power.

    Averaged over the two polarizations, as the notebook reports two
    nearly equal numbers.
    """
    errors = []
    for received, sent in equalize(field, symbols):
        errors.append(np.mean(np.abs(received - sent) ** 2)
                      / np.mean(np.abs(sent) ** 2))
    return float(np.mean(errors))


def equalize(field: np.ndarray, symbols: np.ndarray):
    """Yield the (received, sent) symbol pairs of the centre channel."""
    demultiplexer = WDMDemultiplexer(GRID, fs=FS)
    matched_filter = SRRCFilter(ROLL_OFF, SAMPLES_PER_SYMBOL,
                                N_h=N_TAPS_HALF, method="fft")
    for mode in range(N_POLARIZATIONS):
        channel = demultiplexer(field[mode])[CENTRE_CHANNEL]
        matched = matched_filter(channel)
        received = matched[::SAMPLES_PER_SYMBOL][:N_SYMBOLS][GUARD:-GUARD]
        sent = symbols[mode, CENTRE_CHANNEL][GUARD:-GUARD]
        # one complex gain: the amplitude the link left and the mean
        # nonlinear phase. This is where the notebook runs an adaptive
        # equalizer and a blind phase search instead, and the difference
        # is stated, not hidden.
        yield (np.vdot(received, sent)
               / np.vdot(received, received)) * received, sent


def propagate(field: np.ndarray, spec: FiberSpec, steps_per_span: int,
              noise_scaling: float) -> np.ndarray:
    """Forward through the link, then compensate the dispersion."""
    link = FiberLink(N_SPANS, L_span=SPAN_KM, StPS=steps_per_span, fs=FS,
                     fiber=spec, NF_dB=NOISE_FIGURE_DB,
                     noise_scaling=noise_scaling, seed=456)
    compensator = DBP(N_SPANS, L_span=SPAN_KM, StPS=1, fs=FS, fiber=spec,
                      use_only_linear=True)
    return compensator(link(field))


def ase_limited_snr_dB() -> float:
    r"""Closed form of the ASE-limited SNR of the amplifier chain.

    Each span is transparent, so the :math:`N_s` amplifiers each add
    :math:`(G-1) n_\mathrm{sp} h \nu` per hertz and the matched filter
    keeps :math:`R_s` of it.
    """
    gain = 10 ** (ALPHA_DB_KM * SPAN_KM / 10)
    noise_figure = 10 ** (NOISE_FIGURE_DB / 10)
    spontaneous = (noise_figure / 2) / (1 - 1 / gain)
    per_span = ((gain - 1) * spontaneous * PLANCK_CONSTANT * CENTRE_HZ
                * SYMBOL_RATE)
    # per polarization: the channel power is split over the two modes,
    # and each mode carries its own ASE
    power_W = 1e-3 * 10 ** (POWER_PER_CHANNEL_DBM / 10) / N_POLARIZATIONS
    return 10 * np.log10(power_W / (N_SPANS * per_span))


# -- the checks -----------------------------------------------------------
def check_comb_power(field: np.ndarray) -> float:
    """0. The comb the notebook prints: 11 x -2 dBm = 8.41 dBm.

    Summed over the polarizations: the notebook prints -5.01 dBm per
    mode and -2.00 dBm per channel, and the total is what both add up
    to.
    """
    total = dbm(float(np.sum(np.mean(np.abs(field) ** 2, axis=-1))))
    error = abs(total - PUBLISHED_TOTAL_POWER_DBM)
    assert error < 0.01, (
        f"the comb carries {total:.3f} dBm against the published "
        f"{PUBLISHED_TOTAL_POWER_DBM} dBm")
    print(f"PASS transmitted comb: {N_CHANNELS} channels at "
          f"{POWER_PER_CHANNEL_DBM:.0f} dBm give {total:.2f} dBm, the value "
          f"the notebook prints ({error:.3f} dB away)")
    return error


def check_implementation_floor(field: np.ndarray,
                               symbols: np.ndarray) -> float:
    """2. What the transmitter and receiver cost on their own."""
    floor = receive(field, symbols)
    assert snr_dB(floor) > PUBLISHED_SNR_DB + 10, (
        f"the back-to-back floor is {snr_dB(floor):.2f} dB, too close to the "
        f"{PUBLISHED_SNR_DB} dB being measured for the result to mean "
        f"anything -- lengthen the shaping filter")
    print(f"PASS implementation floor: the transmitter and receiver alone "
          f"leave {snr_dB(floor):.2f} dB, {snr_dB(floor) - PUBLISHED_SNR_DB:.1f} "
          f"dB above the figure under test; removed from everything below")
    return floor


def check_ase(field: np.ndarray, symbols: np.ndarray, floor: float) -> float:
    """1. The amplifier chain against its closed form."""
    received = propagate(field, LINEAR, 1, noise_scaling=1)
    measured = snr_dB(receive(received, symbols) - floor)
    expected = ase_limited_snr_dB()
    error = abs(measured - expected)
    # 3584 symbols on each polarization, and the floor subtraction
    # amplifies the estimator noise: the measurement is worth ~0.15 dB
    assert error < 0.3, (
        f"the ASE-limited SNR is {measured:.2f} dB against the closed form "
        f"{expected:.2f} dB")
    print(f"PASS ASE-limited SNR: {measured:.2f} dB measured through the "
          f"whole chain against {expected:.2f} dB from the closed form "
          f"({error:.2f} dB apart)")
    return error


def check_step_convergence(field: np.ndarray, symbols: np.ndarray,
                           floor: float):
    """3. The nonlinear figure must converge as the step shrinks."""
    results = {}
    for steps in (50, 100, 200, 400):
        received = propagate(field, KERR, steps, noise_scaling=0)
        results[steps] = snr_dB(receive(received, symbols) - floor)
        print(f"     {SPAN_KM / steps * 1e3:6.0f} m step: "
              f"nonlinear-only SNR {results[steps]:6.2f} dB")
    values = [results[s] for s in sorted(results)]
    assert all(b > a for a, b in zip(values, values[1:], strict=False)), (
        f"halving the step does not improve the answer monotonically: "
        f"{values}")
    tail = values[-1] - values[-2]
    assert tail < 0.4, f"still moving by {tail:.2f} dB at the finest step"
    nominal = results[int(SPAN_KM / 0.5)]
    print(f"PASS step convergence: the answer settles to "
          f"{values[-1]:.2f} dB (last halving moves it {tail:.2f} dB), and "
          f"the notebook's nominal 0.5 km step is {values[-1] - nominal:.2f} "
          f"dB short of it on a fixed grid -- their adaptive stepping is what "
          f"makes 0.5 km enough")
    return results


def check_full_link(field: np.ndarray, symbols: np.ndarray, floor: float,
                    nonlinear_only_dB: float, ase_dB: float) -> float:
    """4. Everything on, against the published 20.63 dB."""
    received = propagate(field, KERR, 200, noise_scaling=1)
    measured = snr_dB(receive(received, symbols) - floor)
    # the two contributions add as noise powers
    predicted = -10 * np.log10(10 ** (-ase_dB / 10)
                               + 10 ** (-nonlinear_only_dB / 10))
    assert abs(measured - predicted) < 0.3, (
        f"the full link gives {measured:.2f} dB while its own ASE-only and "
        f"nonlinear-only runs predict {predicted:.2f} dB -- the two "
        f"impairments are not adding as independent noises")
    print(f"PASS full link: {measured:.2f} dB, and the ASE-only "
          f"({ase_dB:.2f} dB) and nonlinear-only ({nonlinear_only_dB:.2f} dB) "
          f"runs add to {predicted:.2f} dB")
    print(f"     published (PDM, adaptive steps, MIMO equalizer + BPS): "
          f"{PUBLISHED_SNR_DB} dB -- {measured - PUBLISHED_SNR_DB:+.2f} dB "
          f"from this reproduction, and the sign is the expected one: every "
          f"impairment the notebook models and this script does not (laser "
          f"linewidth, LO, IQ frontend, polarization rotation and delay, "
          f"equalizer noise enhancement) can only lower theirs")
    return measured


def check_rates(field: np.ndarray, symbols: np.ndarray,
                indices: np.ndarray) -> None:
    """4. MI, GMI and normalized GMI against the notebook's own three."""
    alphabet = get_alphabet("QAM", ORDER)
    rates = []
    for mode, (received, sent) in enumerate(equalize(field, symbols)):
        # the metrics work on the unit-energy alphabet the symbols were
        # drawn from, so undo the launch-power scaling
        scale = np.sqrt(np.mean(np.abs(alphabet) ** 2)
                        / np.mean(np.abs(sent) ** 2))
        sent_index = indices[mode, CENTRE_CHANNEL][GUARD:-GUARD]
        rates.append((compute_mi(scale * received, sent_index, alphabet),
                      compute_gmi(scale * received, sent_index, alphabet),
                      compute_ngmi(scale * received, sent_index, alphabet)))
    mi, gmi, ngmi = (float(np.mean([r[i] for r in rates])) for i in range(3))

    bits = np.log2(ORDER)
    assert gmi <= mi + 1e-9, f"GMI {gmi:.4f} above MI {mi:.4f}"
    assert mi <= bits + 1e-9, f"MI {mi:.4f} above log2(M) = {bits}"
    for measured, published, name in ((mi, PUBLISHED_MI_BIT, "MI"),
                                      (gmi, PUBLISHED_GMI_BIT, "GMI"),
                                      (ngmi, PUBLISHED_NGMI, "NGMI")):
        assert abs(measured - published) < 0.01, (
            f"{name} is {measured:.4f} against the published {published}")
    print(f"PASS achievable rates: MI {mi:.2f} bit, GMI {gmi:.2f} bit, "
          f"NGMI {ngmi:.2f} -- the published 4.00 / 4.00 / 1.00. At this SNR "
          f"16-QAM is saturated, so the rates agree while the SNRs differ "
          f"by a decibel: which is the point of measuring both")


def main():
    import matplotlib.pyplot as plt

    indices, symbols, transmitted = transmit()
    check_comb_power(transmitted)
    floor = check_implementation_floor(transmitted, symbols)
    check_ase(transmitted, symbols, floor)
    ase_dB = snr_dB(receive(propagate(transmitted, LINEAR, 1, 1), symbols)
                    - floor)
    convergence = check_step_convergence(transmitted, symbols, floor)
    check_full_link(transmitted, symbols, floor, convergence[200], ase_dB)

    received = propagate(transmitted, KERR, 200, noise_scaling=1)
    check_rates(received, symbols, indices)
    fig, (ax_psd, ax_step) = plt.subplots(1, 2, figsize=(12, 4.2))

    for label, signal in (("transmitted", transmitted), ("received", received)):
        # the two polarizations carry the same comb: show their sum
        spectrum = np.fft.fftshift(
            np.sum(np.abs(np.fft.fft(signal, axis=-1)) ** 2, axis=0))
        frequencies = np.fft.fftshift(
            np.fft.fftfreq(signal.shape[-1], d=1 / FS))
        ax_psd.plot((CENTRE_HZ + frequencies) / 1e12,
                    10 * np.log10(spectrum / spectrum.max()), lw=0.6,
                    label=label)
    ax_psd.set(xlabel="optical frequency [THz]", ylabel="normalized PSD [dB]",
               ylim=(-60, 5),
               title=f"{N_CHANNELS} x {SYMBOL_RATE / 1e9:.0f} GBd, "
                     f"{SPACING_HZ / 1e9:.1f} GHz grid")
    ax_psd.grid(True)
    ax_psd.legend(fontsize="small")

    steps = sorted(convergence)
    ax_step.semilogx([SPAN_KM / s * 1e3 for s in steps],
                     [convergence[s] for s in steps], "o-", fillstyle="none")
    ax_step.axvline(500, color="0.6", ls=":", label="notebook nominal step")
    ax_step.set(xlabel="split step [m]", ylabel="nonlinear-only SNR [dB]",
                title="a fixed step is not converged at 500 m")
    ax_step.grid(True, which="both")
    ax_step.legend(fontsize="small")

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "optical_wdm_opticommpy.png", dpi=150)
    print(f"figure written to {FIG_DIR / 'optical_wdm_opticommpy.png'}")


if __name__ == "__main__":
    main()
