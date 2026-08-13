r"""Multipath fading: Doppler process and delay profile vs analytic references.

Four independent properties of ``comnumpy.core.fading`` and of
``TappedDelayLineChannel`` are confronted with closed forms (decision D7).

1. **Bessel autocorrelation.** For the classical (Clarke/Jakes) spectrum
   the tap autocorrelation is

       R(tau) = E[h(t) h*(t + tau)] = J0(2 pi f_D tau)

   (Clarke, BSTJ 47(6), 1968; Jakes, *Microwave Mobile Communications*,
   ch. 1). This is the discriminating check: it fixes the *shape* of the
   Doppler spectrum, not just its support.

2. **U-shaped Doppler spectrum.** The density is

       S(f) = 1 / (pi f_D sqrt(1 - (f/f_D)^2)),   |f| < f_D,  0 elsewhere

   whose singularities at +-f_D are integrable, so the comparison is made
   on the *cumulative* power, which is finite and smooth:

       C(f) = arcsin(f / f_D) / pi + 1/2

3. **Power delay profile.** The per-path powers realized in ``channel.h_``
   must reproduce ``profile.to_taps(fs)[1]``, and the measured RMS delay
   spread must reproduce the published figures (EVA 357 ns, ETU 991 ns,
   3GPP TS 36.101 Annex B.2).

4. **Frequency selectivity.** The correlation of the realized transfer
   function must follow the Fourier transform of the delay profile,

       rho(df) = |sum_l gamma_l exp(j 2 pi df tau_l)|

   and its 0.5-crossing must sit within an order of magnitude of the
   coherence bandwidth 1 / (5 sigma_tau) advertised by the profile. That
   last comparison is a **rule of thumb**, so it is checked to a factor 3,
   never as an equality -- and one catalog profile does not honour it
   (see the KNOWN DEVIATION line printed by the script).

Every assertion threshold below is followed by the deviation actually
measured at the working point, so the margin taken is visible.
"""
import pathlib

import numpy as np
from scipy.special import j0

from comnumpy.core.channels import TappedDelayLineChannel
from comnumpy.core.fading import get_delay_profile, rayleigh_process
from comnumpy import style

style.use()

FIG_DIR = pathlib.Path(__file__).parent / "figures"

# -- Doppler working point ---------------------------------------------
# The tap process is generated on its own time grid: 2 kHz is 10x the
# Nyquist rate of a 100 Hz Doppler, and 2048 samples span 1.024 s, i.e.
# ~102 Doppler periods. The process carries ~2 N f_D / f_s = 205 degrees
# of freedom per realization (measured std of the per-realization power:
# 0.087), so 3000 realizations are needed to average R(0) to ~0.2%.
FS_TAP = 2000.0
F_DOPPLER = 100.0
N_TAP = 2048
N_RUNS_TAP = 3000
MAX_LAG = 80                       # f_D * tau up to 4, i.e. 4 Doppler periods
# Window lengths that place f_D at a different position inside the FFT
# bin grid (f_D / df = N / 20 here): edge bins are weighted by the exact
# spectrum the generator has to approximate, so this sweep exposes how
# much it costs. 1760 is the case where f_D lands exactly on a bin centre
# and floating point pushes it just outside the band.
N_TAP_GRID_SWEEP = (1760, 2040, 2048, 2056)
N_RUNS_GRID_SWEEP = 1000

# -- Delay-profile working point ---------------------------------------
# 30.72 MHz is the LTE 20 MHz sampling rate; at 32.6 ns per sample the
# rounding of the tabulated delays onto the sample grid shifts the RMS
# delay spread by at most 1% (EPA +0.54%, EVA -0.93%, ETU +0.14%),
# which is what makes a 3% check against the published figures meaningful.
FS_RF = 30.72e6
N_RUNS_PDP = 6000
N_FFT = 4096                       # 7.5 kHz resolution on the transfer function
PROFILES = ("EPA", "EVA", "ETU")
# Published RMS delay spreads, 3GPP TS 36.101 Annex B.2. EPA is omitted:
# the table gives 43.1 ns while the literature commonly quotes 45 ns, a
# figure the catalog entry itself declines to pin (see fading.py).
TABLE_SPREAD_NS = {"EVA": 357.0, "ETU": 991.0}


# ----------------------------------------------------------------------
# measurements
# ----------------------------------------------------------------------
def doppler_statistics(n_samples, n_runs, *, fs=FS_TAP, f_doppler=F_DOPPLER,
                       max_lag=MAX_LAG, seed=0):
    """Ensemble autocorrelation and averaged periodogram of the tap process.

    The autocorrelation uses the unbiased linear estimator
    ``R(m) = 1/(N-m) sum_n h[n+m] h*[n]``, averaged over realizations and
    normalized by the *measured* ``R(0)`` -- otherwise the comparison
    would also carry the per-realization power fluctuation, which is a
    property of the window length, not of the spectrum shape.
    """
    rng = np.random.default_rng(seed)
    lags = np.arange(max_lag + 1)
    acf = np.zeros(max_lag + 1, dtype=complex)
    psd = np.zeros(n_samples)
    powers = np.empty(n_runs)
    for run in range(n_runs):
        h = rayleigh_process(n_samples, fs, f_doppler, rng=rng)
        powers[run] = np.mean(np.abs(h) ** 2)
        periodogram = np.abs(np.fft.fft(h)) ** 2
        psd += periodogram
        # linear autocorrelation through a zero-padded FFT
        full = np.fft.ifft(np.abs(np.fft.fft(h, 2 * n_samples)) ** 2)
        acf += full[:max_lag + 1] / (n_samples - lags)
    acf /= n_runs
    return {
        "lags_s": lags / fs,
        "acf": acf / acf[0].real,
        "freq": np.fft.fftfreq(n_samples, d=1 / fs),
        "psd": psd / psd.sum(),
        "power_std": float(powers.std()),
    }


def channel_statistics(name, n_runs, *, fs=FS_RF, n_fft=N_FFT, seed=0):
    """Per-path powers and frequency correlation of a realized channel.

    ``f_doppler=0`` (block fading) is deliberate: this measurement is
    about the delay axis, and one independent draw per path per call is
    the cheapest way to average the Rayleigh power. The Doppler axis is
    covered by :func:`doppler_statistics`.
    """
    profile = get_delay_profile(name)
    delays, powers = profile.to_taps(fs)
    channel = TappedDelayLineChannel(profile, fs=fs, f_doppler=0.0, seed=seed)
    x = np.ones(int(delays[-1]) + 64, dtype=complex)

    tap_power = np.zeros(delays.size)
    freq_corr = np.zeros(n_fft, dtype=complex)
    for _ in range(n_runs):
        channel(x)
        gains = channel.h_[:, 0]           # block fading: constant over the call
        tap_power += np.abs(gains) ** 2
        response = np.zeros(n_fft, dtype=complex)
        response[delays] = gains
        transfer = np.fft.fft(response)
        # circular correlation of H along the frequency axis:
        # sum_j H[j] H*[j+k] / n_fft, obtained from |FFT(H)|^2
        freq_corr += np.fft.fft(
            np.abs(np.fft.fft(transfer)) ** 2) / n_fft ** 2
    return {
        "profile": profile,
        "delays": delays,
        "powers_ref": powers,
        "powers_meas": tap_power / n_runs,
        "freq_axis": np.arange(n_fft) * fs / n_fft,
        "freq_corr": np.abs(freq_corr) / np.abs(freq_corr[0]),
    }


def rms_delay_spread_ns(delays, powers, fs):
    """RMS delay spread of a sampled profile, in nanoseconds."""
    tau = delays * 1e9 / fs
    weights = powers / powers.sum()
    mean = float(np.sum(weights * tau))
    return float(np.sqrt(np.sum(weights * tau ** 2) - mean ** 2))


def half_correlation_bandwidth(freq_axis, corr):
    """First frequency at which the correlation drops below 0.5, by interpolation."""
    below = np.flatnonzero(corr < 0.5)
    if below.size == 0:
        return float("inf")
    i = int(below[0])
    f0, f1 = freq_axis[i - 1], freq_axis[i]
    c0, c1 = corr[i - 1], corr[i]
    return float(f0 + (0.5 - c0) * (f1 - f0) / (c1 - c0))


# ----------------------------------------------------------------------
def main():
    # == property 1: autocorrelation follows J0 ========================
    doppler = doppler_statistics(N_TAP, N_RUNS_TAP, seed=1)
    bessel = j0(2 * np.pi * F_DOPPLER * doppler["lags_s"])
    acf_err = np.abs(doppler["acf"] - bessel)
    # Measured: max deviation over f_D*tau in [0, 4] is 0.0048 here and
    # spans 0.0033 to 0.0056 over five seeds (3000 runs each); 0.0026 of
    # it is the deterministic edge-bin residual of this window length,
    # quantified by the sweep below, the rest is Monte-Carlo. Threshold
    # 0.02 = 3.5x the worst measurement; for scale, a flat Doppler
    # spectrum instead of the classical one would miss J0 by 0.31.
    assert acf_err.max() < 0.02, (
        f"tap autocorrelation departs from J0(2 pi f_D tau) by "
        f"{acf_err.max():.4f} at f_D*tau = "
        f"{F_DOPPLER * doppler['lags_s'][acf_err.argmax()]:.2f} "
        f"(f_D={F_DOPPLER} Hz, fs={FS_TAP} Hz, {N_RUNS_TAP} realizations); "
        f"the Doppler spectrum shape is wrong, not just its support.")

    # The generator weights every bin that *overlaps* the band by its exact
    # covered power, so the result must not depend on where f_D happens to
    # fall in the bin grid. This sweep is the regression guard for that:
    # an earlier version tested bin *centres* and dropped edge bins whole,
    # losing up to 8.3% of the band power near the singularity.
    grid_errors = {}
    for n_samples in N_TAP_GRID_SWEEP:
        stats = doppler_statistics(n_samples, N_RUNS_GRID_SWEEP, seed=2)
        ref = j0(2 * np.pi * F_DOPPLER * stats["lags_s"])
        grid_errors[n_samples] = float(np.abs(stats["acf"] - ref).max())
    worst_grid = max(grid_errors, key=grid_errors.get)
    # Measured after the overlap fix: 0.0021 (N=1760), 0.0017 (N=2040),
    # 0.0032 (N=2048), 0.0053 (N=2056) -- all at the Monte-Carlo floor,
    # with no grid dependence left. The band power itself is exact to
    # 9.5e-9 for every N in [1000, 4100]. Threshold 0.02 = 4x the worst
    # measurement; before the fix N=1760 alone reached 0.065.
    assert grid_errors[worst_grid] < 0.02, (
        f"the Doppler autocorrelation depends on the FFT grid: "
        f"{grid_errors[worst_grid]:.4f} at N={worst_grid}, expected at most "
        f"0.02. An edge bin is being kept or dropped whole instead of "
        f"weighted by the band fraction it covers. Full sweep: {grid_errors}")

    # == property 2: U-shaped spectrum ================================
    order = np.argsort(doppler["freq"])
    freq, psd = doppler["freq"][order], doppler["psd"][order]
    out_of_band = float(psd[np.abs(freq) > F_DOPPLER].sum())
    # Measured: 8.1e-32. The generator zeroes the out-of-band bins, so
    # this is exact to machine precision; 1e-12 catches any leakage.
    assert out_of_band < 1e-12, (
        f"the Doppler process carries {out_of_band:.2e} of its power "
        f"outside |f| <= f_D = {F_DOPPLER} Hz; it is not band-limited.")

    cumulative = np.cumsum(psd)
    probe = np.linspace(-0.95, 0.95, 39) * F_DOPPLER
    cum_meas = np.interp(probe, freq, cumulative)
    cum_theo = np.arcsin(probe / F_DOPPLER) / np.pi + 0.5
    cum_err = np.abs(cum_meas - cum_theo)
    # Measured: 0.0043 here, 0.0042 to 0.0059 over five seeds, on the
    # cumulative power (integrated bands, not bin by bin -- the density
    # itself diverges at +-f_D). Threshold 0.02 = 3.4x the worst
    # measurement; for scale, a flat spectrum would deviate by 0.105.
    assert cum_err.max() < 0.02, (
        f"cumulative Doppler power departs from arcsin(f/f_D)/pi + 1/2 by "
        f"{cum_err.max():.4f} at f/f_D = {probe[cum_err.argmax()] / F_DOPPLER:+.2f}; "
        f"the spectrum is band-limited but not U-shaped.")

    # == properties 3 and 4: delay profile and selectivity =============
    results = {}
    for name in PROFILES:
        stats = channel_statistics(name, N_RUNS_PDP, seed=10)
        profile, delays = stats["profile"], stats["delays"]
        ref, meas = stats["powers_ref"], stats["powers_meas"]

        power_err = np.abs(meas / ref - 1)
        # Measured: 2.3% (EPA), 1.9% (EVA), 1.9% (ETU) as the worst path
        # of each profile, 1.8% to 2.8% over three seeds. Each path power
        # is an average of N_RUNS_PDP exponential draws, so its relative
        # standard deviation is 1/sqrt(6000) = 1.3% and the worst of 9
        # paths sits near 2 sigma. Threshold 8% = 6 sigma; a mis-normalized
        # or mis-ordered profile is off by tens of percent, not by a few.
        assert power_err.max() < 0.08, (
            f"{name}: realized path powers depart from profile.to_taps(fs)[1] "
            f"by {power_err.max():.1%} (path {int(power_err.argmax())}, "
            f"{N_RUNS_PDP} realizations)\n  measured  {np.round(meas, 5)}"
            f"\n  reference {np.round(ref, 5)}")
        total = float(meas.sum())
        assert abs(total - 1.0) < 0.03, (
            f"{name}: the realized profile carries {total:.4f} of total power, "
            f"expected 1.0 -- the normalization sum_l gamma_l = 1 is broken.")

        spread_meas = rms_delay_spread_ns(delays, meas, FS_RF)
        spread_grid = rms_delay_spread_ns(delays, ref, FS_RF)
        results[name] = {**stats, "spread_meas": spread_meas,
                         "spread_grid": spread_grid,
                         "power_err": float(power_err.max())}

        if name in TABLE_SPREAD_NS:
            table = TABLE_SPREAD_NS[name]
            rel = abs(spread_meas / table - 1)
            # Measured: EVA 0.8% (354.2 ns vs 357 ns), ETU 0.4%
            # (995.2 ns vs 991 ns); worst over three seeds 1.0%. Two
            # effects add up: rounding the tabulated delays onto the
            # 32.6 ns sample grid (-0.93% for EVA, +0.14% for ETU, see
            # FS_RF above) and the Monte-Carlo error on the path powers
            # (~1%). Threshold 3% = 3x the worst measurement.
            assert rel < 0.03, (
                f"{name}: measured RMS delay spread {spread_meas:.1f} ns, "
                f"3GPP TS 36.101 Annex B.2 publishes {table:.0f} ns "
                f"({rel:.1%} away); on-grid value {spread_grid:.1f} ns.")

        # property 4: measured frequency correlation vs the Fourier
        # transform of the delay profile -- an exact analytic reference.
        corr_theo = np.abs(np.sum(
            ref[None, :] * np.exp(2j * np.pi * stats["freq_axis"][:, None]
                                  * delays[None, :] / FS_RF), axis=1))
        corr_err = np.abs(stats["freq_corr"] - corr_theo)
        # Measured: 0.006 (EPA), 0.004 (EVA), 0.005 (ETU) over the whole
        # band, 0.010 worst over three seeds; it is the same Monte-Carlo
        # error as the path powers, seen through the Fourier transform.
        # Threshold 0.03 = 3x the worst measurement.
        assert corr_err.max() < 0.03, (
            f"{name}: the frequency correlation of the realized channel "
            f"departs from the Fourier transform of its delay profile by "
            f"{corr_err.max():.3f}.")

        bc_meas = half_correlation_bandwidth(stats["freq_axis"], stats["freq_corr"])
        bc_rule = profile.coherence_bandwidth_hz
        results[name]["bc_meas"] = bc_meas
        results[name]["bc_rule"] = bc_rule
        results[name]["bc_ratio"] = bc_meas / bc_rule
        results[name]["corr_err"] = float(corr_err.max())

    # The 1/(5 sigma_tau) coherence bandwidth is a RULE OF THUMB, so it is
    # checked to a factor 3 either way, never as an equality. It assumes a
    # compact, roughly exponential delay profile.
    for name in ("EPA", "EVA"):
        ratio = results[name]["bc_ratio"]
        # Measured: EPA 1.00 (1.00-1.01 over three seeds), EVA 1.96
        # (1.95-1.98). ETU sits at 4.9 and is handled below.
        assert 1 / 3 < ratio < 3, (
            f"{name}: the realized channel decorrelates at "
            f"{results[name]['bc_meas'] / 1e3:.0f} kHz, which is {ratio:.2f}x "
            f"the coherence bandwidth 1/(5 sigma_tau) = "
            f"{results[name]['bc_rule'] / 1e3:.0f} kHz announced by the "
            f"profile -- more than the factor 3 this rule of thumb is worth.")
    # ETU is the documented exception, checked against the exact Fourier
    # transform above instead. Asserting the factor 3 here would fail
    # (measured 4.92x), and relaxing the factor until it passes would
    # hide a real property of the profile, so the deviation is printed.
    etu_ratio = results["ETU"]["bc_ratio"]

    make_figure(doppler, bessel, freq, psd, cumulative, results)

    print(f"PASS fading Doppler: autocorrelation within {acf_err.max():.4f} of "
          f"J0 over 4 Doppler periods ({N_RUNS_TAP} realizations, "
          f"per-realization power std {doppler['power_std']:.3f}); cumulative "
          f"Doppler power within {cum_err.max():.4f} of arcsin(f/f_D)/pi + 1/2, "
          f"out-of-band power {out_of_band:.1e}")
    for name in PROFILES:
        r = results[name]
        table = TABLE_SPREAD_NS.get(name)
        table_txt = (f", table {table:.0f} ns"
                     f" ({abs(r['spread_meas'] / table - 1):.1%} away)"
                     if table else " (no published figure)")
        print(f"PASS {name}: path powers within {r['power_err']:.1%} of the "
              f"profile, RMS delay spread {r['spread_meas']:.1f} ns{table_txt}; "
              f"frequency correlation within {r['corr_err']:.3f} of the "
              f"profile transform, 0.5-crossing {r['bc_meas'] / 1e3:.0f} kHz "
              f"= {r['bc_ratio']:.2f} x 1/(5 sigma_tau)")
    print(f"KNOWN DEVIATION ETU: the 0.5-coherence bandwidth is "
          f"{etu_ratio:.2f}x the 1/(5 sigma_tau) rule of thumb, outside the "
          f"factor 3 this script asks of EPA and EVA. ETU's RMS delay spread "
          f"is inflated by the -7 dB path at 5 us while its correlation is set "
          f"by the 0-500 ns cluster carrying 84% of the power: the rule of "
          f"thumb, not the simulation, is what fails here (the exact profile "
          f"transform is matched to {results['ETU']['corr_err']:.3f}).")
    print(f"PASS Doppler grid independence: the normalized autocorrelation "
          f"stays within {min(grid_errors.values()):.4f} to "
          f"{max(grid_errors.values()):.4f} of J0 across the bin grids "
          f"{dict((k, round(v, 4)) for k, v in grid_errors.items())}, i.e. at "
          f"the Monte-Carlo floor wherever f_D falls between two bins.")


def make_figure(doppler, bessel, freq, psd, cumulative, results):
    """Four panels: J0, U spectrum, delay profile, frequency correlation."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    tau = F_DOPPLER * doppler["lags_s"]
    fine = np.linspace(0, tau[-1], 400)
    ax.plot(fine, j0(2 * np.pi * fine), "-", label=r"theory: $J_0(2\pi f_D \tau)$")
    ax.plot(tau[::2], doppler["acf"].real[::2], "o", fillstyle="none", ms=4,
            label=f"simulation ({N_RUNS_TAP} runs)")
    ax.plot(tau, 20 * (doppler["acf"].real - bessel), ":",
            label=r"residual ($\times 20$)")
    ax.set_xlabel(r"normalized lag $f_D \tau$")
    ax.set_ylabel(r"$R(\tau) / R(0)$")
    ax.set_title(f"Clarke autocorrelation, $f_D$ = {F_DOPPLER:.0f} Hz")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[0, 1]
    df = FS_TAP / N_TAP
    band = np.abs(freq) <= F_DOPPLER
    ax.plot(freq[band] / F_DOPPLER, psd[band] / df * F_DOPPLER, "o",
            fillstyle="none", ms=3, label="simulation (per-bin power / $\\Delta f$)")
    fine = np.linspace(-0.995, 0.995, 800)
    ax.plot(fine, 1 / (np.pi * np.sqrt(1 - fine ** 2)), "-",
            label=r"theory: $1/(\pi f_D \sqrt{1-(f/f_D)^2})$")
    ax.set_ylim(0, 2.5)
    ax.set_xlabel(r"$f / f_D$")
    ax.set_ylabel(r"$f_D \, S(f)$")
    ax.set_title("Classical Doppler spectrum")
    ax.legend(fontsize=8)
    ax.grid(True)
    inset = ax.inset_axes((0.62, 0.13, 0.35, 0.32))
    probe = np.linspace(-0.98, 0.98, 60) * F_DOPPLER
    inset.plot(probe / F_DOPPLER, np.arcsin(probe / F_DOPPLER) / np.pi + 0.5, "-")
    inset.plot(probe[::4] / F_DOPPLER, np.interp(probe[::4], freq, cumulative),
               "o", fillstyle="none", ms=3)
    inset.set_title("cumulative power", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True)

    ax = axes[1, 0]
    for name, marker in zip(("EVA", "ETU"), ("o", "s"), strict=True):
        r = results[name]
        tau_ns = r["delays"] * 1e9 / FS_RF
        ax.stem(tau_ns / 1e3, 10 * np.log10(r["powers_ref"]),
                bottom=-25, linefmt="C7-", markerfmt=" ", basefmt=" ")
        ax.plot(tau_ns / 1e3, 10 * np.log10(r["powers_meas"]), marker,
                fillstyle="none",
                label=f"{name} simulation ($\\sigma_\\tau$ = {r['spread_meas']:.0f} ns)")
    ax.set_ylim(-25, 0)
    ax.set_xlabel(r"path delay $\tau_l$ [$\mu$s]")
    ax.set_ylabel(r"path power $\gamma_l$ [dB]")
    ax.set_title("Power delay profile: stems = 3GPP table on the sample grid")
    ax.legend(fontsize=8)
    ax.grid(True)

    ax = axes[1, 1]
    for i, name in enumerate(PROFILES):
        r = results[name]
        keep = r["freq_axis"] <= 6e6
        ax.plot(r["freq_axis"][keep] / 1e6, r["freq_corr"][keep], f"C{i}-",
                label=f"{name} simulation")
        ax.axvline(r["bc_rule"] / 1e6, color=f"C{i}", ls=":", lw=1)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"frequency separation $\Delta f$ [MHz]")
    ax.set_ylabel(r"$|\rho(\Delta f)|$")
    ax.set_title(r"Frequency selectivity (dotted: $1/(5\sigma_\tau)$ rule of thumb)")
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "fading_doppler.png", dpi=150)


if __name__ == "__main__":
    main()
