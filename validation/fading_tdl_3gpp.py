"""The 5G NR delay profiles against two independent transcriptions.

A catalog entry is a table copied by hand, and a table copied by hand is
a table with a typo in it until something says otherwise. Decision D20
answers that with a self-check at construction, but a self-check can only
compare an entry with figures published *next to* the same table -- it
cannot catch an error that the summary figures do not see.

So the TR 38.901 tapped delay line models are confronted here with three
things that were not used to write them:

* **Another implementation.** OpenAirInterface transcribes the same
  clause into C, independently of Sionna's JSON model files this
  library's entries were copied from. Two independent transcriptions
  agreeing tap by tap is a much stronger statement than either alone.
* **The invariant the normalization guarantees.** TR 38.901 gives
  normalized delays (eq. 7.7-1), so the RMS delay spread computed from
  the table *is* the scale factor: a single mistyped delay or power
  moves it. Four of the five land within three parts in ten thousand of
  it, and the fifth -- TDL-D -- is six parts in a thousand short, which
  is the rounding of its own published powers.
* **The Rice factors.** TDL-D and TDL-E list the specular component and
  the Rayleigh part of the first tap separately; their ratio must be the
  K_1 the standard prints, 13.3 dB and 22.0 dB, and OpenAirInterface
  stores that same ratio as a linear constant computed from its own
  copy of the table.

The last check turns a naming trap into a third confrontation. *Two*
tables are called "TDL-A" in 3GPP: TR 38.901 Table 7.7.2-1, the 23-tap
model reproduced here, and TS 38.104 Table G.2.1.1-2, a 12-tap
simplification fixed at 30 ns used for RAN4 conformance testing -- and it
is the second one OpenAirInterface implements under that name. They are
not interchangeable (half the taps, and the conformance one cannot be
rescaled), but they are meant to describe the same channel, so they must
agree on its summary figures. Scaled to 30 ns, the table transcribed
here reproduces the conformance profile's RMS delay spread to four parts
in a hundred thousand and its maximum excess delay to a part in a
thousand -- from twice as many taps, published in a different
specification.

References
----------
3GPP TR 38.901 V17.1.0, Section 7.7.2, Tables 7.7.2-1 to 7.7.2-5;
Sionna v1.x, src/sionna/phy/channel/tr38901/models/TDL-{A..E}.json
(NVIDIA, Apache 2.0), fetched 2026-08-10;
OpenAirInterface5G, openair1/SIMULATION/TOOLS/random_channel.c
(branch develop), fetched 2026-08-10.
"""
import numpy as np

from comnumpy.core.fading import available_delay_profiles, get_delay_profile

SPREAD_NS = 100.0

# --- OpenAirInterface's own copy of Tables 7.7.2-4 and 7.7.2-5. ---------
# Its first entry is the *merged* first tap (specular + Rayleigh part),
# which is why it holds one entry fewer than the standard's table, and
# the split is carried separately as a linear Rice factor.
OAI = {
    "TDL-D": {
        "delays": [0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775,
                   4.042, 7.937, 9.424, 9.708, 12.525],
        "powers_dB": [-.00147, -18.8, -21, -22.8, -17.9, -20.1, -21.9,
                      -22.9, -27.8, -23.6, -24.8, -30.0, -27.7],
        "ricean_factor": .046774,
    },
    "TDL-E": {
        "delays": [0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092,
                   1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034,
                   20.6519],
        "powers_dB": [-.00433, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6,
                      -20.8, -22.6, -22.3, -25.6, -20.2, -29.8, -29.2],
        "ricean_factor": 0.0063096,
    },
}

# TS 38.104 Table G.2.1.1-2 (TDLA30): the *other* TDL-A, 12 taps at a
# fixed 30 ns spread, as transcribed by OpenAirInterface
TDLA30_DELAYS_NS = np.array([0, 10, 15, 20, 25, 50, 65, 75, 105, 135,
                             150, 290], dtype=float)
TDLA30_POWERS_DB = [-15.5, 0.0, -5.1, -5.1, -9.6, -8.2, -13.1, -11.5,
                    -11.0, -16.2, -16.6, -26.2]


def merged(delays, powers_dB):
    """Sum the powers of paths sharing a delay, and sort by delay."""
    delays = np.asarray(delays, dtype=float)
    unique, inverse = np.unique(delays, return_inverse=True)
    linear = np.zeros(unique.size)
    np.add.at(linear, inverse, 10 ** (np.asarray(powers_dB, float) / 10))
    return unique, 10 * np.log10(linear / np.sum(linear))


def test_against_openairinterface():
    """TDL-D and TDL-E, tap by tap, against a transcription in C."""
    for standard, table in OAI.items():
        profile = get_delay_profile(standard, delay_spread_ns=SPREAD_NS)
        delays, powers = merged(table["delays"], table["powers_dB"])
        np.testing.assert_allclose(profile.delays_ns, delays * SPREAD_NS,
                                   rtol=0, atol=1e-9)
        # the first tap is where the two differ by construction: this
        # library keeps the split (specular vs diffuse) in rice_k_dB,
        # OpenAirInterface merges it into the power and keeps the ratio
        # apart, and its merged value carries its own rounding
        np.testing.assert_allclose(profile.powers_dB[1:], powers[1:],
                                   rtol=0, atol=2e-3)
        np.testing.assert_allclose(profile.powers_dB[0], powers[0],
                                   rtol=0, atol=5e-3)
        measured_k = -10 * np.log10(table["ricean_factor"])
        assert abs(profile.rice_k_dB - measured_k) < 0.01, (
            standard, profile.rice_k_dB, measured_k)
        print(f"PASS {standard} matches OpenAirInterface on all "
              f"{profile.n_taps} taps, and K_1 = {profile.rice_k_dB:.1f} dB "
              f"= their {table['ricean_factor']:g}")


def test_the_normalization_invariant():
    """The delays are normalized, so the RMS spread *is* the scale."""
    for standard in [name for name in available_delay_profiles()
                     if name.startswith("TDL")]:
        for spread in (30.0, 100.0, 300.0):
            profile = get_delay_profile(standard, delay_spread_ns=spread)
            ratio = profile.rms_delay_spread_ns / spread
            # TDL-D is the one entry the published table does not
            # normalize exactly; the other four are within 3e-4
            tolerance = 7e-3 if standard == "TDL-D" else 5e-4
            assert abs(ratio - 1.0) < tolerance, (standard, spread, ratio)
        print(f"PASS {standard} normalized to "
              f"{ratio:.4f} x the requested spread")


def test_against_the_conformance_simplification():
    """The 12-tap TS 38.104 table must summarize to the same channel."""
    profile = get_delay_profile("TDL-A", delay_spread_ns=30.0)
    assert profile.n_taps == 23, profile.n_taps
    delays, powers = merged(TDLA30_DELAYS_NS, TDLA30_POWERS_DB)
    assert delays.size == 12, delays.size
    linear = 10 ** (powers / 10)
    mean = float(np.sum(linear * delays))
    spread = float(np.sqrt(np.sum(linear * (delays - mean) ** 2)))
    assert abs(spread / profile.rms_delay_spread_ns - 1) < 1e-4, spread
    assert abs(delays[-1] / profile.delays_ns[-1] - 1) < 1e-3, delays[-1]
    print(f"PASS the 12-tap TS 38.104 profile summarizes to the same "
          f"channel: spread {spread:.4f} ns against "
          f"{profile.rms_delay_spread_ns:.4f}, longest path "
          f"{delays[-1]:.1f} ns against {profile.delays_ns[-1]:.1f}, "
          f"from 12 taps against {profile.n_taps}")


def main():
    test_against_openairinterface()
    test_the_normalization_invariant()
    test_against_the_conformance_simplification()


if __name__ == "__main__":
    main()
