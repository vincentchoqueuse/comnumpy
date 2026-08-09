"""Golden tests for the three analytical references of the ``ofdm`` module.

1. The CCDF of the PAPR follows ``1 - (1 - exp(-gamma))**N`` at the
   Nyquist rate, and is under-estimated by that formula once the signal
   is oversampled (validation script ``validation/ofdm_papr_ccdf.py``).
2. A cyclic prefix at least as long as the channel memory turns the
   linear convolution into a circular one, so the one-tap frequency-domain
   equalizer inverts the channel *exactly* -- not approximately.
3. The orthonormal (I)FFT conserves power (Parseval), hence the
   per-subcarrier SNR equals the time-domain SNR times the occupancy
   factor ``N_fft / (N_data + N_pilots)`` -- and nothing else.

Every threshold below is annotated with the measurement that justifies
it, taken over seeds 1..5 at the sample counts used here.
"""
import unittest

import numpy as np

from comnumpy import AWGN, Sequential, SymbolGenerator
from comnumpy.core.channels import FIRChannel
from comnumpy.core.mappers import SymbolMapper
from comnumpy.core.metrics import compute_ccdf, compute_ser
from comnumpy.core.processors import Parallel2Serial, Serial2Parallel
from comnumpy.core.utils import get_alphabet
from comnumpy.ofdm.allocation import get_allocation
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from comnumpy.ofdm.metrics import compute_PAPR
from comnumpy.ofdm.processors import (CarrierAllocator, CarrierExtractor,
                                      CyclicPrefixer, CyclicPrefixRemover,
                                      FFTProcessor, IFFTProcessor)

SEED = 1


# ---------------------------------------------------------------------------
# 1. PAPR distribution
# ---------------------------------------------------------------------------

class TestPAPRDistribution(unittest.TestCase):
    """CCDF of the PAPR against 1 - (1 - exp(-gamma))^N."""

    N_SUB = 256
    N_SYMBOLS = 20_000     # 200 OFDM symbols above the 1e-2 CCDF level

    @classmethod
    def setUpClass(cls):
        cls.papr = {os: cls.papr_dB(os) for os in (1, 4)}

    @classmethod
    def papr_dB(cls, os):
        """PAPR of QPSK OFDM symbols, in dB, one value per symbol."""
        carrier_type = np.zeros(os * cls.N_SUB, dtype=int)
        carrier_type[:cls.N_SUB] = 1
        chain = Sequential([
            SymbolGenerator(4),
            SymbolMapper(get_alphabet("PSK", 4)),
            Serial2Parallel(cls.N_SUB),
            CarrierAllocator(carrier_type=carrier_type),
            IFFTProcessor(),
        ])
        chain.seed(SEED)
        return compute_PAPR(chain(cls.N_SYMBOLS * cls.N_SUB), unit="dB", axis=-1)

    @staticmethod
    def gap_dB(papr, N, level):
        """Horizontal distance to the reference CCDF, at a fixed level."""
        sorted_papr, ccdf = compute_ccdf(papr)
        measured = sorted_papr[np.searchsorted(-ccdf, -level)]  # ccdf decreases
        theory = 10 * np.log10(-np.log(1 - (1 - level) ** (1.0 / N)))
        return measured - theory

    def test_nyquist_rate_matches_the_closed_form(self):
        """At the Nyquist rate the N samples behave as N independent ones."""
        # Measured over seeds 1..5: gap in [-0.045, -0.010] dB at CCDF=1e-1
        # and in [-0.069, +0.012] dB at CCDF=1e-2. The 0.20 dB threshold is
        # ~3x the worst deviation seen.
        for level in (1e-1, 1e-2):
            gap = self.gap_dB(self.papr[1], self.N_SUB, level)
            self.assertLess(abs(gap), 0.20,
                            f"Nyquist PAPR off the reference by {gap:+.3f} dB "
                            f"at CCDF={level:.0e}")

    def test_oversampling_reveals_peaks_the_formula_misses(self):
        """The formula under-estimates the continuous-time PAPR."""
        # 4x oversampling is the usual proxy for continuous time. Measured
        # over seeds 1..5: gap in [+0.365, +0.424] dB at CCDF=1e-2, i.e. the
        # true PAPR is *above* the reference. The [0.20, 0.80] window keeps
        # both bounds more than 5 sigma (spread 0.02 dB) from the data.
        gap = self.gap_dB(self.papr[4], self.N_SUB, 1e-2)
        self.assertGreater(gap, 0.20, f"got {gap:+.3f} dB")
        self.assertLess(gap, 0.80, f"got {gap:+.3f} dB")

    def test_oversampled_papr_is_larger_on_average(self):
        """Nyquist sampling misses the maxima between the samples."""
        # Measured: 7.804 dB (Nyquist) vs 8.398 dB (4x), i.e. +0.594 dB,
        # stable to better than 0.01 dB across seeds. Threshold 0.3 dB.
        delta = float(np.mean(self.papr[4]) - np.mean(self.papr[1]))
        self.assertGreater(delta, 0.3, f"got {delta:+.3f} dB")


# ---------------------------------------------------------------------------
# 2. Cyclic prefix and the one-tap equalizer
# ---------------------------------------------------------------------------

class TestCyclicPrefixExactness(unittest.TestCase):
    """L <= N_cp + 1 makes the one-tap ZF equalizer an exact inverse."""

    N_CP = 16
    N_OFDM_SYMBOLS = 100

    def setUp(self):
        self.alloc = get_allocation("802.11a")
        self.alphabet = get_alphabet("QAM", 16)

    def run_chain(self, L):
        """Noiseless OFDM round trip through an L-tap channel.

        Returns the NMSE on the equalized symbols and the SER.
        """
        rng = np.random.default_rng(SEED)
        h = rng.normal(size=L) + 1j * rng.normal(size=L)  # uniform PDP: every
        h = h / np.linalg.norm(h)                         # tap carries energy
        chain = Sequential([
            SymbolGenerator(16, name="tx"),
            SymbolMapper(self.alphabet),
            Serial2Parallel(self.alloc.N_data),
            CarrierAllocator(self.alloc, pilots=1.0),
            IFFTProcessor(),
            CyclicPrefixer(self.N_CP),
            Parallel2Serial(),
            FIRChannel(h),
            Serial2Parallel(self.alloc.N_fft + self.N_CP),
            CyclicPrefixRemover(self.N_CP),
            FFTProcessor(),
            FrequencyDomainEqualizer(h=h),
            CarrierExtractor(self.alloc),
            Parallel2Serial(),
        ], taps=["tx"])
        chain.seed(SEED)
        n = self.N_OFDM_SYMBOLS * self.alloc.N_data
        # the FIR channel lengthens the stream by L-1 samples, so the
        # receiver's Serial2Parallel emits one extra (zero-padded) block
        y = chain(n)[:n]
        tx = chain.tap("tx")
        x = self.alphabet[tx]
        nmse = float(np.sum(np.abs(y - x) ** 2) / np.sum(np.abs(x) ** 2))
        decided = np.argmin(np.abs(y[:, None] - self.alphabet[None, :]), axis=-1)
        return nmse, compute_ser(tx, decided)

    def test_short_channel_is_inverted_exactly(self):
        """No approximation: the residual is at machine precision, SER is 0."""
        # Measured over seeds 1..5 and L in {1, 9, 17}: NMSE in
        # [4.8e-32, 3.2e-30], SER exactly 0. The 1e-25 threshold sits five
        # decades above the worst residual and 27 decades below the
        # smallest ISI residual measured in the breaking case (7.1e-4).
        for L in (1, 9, self.N_CP + 1):
            with self.subTest(L=L):
                nmse, ser = self.run_chain(L)
                self.assertLess(nmse, 1e-25, f"L={L}: NMSE={nmse:.3e}")
                self.assertEqual(ser, 0.0, f"L={L}: SER={ser}")

    def test_channel_longer_than_the_prefix_breaks_orthogonality(self):
        """One tap too many and the exactness is gone -- the contrast proves
        the previous test is testing something."""
        # Measured over seeds 1..5: L = N_cp+2 gives NMSE in [7.1e-4, 1.9e-2],
        # i.e. at least 26 decades above the L <= N_cp+1 case. Threshold 1e-5
        # is ~70x below the smallest measured value.
        nmse, _ = self.run_chain(self.N_CP + 2)
        self.assertGreater(nmse, 1e-5, f"expected ISI, got NMSE={nmse:.3e}")

    def test_ser_degrades_when_the_prefix_is_too_short(self):
        """A clearly under-sized prefix costs symbol errors."""
        # Measured over seeds 1..5 at L = 24 (N_cp = 16): SER in
        # [0.0706, 0.1269]. Threshold 0.01 is 7x below the worst case.
        _, ser = self.run_chain(24)
        self.assertGreater(ser, 0.01, f"got SER={ser}")


# ---------------------------------------------------------------------------
# 3. Parseval and the per-subcarrier SNR
# ---------------------------------------------------------------------------

class TestParsevalAndSNR(unittest.TestCase):
    """norm="ortho" conserves power, so the SNR only pays the occupancy."""

    N_OFDM_SYMBOLS = 2000

    def setUp(self):
        self.alloc = get_allocation("802.11a")
        self.alphabet = get_alphabet("QAM", 16)
        self.occupancy = self.alloc.N_fft / (self.alloc.N_data + self.alloc.N_pilots)

    def test_ifft_conserves_power_exactly(self):
        """Parseval, at machine precision -- not a statistical statement."""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(50, 64)) + 1j * rng.normal(size=(50, 64))
        x = IFFTProcessor()(X)
        # Measured: the relative power error is exactly 0.0 (the two sums
        # round to the same float), the round-trip error is 1.3e-15.
        p_freq = float(np.sum(np.abs(X) ** 2))
        p_time = float(np.sum(np.abs(x) ** 2))
        self.assertLess(abs(p_time - p_freq) / p_freq, 1e-12)
        self.assertLess(float(np.max(np.abs(FFTProcessor()(x) - X))), 1e-12)

    def snr_ratio(self, N_cp, snr_dB=10):
        """Ratio of the per-subcarrier SNR to the time-domain SNR."""
        chain = Sequential([
            SymbolGenerator(16, name="tx"),
            SymbolMapper(self.alphabet),
            Serial2Parallel(self.alloc.N_data),
            CarrierAllocator(self.alloc, pilots=1.0),
            IFFTProcessor(),
            CyclicPrefixer(N_cp),
            Parallel2Serial(name="tx_time"),
            AWGN(snr_dB=snr_dB, name="noise"),
            Serial2Parallel(self.alloc.N_fft + N_cp),
            CyclicPrefixRemover(N_cp),
            FFTProcessor(),
            CarrierExtractor(self.alloc),
            Parallel2Serial(),
        ], taps=["tx", "tx_time"])
        chain.seed(SEED)
        n = self.N_OFDM_SYMBOLS * self.alloc.N_data
        y = chain(n)[:n]
        x = self.alphabet[chain.tap("tx")]
        # the AWGN block sets sigma2 from the power it measures on its own
        # input, i.e. on the time-domain stream, cyclic prefix included
        snr_time = float(np.mean(np.abs(chain.tap("tx_time")) ** 2)) \
            / chain["noise"].sigma2_
        snr_sub = float(np.mean(np.abs(x) ** 2)) \
            / float(np.mean(np.abs(y - x) ** 2))
        return snr_sub / snr_time

    def test_subcarrier_snr_gains_exactly_the_occupancy_factor(self):
        """SNR_sub = SNR_time * N_fft / (N_data + N_pilots)."""
        # Measured over seeds 1..5 and N_cp in {4, 16, 32}: the relative
        # deviation from N_fft/occupied = 1.23077 stays within
        # [-0.27%, +0.55%] -- pure Monte-Carlo scatter on 96,000 symbols.
        # The 1.5% threshold is ~3x the worst deviation.
        for N_cp in (4, 16, 32):
            with self.subTest(N_cp=N_cp):
                ratio = self.snr_ratio(N_cp)
                self.assertLess(abs(ratio - self.occupancy) / self.occupancy, 0.015,
                                f"N_cp={N_cp}: ratio={ratio:.5f}, "
                                f"expected {self.occupancy:.5f}")

    def test_cyclic_prefix_does_not_dilute_the_subcarrier_snr(self):
        """The intuitive extra factor N_fft/(N_fft+N_cp) is *not* there.

        The prefix repeats samples of identical average power, so it moves
        the measured time-domain power not at all; and the receiver drops
        the prefix samples without changing the noise variance of the ones
        it keeps. The prefix costs energy per information symbol, but not
        SNR under the "measured time-domain power" convention used by the
        AWGN block.
        """
        # Decisive measurement: the ratio is 1.2331 / 1.2311 / 1.2309 for
        # N_cp = 4 / 16 / 32 (spread <= 0.002), while the CP-diluted
        # prediction would be 1.1584 / 0.9846 / 0.8205 -- 6%, 20% and 33%
        # away. Requiring the two ends to agree within 2% rejects the wrong
        # formula, whose two ends differ by 29%.
        ratio_4, ratio_32 = self.snr_ratio(4), self.snr_ratio(32)
        self.assertLess(abs(ratio_32 - ratio_4) / ratio_4, 0.02,
                        f"the ratio depends on N_cp: {ratio_4:.5f} vs {ratio_32:.5f}")
        for N_cp, ratio in ((4, ratio_4), (32, ratio_32)):
            diluted = self.occupancy * self.alloc.N_fft / (self.alloc.N_fft + N_cp)
            self.assertGreater(abs(ratio - diluted) / diluted, 0.03,
                               f"N_cp={N_cp}: cannot tell the CP-diluted "
                               f"formula apart (ratio={ratio:.5f})")


if __name__ == "__main__":
    unittest.main()
