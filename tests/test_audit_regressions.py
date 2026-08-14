r"""Regressions from the v1 code audit.

Every test here pins a defect found by the full-library audit: five
reviewers swept the code hunting for wrong axes, dtype loss, RNG misuse,
validation gaps and numerical blow-ups, and each confirmed finding was
fixed and locked by one test below. The names say what used to go wrong.
"""
import unittest
import warnings

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.filters import BWFilter, SRRCFilter
from comnumpy.core.frames import Deframer, FieldRole, FrameField, FrameStructure
from comnumpy.core.generators import GaussianGenerator, SymbolGenerator
from comnumpy.core.metrics import (ErrorCounter, compute_acpr, compute_ser,
                                   compute_ser_awgn_qam)
from comnumpy.core.processors import (Complex2Real, DataExtractor, Downsampler,
                                      Parallel2Serial, SampleRemover,
                                      Serial2Parallel, Upsampler)
from comnumpy.core.shaping import (DistributionDematcher, DistributionMatcher,
                                   SphereShaper)
from comnumpy.core.utils import get_alphabet, soft_projector
from comnumpy.data import format_data
from comnumpy.exceptions import ShapeError
from comnumpy.serialization import from_json, to_json


class TestChainMachinery(unittest.TestCase):

    def test_seed_reaches_a_nested_sequential(self):
        chain = Sequential([SymbolGenerator(4),
                            Sequential([AWGN(sigma2=0.1)], name="inner")])
        first = chain.seed(7)(8)
        second = chain.seed(7)(8)
        np.testing.assert_array_equal(first, second)

    def test_seed_reaches_a_wrapper_holding_a_chain(self):
        from comnumpy.ofdm.chains import OFDMTransmitter
        chain = Sequential([SymbolGenerator(4), OFDMTransmitter(4, 1)])
        first = chain.seed(3)(8)
        second = chain.seed(3)(8)
        np.testing.assert_array_equal(first, second)

    def test_seed_of_a_flat_chain_is_unchanged(self):
        # the recursion must not move the child seeds of the existing scheme
        chain = Sequential([SymbolGenerator(4), AWGN(sigma2=0.1)])
        chain.seed(7)
        expected = [int(c.generate_state(1)[0])
                    for c in np.random.SeedSequence(7).spawn(2)]
        self.assertEqual([chain[0].seed, chain[1].seed], expected)

    def test_summary_feeds_wiring_and_records_taps(self):
        from comnumpy.core.compensators import DataAidedPhaseCompensator
        wired = Sequential([SymbolGenerator(4, name="generator"),
                            AWGN(sigma2=0.01),
                            DataAidedPhaseCompensator(name="comp")],
                           wiring={"comp.reference": "generator"})
        rows = wired.summary(64, print_out=False)
        self.assertEqual(len(rows), 3)

        observed = Sequential([SymbolGenerator(4, name="generator"),
                             AWGN(sigma2=0.1)], observations=["generator"])
        observed.summary(8, print_out=False)
        self.assertEqual(observed.observation("generator").shape, (8,))

    def test_an_int_keyed_callback_fires(self):
        fired = []
        Sequential([SymbolGenerator(4, seed=1)],
                   callbacks={0: fired.append})(5)
        self.assertEqual(len(fired), 1)

    def test_a_tap_does_not_survive_its_reconfiguration(self):
        chain = Sequential([SymbolGenerator(4, name="generator"),
                            AWGN(sigma2=0.1)], observations=["generator", "awgn"])
        chain(8)
        chain.observations = ["awgn"]
        chain(8)
        with self.assertRaises(KeyError):
            chain.observation("generator")

    def test_a_nested_chain_round_trips_through_json(self):
        chain = Sequential([SymbolGenerator(4, seed=1),
                            Sequential([AWGN(sigma2=0.1, seed=2)],
                                       name="inner")])
        rebuilt = from_json(to_json(chain))
        np.testing.assert_array_equal(rebuilt(8), chain(8))


class TestCoreBlocks(unittest.TestCase):

    def test_framefield_freezes_a_copy_not_the_callers_array(self):
        values = np.array([1.0, -1.0])
        field = FrameField("S", FieldRole.SYNC, values)
        values[0] = 5.0                     # must stay writable
        assert field.values is not None
        self.assertEqual(field.values[0], 1.0)

    def test_deframer_fields_own_their_data(self):
        frame = FrameStructure((
            FrameField("SYNC", FieldRole.SYNC, np.array([1.0, -1.0])),
            FrameField("PAYLOAD", FieldRole.PAYLOAD, length=3)),
            standard="example")
        deframer = Deframer(frame)
        buffer = np.array([[1.0, -1.0, 1.0, 2.0, 3.0]])
        deframer(buffer)
        buffer[:] = 999.0
        np.testing.assert_array_equal(deframer.get_field("SYNC"),
                                      [[1.0, -1.0]])

    def test_complex2real_rejects_an_unknown_part(self):
        with self.assertRaisesRegex(ValueError, "real.*imag"):
            Complex2Real(part="magnitude")(np.array([1 + 2j]))  # type: ignore[arg-type]

    def test_serial2parallel_validates_method_at_construction(self):
        with self.assertRaisesRegex(ValueError, "zero-padding"):
            Serial2Parallel(3, method="trunc")  # type: ignore[arg-type]

    def test_parallel2serial_refuses_a_serial_signal(self):
        with self.assertRaises(ShapeError):
            Parallel2Serial()(np.arange(5))

    def test_upsampler_validates_the_phase(self):
        with self.assertRaisesRegex(ValueError, "phase"):
            Upsampler(L=2, phase=2)

    def test_sampleremover_validates_the_window(self):
        with self.assertRaises(ShapeError):
            SampleRemover(N_start=4, length=3)(np.arange(5))

    def test_dataextractor_keeps_its_declared_selector(self):
        extractor = DataExtractor((2, 8))
        extractor(np.arange(10))
        self.assertEqual(extractor.selector, (2, 8))

    def test_generators_accept_numpy_sizes_and_refuse_bool(self):
        self.assertEqual(SymbolGenerator(4, seed=1)(np.int64(5)).shape, (5,))
        with self.assertRaises(ValueError):
            GaussianGenerator()(True)

    def test_awgn_rejects_a_negative_variance(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            AWGN(sigma2=-0.1, seed=0)(np.zeros(4, dtype=complex))

    def test_the_matcher_pair_handles_an_empty_stream(self):
        shaper = SphereShaper(np.array([1.0, 3.0]), length=8, max_energy=3)
        empty_bits = np.array([], dtype=int)
        self.assertEqual(DistributionMatcher(shaper)(empty_bits).shape, (0,))
        self.assertEqual(DistributionDematcher(shaper)(empty_bits).shape, (0,))

    def test_the_antialias_filter_keeps_a_real_signal_real(self):
        x = np.cos(2 * np.pi * 0.01 * np.arange(64))
        self.assertFalse(np.iscomplexobj(Downsampler(L=4, use_filter=True)(x)))
        self.assertFalse(np.iscomplexobj(BWFilter(0.5)(x)))


class TestFiltersAndNumerics(unittest.TestCase):

    def test_srrc_at_zero_rolloff_is_the_sinc_pulse(self):
        h = SRRCFilter(0.0, 4).h()
        n = np.arange(-40, 41)
        sinc = np.sinc(n / 4)
        sinc = sinc / np.sqrt(np.sum(sinc ** 2))
        np.testing.assert_allclose(h, sinc, atol=1e-12)

    def test_srrc_fft_method_names_the_minimum_length(self):
        with self.assertRaisesRegex(ShapeError, "method='lfilter'"):
            SRRCFilter(0.25, 4, method="fft")(np.ones(64, dtype=complex))

    def test_soft_projector_survives_the_high_snr_regime(self):
        alphabet = get_alphabet("QAM", 4)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate = soft_projector(np.array([5 + 5j]), alphabet, 0.01)
        self.assertFalse(np.isnan(estimate).any())


class TestMetrics(unittest.TestCase):

    def test_errorcounter_truncates_along_the_last_axis(self):
        reference = np.tile(np.arange(10), (2, 1))
        detected = reference[:, :9]          # one symbol of filter delay
        counter = ErrorCounter().update(reference, detected)
        self.assertEqual(counter.n_errors, 0)

    def test_ser_awgn_qam_rejects_cross_constellations(self):
        with self.assertRaisesRegex(ValueError, "square"):
            compute_ser_awgn_qam(32, 10.0)

    def test_compute_ser_refuses_an_empty_comparison(self):
        with self.assertRaisesRegex(ValueError, "no symbols"):
            compute_ser(np.array([]), np.array([]))

    def test_acpr_counts_an_edge_bin_once(self):
        # a tone exactly on the shared band edge belongs to the main
        # channel only; counted twice it read as acpr = 0 dB
        tone = np.exp(2j * np.pi * 10 * np.arange(1000) / 100)
        acpr_right, _ = compute_acpr(tone, 20.0, 100.0)
        self.assertLess(acpr_right, -100.0)


class TestOFDM(unittest.TestCase):

    def test_receiver_reads_nfft_off_a_scattered_mask(self):
        from comnumpy.ofdm.chains import OFDMReceiver, OFDMTransmitter
        mask = np.array([[1, 2, 1, 1], [1, 1, 2, 1]])
        transmitter = OFDMTransmitter(3, 1, carrier_type=mask,
                                      pilots=np.array([1.0 + 0j]))
        receiver = OFDMReceiver(3, 1, carrier_type=mask)
        x = np.arange(12) + 0j
        np.testing.assert_allclose(receiver(transmitter(x)), x, atol=1e-12)

    def test_allocator_keeps_complex_pilots_on_real_data(self):
        from comnumpy.ofdm.processors import CarrierAllocator
        allocator = CarrierAllocator(carrier_type=np.array([1, 2, 1]),
                                     pilots=np.array([1j]))
        np.testing.assert_array_equal(allocator(np.array([1.0, 2.0])),
                                      np.array([1.0, 1j, 2.0]))

    def test_prefixer_refuses_a_prefix_longer_than_the_block(self):
        from comnumpy.ofdm.processors import CyclicPrefixer
        with self.assertRaises(ShapeError):
            CyclicPrefixer(N_cp=6)(np.arange(4))


class TestDataDisplay(unittest.TestCase):

    def test_degenerate_tables_render_their_header(self):
        empty = {"x": [], "curves": {"a": []}}
        self.assertIn("a", format_data(empty))
        self.assertIn("a", format_data(empty, transpose=True))
        self.assertIn("x", format_data({"x": [1, 2], "curves": {}},
                                       transpose=True))


class TestMIMOAndFEC(unittest.TestCase):

    def test_amp_no_longer_diverges_to_nan(self):
        from comnumpy.mimo.detectors import ApproximateMessagePassingDetector
        alphabet = get_alphabet("QAM", 4)
        alphabet = alphabet / np.sqrt(np.mean(np.abs(alphabet) ** 2))
        for seed in range(10):
            rng = np.random.default_rng(seed)
            H = (rng.normal(size=(4, 4))
                 + 1j * rng.normal(size=(4, 4))) / np.sqrt(8)
            sent = alphabet[rng.integers(0, 4, size=(4, 10))]
            Y = H @ sent + np.sqrt(0.25) * (rng.normal(size=(4, 10))
                                            + 1j * rng.normal(size=(4, 10)))
            detector = ApproximateMessagePassingDetector(
                alphabet, H=H, sigma2=0.5, N_it=20)
            detector(Y)
            for n in range(10):
                soft = detector.fit(Y[:, n])
                self.assertFalse(np.isnan(soft).any(),
                                 f"AMP state went NaN at seed {seed}")

    def test_oamp_mmse_raises_no_complex_warning(self):
        from comnumpy.mimo.detectors import (
            OrthogonalApproximateMessagePassingDetector)
        alphabet = get_alphabet("QAM", 4)
        alphabet = alphabet / np.sqrt(np.mean(np.abs(alphabet) ** 2))
        rng = np.random.default_rng(1)
        H = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / 2
        Y = H @ alphabet[rng.integers(0, 4, size=(4, 5))]
        detector = OrthogonalApproximateMessagePassingDetector(
            alphabet, H=H, sigma2=0.5, type="MMSE", N_it=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector(Y)

    def test_viterbi_round_trips_an_empty_message(self):
        from comnumpy.fec.convolutional import (ConvolutionalEncoder,
                                                ViterbiDecoder)
        encoder = ConvolutionalEncoder((0o5, 0o7))
        decoder = ViterbiDecoder((0o5, 0o7))
        decoded = decoder(encoder(np.array([], dtype=int)))
        self.assertEqual(decoded.shape, (0,))


class TestOptical(unittest.TestCase):

    def test_logarithmic_steps_of_a_lossless_fiber_are_uniform(self):
        from comnumpy.optical.utils import get_logarithmic_step_size
        steps = get_logarithmic_step_size(80.0, 5, alpha_dB=0.0)
        np.testing.assert_allclose(steps, 16.0)

    def test_the_fir_cd_compensator_degrades_to_identity(self):
        from comnumpy.optical.compensators import (
            ChromaticDispersionFIRCompensator)
        for z_km in (0.0, 1.0, 10.0):
            compensator = ChromaticDispersionFIRCompensator(z_km, fs=20e9)
            assert compensator.h is not None
            self.assertEqual(len(compensator.h), 1)
            self.assertAlmostEqual(float(np.sum(np.abs(compensator.h) ** 2)),
                                   1.0)

    def test_the_lo_broadcasts_over_a_polarization_pair(self):
        from comnumpy.optical.devices import (Laser, MachZehnderModulator,
                                              Optical90HybridCircuit)
        pair = np.ones((2, 8), dtype=complex)
        self.assertEqual(Optical90HybridCircuit(is_ideal=False)(pair).shape,
                         (2, 8))
        modulator = MachZehnderModulator(is_ideal=True,
                                         laser_in=Laser(0, theta0=0.0))
        self.assertEqual(modulator(pair).shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
