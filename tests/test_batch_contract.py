"""The batch promise, verified over the whole catalogue.

CONVENTIONS.md (D51) promises that the axes ahead of a block's event
axes are batch axes: independent trials in one call. A promise over a
catalogue is only worth the sweep that checks it, so this file is a
*ratchet*, like the pyright one (D37): every ``Processor`` subclass in
the library must appear in exactly one register below, and a discovery
test fails on any block that has not declared its batch behaviour.
Adding a block therefore *forces* the author to say -- and prove --
what a batch means for it.

Three verified families, one honest bucket:

* ``BROADCAST`` -- deterministic: row ``i`` of a batched call equals
  the block applied to row ``i`` alone. The strongest property, checked
  by equality.
* ``INDEPENDENT`` -- stochastic: the batched call runs, keeps the
  leading axis, and two rows do not share a realization.
* ``REFUSES`` -- the block cannot read a batch one way and must say so:
  a ``ShapeError`` (or documented ``ValueError``), never a silent
  misread.
* ``EXEMPT`` -- blocks the sweep cannot exercise generically, each with
  the reason written next to it. An exemption is visible debt, not a
  pass.
"""
import importlib
import inspect
import pkgutil
import unittest

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.core.utils import Constellation

RNG = np.random.default_rng(0)
N = 64
SERIAL = RNG.standard_normal((3, N)) + 1j * RNG.standard_normal((3, N))
INDICES = RNG.integers(0, 4, (3, N))
BITS = RNG.integers(0, 2, (3, N))
QPSK = Constellation("PSK", 4)
PAIR = RNG.standard_normal((3, 2, N)) + 1j * RNG.standard_normal((3, 2, N))


def _alamouti_pieces():
    # ONE 2-D channel broadcast over the batch: the row-equality check
    # runs the same factory on the batch and on one row, so the config
    # must not carry the batch itself (the stacked-H path is locked in
    # tests/mimo/test_stacked_channel.py)
    from comnumpy.mimo.channels import FlatMIMOChannel
    from comnumpy.mimo.coding import SpaceTimeEncoder, get_code
    from comnumpy.mimo.utils import rayleigh_channel
    channel = rayleigh_channel(1, 2, seed=3)
    encoded = SpaceTimeEncoder(get_code("alamouti"))(SERIAL)
    return channel, FlatMIMOChannel(channel)(encoded)


ALAMOUTI_H, ALAMOUTI_RX = _alamouti_pieces()
# a rotated QPSK frame set, for the blind trackers
QPSK_POINTS = np.asarray(QPSK.alphabet)
ROTATED_QPSK = QPSK_POINTS[RNG.integers(0, 4, (3, 200))] * np.exp(1j * 0.3)


def _fs():
    return {"fs": 32e9}


# --- the registers ---------------------------------------------------
# name: (factory, batched input). The factory builds a FRESH block each
# call -- the equality check runs it twice.

BROADCAST = {
    "core.channels.FIRChannel":
        (lambda: _cls("core.channels", "FIRChannel")(np.array([1.0, 0.4])),
         SERIAL),
    "core.compensators.LinearEqualizer":
        (lambda: _cls("core.compensators", "LinearEqualizer")(
            np.array([1.0, 0.4])), SERIAL),
    "core.compensators.BlindIQCompensator":
        (lambda: _cls("core.compensators", "BlindIQCompensator")(), SERIAL),
    "core.compensators.DCCorrector":
        (lambda: _cls("core.compensators", "DCCorrector")(), SERIAL),
    "core.devices.RappAmplifier":
        (lambda: _cls("core.devices", "RappAmplifier")(1.0), SERIAL),
    "core.devices.SalehAmplifier":
        (lambda: _cls("core.devices", "SalehAmplifier")(), SERIAL),
    "core.filters.BWFilter":
        (lambda: _cls("core.filters", "BWFilter")(0.3), SERIAL),
    "core.filters.SRRCFilter":
        (lambda: _cls("core.filters", "SRRCFilter")(0.25, 2, N_h=8), SERIAL),
    "core.impairments.CFO":
        (lambda: _cls("core.impairments", "CFO")(1e-3), SERIAL),
    "core.impairments.Delay":
        (lambda: _cls("core.impairments", "Delay")(3), SERIAL),
    "core.impairments.IQImbalance":
        (lambda: _cls("core.impairments", "IQImbalance")(1.0, 0.1), SERIAL),
    "core.mappers.SymbolMapper":
        (lambda: _cls("core.mappers", "SymbolMapper")(QPSK), INDICES),
    "core.mappers.SymbolDemapper":
        (lambda: _cls("core.mappers", "SymbolDemapper")(QPSK), SERIAL),
    "core.processors.Amplifier":
        (lambda: _cls("core.processors", "Amplifier")(2.0), SERIAL),
    "core.processors.Clipper":
        (lambda: _cls("core.processors", "Clipper")(0.8), SERIAL),
    "core.processors.Complex2Real":
        (lambda: _cls("core.processors", "Complex2Real")(), SERIAL),
    "core.processors.Downsampler":
        (lambda: _cls("core.processors", "Downsampler")(2), SERIAL),
    "core.processors.Upsampler":
        (lambda: _cls("core.processors", "Upsampler")(2), SERIAL),
    "core.processors.DelayRemover":
        (lambda: _cls("core.processors", "DelayRemover")(3), SERIAL),
    "core.processors.BlindPhaseTracker":
        (lambda: _cls("core.processors", "BlindPhaseTracker")(
            16, QPSK_POINTS), ROTATED_QPSK),
    "core.processors.Serial2Parallel":
        (lambda: _cls("core.processors", "Serial2Parallel")(8), SERIAL),
    "core.processors.Parallel2Serial":
        (lambda: _cls("core.processors", "Parallel2Serial")(),
         SERIAL.reshape(3, 8, 8)),
    "fec.convolutional.ConvolutionalEncoder":
        (lambda: _cls("fec.convolutional", "ConvolutionalEncoder")(), BITS),
    "mimo.channels.FlatMIMOChannel":
        (lambda: _cls("mimo.channels", "FlatMIMOChannel")(
            _rayleigh(2, 2, seed=1)), PAIR),
    "mimo.coding.SpaceTimeEncoder":
        (lambda: _cls("mimo.coding", "SpaceTimeEncoder")(
            _cls("mimo.coding", "get_code")("alamouti")), SERIAL),
    "mimo.coding.SpaceTimeDecoder":
        (lambda: _cls("mimo.coding", "SpaceTimeDecoder")(
            _cls("mimo.coding", "get_code")("alamouti"), H=ALAMOUTI_H),
         ALAMOUTI_RX),
    "mimo.detectors.LinearDetector":
        (lambda: _cls("mimo.detectors", "LinearDetector")(
            QPSK, H=_rayleigh(2, 2, seed=1), method="zf"), PAIR),
    "ofdm.chains.OFDMTransmitter":
        (lambda: _cls("ofdm.chains", "OFDMTransmitter")(16, 4), SERIAL),
    "ofdm.chains.OFDMReceiver":
        (lambda: _cls("ofdm.chains", "OFDMReceiver")(16, 4),
         SERIAL[:, :60].repeat(2, axis=-1)[:, :80].reshape(3, 80)),
    "ofdm.compensators.FrequencyDomainEqualizer":
        (lambda: _cls("ofdm.compensators", "FrequencyDomainEqualizer")(
            h=np.array([1.0, 0.3])), SERIAL.reshape(3, 4, 16)),
    "ofdm.processors.FFTProcessor":
        (lambda: _cls("ofdm.processors", "FFTProcessor")(),
         SERIAL.reshape(3, 4, 16)),
    "ofdm.processors.IFFTProcessor":
        (lambda: _cls("ofdm.processors", "IFFTProcessor")(),
         SERIAL.reshape(3, 4, 16)),
    "ofdm.processors.CyclicPrefixer":
        (lambda: _cls("ofdm.processors", "CyclicPrefixer")(4),
         SERIAL.reshape(3, 4, 16)),
    "ofdm.processors.CyclicPrefixRemover":
        (lambda: _cls("ofdm.processors", "CyclicPrefixRemover")(4),
         SERIAL.reshape(3, 4, 16)),
    "ofdm.predistorders.HardClipper":
        (lambda: _cls("ofdm.predistorders", "HardClipper")(0.8), SERIAL),
    "optical.channels.ChromaticDispersion":
        (lambda: _cls("optical.channels", "ChromaticDispersion")(
            100, **_fs()), SERIAL),
    "optical.channels.KerrNonLinearity":
        (lambda: _cls("optical.channels", "KerrNonLinearity")(1e-3), SERIAL),
    "optical.compensators.ChromaticDispersionFIRCompensator":
        (lambda: _cls("optical.compensators",
                      "ChromaticDispersionFIRCompensator")(100, **_fs()),
         SERIAL),
    "optical.compensators.ChromaticDispersionLSFIRCompensator":
        (lambda: _cls("optical.compensators",
                      "ChromaticDispersionLSFIRCompensator")(
            100, 15, w_vect=[-np.pi, np.pi], **_fs()), SERIAL),
}

INDEPENDENT = {
    "core.channels.AWGN":
        (lambda: _cls("core.channels", "AWGN")(sigma2=1.0),
         np.zeros((3, 4096), dtype=complex)),
    "core.generators.SymbolGenerator":
        (lambda: _cls("core.generators", "SymbolGenerator")(4), (3, 256)),
    "core.generators.GaussianGenerator":
        (lambda: _cls("core.generators", "GaussianGenerator")(), (3, 256)),
    "optical.channels.PhaseNoise":
        (lambda: _cls("optical.channels", "PhaseNoise")(1e-3, per="row"),
         np.ones((3, 512), dtype=complex)),
}

REFUSES = {
    "core.compensators.BlindCFOCompensator":
        (lambda: _cls("core.compensators", "BlindCFOCompensator")(),
         np.ones((3, N), dtype=complex)),
    "core.compensators.DataAidedPhaseCompensator":
        (lambda: _cls("core.compensators", "DataAidedPhaseCompensator")(
            reference=SERIAL[0]), SERIAL),
    "core.compensators.DataAidedComplexGainCompensator":
        (lambda: _cls("core.compensators",
                      "DataAidedComplexGainCompensator")(
            reference=SERIAL[0]), SERIAL),
    "core.compensators.DataAidedFIRCompensator":
        (lambda: _cls("core.compensators", "DataAidedFIRCompensator")(
            reference=SERIAL[0]), SERIAL),
}

EXEMPT = {
    # abstract bases and containers
    "mimo.channels.BaseMIMOChannel": "abstract base, forward raises",
    "core.compensators.Normalizer": "the gain is ONE scalar measured "
        "over the whole array -- a documented estimand, batch rows "
        "pooled by design; the docstring says so",
    # generic event = the whole 1-D stream by declared design; a batch
    # has no meaning the docstring does not already refuse
    "core.compensators.DataAidedFineSynchronizer":
        "estimates one delay against one 1-D reference",
    "core.compensators.DataAidedSimpleSynchronizer":
        "estimates one delay against one 1-D reference",
    "core.frames.Framer": "frames (..., T, N) against a "
        "FrameStructure; needs one, not generically constructible here",
    "core.frames.Deframer": "same FrameStructure dependency",
    "core.processors.AutoConcatenator": "its masks are configured by a "
        "companion block (HermitianPrefixer); not generically "
        "constructible",

    "core.processors.DataAdder": "adds a stored companion signal of the "
        "un-batched shape",
    "core.processors.DataExtractor": "extracts against stored 1-D markers",

    "core.processors.SampleRemover": "drops N samples of one stream",
    "core.processors.Resampler": "scipy.signal.resample along a declared "
        "axis; not swept here because its output length depends on the "
        "rate argument",
    "core.processors.WeightAmplifier": "per-branch gains tied to the "
        "un-batched branch axis",
    "core.compensators.BlindPhaseCompensation": "legacy single-signal "
        "moment estimator (superseded by BlindPhaseSearchCompensator)",
    "core.shaping.AmplitudeMapper": "shaping operates on one 1-D "
        "amplitude stream",
    "core.shaping.AmplitudeDemapper": "shaping operates on one 1-D "
        "amplitude stream",
    "core.shaping.DistributionMatcher": "blocks of n_bits along the "
        "last axis, leading axes carried through; not swept for want of "
        "a canonical probe distribution",
    "core.shaping.DistributionDematcher": "blocks of `length` along the "
        "last axis, leading axes carried through; same",
    "core.channels.TappedDelayLineChannel": "frozen-realization channel: "
        "one seeded draw is the configuration (D51), sounded via "
        "impulse_response",
    "fec.convolutional.ViterbiDecoder": "decodes (n_cw, n) codeword "
        "blocks -- its batch axis is the codeword axis, locked in "
        "tests/fec",
    "fec.ldpc.LDPCEncoder": "operates on (k, n_cw) codeword blocks, "
        "locked in tests/fec",
    "fec.ldpc.LDPCDecoder": "operates on (n, n_cw) LLR blocks, locked "
        "in tests/fec",
    "mimo.channels.SelectiveMIMOChannel": "builds an (L N_t, N) stacked "
        "convolution matrix for one frame",
    "mimo.compensators.BlindDualMIMOCompensator": "adaptive per pair; its "
        "batch (one butterfly per pair) is locked in "
        "tests/core/test_batch_axes.py",
    "core.compensators.BlindPhaseSearchCompensator": "adaptive per row; "
        "its per-row trajectories are locked in "
        "tests/optical/test_pmd_bps.py",
    "mimo.detectors.MaximumLikelihoodDetector": "stacked-H batch locked "
        "in tests/mimo/test_stacked_channel.py",
    "mimo.detectors.SphereDecoder": "stacked-H batch locked in "
        "tests/mimo/test_stacked_channel.py",
    "mimo.detectors.OrderedSuccessiveInterferenceCancellationDetector":
        "stacked-H batch locked in tests/mimo/test_stacked_channel.py",
    "mimo.detectors.ApproximateMessagePassingDetector": "iterative "
        "per-frame message passing against one H",
    "mimo.detectors.OrthogonalApproximateMessagePassingDetector":
        "iterative per-frame message passing against one H",
    "ofdm.processors.CarrierAllocator": "validated (..., T, F_data) "
        "layout, locked in tests/ofdm",
    "ofdm.processors.CarrierExtractor": "validated (..., T, F) layout, "
        "locked in tests/ofdm",
    "ofdm.processors.HermitianPrefixer": "builds the Hermitian spectrum "
        "of one block set",
    "ofdm.predistorders.IctPaprReductor": "iterative clipping on one "
        "block set",
    "ofdm.predistorders.PtsPaprReductor": "phase search over one block "
        "set",
    "optical.channels.PMDEmulator": "frozen-realization channel (D51): "
        "polarization pair is the event, locked in "
        "tests/optical/test_pmd_bps.py",
    "optical.dbp.DBP": "SSFM split-step over one field, span state along "
        "the last axis; validated in validation/",
    "optical.devices.ErbiumDopedFiberAmplifier": "gain plus ASE draw "
        "sized on the whole input; the optical event is the field",
    "optical.devices.Laser": "one laser per instance by physics; its "
        "sharing convention lives in PhaseNoise per=",
    "optical.devices.MachZehnderModulator": "drives one modulator with "
        "one electrical signal",
    "optical.devices.Optical90HybridCircuit": "mixes one signal pair by "
        "construction",
    "optical.devices.PowerControl": "normalizes one field's power",
    "optical.links.FiberLink": "SSFM propagation of one field; EDFA "
        "noise inside; validated against closed forms in validation/",
    "optical.wdm.WDMMultiplexer": "combines a channel *set* -- its "
        "leading axis is the WDM channel, not a batch",
    "optical.wdm.WDMDemultiplexer": "splits into a channel set",
}


def _cls(module: str, name: str):
    return getattr(importlib.import_module(f"comnumpy.{module}"), name)


def _rayleigh(n_r: int, n_t: int, seed: int) -> np.ndarray:
    from comnumpy.mimo.utils import rayleigh_channel
    return rayleigh_channel(n_r, n_t, seed=seed)


def discover() -> dict[str, type]:
    """Every Processor subclass defined in the library, by short name."""
    found = {}
    for pkg in ("core", "optical", "mimo", "ofdm", "fec"):
        package = importlib.import_module(f"comnumpy.{pkg}")
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("_"):
                continue
            module = importlib.import_module(f"comnumpy.{pkg}.{info.name}")
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (issubclass(cls, Processor) and cls is not Processor
                        and cls.__module__ == module.__name__):
                    short = f"{pkg}.{info.name}.{name}"
                    found[short] = cls
    return found


class TestTheCatalogueIsCovered(unittest.TestCase):
    """The ratchet: a new block must declare its batch behaviour."""

    def test_every_block_is_declared_exactly_once(self):
        declared = (set(BROADCAST) | set(INDEPENDENT) | set(REFUSES)
                    | set(EXEMPT))
        discovered = set(discover())
        missing = sorted(discovered - declared)
        self.assertFalse(
            missing,
            f"blocks with no declared batch behaviour: {missing} -- add "
            f"each to BROADCAST, INDEPENDENT or REFUSES in "
            f"tests/test_batch_contract.py with a probe input, or to "
            f"EXEMPT with the reason written out (D51)")
        stale = sorted(declared - discovered)
        self.assertFalse(stale, f"declared but no longer found: {stale}")
        for name in sorted(declared):
            registers = [name in r for r in
                         (BROADCAST, INDEPENDENT, REFUSES, EXEMPT)]
            self.assertEqual(sum(registers), 1,
                             f"{name} appears in {sum(registers)} registers")


class TestBroadcastFamily(unittest.TestCase):
    """Row i of a batched call equals the block applied to row i."""

    def test_row_equality(self):
        for name, (factory, x) in BROADCAST.items():
            with self.subTest(block=name):
                batched = factory()(np.asarray(x))
                single = factory()(np.asarray(x)[1])
                self.assertEqual(batched.shape[0], 3,
                                 f"{name} lost the batch axis")
                np.testing.assert_allclose(
                    np.asarray(batched)[1], np.asarray(single),
                    err_msg=f"{name}: batched row 1 differs from the "
                            f"block applied to row 1 alone")


class TestIndependentFamily(unittest.TestCase):
    """Stochastic blocks draw independently per trial."""

    def test_rows_do_not_share_a_realization(self):
        for name, (factory, x) in INDEPENDENT.items():
            with self.subTest(block=name):
                y = np.asarray(factory()(x))
                self.assertEqual(y.shape[0], 3,
                                 f"{name} lost the batch axis")
                self.assertFalse(
                    np.array_equal(y[0], y[1]),
                    f"{name}: two trials of a batch share a realization")


class TestRefusesFamily(unittest.TestCase):
    """What cannot be read one way is refused, never misread."""

    def test_the_batch_is_refused(self):
        for name, (factory, x) in REFUSES.items():
            with self.subTest(block=name):
                with self.assertRaises(ValueError,
                                       msg=f"{name} accepted a batch it "
                                           f"cannot interpret"):
                    factory()(x)


if __name__ == "__main__":
    unittest.main()
