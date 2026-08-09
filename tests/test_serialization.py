"""Normative round-trip test for chain serialization (decision D32).

``from_json(to_json(chain))`` re-run with equal seeds must reproduce the
same signal. This single test covers types, parameters, arrays and
ordering; it is a merge condition for any format evolution.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from comnumpy import (AWGN, Sequential, SymbolDemapper, SymbolGenerator,
                      SymbolMapper, compute_ser, get_alphabet)
from comnumpy.core.compensators import TrainedBasedPhaseCompensator
from comnumpy.core.processors import Amplifier
from comnumpy.core.frames import (Deframer, FieldRole, FrameField,
                                  Framer, FrameStructure)
from comnumpy.core.processors import Parallel2Serial, Serial2Parallel
from comnumpy.ofdm.allocation import CarrierAllocation, get_allocation
from comnumpy.ofdm.processors import (CarrierAllocator, CarrierExtractor,
                                      CyclicPrefixer, CyclicPrefixRemover,
                                      FFTProcessor, IFFTProcessor)
from comnumpy.serialization import from_json, to_json


class TestRoundTrip(unittest.TestCase):

    def run_twice_and_compare(self, chain, stimulus, npz=False):
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "arrays.npz" if npz else None
            text = to_json(chain, npz_path=npz_path)
            rebuilt = from_json(text, npz_path=npz_path)
        y_ref = chain(stimulus)
        y_new = rebuilt(stimulus)
        np.testing.assert_array_equal(y_ref, y_new)
        return text

    def test_scalar_chain_roundtrip(self):
        chain = Sequential([
            SymbolGenerator(16, seed=11),
            AWGN(snr_dB=12, seed=12),
        ])
        # identical seeds -> identical noise -> identical signal
        text = self.run_twice_and_compare(chain, 1000)
        document = json.loads(text)
        self.assertEqual(document["comnumpy"], "1.0")
        self.assertEqual([b["id"] for b in document["blocks"]],
                         ["generator", "awgn"])
        # intent, not derived state: no rng, no sigma2_ in the JSON
        self.assertNotIn("rng", text)
        self.assertNotIn("sigma2_", text)
        # explicit inputs field from day one (D31)
        self.assertEqual(document["blocks"][1]["inputs"], ["generator"])

    def test_array_params_roundtrip_via_npz(self):
        alphabet = get_alphabet("QAM", 16)
        chain = Sequential([
            SymbolGenerator(16, seed=3),
            SymbolMapper(alphabet),
            AWGN(sigma2=0.01, seed=4),
            SymbolDemapper(alphabet),
        ])
        self.run_twice_and_compare(chain, 500, npz=True)

    def test_array_params_require_npz(self):
        chain = Sequential([SymbolMapper(get_alphabet("QAM", 4))])
        with self.assertRaises(ValueError):
            to_json(chain)

    def test_ofdm_chain_roundtrip(self):
        carrier_type = np.zeros(16)
        carrier_type[[1, 2, 3, 5, 6, 7]] = 1
        carrier_type[[4, 12]] = 2
        chain = Sequential([
            Serial2Parallel(6),
            CarrierAllocator(carrier_type, pilots=1.0),
            IFFTProcessor(),
            CyclicPrefixer(4),
            Parallel2Serial(),
            Serial2Parallel(20),
            CyclicPrefixRemover(4),
            FFTProcessor(),
            CarrierExtractor(carrier_type),
            Parallel2Serial(),
        ])
        rng = np.random.default_rng(7)
        stimulus = rng.normal(size=60) + 1j * rng.normal(size=60)
        self.run_twice_and_compare(chain, stimulus, npz=True)

    def test_shared_allocation_object_roundtrip(self):
        """The D18 form -- one CarrierAllocation shared TX/RX -- round trips.

        The mask form above is not enough: an allocation object is a
        *value dataclass* passed as a block parameter, a different decode
        path (it is not a Processor).
        """
        alloc = get_allocation("802.11a")
        chain = Sequential([
            Serial2Parallel(alloc.N_data),
            CarrierAllocator(alloc, pilots=1.0),
            IFFTProcessor(),
            FFTProcessor(),
            CarrierExtractor(alloc),
            Parallel2Serial(),
        ])
        rng = np.random.default_rng(3)
        n = 4 * alloc.N_data
        stimulus = rng.normal(size=n) + 1j * rng.normal(size=n)
        self.run_twice_and_compare(chain, stimulus, npz=True)

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "arrays.npz"
            rebuilt = from_json(to_json(chain, npz_path=npz), npz_path=npz)
        # the allocation comes back as an allocation, metadata included,
        # so the rebuilt chain still documents itself
        rebuilt_alloc = rebuilt.module_list[1].carrier_type
        self.assertIsInstance(rebuilt_alloc, CarrierAllocation)
        self.assertEqual(rebuilt_alloc.standard, "802.11a")

    def test_frame_structure_roundtrip(self):
        """FrameStructure/FrameField are value dataclasses too (D28)."""
        frame = FrameStructure([
            FrameField("sync", FieldRole.SYNC, values=np.ones(4, dtype=complex)),
            FrameField("payload", FieldRole.PAYLOAD, length=8),
        ])
        chain = Sequential([Framer(frame), Deframer(frame)])
        self.run_twice_and_compare(chain, np.arange(8) + 0j, npz=True)

    def test_field_role_survives_as_an_enum(self):
        """A JSON int must come back as FieldRole, not a bare int."""
        frame = FrameStructure([
            FrameField("sync", FieldRole.SYNC, values=np.ones(2, dtype=complex)),
            FrameField("payload", FieldRole.PAYLOAD, length=4),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "arrays.npz"
            rebuilt = from_json(to_json(Sequential([Framer(frame)]), npz_path=npz),
                                npz_path=npz)
        role = rebuilt.module_list[0].frame.fields[0].role
        self.assertIsInstance(role, FieldRole)
        self.assertEqual(role.name, "SYNC")

    def test_chain_metadata_survives(self):
        """taps and wiring are intent: a rebuilt chain still records/feeds."""
        alphabet = get_alphabet("QAM", 16)
        chain = Sequential([
            SymbolGenerator(16, seed=5, name="tx"),
            SymbolMapper(alphabet, name="ref"),
            Amplifier(np.exp(1j * 0.4)),
            AWGN(sigma2=0.01, seed=6),
            TrainedBasedPhaseCompensator(target_data=np.zeros(1), name="comp"),
            SymbolDemapper(alphabet),
        ], taps=["tx"], wiring={"comp.target_data": "ref"})
        self.run_twice_and_compare(chain, 500, npz=True)

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "arrays.npz"
            rebuilt = from_json(to_json(chain, npz_path=npz), npz_path=npz)
        self.assertEqual(rebuilt.taps, ["tx"])
        self.assertEqual(rebuilt.wiring, {"comp.target_data": "ref"})
        # the rebuilt chain is still usable end to end
        y = rebuilt(500)
        self.assertEqual(compute_ser(rebuilt.tap("tx"), y), 0.0)

    def test_complex_scalar_params_roundtrip(self):
        """Complex gains are ordinary in this domain; JSON has no complex."""
        gain = 0.6 - 0.8j
        chain = Sequential([Amplifier(gain, name="amp")])
        text = to_json(chain)
        self.assertIn("__complex__", text)
        rebuilt = from_json(text)
        self.assertEqual(rebuilt["amp"].gain, gain)

    def test_unknown_type_names_the_registration_hook(self):
        with self.assertRaises(KeyError) as ctx:
            from_json('{"comnumpy": "1.0", "blocks": ['
                      '{"id": "x", "type": "NoSuchBlock", "params": {}, "inputs": []}]}')
        self.assertIn("register_block", str(ctx.exception))

    def test_callable_params_are_a_documented_frontier(self):
        bad = Sequential([SymbolGenerator(4, seed=1)])
        bad.module_list.append(
            CarrierAllocator(np.array([1, 1]), name="alloc"))
        bad.module_list[-1].pilots = lambda: None  # a callable parameter
        with self.assertRaises(TypeError):
            to_json(bad, npz_path="unused.npz")


if __name__ == "__main__":
    unittest.main()
