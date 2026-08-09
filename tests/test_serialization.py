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

from comnumpy import (AWGN, Sequential, SymbolDemapper,
                      SymbolGenerator, SymbolMapper, get_alphabet)
from comnumpy.core.processors import Parallel2Serial, Serial2Parallel
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

    def test_callable_params_are_a_documented_frontier(self):
        bad = Sequential([SymbolGenerator(4, seed=1)])
        bad.module_list.append(
            CarrierAllocator(np.array([1, 1]), name="alloc"))
        bad.module_list[-1].pilots = lambda: None  # a callable parameter
        with self.assertRaises(TypeError):
            to_json(bad, npz_path="unused.npz")


if __name__ == "__main__":
    unittest.main()
