"""Chain wiring: feeding a data-aided block from an upstream block.

Some estimators need a reference signal produced *inside* the chain (the
transmitted symbols, a training field). A frozen ``reference=`` array
cannot express that: on the second run it silently refers to the previous
run's data. ``wiring`` declares the edge as chain metadata, so the block
receives the current pass's signal without ever holding a reference to
another block.
"""
import unittest

import numpy as np

from comnumpy import (AWGN, Sequential, SymbolDemapper, SymbolGenerator,
                      SymbolMapper, compute_ser, get_alphabet)
from comnumpy.core.compensators import DataAidedPhaseCompensator
from comnumpy.core.processors import Amplifier

PHASE = 0.4


class TestWiring(unittest.TestCase):

    def build(self, **kwargs):
        alphabet = get_alphabet("QAM", 16)
        return Sequential([
            SymbolGenerator(16, name="tx"),
            SymbolMapper(alphabet, name="ref"),
            Amplifier(np.exp(1j * PHASE)),
            AWGN(sigma2=0.01, name="noise"),
            DataAidedPhaseCompensator(reference=np.zeros(1), name="comp"),
            SymbolDemapper(alphabet),
        ], **kwargs)

    def test_reference_follows_the_current_run(self):
        """The regression this exists for: no stale reference across runs."""
        chain = self.build(taps=["tx"], wiring={"comp.reference": "ref"})
        for _ in range(3):
            y = chain(2000)
            # the compensator saw this pass's reference, not a previous one
            np.testing.assert_array_equal(chain["comp"].reference,
                                          chain.tapped_["ref"])
            self.assertAlmostEqual(chain["comp"].theta_, -PHASE, places=1)
            self.assertEqual(compute_ser(chain.tap("tx"), y), 0.0)

    def test_frozen_reference_would_be_stale(self):
        """Without wiring the reference is frozen -- documents the trap."""
        chain = self.build()
        first = chain(500)
        del first
        frozen = chain["comp"].reference.copy()
        chain(500)
        np.testing.assert_array_equal(chain["comp"].reference, frozen)

    def test_source_is_tapped_automatically(self):
        chain = self.build(wiring={"comp.reference": "ref"})
        chain(100)
        self.assertIn("ref", chain.tapped_)

    def test_module_list_stays_pure(self):
        chain = self.build(wiring={"comp.reference": "ref"})
        self.assertEqual(len(chain.module_list), 6)
        self.assertNotIn("wiring", repr(chain.module_list))

    def test_backward_edge_rejected(self):
        """A source running after its target would serve the previous run."""
        alphabet = get_alphabet("QAM", 4)
        chain = Sequential([
            SymbolGenerator(4, name="tx"),
            DataAidedPhaseCompensator(reference=np.zeros(1), name="comp"),
            SymbolMapper(alphabet, name="ref"),
        ], wiring={"comp.reference": "ref"})
        with self.assertRaises(ValueError) as ctx:
            chain(10)
        self.assertIn("previous run", str(ctx.exception))

    def test_graph_shows_the_data_edge(self):
        """A picture of the chain must show every edge it actually has."""
        chain = self.build(taps=["tx"], wiring={"comp.reference": "ref"})
        model = chain.graph()
        self.assertEqual(model["data_edges"], [("ref", "comp", "reference")])
        self.assertEqual(len(model["signal_edges"]), 5)
        self.assertEqual(model["taps"], ["tx"])

        mermaid = chain.to_mermaid()
        self.assertIn("ref -.->|reference| comp", mermaid)   # dashed, labelled
        self.assertIn("class tx tapped", mermaid)
        # the signal path is still drawn with solid arrows
        self.assertIn("tx --> ref", mermaid)

    def test_graph_without_wiring_has_no_data_edge(self):
        model = self.build().graph()
        self.assertEqual(model["data_edges"], [])
        self.assertEqual(model["taps"], [])

    def test_unknown_ids_and_malformed_keys_rejected(self):
        for wiring in ({"comp.reference": "nope"},
                       {"nope.reference": "ref"},
                       {"comp": "ref"}):
            with self.subTest(wiring=wiring):
                with self.assertRaises(KeyError):
                    self.build(wiring=wiring)(10)


if __name__ == "__main__":
    unittest.main()
