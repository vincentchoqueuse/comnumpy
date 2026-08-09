"""Chain taps: signal probes as chain metadata, not in-chain blocks."""
import unittest

import numpy as np

from comnumpy import AWGN, Recorder, Sequential, SymbolDemapper, \
    SymbolGenerator, SymbolMapper, get_alphabet


class TestTaps(unittest.TestCase):

    def build(self, **kwargs):
        alphabet = get_alphabet("QAM", 16)
        return Sequential([
            SymbolGenerator(16),
            SymbolMapper(alphabet),
            AWGN(snr_dB=15, name="noise"),
            SymbolDemapper(alphabet),
        ], **kwargs)

    def test_tap_equals_recorder(self):
        """A tap records exactly what an in-chain Recorder would."""
        alphabet = get_alphabet("QAM", 16)
        tapped = self.build(taps=["generator"]).seed(3)
        recorded = Sequential([
            SymbolGenerator(16),
            Recorder(name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=15, name="noise"),
            SymbolDemapper(alphabet),
        ]).seed(3)
        y_tap = tapped(1000)
        y_rec = recorded(1000)
        np.testing.assert_array_equal(y_tap, y_rec)
        np.testing.assert_array_equal(tapped.tap("generator"),
                                      recorded["tx"].get_data())

    def test_module_list_stays_pure(self):
        """The chain description contains communication blocks only."""
        chain = self.build(taps=["generator", "noise"])
        self.assertEqual(len(chain.module_list), 4)
        chain(100)
        self.assertEqual(sorted(chain.tapped_), ["generator", "noise"])

    def test_unknown_tap_rejected_at_run(self):
        chain = self.build(taps=["nope"])
        with self.assertRaises(KeyError):
            chain(10)

    def test_tap_before_run_raises(self):
        chain = self.build(taps=["generator"])
        with self.assertRaises(KeyError):
            chain.tap("generator")


if __name__ == "__main__":
    unittest.main()
