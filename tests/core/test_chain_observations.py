"""Chain observations: retained outputs as chain metadata, not in-chain blocks."""
import unittest

import numpy as np

from comnumpy import AWGN, Sequential, SymbolDemapper, \
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

    def test_tap_records_the_block_output(self):
        """An observation retains exactly the observed block output, untouched."""
        alphabet = get_alphabet("QAM", 16)
        chain = self.build(observations=["generator", "symbol_mapper"]).seed(3)
        chain(1000)
        symbols = chain.observation("generator")
        # the mapper observation must be the exact image of the generator one
        np.testing.assert_array_equal(chain.observation("symbol_mapper"),
                                      alphabet[symbols])
        # observations do not change the chain output: same seed, none declared
        y_ref = self.build().seed(3)(1000)
        np.testing.assert_array_equal(
            self.build(observations=["generator"]).seed(3)(1000), y_ref)

    def test_module_list_stays_pure(self):
        """The chain description contains communication blocks only."""
        chain = self.build(observations=["generator", "noise"])
        self.assertEqual(len(chain.module_list), 4)
        chain(100)
        self.assertEqual(sorted(chain.observed_), ["generator", "noise"])

    def test_unknown_tap_rejected_at_run(self):
        chain = self.build(observations=["nope"])
        with self.assertRaises(KeyError):
            chain(10)

    def test_tap_before_run_raises(self):
        chain = self.build(observations=["generator"])
        with self.assertRaises(KeyError):
            chain.observation("generator")


if __name__ == "__main__":
    unittest.main()
