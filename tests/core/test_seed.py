"""Chain seeding is deterministic and per-block independent (decision D6)."""
import unittest

import numpy as np

from comnumpy import AWGN, Sequential, SymbolGenerator


class TestChainSeed(unittest.TestCase):

    def build(self):
        return Sequential([SymbolGenerator(4), AWGN(snr_dB=10)])

    def test_same_seed_same_signal(self):
        chain = self.build()
        y1 = chain.seed(42)(1000)
        y2 = chain.seed(42)(1000)
        np.testing.assert_array_equal(y1, y2)

    def test_two_chains_same_seed_same_signal(self):
        y1 = self.build().seed(7)(500)
        y2 = self.build().seed(7)(500)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        chain = self.build()
        y1 = chain.seed(1)(1000)
        y2 = chain.seed(2)(1000)
        self.assertFalse(np.array_equal(y1, y2))

    def test_blocks_get_distinct_seeds(self):
        chain = self.build()
        chain.seed(3)
        self.assertNotEqual(chain.module_list[0].seed, chain.module_list[1].seed)


if __name__ == "__main__":
    unittest.main()
