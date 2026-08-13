"""Addressing a block's parameter, in the two spellings of one address.

`set_params` was borrowed from scikit-learn but not its separator, so a
parameter written by hand cost a `**{...}` wrapper around a string --
`chain.set_params(**{"fibre.use_only_linear": True})` for one boolean.
The double underscore is the same address spelled as an identifier, which
is what scikit-learn uses, so it can be a plain keyword argument.

The dotted form stays: `monte_carlo` builds its addresses as strings, and a
string is what you want when the address is computed rather than typed.
"""
import unittest

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator


class TestTheTwoSpellings(unittest.TestCase):

    def setUp(self):
        self.chain = Sequential([SymbolGenerator(4, name="source"),
                                 AWGN(sigma2=0.1, name="noise")])

    def test_the_dotted_form_still_works(self):
        self.chain.set_params(**{"noise.sigma2": 0.02})
        self.assertEqual(self.chain["noise"].sigma2, 0.02)

    def test_the_underscore_form_sets_the_same_field(self):
        self.chain.set_params(noise__sigma2=0.02)
        self.assertEqual(self.chain["noise"].sigma2, 0.02)

    def test_several_blocks_in_one_call(self):
        self.chain.set_params(noise__sigma2=0.02, source__M=16)
        self.assertEqual(self.chain["noise"].sigma2, 0.02)
        self.assertEqual(self.chain["source"].M, 16)

    def test_the_call_returns_the_chain(self):
        self.assertIs(self.chain.set_params(noise__sigma2=0.02), self.chain)


class TestWhatItRefuses(unittest.TestCase):

    def setUp(self):
        self.chain = Sequential([AWGN(sigma2=0.1, name="noise")])

    def test_an_unknown_block_is_named(self):
        with self.assertRaises(KeyError) as caught:
            self.chain.set_params(nosuch__sigma2=1.0)
        self.assertIn("nosuch", str(caught.exception))

    def test_an_unknown_field_lists_the_real_ones(self):
        with self.assertRaises(AttributeError) as caught:
            self.chain.set_params(noise__nosuch=1.0)
        self.assertIn("sigma2", str(caught.exception))

    def test_an_address_with_no_field_is_refused(self):
        with self.assertRaises(ValueError):
            self.chain.set_params(noise=1.0)


class TestTheSplitIsUnambiguous(unittest.TestCase):
    """The first ``__`` is the separator, and it can be.

    That holds because `block_ids` collapses every run of
    non-alphanumerics into a *single* underscore, so no id can contain a
    double one. The rule is what makes the underscore spelling safe, so
    it is pinned here rather than left as a reading of the regex.
    """

    def test_an_underscored_block_id_is_addressed(self):
        chain = Sequential([AWGN(sigma2=0.1, name="data_rx_eq")])
        chain.set_params(data_rx_eq__sigma2=0.05)
        self.assertEqual(chain["data_rx_eq"].sigma2, 0.05)

    def test_no_block_id_can_contain_a_double_underscore(self):
        chain = Sequential([AWGN(sigma2=0.1, name="odd__name"),
                            AWGN(sigma2=0.1, name="two  spaces"),
                            AWGN(sigma2=0.1, name="Mixed-Case__Thing")])
        for block_id in chain.block_ids():
            self.assertNotIn("__", block_id)
        self.assertEqual(chain.block_ids(),
                         ["odd_name", "two_spaces", "mixed_case_thing"])


class TestThePrecomputationIsReRun(unittest.TestCase):

    def test_a_changed_variance_is_actually_applied(self):
        """`sigma2_` is derived in __post_init__, not stored by the caller."""
        chain = Sequential([AWGN(sigma2=1.0, name="noise")])
        chain.seed(0)
        loud = chain(np.zeros(4096, dtype=complex))
        chain.set_params(noise__sigma2=0.01)
        chain.seed(0)
        quiet = chain(np.zeros(4096, dtype=complex))
        self.assertLess(np.var(quiet), np.var(loud) / 10)


if __name__ == "__main__":
    unittest.main()
