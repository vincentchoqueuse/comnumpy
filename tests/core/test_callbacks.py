"""The callback table of ``Sequential``.

Callbacks predate the taps of D42 and stay for side effects a tap
cannot express (a progress line, a live plot). The table is optional,
and passing ``callbacks=None`` explicitly used to raise
``TypeError: argument of type 'NoneType' is not iterable`` from inside
the pass -- the annotation said the field was optional and the code
never allowed it.
"""
import unittest

import numpy as np

from comnumpy import AWGN, Sequential, SymbolGenerator


class TestCallbackTable(unittest.TestCase):

    def chain(self, **kwargs):
        return Sequential([SymbolGenerator(4, seed=1),
                           AWGN(snr_dB=10, name="noise", seed=2)], **kwargs)

    def test_an_explicit_none_runs_the_chain(self):
        self.assertEqual(len(self.chain(callbacks=None)(8)), 8)

    def test_an_explicit_none_matches_the_default(self):
        np.testing.assert_array_equal(self.chain(callbacks=None)(8),
                                      self.chain()(8))

    def test_a_callback_sees_the_output_of_its_block(self):
        seen = []
        chain = self.chain(callbacks={"noise": seen.append})
        y = chain(8)
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], y)

    def test_a_callback_on_an_unknown_name_is_simply_never_called(self):
        seen = []
        self.chain(callbacks={"nonexistent": seen.append})(8)
        self.assertEqual(seen, [])


if __name__ == "__main__":
    unittest.main()
