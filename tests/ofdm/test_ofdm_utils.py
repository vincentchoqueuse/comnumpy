"""Guards on the deprecated allocation helper.

``get_standard_carrier_allocation`` is superseded by
:func:`comnumpy.ofdm.allocation.get_allocation`, but it stays until the
deprecation window closes and it should fail readably meanwhile: asking
for ``"Custom"`` without ``custom=`` used to unpack ``None`` and raise a
bare ``TypeError`` from inside the function body.
"""
import unittest
import warnings

import numpy as np

from comnumpy.ofdm.utils import get_standard_carrier_allocation


def allocation(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return get_standard_carrier_allocation(*args, **kwargs)


class TestArgumentGuards(unittest.TestCase):

    def test_custom_without_the_parameter_says_what_to_pass(self):
        with self.assertRaises(ValueError) as ctx:
            allocation("Custom")
        message = str(ctx.exception)
        self.assertIn("custom=", message)
        self.assertIn("N_nulled_DC", message)

    def test_a_custom_of_the_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            allocation("Custom", custom=[64, 0, 0, 0])
        self.assertIn("expected 5", str(ctx.exception))

    def test_an_unknown_name_lists_the_known_ones(self):
        with self.assertRaises(KeyError) as ctx:
            allocation("802.11ah_1024")
        self.assertIn("802.11ah_512", str(ctx.exception))

    def test_a_valid_custom_still_works(self):
        carrier_type = allocation("Custom", custom=[16, 0, 0, 0, [2, 5]],
                                  shift=True)
        self.assertEqual(len(carrier_type), 16)
        np.testing.assert_array_equal(np.where(carrier_type == 2)[0], [2, 5])


class TestStandardConfigurations(unittest.TestCase):

    def test_the_length_follows_the_oversampling(self):
        for os in (1, 2, 4):
            with self.subTest(os=os):
                self.assertEqual(len(allocation("NoPilot_Full_64", os=os)),
                                 64 * os)

    def test_the_pilot_count_matches_the_table(self):
        carrier_type = allocation("802.11ah_64", shift=True)
        self.assertEqual(int(np.sum(carrier_type == 2)), 4)


if __name__ == "__main__":
    unittest.main()
