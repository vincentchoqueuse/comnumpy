"""A rate change must not carry its anti-alias filter written by hand.

The regression this exists for was real and silent. Two optical examples
band-limited with ``BWFilter(1 / oversampling_sim)`` and then decimated by
``oversampling_ratio`` -- a different number. The mask kept +/-16 GHz of a
35.2 GHz root-raised-cosine spectrum, so it cut the roll-off shoulders off
the signal it was meant to pass, and every curve measured through that
chain sat on a 22.9 dB distortion floor that looked like a channel
impairment.

Nothing in the chain connected the two numbers, because the cutoff was
stated twice: once as ``L``, once as ``wn``. That is the defect D41 exists
to prevent, and ``Downsampler(L, use_filter=True)`` already derives one
from the other. ``Sequential`` now says so when it sees the pair.
"""
import logging
import unittest

from comnumpy.core import Sequential
from comnumpy.core.filters import BWFilter
from comnumpy.core.processors import Downsampler, Upsampler


class TestResamplingFilterCheck(unittest.TestCase):

    def build(self, modules):
        """Build a chain and return whatever it warned about."""
        with self.assertLogs("comnumpy.core.generics", level="WARNING") as log:
            Sequential(modules)
        return "\n".join(log.output)

    def silent(self, modules):
        """Build a chain that must warn about nothing."""
        logger = logging.getLogger("comnumpy.core.generics")
        with self.assertNoLogs(logger, level="WARNING"):
            Sequential(modules)

    def test_the_regression_itself(self):
        """BWFilter(1/6) before Downsampler(3): the examples' actual bug."""
        message = self.build([BWFilter(1 / 6), Downsampler(3)])
        self.assertIn("0.166667", message)          # the cutoff written
        self.assertIn("0.333333", message)          # the cutoff needed
        self.assertIn("throwing signal away", message)
        self.assertIn("use_filter=True", message)

    def test_a_mask_wider_than_the_rate_change_names_the_folding(self):
        message = self.build([BWFilter(0.9), Downsampler(3)])
        self.assertIn("folds", message)

    def test_the_anti_imaging_side_is_checked_too(self):
        """An interpolator's filter comes after it, and can be as wrong."""
        message = self.build([Upsampler(4), BWFilter(1 / 6)])
        self.assertIn("Upsampler(4, use_filter=True)", message)
        self.assertIn("anti-imaging", message)

    def test_a_correct_hand_written_pair_is_still_flagged(self):
        """Right today, and one edit from being wrong tomorrow."""
        message = self.build([BWFilter(1 / 3), Downsampler(3)])
        self.assertIn("written once", message)
        self.assertNotIn("cutoff is wrong", message)

    def test_the_argument_form_is_silent(self):
        self.silent([Downsampler(3, use_filter=True)])
        self.silent([Upsampler(4, use_filter=True)])

    def test_an_unrelated_adjacency_is_silent(self):
        """A filter *after* a decimation is not that decimation's filter."""
        self.silent([Downsampler(3), BWFilter(1 / 6)])
        self.silent([BWFilter(0.5), Upsampler(2)])

    def test_a_deliberate_second_filter_is_left_alone(self):
        """use_filter=True already anti-aliases; the mask does another job."""
        self.silent([BWFilter(0.05), Downsampler(3, use_filter=True)])

    def test_the_check_does_not_disturb_the_chain(self):
        """It inspects the module list and changes nothing in it."""
        modules = [BWFilter(1 / 6), Downsampler(3)]
        with self.assertLogs("comnumpy.core.generics", level="WARNING"):
            chain = Sequential(modules)
        self.assertEqual(chain.module_list, modules)
        self.assertEqual(modules[0].wn, 1 / 6)
        self.assertFalse(modules[1].use_filter)


if __name__ == "__main__":
    unittest.main()
