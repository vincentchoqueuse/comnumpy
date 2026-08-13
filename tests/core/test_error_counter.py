"""Counting errors across the trials of a Monte-Carlo point.

Four example scripts accumulate errors over trials, each with its own
loop, and three of them wrote ``+= compute_ser(...) / n_trials``. That
averages *rates*, which is only the same answer when every trial has the
same length, and it discards the count the point's credibility rests on.
The counter keeps the totals; the loop stays where it was.
"""
import unittest

import numpy as np

from comnumpy.core.metrics import ErrorCounter, compute_ber, compute_ser
from comnumpy.exceptions import NotFittedError


class TestErrorCounter(unittest.TestCase):

    def test_one_trial_agrees_with_compute_ser(self):
        rng = np.random.default_rng(0)
        sent = rng.integers(0, 16, 500)
        received = sent.copy()
        received[rng.choice(500, 20, replace=False)] ^= 1
        counter = ErrorCounter().update(sent, received)
        self.assertAlmostEqual(counter.rate, compute_ser(sent, received))

    def test_bits_agree_with_compute_ber(self):
        rng = np.random.default_rng(1)
        sent = rng.integers(0, 16, 500)
        received = rng.integers(0, 16, 500)
        counter = ErrorCounter(width=4).update(sent, received)
        self.assertAlmostEqual(counter.rate, compute_ber(sent, received, 4))

    def test_it_is_a_ratio_of_totals_not_a_mean_of_rates(self):
        """The regression this exists for: unequal trials.

        One error in ten symbols, then none in a thousand. The honest
        rate is 1/1010; averaging the two rates gives 1/20, fifty times
        larger, and that is what a loop dividing by the trial count
        computes.
        """
        counter = ErrorCounter()
        counter.update(np.zeros(10, dtype=int),
                       np.array([1] + [0] * 9))
        counter.update(np.zeros(1000, dtype=int), np.zeros(1000, dtype=int))
        self.assertAlmostEqual(counter.rate, 1 / 1010)
        mean_of_rates = (0.1 + 0.0) / 2
        self.assertNotAlmostEqual(counter.rate, mean_of_rates)

    def test_equal_trials_reproduce_the_loop_being_replaced(self):
        """Where the examples stand today, the answer must not move."""
        rng = np.random.default_rng(2)
        counter = ErrorCounter()
        by_hand = 0.0
        n_trials = 4
        for _ in range(n_trials):
            sent = rng.integers(0, 4, 250)
            received = rng.integers(0, 4, 250)
            counter.update(sent, received)
            by_hand += compute_ser(sent, received) / n_trials
        self.assertAlmostEqual(counter.rate, by_hand)

    def test_counts_are_reported_beside_the_rate(self):
        counter = ErrorCounter()
        counter.update(np.zeros(100, dtype=int), np.zeros(100, dtype=int))
        counter.update(np.zeros(100, dtype=int),
                       np.array([1, 1] + [0] * 98))
        self.assertEqual((counter.n_errors, counter.n_symbols,
                          counter.n_trials), (2, 200, 2))

    def test_a_shorter_estimate_sets_the_length(self):
        """A chain with a filter delay returns fewer symbols than it got."""
        counter = ErrorCounter().update(np.zeros(100, dtype=int),
                                        np.zeros(90, dtype=int))
        self.assertEqual(counter.n_symbols, 90)

    def test_update_chains_and_reset_forgets(self):
        counter = ErrorCounter(width=2)
        counter.update(np.zeros(8, dtype=int), np.zeros(8, dtype=int))
        self.assertIs(counter.reset(), counter)
        self.assertEqual((counter.n_errors, counter.n_trials), (0, 0))
        self.assertEqual(counter.width, 2)      # reset keeps the unit

    def test_a_rate_before_any_trial_is_refused(self):
        """An empty average is a NaN waiting to reach a figure (D38)."""
        with self.assertRaises(NotFittedError) as ctx:
            _ = ErrorCounter().rate
        self.assertIn("update", str(ctx.exception))

    def test_str_says_counts_and_rate(self):
        counter = ErrorCounter()
        self.assertIn("nothing counted", str(counter))
        counter.update(np.zeros(1000, dtype=int),
                       np.array([1] + [0] * 999))
        self.assertEqual(str(counter),
                         "ErrorCounter(1 error / 1000 symbols, rate 1.00e-03)")

    def test_refusals(self):
        with self.assertRaises(ValueError):
            ErrorCounter(width=0)
        with self.assertRaises(ValueError):
            ErrorCounter().update(np.zeros(0, dtype=int), np.zeros(5, dtype=int))


if __name__ == "__main__":
    unittest.main()
