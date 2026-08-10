r"""A parallel sweep must be the *same* sweep (decision D35).

The points of a sweep are independent -- each one reconfigures and
reseeds the chain from scratch -- so running them at once is a
scheduling decision and nothing else. That is a testable claim, and it
is the only one that matters here: a curve computed on four processes
must equal the serial one **value for value**, not in distribution.
Everything else in this file is about the boundary a worker process
puts around the computation, and about saying so clearly when something
cannot cross it.
"""
import unittest

import numpy as np

from comnumpy import sweep
from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolDemapper, SymbolMapper
from comnumpy.core.metrics import compute_ser
from comnumpy.core.utils import get_alphabet

SNR_DB = [0.0, 4.0, 8.0, 12.0]


def build_chain(order=4):
    alphabet = get_alphabet("PSK", order)
    return Sequential([
        SymbolGenerator(order, name="tx"),
        SymbolMapper(alphabet),
        AWGN(snr_dB=0.0, name="noise"),
        SymbolDemapper(alphabet),
    ], taps=["tx"])


def mean_power(y):
    """A module-level metric: a lambda could not reach a worker."""
    return float(np.mean(np.abs(y) ** 2))


class TestParallelIsTheSameSweep(unittest.TestCase):

    def test_the_curve_is_identical_value_for_value(self):
        serial = sweep(build_chain(), "noise.snr_dB", SNR_DB,
                       {"ser": compute_ser}, 2000, reference="tx", seed=5)
        parallel = sweep(build_chain(), "noise.snr_dB", SNR_DB,
                         {"ser": compute_ser}, 2000, reference="tx", seed=5,
                         n_jobs=2)
        np.testing.assert_array_equal(parallel["ser"], serial["ser"])

    def test_the_order_of_the_points_is_preserved(self):
        """Workers finish out of order; the arrays must not."""
        serial = sweep(build_chain(), "noise.snr_dB", SNR_DB,
                       {"power": mean_power}, 500, seed=1)
        parallel = sweep(build_chain(), "noise.snr_dB", SNR_DB,
                         {"power": mean_power}, 500, seed=1, n_jobs=-1)
        np.testing.assert_array_equal(parallel["power"], serial["power"])
        self.assertEqual(parallel["power"].shape, (len(SNR_DB),))

    def test_more_workers_than_points_is_not_an_error(self):
        out = sweep(build_chain(), "noise.snr_dB", [3.0, 6.0],
                    {"power": mean_power}, 200, seed=2, n_jobs=8)
        self.assertEqual(out["power"].shape, (2,))


class TestWhatCannotCrossIntoAWorker(unittest.TestCase):

    def test_a_sweep_without_a_seed_is_refused(self):
        """Without a seed, what a point draws would depend on the worker."""
        with self.assertRaises(ValueError) as ctx:
            sweep(build_chain(), "noise.snr_dB", SNR_DB,
                  {"power": mean_power}, 100, n_jobs=2)
        self.assertIn("seed=", str(ctx.exception))

    def test_a_lambda_metric_is_named_rather_than_left_to_pickle(self):
        with self.assertRaises(ValueError) as ctx:
            sweep(build_chain(), "noise.snr_dB", SNR_DB,
                  {"power": lambda y: float(np.mean(np.abs(y)))}, 100,
                  seed=3, n_jobs=2)
        message = str(ctx.exception)
        self.assertIn("metrics", message)
        self.assertIn("lambda", message)

    def test_the_serial_path_still_accepts_a_lambda(self):
        """n_jobs=1 has no boundary to cross, and must stay permissive."""
        out = sweep(build_chain(), "noise.snr_dB", [5.0],
                    {"power": lambda y: float(np.mean(np.abs(y)))}, 100,
                    seed=3)
        self.assertEqual(out["power"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
