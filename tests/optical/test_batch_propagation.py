"""Propagating several realizations in one call.

`is_polarization_pair` reads axes -2 and -1 only, so the leading axes of
a field are free and `FiberLink` broadcasts over them. That has always
been true and nothing pinned it, which for a capability is the same as
not having it: the next refactor of `prepare()` could take it away and
every test would still pass.

The claim is stronger than "it runs". A batch must give *the same
answer* as the loop it replaces, span for span and step for step, or it
is a different simulation wearing the same name. With the amplifier
noise off that is bit-for-bit; with it on it cannot be, because one draw
of `B x N` samples is not `B` draws of `N`, and the test says which of
the two it checks.
"""
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from comnumpy.optical.dbp import DBP
from comnumpy.optical.links import FiberLink

N = 512


def field(shape, seed=0):
    rng = np.random.default_rng(seed)
    return 1e-3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


class TestBatchIsTheSameSimulation(unittest.TestCase):

    def link(self, **kwargs):
        params = dict(L_span=80.0, StPS=8, fs=10e9, noise_scaling=0.0)
        params.update(kwargs)
        return FiberLink(3, **params)

    def test_a_batch_of_one_polarization_equals_the_loop(self):
        """Bit for bit, which is what makes the batch a shortcut."""
        rows = [field(N, seed=index) for index in range(4)]
        batch = np.stack(rows)[:, None, :]              # (4, 1, N)
        link = self.link()
        together = link(batch.copy())
        self.assertEqual(together.shape, (4, 1, N))
        for index, row in enumerate(rows):
            with self.subTest(item=index):
                np.testing.assert_array_equal(together[index, 0],
                                              link(row.copy()))

    def test_a_batch_of_pairs_equals_the_loop(self):
        """The Manakov coupling stays inside an item, never across items."""
        pairs = [field((2, N), seed=index) for index in range(3)]
        link = self.link()
        together = link(np.stack(pairs).copy())
        self.assertEqual(together.shape, (3, 2, N))
        for index, pair in enumerate(pairs):
            with self.subTest(item=index):
                np.testing.assert_array_equal(together[index],
                                              link(pair.copy()))

    def test_one_item_of_a_batch_equals_the_bare_signal(self):
        """`(1, 1, N)` and `(N,)` are the same field written two ways."""
        signal = field(N)
        link = self.link()
        np.testing.assert_array_equal(
            link(signal.reshape(1, 1, N).copy())[0, 0], link(signal.copy()))

    def test_the_kerr_term_does_not_leak_between_items(self):
        """The nonlinearity is pointwise, so a loud item must not be felt.

        Two realizations, one of them a hundred times stronger. If the
        intensity were summed over the batch instead of over the
        polarizations, the quiet one would come back with the loud
        one's phase.
        """
        quiet, loud = field(N, seed=1), 100 * field(N, seed=2)
        link = self.link()
        batch = np.stack([quiet, loud])[:, None, :]
        together = link(batch.copy())
        np.testing.assert_array_equal(together[0, 0], link(quiet.copy()))

    def test_dbp_batches_the_same_way(self):
        """A receiver has to accept what the link produced."""
        rows = [field(N, seed=index) for index in range(3)]
        dbp = DBP(3, L_span=80.0, StPS=8, fs=10e9)
        together = dbp(np.stack(rows)[:, None, :].copy())
        for index, row in enumerate(rows):
            with self.subTest(item=index):
                np.testing.assert_array_equal(together[index, 0],
                                              dbp(row.copy()))

    def test_the_round_trip_survives_a_batch(self):
        """Propagate a batch, back-propagate it, recover every item."""
        rows = np.stack([field(N, seed=index) for index in range(3)])[:, None, :]
        link = self.link(StPS=200)
        back = DBP(3, L_span=80.0, StPS=200, fs=10e9)
        recovered = back(link(rows.copy()))
        np.testing.assert_allclose(recovered, rows, rtol=0, atol=1e-9)

    def test_noise_is_drawn_for_the_whole_batch(self):
        """With ASE on the batch is not the loop, and that is not a bug.

        One draw of `B x N` Gaussian samples is not `B` draws of `N`, so
        the realizations differ. What must hold is that each item still
        carries the noise the link budgets for it.
        """
        link = self.link(noise_scaling=1.0, NF_dB=5.0, seed=0)
        rows = np.zeros((64, 1, 4096), dtype=complex)
        noise = link(rows)
        per_item = np.var(noise, axis=-1).ravel()
        expected = link.budget(link.fs)["ase_power_W"]
        self.assertAlmostEqual(float(np.mean(per_item)) / expected, 1.0,
                               delta=0.05)
        # every item got noise, none was left silent
        self.assertGreater(float(np.min(per_item)) / expected, 0.5)

    def test_a_two_dimensional_batch_is_refused_and_says_how(self):
        """`(B, N)` is what a user reaches for first; the message helps."""
        with self.assertRaises(ShapeError) as ctx:
            self.link()(np.zeros((32, N), dtype=complex))
        message = str(ctx.exception)
        self.assertIn(f"(32, 1, {N})", message)      # the shape to write
        self.assertIn("WDMMultiplexer", message)     # the other reading


if __name__ == "__main__":
    unittest.main()
