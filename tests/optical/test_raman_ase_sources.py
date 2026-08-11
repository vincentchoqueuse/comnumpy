"""Where the Raman ASE comes from, by superposition rather than by re-running."""
import unittest

import numpy as np

from comnumpy.optical.raman import get_gain_spectrum, solve_raman

SPECTRUM = get_gain_spectrum("blow-wood")


def solve(n_signals=8, **kwargs):
    return solve_raman(
        length_km=80.0, gain_peak_W_km=0.4,
        signal_W=np.full(n_signals, 1e-3),
        wavelength_signal_nm=np.linspace(1540.0, 1560.0, n_signals),
        wavelength_pump_nm=np.array([1450.0]),
        alpha_signal_dB_km=0.2, alpha_pump_dB_km=0.25,
        spectrum=SPECTRUM, bandwidth_Hz=32e9, n_nodes=401, **kwargs)


class TestTheDecompositionIsExact(unittest.TestCase):
    r"""The ASE equation is linear in the ASE, so the parts add up.

    :math:`\mathrm{d}A_i/\mathrm{d}z = g_i A_i + \sum_j S_{ij} P_j` gives
    :math:`A_i = \sum_j A_i^{(j)}` identically. The split is arithmetic
    on profiles already computed, not a second solve and not a model, so
    the only error it may carry is the quadrature's.
    """

    def test_the_two_groups_add_back_to_the_total(self):
        for direction in ("pump_backward_W", "pump_forward_W"):
            with self.subTest(direction=direction):
                solution = solve(**{direction: np.array([0.3])})
                total = np.atleast_2d(solution.ase_W)[:, -1]
                parts = (np.atleast_2d(solution.ase_from_pumps_W)[:, -1]
                         + np.atleast_2d(solution.ase_from_signals_W)[:, -1])
                self.assertLess(float(np.abs(parts / total - 1).max()), 1e-4)

    def test_it_holds_along_the_whole_span_not_only_at_the_end(self):
        solution = solve(pump_backward_W=np.array([0.3]))
        total = np.atleast_2d(solution.ase_W)[:, 1:]
        parts = (np.atleast_2d(solution.ase_from_pumps_W)[:, 1:]
                 + np.atleast_2d(solution.ase_from_signals_W)[:, 1:])
        self.assertLess(float(np.abs(parts / total - 1).max()), 1e-3)


class TestTheSplitIsPhysical(unittest.TestCase):
    """A correct total can still be split wrongly; this pins the split."""

    def test_the_bluest_channel_is_seeded_by_no_channel_at_all(self):
        """Raman only flows downhill in frequency, so the top row gets nothing.

        This is the assertion a sign error or a transposed index cannot
        survive, and it needs no reference value.
        """
        solution = solve(pump_backward_W=np.array([0.3]))
        from_signals = np.atleast_2d(solution.ase_from_signals_W)
        self.assertEqual(float(from_signals[0, -1]), 0.0)
        self.assertGreater(float(from_signals[-1, -1]), 0.0)

    def test_the_inter_channel_share_grows_down_the_comb(self):
        """Each channel further down has more channels above it to seed it."""
        solution = solve(pump_backward_W=np.array([0.3]))
        from_signals = np.atleast_2d(solution.ase_from_signals_W)[:, -1]
        self.assertTrue(np.all(np.diff(from_signals) > 0), from_signals)

    def test_with_one_signal_nothing_is_left_for_the_channels(self):
        solution = solve(n_signals=1, pump_backward_W=np.array([0.3]))
        self.assertEqual(float(np.max(solution.ase_from_signals_W)), 0.0)
        self.assertLess(
            float(np.abs(np.asarray(solution.ase_from_pumps_W)[-1]
                         / np.asarray(solution.ase_W)[-1] - 1)), 1e-4)


if __name__ == "__main__":
    unittest.main()
