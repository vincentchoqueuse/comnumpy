"""The constructed alphabets must be the tables they replaced, index for index.

``get_alphabet`` used to read thirty-six CSV files. It now builds PSK,
PAM and square QAM from their definitions, which is why BPSK, 8-PSK,
1024-QAM and any other power-of-two order finally exist -- the tables
only covered 4, 16, 32, 64, 128 and 256.

Replacing data by a formula is only safe if the formula is *checked
against the data it replaces*, so the thirty-two superseded tables are
kept here, under ``data_reference/``, purely as test fixtures. Index
order is the whole point: a set-equal check would pass a different
bit-to-symbol mapping, and a different mapping is a different BER,
silently.

The two cross constellations, 32-QAM and 128-QAM, are not square and so
are not the product of two PAM axes; they stay tabulated in the package.
"""
import unittest
from pathlib import Path

import numpy as np

from comnumpy.core.utils import _construct_alphabet, get_alphabet

REFERENCE_DIR = Path(__file__).parent / "data_reference"
TABULATED = (4, 16, 32, 64, 128, 256)
SQUARE_QAM = (4, 16, 64, 256)


def load_reference(family, order, mapping):
    data = np.loadtxt(REFERENCE_DIR / f"{family}_{order}_{mapping}.csv",
                      delimiter=",", skiprows=1)
    return data[:, 1] + 1j * data[:, 2]


class TestConstructionMatchesTheTables(unittest.TestCase):

    def test_the_tables_are_reproduced_entry_by_entry(self):
        """Tolerance 1e-6, which is the *tables'* precision, not ours.

        The CSVs stored six decimals (0.999699 where the exact value is
        0.99969882...), so the construction cannot be checked any
        tighter than that -- and it is the construction that is exact
        here, the table that was rounded. That is the argument for
        dropping the files.
        """
        for family, orders in (("PSK", TABULATED), ("PAM", TABULATED),
                               ("QAM", SQUARE_QAM)):
            for order in orders:
                for mapping in ("gray", "bin"):
                    with self.subTest(alphabet=f"{family}-{order}-{mapping}"):
                        tabulated = load_reference(family, order, mapping)
                        built = _construct_alphabet(family, order, mapping)
                        np.testing.assert_allclose(built, tabulated, atol=1e-6)

    def test_the_reference_fixtures_are_all_there(self):
        """A missing fixture would silently shrink the check above."""
        expected = {f"{family}_{order}_{mapping}.csv"
                    for family, orders in (("PSK", TABULATED), ("PAM", TABULATED),
                                           ("QAM", SQUARE_QAM))
                    for order in orders for mapping in ("gray", "bin")}
        found = {path.name for path in REFERENCE_DIR.glob("*.csv")}
        self.assertEqual(found, expected)

    def test_the_cross_constellations_are_still_shipped(self):
        """32-QAM and 128-QAM have no construction, so they need their file."""
        for order in (32, 128):
            with self.subTest(order=order):
                alphabet = get_alphabet("QAM", order, norm=False)
                self.assertEqual(len(alphabet), order)
                # a cross constellation is not a square grid: it has more
                # distinct in-phase levels than sqrt(M)
                levels = len(set(np.round(alphabet.real).astype(int)))
                self.assertGreater(levels, int(np.sqrt(order)))


class TestUntabulatedOrders(unittest.TestCase):

    def test_bpsk_is_the_real_pair(self):
        np.testing.assert_allclose(get_alphabet("PSK", 2), [1.0, -1.0],
                                   atol=1e-12)

    def test_8psk_sits_on_the_unit_circle_with_gray_neighbours(self):
        alphabet = get_alphabet("PSK", 8)
        np.testing.assert_allclose(np.abs(alphabet), 1.0)
        # Gray: adjacent points on the circle differ in exactly one bit
        angle_order = np.argsort(np.angle(alphabet) % (2 * np.pi))
        for a, b in zip(angle_order, np.roll(angle_order, -1), strict=True):
            self.assertEqual(bin(int(a) ^ int(b)).count("1"), 1)

    def test_1024qam_is_a_32_by_32_grid(self):
        alphabet = get_alphabet("QAM", 1024, norm=False)
        self.assertEqual(len(alphabet), 1024)
        self.assertEqual(len(set(np.round(alphabet.real).astype(int))), 32)
        self.assertEqual(len(set(np.round(alphabet.imag).astype(int))), 32)

    def test_gray_qam_neighbours_differ_in_one_bit(self):
        """The property the labelling exists for, on the built alphabets."""
        for order in (16, 64, 1024):
            with self.subTest(order=order):
                alphabet = get_alphabet("QAM", order, norm=False)
                root = int(np.sqrt(order))
                spacing = 2.0
                for m, symbol in enumerate(alphabet):
                    for step in (spacing, -spacing, 1j * spacing, -1j * spacing):
                        neighbour = symbol + step
                        match = np.flatnonzero(np.isclose(alphabet, neighbour))
                        if match.size:
                            self.assertEqual(
                                bin(m ^ int(match[0])).count("1"), 1,
                                f"QAM-{order}: {m} and {match[0]} are adjacent "
                                f"but differ in more than one bit")
                self.assertEqual(root * root, order)

    def test_every_alphabet_has_unit_energy_by_default(self):
        for family, order in (("PSK", 2), ("PSK", 8), ("PSK", 512),
                              ("PAM", 2), ("PAM", 8), ("QAM", 16),
                              ("QAM", 32), ("QAM", 1024)):
            with self.subTest(alphabet=f"{family}-{order}"):
                energy = float(np.mean(np.abs(get_alphabet(family, order)) ** 2))
                self.assertAlmostEqual(energy, 1.0, places=12)

    def test_rejects_an_order_that_is_not_a_power_of_two(self):
        for bad in (0, 1, 3, 6, 100):
            with self.subTest(order=bad):
                with self.assertRaises(ValueError):
                    get_alphabet("PSK", bad)

    def test_rejects_a_non_square_qam_with_no_table(self):
        with self.assertRaises(ValueError) as ctx:
            get_alphabet("QAM", 512)
        self.assertIn("cross constellations", str(ctx.exception))

    def test_reports_an_unknown_family_rather_than_a_file_error(self):
        with self.assertRaises(ValueError):
            get_alphabet("FSK", 8)


if __name__ == "__main__":
    unittest.main()
