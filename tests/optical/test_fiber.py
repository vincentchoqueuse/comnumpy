"""The fibre spec, its catalog, and the guards against a unit mistake (D46).

Written after making the exact error the last class here checks for: a
second, hand-written copy of the beta2 conversion that was a thousand
times too small. Two things caught it or would have -- the D20
self-check on the catalog entry, which did, and the plausibility bounds,
which cover the values the catalog never sees because a user typed them.

The structural fix matters more than either: `beta2` now delegates to
the one conversion in `optical.utils`, so there is no second formula to
get wrong. Guards catch mistakes; not duplicating catches the class.
"""
import dataclasses
import unittest

import numpy as np

from comnumpy.optical import fiber as fiber_module
from comnumpy.optical.fiber import (FiberSpec, available_fibers, get_fiber,
                                    register_fiber)
from comnumpy.optical.utils import compute_beta2


class TestDerivedQuantities(unittest.TestCase):

    def test_beta2_agrees_with_the_single_conversion(self):
        """One formula, one place -- this pins the delegation."""
        for wavelength, dispersion in ((1550.0, 17.0), (1310.0, 0.0),
                                       (1550.0, -100.0), (1625.0, 22.0)):
            with self.subTest(wavelength=wavelength, D=dispersion):
                spec = FiberSpec(0.2, cd_coefficient=dispersion,
                                 wavelength_nm=wavelength)
                self.assertAlmostEqual(
                    spec.beta2,
                    compute_beta2(wavelength, dispersion, 299792458), places=12)

    def test_the_carrier_frequency_cannot_disagree_with_the_wavelength(self):
        """It is derived, which is the point: they were two arguments."""
        for wavelength in (1310.0, 1550.0, 1625.0):
            with self.subTest(wavelength=wavelength):
                spec = FiberSpec(wavelength_nm=wavelength)
                self.assertAlmostEqual(
                    spec.carrier_frequency_Hz * wavelength * 1e-9,
                    299792458, places=3)

    def test_beta2_changes_sign_with_the_dispersion(self):
        self.assertLess(FiberSpec(cd_coefficient=17.0).beta2, 0)
        self.assertGreater(FiberSpec(cd_coefficient=-100.0).beta2, 0)
        self.assertEqual(FiberSpec(cd_coefficient=0.0).beta2, 0)

    def test_effective_length_is_the_span_for_a_lossless_fibre(self):
        self.assertAlmostEqual(FiberSpec(0.0).effective_length_km(80.0), 80.0)

    def test_effective_length_saturates_for_a_long_span(self):
        r"""It tends to :math:`1/\alpha`, which is 21.7 km at 0.2 dB/km."""
        spec = FiberSpec(0.2)
        self.assertAlmostEqual(spec.effective_length_km(100_000),
                               1 / spec.alpha_per_km, places=6)
        self.assertLess(spec.effective_length_km(80.0),
                        spec.effective_length_km(160.0))

    def test_alpha_per_km_matches_the_decibel_definition(self):
        """A 0.2 dB/km fibre loses 16 dB over 80 km, in either unit."""
        spec = FiberSpec(0.2)
        power_ratio = np.exp(-spec.alpha_per_km * 80.0)
        self.assertAlmostEqual(-10 * np.log10(power_ratio), spec.loss_dB(80.0),
                               places=9)


class TestCatalog(unittest.TestCase):
    """The registry is process-global, so restore it around each test."""

    def setUp(self):
        self._registry = dict(fiber_module._FIBER_REGISTRY)

    def tearDown(self):
        fiber_module._FIBER_REGISTRY.clear()
        fiber_module._FIBER_REGISTRY.update(self._registry)

    def test_smf_reproduces_its_published_beta2(self):
        """-21.68 against the -21.7 quoted separately in the same source."""
        self.assertAlmostEqual(get_fiber("SMF").beta2, -21.7, delta=0.05)

    def test_every_entry_lands_in_a_physically_sane_range(self):
        """A future catalog entry with a unit slip fails here."""
        for name in available_fibers():
            with self.subTest(fiber=name):
                spec = get_fiber(name)
                self.assertGreater(spec.alpha_dB, 0.0)
                self.assertLess(spec.alpha_dB, 2.0)          # dB/km
                self.assertGreater(spec.gamma, 0.0)
                self.assertLess(spec.gamma, 50.0)            # rad/W/km
                self.assertLess(abs(spec.beta2), 500.0)      # ps^2/km
                self.assertGreater(spec.carrier_frequency_Hz, 100e12)
                self.assertLess(spec.carrier_frequency_Hz, 400e12)

    def test_every_entry_carries_its_provenance(self):
        for name in available_fibers():
            with self.subTest(fiber=name):
                spec = get_fiber(name)
                self.assertNotEqual(spec.standard, "custom")
                self.assertTrue(spec.reference)

    def test_the_compensating_fibre_has_the_opposite_dispersion(self):
        self.assertLess(get_fiber("SMF").beta2 * get_fiber("DCF").beta2, 0)

    def test_the_self_check_catches_a_mistyped_entry(self):
        @register_fiber("_mistyped")
        def _entry():
            return fiber_module._check_expect(
                FiberSpec(0.2, cd_coefficient=1.7, standard="_mistyped",
                          reference="none"),
                {"beta2": -21.7})
        with self.assertRaises(ValueError) as ctx:
            get_fiber("_mistyped")
        self.assertIn("mistyped", str(ctx.exception))

    def test_an_unknown_name_lists_what_exists(self):
        with self.assertRaises(KeyError) as ctx:
            get_fiber("SMF-28e+")
        self.assertIn("SMF", str(ctx.exception))


class TestUnitGuards(unittest.TestCase):
    """The bound is the width of a unit mistake, not a law of physics."""

    def test_rejects_a_value_a_thousand_times_too_large(self):
        cases = {
            "alpha_dB": 200.0,          # dB/m read as dB/km
            "gamma": 1300.0,            # rad/W/m read as rad/W/km
            "cd_coefficient": 17_000.0,
            "raman_gain_W_km": 4_000.0,
        }
        for name, value in cases.items():
            with self.subTest(parameter=name):
                with self.assertRaises(ValueError) as ctx:
                    FiberSpec(**{name: value})
                self.assertIn("unit mistake", str(ctx.exception))

    def test_rejects_micrometres_where_nanometres_are_expected(self):
        with self.assertRaises(ValueError) as ctx:
            FiberSpec(wavelength_nm=1.55)
        self.assertIn("micrometres", str(ctx.exception))

    def test_the_message_names_the_unit_it_wants(self):
        with self.assertRaises(ValueError) as ctx:
            FiberSpec(200.0)
        self.assertIn("dB/km", str(ctx.exception))

    def test_the_bounds_leave_room_for_every_real_fibre(self):
        """A guard that rejected a legitimate fibre would be worse."""
        for name in available_fibers():
            with self.subTest(fiber=name):
                get_fiber(name)          # must not raise
        FiberSpec(0.0, gamma=0.0, cd_coefficient=0.0)   # the ideal fibre
        FiberSpec(0.6, gamma=20.0, cd_coefficient=-200.0)  # a strong DCF
        FiberSpec(2.0, gamma=1.3, wavelength_nm=850.0)     # multimode window

    def test_rejects_the_physically_impossible(self):
        for kwargs in ({"alpha_dB": -0.1}, {"gamma": -1.0},
                       {"wavelength_nm": 0.0}, {"raman_gain_W_km": -1.0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    FiberSpec(**kwargs)


class TestValueObject(unittest.TestCase):

    def test_is_frozen_and_hashable(self):
        spec = get_fiber("SMF")
        self.assertIsInstance(hash(spec), int)
        self.assertEqual(spec, get_fiber("SMF"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.alpha_dB = 0.3          # type: ignore[misc]

    def test_repr_shows_the_derived_dispersion(self):
        text = repr(get_fiber("SMF"))
        self.assertIn("SMF", text)
        self.assertIn("-21.68", text)
        self.assertIn("dB/km", text)

    def test_survives_a_json_round_trip(self):
        from comnumpy.core import Sequential
        from comnumpy.optical import FiberLink
        from comnumpy.serialization import from_json, to_json
        chain = Sequential([FiberLink(2, L_span=80.0, fs=100e9,
                                      fiber=get_fiber("DCF"))])
        restored = from_json(to_json(chain))
        self.assertEqual(restored[0].fiber, chain[0].fiber)


if __name__ == "__main__":
    unittest.main()
