"""Tests for the spectral allocation objects (decisions D15-D21)."""
import dataclasses
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from src.comnumpy.ofdm.allocation import (CarrierAllocation, CarrierType,
                                          available_allocations,
                                          band_allocation, get_allocation,
                                          scattered_allocation)
from src.comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor


class TestCarrierAllocation(unittest.TestCase):

    def test_catalog_entries_self_check(self):
        """Every catalog entry builds and passes its expect= self-check (D20)."""
        for name in available_allocations():
            alloc = get_allocation(name)
            self.assertGreater(alloc.N_data, 0, name)

    def test_wifi_11a_matches_standard_table(self):
        alloc = get_allocation("802.11a")
        self.assertEqual(alloc.N_fft, 64)
        self.assertEqual(alloc.N_data, 48)
        self.assertEqual(alloc.N_pilots, 4)
        # pilots at +/-7 and +/-21 (IEEE 802.11-2020, Table 17-5)
        pilot_k = alloc.k[alloc.carrier_type[0] == CarrierType.PILOT]
        np.testing.assert_array_equal(pilot_k, [-21, -7, 7, 21])
        # DC is null
        self.assertEqual(alloc.carrier_type[0][alloc.k == 0], CarrierType.NULL)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            CarrierAllocation(np.array([[0, 1, 7]]))

    def test_non_constant_data_count_rejected(self):
        with self.assertRaises(ValueError):
            CarrierAllocation(np.array([[1, 1, 0], [1, 0, 0]]))

    def test_frozen_and_readonly(self):
        alloc = get_allocation("802.11a")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            alloc.standard = "other"
        with self.assertRaises(ValueError):
            alloc.carrier_type[0, 0] = 1

    def test_to_fft_order_roundtrip(self):
        from scipy.fft import fftshift
        alloc = get_allocation("802.11a")
        np.testing.assert_array_equal(
            fftshift(alloc.to_fft_order(), axes=1), alloc.carrier_type)

    def test_repr_shows_spectral_map(self):
        text = repr(get_allocation("802.11a"))
        self.assertIn("802.11a", text)
        self.assertIn("data 48 | pilots 4", text)
        self.assertIn("P", text)  # pilots visible in the ASCII map


class TestAllocatorRoundTrip(unittest.TestCase):

    def test_shared_allocation_roundtrip(self):
        """TX and RX share one CarrierAllocation object (D18)."""
        alloc = get_allocation("802.11a")
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, alloc.N_data)) + 1j * rng.normal(size=(10, alloc.N_data))

        allocator = CarrierAllocator(alloc, pilots=1.0)
        extractor = CarrierExtractor(alloc)
        Y = allocator(X)
        self.assertEqual(Y.shape, (10, alloc.N_fft))
        np.testing.assert_allclose(extractor(Y), X)

    def test_scattered_allocation_roundtrip(self):
        alloc = scattered_allocation(
            16, k_used=(-8, 7), period=2,
            rule=lambda line, k: (k + 8 + 2 * line) % 4 == 0)
        T = 6
        X = np.arange(T * alloc.N_data, dtype=complex).reshape(T, alloc.N_data)
        allocator = CarrierAllocator(alloc, pilots=-1.0)
        extractor = CarrierExtractor(alloc)
        Y = allocator(X)
        np.testing.assert_allclose(extractor(Y), X)
        # pilot positions rotate with the symbol index
        row0, row1 = allocator.mask[0], allocator.mask[1]
        self.assertFalse(np.array_equal(row0, row1))
        np.testing.assert_allclose(Y[0, row0 == 2], -1.0)
        np.testing.assert_allclose(Y[1, row1 == 2], -1.0)

    def test_extractor_exposes_pilots_as_estimated_attribute(self):
        """Pilot content is an estimated attribute, not a recorder block."""
        alloc = get_allocation("802.11a")
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4, alloc.N_data)) + 1j * rng.normal(size=(4, alloc.N_data))
        extractor = CarrierExtractor(alloc)
        self.assertIsNone(extractor.pilots_)  # nothing before the first run
        extractor(CarrierAllocator(alloc, pilots=3.0)(X))
        np.testing.assert_allclose(extractor.pilots_, 3.0)
        self.assertEqual(np.shape(extractor.pilots_), (4, alloc.N_pilots))

    def test_wrong_input_size_raises_shape_error(self):
        allocator = CarrierAllocator(get_allocation("802.11a"))
        with self.assertRaises(ShapeError) as ctx:
            allocator(np.ones((10, 47)))
        self.assertIn("N_data=48", str(ctx.exception))

    def test_pilot_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            allocator = CarrierAllocator(get_allocation("802.11a"),
                                         pilots=np.array([1.0, -1.0]))
            allocator(np.ones((2, 48)))


class TestBandAllocationConstructor(unittest.TestCase):

    def test_expect_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            band_allocation(8, k_used=(-3, 3), n_dc=1,
                            expect={"data": 99, "pilots": 0})
        self.assertIn("expected 99", str(ctx.exception))


class TestFullFieldGuard(unittest.TestCase):

    def test_fiberlink_refuses_multichannel_arrays(self):
        from comnumpy.optical.links import FiberLink
        with self.assertRaises(ShapeError) as ctx:
            FiberLink(1, noise_scaling=0)(np.ones((2, 64), dtype=complex))
        self.assertIn("full-field", str(ctx.exception))

    def test_dbp_refuses_multichannel_arrays(self):
        from comnumpy.optical.dbp import DBP
        with self.assertRaises(ShapeError):
            DBP(1)(np.ones((2, 64), dtype=complex))


if __name__ == "__main__":
    unittest.main()
