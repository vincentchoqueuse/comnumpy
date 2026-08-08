"""Property tests for synchronization sequences (D29) and frames (D28)."""
import unittest

import numpy as np

from comnumpy.core.frames import (Deframer, FieldRole, FrameField,
                                  FrameStructure, Framer)
from comnumpy.core.sequences import (barker, golay_pair, m_sequence,
                                     schmidl_cox_preamble, zadoff_chu)
from comnumpy.exceptions import ShapeError


class TestSequences(unittest.TestCase):

    def test_zadoff_chu_cazac_properties(self):
        x = zadoff_chu(25, 139)
        # constant amplitude
        np.testing.assert_allclose(np.abs(x), 1.0, atol=1e-12)
        # ideal periodic autocorrelation: zero at every nonzero lag
        for lag in (1, 7, 50):
            r = np.vdot(x, np.roll(x, lag))
            self.assertLess(abs(r), 1e-9, f"lag {lag}")

    def test_zadoff_chu_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            zadoff_chu(2, 64)   # even length
        with self.assertRaises(ValueError):
            zadoff_chu(3, 9)    # gcd != 1

    def test_schmidl_cox_half_repetition(self):
        x = schmidl_cox_preamble(64, seed=3)
        np.testing.assert_allclose(x[32:], x[:32], atol=1e-12)
        self.assertAlmostEqual(float(np.mean(np.abs(x) ** 2)), 1.0, places=6)

    def test_barker_sidelobes(self):
        for length in (2, 3, 4, 5, 7, 11, 13):
            code = barker(length)
            r = np.correlate(code, code, "full")
            sidelobes = np.delete(r, length - 1)
            self.assertLessEqual(np.max(np.abs(sidelobes)), 1, f"length {length}")
            self.assertEqual(r[length - 1], length)

    def test_golay_complementarity(self):
        for length in (2, 4, 8, 16, 32, 64, 128):
            a, b = golay_pair(length)
            r_sum = np.correlate(a, a, "full") + np.correlate(b, b, "full")
            expected = np.zeros(2 * length - 1)
            expected[length - 1] = 2 * length
            np.testing.assert_allclose(r_sum, expected, atol=1e-9)

    def test_m_sequence_properties(self):
        for degree in range(2, 11):
            x = m_sequence(degree)
            N = 2**degree - 1
            self.assertEqual(len(x), N)
            # balance: one more -1 (bit 1) than +1
            self.assertEqual(int(np.sum(x == -1)) - int(np.sum(x == 1)), 1)
            # two-valued periodic autocorrelation: N at lag 0, -1 elsewhere
            for lag in range(1, N):
                self.assertEqual(int(np.dot(x, np.roll(x, lag))), -1)


class TestFrameStructure(unittest.TestCase):

    def build(self, payload=1000):
        return FrameStructure((
            FrameField("STF", FieldRole.SYNC, np.ones(16)),
            FrameField("LTF", FieldRole.TRAINING, zadoff_chu(1, 63)),
            FrameField("PAYLOAD", FieldRole.PAYLOAD, length=payload),
            FrameField("TAIL", FieldRole.TAIL, np.zeros(6)),
        ), standard="testframe")

    def test_lengths_and_slices(self):
        frame = self.build()
        self.assertEqual(frame.frame_length, 16 + 63 + 1000 + 6)
        self.assertEqual(frame.payload_length, 1000)
        self.assertEqual(frame.slice_of("LTF"), slice(16, 79))
        self.assertEqual(len(frame.fields_by_role(FieldRole.SYNC)), 1)

    def test_exactly_one_payload(self):
        with self.assertRaises(ValueError):
            FrameStructure((FrameField("A", FieldRole.SYNC, np.ones(4)),))
        with self.assertRaises(ValueError):
            FrameStructure((
                FrameField("P1", FieldRole.PAYLOAD, length=8),
                FrameField("P2", FieldRole.PAYLOAD, length=8),
            ))

    def test_known_fields_need_values(self):
        with self.assertRaises(ValueError):
            FrameStructure((
                FrameField("STF", FieldRole.SYNC, length=16),  # no values
                FrameField("PAYLOAD", FieldRole.PAYLOAD, length=8),
            ))

    def test_repr_shows_frame_map(self):
        text = repr(self.build())
        self.assertIn("--STF--", text)
        self.assertIn("PAYLOAD", text)
        self.assertIn("unknown at TX", text)


class TestFramerDeframer(unittest.TestCase):

    def setUp(self):
        self.frame = FrameStructure((
            FrameField("SYNC", FieldRole.SYNC, zadoff_chu(1, 63)),
            FrameField("PAYLOAD", FieldRole.PAYLOAD, length=100),
        ), standard="testframe")

    def test_roundtrip_and_field_access(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 100)) + 1j * rng.normal(size=(5, 100))
        framer = Framer(self.frame)
        deframer = Deframer(self.frame)
        Y = framer(X)
        self.assertEqual(Y.shape, (5, 163))
        np.testing.assert_allclose(deframer(Y), X)
        # every frame carries the same known SYNC samples
        sync_rx = deframer.get_field("SYNC")
        np.testing.assert_allclose(sync_rx[3], zadoff_chu(1, 63))

    def test_framer_message_names_the_fix(self):
        framer = Framer(self.frame)
        with self.assertRaises(ShapeError) as ctx:
            framer(np.ones((5, 99)))
        message = str(ctx.exception)
        self.assertIn("payload field length 100", message)
        self.assertIn("Serial2Parallel(N_sub=100)", message)

    def test_deframer_validates_frame_length(self):
        with self.assertRaises(ShapeError):
            Deframer(self.frame)(np.ones((5, 100)))


if __name__ == "__main__":
    unittest.main()
