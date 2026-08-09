
import ast
import inspect
import pathlib
import unittest
import numpy as np
from comnumpy.exceptions import NotFittedError
from comnumpy.core import compensators as compensators_module
from comnumpy.core.compensators import (
    BlindCFOCompensator,
    BlindIQCompensator,
    DataAidedFIRCompensator,
    DataAidedFineSynchronizer,
    DataAidedSimpleSynchronizer,
    Normalizer,
)


QPSK = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)


def make_frame(gain, delay, n_preamble=64, n_payload=64, seed=0):
    """Build (preamble, payload, received record) for a flat channel."""
    rng = np.random.default_rng(seed)
    preamble = QPSK[rng.integers(0, 4, n_preamble)]
    payload = QPSK[rng.integers(0, 4, n_payload)]
    x = np.hstack([np.zeros(delay, dtype=complex),
                   gain * np.hstack([preamble, payload])])
    return preamble, payload, x


class TestNormalizer(unittest.TestCase):

    def test_normalizer(self):

        X = np.array([1, 2, 3, 4])

        # test max
        normalizer = Normalizer(method='max', value=2.0)
        Y = normalizer(X)
        Y_ref = np.array([0.5, 1., 1.5, 2.])
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

        # test var
        normalizer = Normalizer(method='var', value=2.0)
        Y = normalizer(X)

        scale = np.sqrt(np.var(X))
        Y_ref = (X / scale) * np.sqrt(2.0)
        np.testing.assert_allclose(Y, Y_ref, atol=1e-8)

    def test_method_is_the_first_positional_argument(self):
        # Normalizer('max') used to set the inherited `gain` parameter and
        # silently keep method='amp'
        X = np.array([1., 2., 3., 4.])
        np.testing.assert_allclose(Normalizer('max', value=2.0)(X),
                                   Normalizer(method='max', value=2.0)(X),
                                   atol=1e-12)
        self.assertEqual(Normalizer('max').method, 'max')

    def test_unknown_method_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            Normalizer('maximum')
        self.assertIn("expected one of", str(ctx.exception))
        # a numeric first positional argument (the old `gain`) is caught too
        with self.assertRaises(ValueError):
            Normalizer(2.0)

    def test_non_positive_value_is_rejected(self):
        with self.assertRaises(ValueError):
            Normalizer('var', value=0.0)

    def test_estimated_gain_carries_a_trailing_underscore(self):
        # D23: the gain of a Normalizer is measured, not configured
        normalizer = Normalizer('abs', value=2.0)
        normalizer(np.array([1., 2., 3., 4.]))
        self.assertAlmostEqual(normalizer.gain_, 0.5)
        self.assertNotIn('gain', inspect.signature(Normalizer).parameters)


class TestSimpleSynchronizer(unittest.TestCase):
    """The scale correction must invert the channel gain, not repeat it."""

    def test_restores_the_transmitted_signal(self):
        gain = 0.5 * np.exp(1j * 0.3)
        preamble, payload, x = make_frame(gain, delay=7)

        synchronizer = DataAidedSimpleSynchronizer(preamble)
        y = synchronizer(x)

        self.assertEqual(int(synchronizer.delay_), 7)
        # the applied scale is the inverse of the channel gain
        np.testing.assert_allclose(synchronizer.scale_, 1 / gain, rtol=1e-10)
        # the output is the transmitted frame, not a doubly-distorted copy
        np.testing.assert_allclose(y, np.hstack([preamble, payload]), atol=1e-10)

    def test_amplitude_is_compensated_not_amplified(self):
        # explicit non-regression on the inverted correction: an attenuation
        # a used to come out as a^2, i.e. |y| = |a|^2 |d| instead of |d|
        gain = 0.5
        preamble, payload, x = make_frame(gain, delay=3)
        y = DataAidedSimpleSynchronizer(preamble)(x)
        self.assertAlmostEqual(float(np.mean(np.abs(y))), 1.0, places=10)

    def test_scale_is_independent_of_the_preamble_power(self):
        # the correlation is normalized by the preamble energy, so a
        # preamble of arbitrary power still yields scale_ = 1/a
        gain = 0.5 * np.exp(1j * 0.3)
        preamble, payload, _ = make_frame(gain, delay=0)
        preamble = 3.0 * preamble
        x = np.hstack([np.zeros(4, dtype=complex),
                       gain * np.hstack([preamble, payload])])
        synchronizer = DataAidedSimpleSynchronizer(preamble)
        y = synchronizer(x)
        np.testing.assert_allclose(synchronizer.scale_, 1 / gain, rtol=1e-10)
        np.testing.assert_allclose(y, np.hstack([preamble, payload]), atol=1e-10)

    def test_scale_correction_disabled_only_realigns(self):
        gain = 0.5 * np.exp(1j * 0.3)
        preamble, payload, x = make_frame(gain, delay=5)
        synchronizer = DataAidedSimpleSynchronizer(preamble, False)
        y = synchronizer(x)
        self.assertEqual(synchronizer.scale_, 1)
        np.testing.assert_allclose(y, gain * np.hstack([preamble, payload]), atol=1e-10)

    def test_zero_energy_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            DataAidedSimpleSynchronizer(np.zeros(4))(np.ones(8))

    def test_estimated_attributes_carry_a_trailing_underscore(self):
        preamble, _, x = make_frame(1.0, delay=2)
        synchronizer = DataAidedSimpleSynchronizer(preamble)
        synchronizer(x)
        for name in ("delay_", "scale_", "cross_corr_", "n_vect_"):
            self.assertIsNotNone(getattr(synchronizer, name))
        for name in ("delay", "scale", "cross_corr", "n_vect"):
            with self.assertRaises(AttributeError):
                getattr(synchronizer, name)


class TestFineSynchronizer(unittest.TestCase):

    def test_restores_the_transmitted_amplitude(self):
        gain = 0.5 * np.exp(1j * 0.3)
        preamble, payload, x = make_frame(gain, delay=5)

        synchronizer = DataAidedFineSynchronizer(preamble, up_factor=2, signal_len=128)
        y = synchronizer(x)

        self.assertEqual(int(synchronizer.delay_), 5 * synchronizer.up_factor)
        # the polyphase resampling limits the accuracy, hence the tolerance;
        # what matters is that the gain is inverted (|scale_ * a| = 1) and
        # not repeated (|scale_ * a| = |a|^2 = 0.25 before the fix)
        self.assertAlmostEqual(abs(synchronizer.scale_ * gain), 1.0, places=2)
        self.assertAlmostEqual(float(np.mean(np.abs(y))), 1.0, places=1)

    def test_estimated_attributes_carry_a_trailing_underscore(self):
        preamble, _, x = make_frame(1.0, delay=2)
        synchronizer = DataAidedFineSynchronizer(preamble, up_factor=2)
        synchronizer(x)
        for name in ("delay_", "scale_", "cross_corr_", "n_vect_"):
            self.assertIsNotNone(getattr(synchronizer, name))
        for name in ("delay", "scale", "cross_corr", "n_vect"):
            with self.assertRaises(AttributeError):
                getattr(synchronizer, name)


class TestBlindCFOCompensator(unittest.TestCase):

    def _signal(self, w0=0.01, N=500, seed=2):
        rng = np.random.default_rng(seed)
        s = QPSK[rng.integers(0, 4, N)]
        return s, s * np.exp(1j * w0 * np.arange(N))

    def test_not_fitted_raises(self):
        _, x = self._signal()
        compensator = BlindCFOCompensator(should_fit=False)
        with self.assertRaises(NotFittedError):
            compensator(x)

    def test_reused_estimate_after_fit(self):
        _, x = self._signal()
        compensator = BlindCFOCompensator(should_fit=False)
        compensator.fit(x, 0.0)
        self.assertAlmostEqual(float(compensator.w0_), 0.01, places=4)
        compensator(x)  # no exception once fitted

    def test_estimated_attribute_carries_a_trailing_underscore(self):
        _, x = self._signal()
        compensator = BlindCFOCompensator()
        compensator(x)
        self.assertAlmostEqual(float(compensator.w0_), 0.01, places=4)
        self.assertFalse(hasattr(compensator, "w0"))


class TestBlindIQCompensator(unittest.TestCase):

    def test_estimated_attributes_carry_a_trailing_underscore(self):
        rng = np.random.default_rng(0)
        s = (rng.integers(0, 2, 2000) * 2 - 1) + 1j * (rng.integers(0, 2, 2000) * 2 - 1)
        x = 2 * s.real + 1j * (0.5 * s.real + s.imag)
        compensator = BlindIQCompensator(coef=1.0)
        y = compensator(x)
        self.assertAlmostEqual(float(np.mean(np.abs(y)**2)), 1.0, places=6)
        self.assertIsNotNone(compensator.alpha_)
        self.assertIsNotNone(compensator.beta_)
        for name in ("alpha", "beta"):
            with self.assertRaises(AttributeError):
                getattr(compensator, name)

    def test_no_dead_code_left_in_the_module(self):
        # the GSOP implementation of Fatadin et al. used to sit between two
        # methods as a bare string literal: valid Python, dead code
        source = pathlib.Path(compensators_module.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Module)):
                continue
            for statement in node.body[1:]:   # body[0] may be the docstring
                self.assertFalse(
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str),
                    f"bare string statement (dead code) in {getattr(node, 'name', 'module')}")


class TestDataAidedFIRCompensator(unittest.TestCase):

    def test_estimates_the_channel(self):
        x_ref = np.array([1.0, -1.0, 1.0, 1.0])
        received = np.convolve(x_ref, [1.0, 0.5])[:4]
        compensator = DataAidedFIRCompensator(reference=x_ref)
        y = compensator(received)
        np.testing.assert_allclose(y, x_ref, atol=1e-8)
        np.testing.assert_allclose(compensator.h_[:2], [1.0, 0.5], atol=1e-8)

    def test_not_fitted_raises(self):
        x_ref = np.array([1.0, -1.0, 1.0, 1.0])
        received = np.convolve(x_ref, [1.0, 0.5])[:4]
        compensator = DataAidedFIRCompensator(reference=x_ref, should_fit=False)
        with self.assertRaises(NotFittedError):
            compensator(received)

    def test_estimated_attribute_carries_a_trailing_underscore(self):
        x_ref = np.array([1.0, -1.0, 1.0, 1.0])
        compensator = DataAidedFIRCompensator(reference=x_ref)
        compensator(np.convolve(x_ref, [1.0, 0.5])[:4])
        self.assertFalse(hasattr(compensator, "h"))


if __name__ == '__main__':
    unittest.main()
