r"""Dual-polarization propagation: the Manakov equation (D47).

A field shaped ``(..., 2, N)`` -- the antenna axis of D2 -- is
propagated with

.. math::

    \frac{\partial E_{x,y}}{\partial z} =
    -\frac{\alpha}{2} E_{x,y}
    - j \frac{\beta_2}{2} \frac{\partial^2 E_{x,y}}{\partial t^2}
    + j \frac{8\gamma}{9}\left(|E_x|^2 + |E_y|^2\right) E_{x,y}

instead of the scalar NLSE. Which equation is integrated is read off
the shape of the field, the way which pumps are on is read off their
powers (D41), so a link and its compensator cannot end up integrating
different models.

The two properties worth testing are the ones that *define* the model
rather than restate the code: an empty second polarization must give
back the scalar answer with :math:`8\gamma/9`, and the answer must be
invariant under a constant unitary rotation of the input pair -- which
is precisely what averaging over the fibre's random birefringence buys,
and what a wrong cross-term would destroy.
"""
import unittest

import numpy as np

from comnumpy.exceptions import ShapeError
from comnumpy.optical import DBP, FiberLink
from comnumpy.optical.fiber import FiberSpec

FS = 100e9
N = 512
SPAN_KM = 80.0
STEPS = 8
GAMMA = 1.3


def field(seed=0, power_W=2e-3, shape=(2, N)):
    rng = np.random.default_rng(seed)
    return ((rng.normal(size=shape) + 1j * rng.normal(size=shape))
            * np.sqrt(power_W / 2))


def link(gamma=GAMMA, **kwargs):
    kwargs.setdefault("noise_scaling", 0)
    return FiberLink(2, L_span=SPAN_KM, StPS=STEPS, fs=FS,
                     fiber=FiberSpec(0.2, gamma=gamma), **kwargs)


def jones(theta, phi):
    """An arbitrary constant unitary rotation of the polarization pair."""
    return np.array([[np.cos(theta), np.sin(theta) * np.exp(1j * phi)],
                     [-np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]])


class TestModelSelection(unittest.TestCase):

    def test_a_pair_selects_the_manakov_model(self):
        chain = link()
        chain.prepare(field())
        self.assertTrue(chain.manakov_)

    def test_a_single_field_stays_scalar(self):
        for x in (field(shape=(N,)), field(shape=(1, N))):
            with self.subTest(shape=x.shape):
                chain = link()
                chain.prepare(x)
                self.assertFalse(chain.manakov_)

    def test_the_output_keeps_the_shape_it_was_given(self):
        for shape in ((N,), (1, N), (2, N), (3, 2, N)):
            with self.subTest(shape=shape):
                self.assertEqual(link()(field(shape=shape)).shape, shape)

    def test_any_other_polarization_count_is_refused(self):
        with self.assertRaises(ShapeError) as ctx:
            link()(field(shape=(4, N)))
        message = str(ctx.exception)
        self.assertIn("4 on the polarization axis", message)
        self.assertIn("WDMMultiplexer", message)

    def test_the_compensator_reads_the_shape_the_same_way(self):
        compensator = DBP(2, L_span=SPAN_KM, StPS=STEPS, fs=FS)
        compensator.prepare(field())
        self.assertTrue(compensator.manakov_)


class TestReductionToTheScalarModel(unittest.TestCase):
    r"""With :math:`E_y = 0` the pair must reproduce the scalar NLSE.

    Not with :math:`\gamma` but with :math:`8\gamma/9`: the factor is a
    property of the *model*, so it applies to a lone polarization inside
    a Manakov solve too. That is the whole reason it does not live in
    ``FiberSpec``.
    """

    def test_an_empty_second_polarization_gives_the_scalar_answer(self):
        x = field()
        pair = np.stack([x[0], np.zeros(N, dtype=complex)])
        propagated = link()(pair)
        scalar = link(gamma=8 / 9 * GAMMA)(x[0])
        np.testing.assert_allclose(propagated[0], scalar, rtol=1e-12,
                                   atol=1e-14)
        np.testing.assert_array_equal(propagated[1], np.zeros(N))

    def test_the_fibre_gamma_is_not_rescaled_by_the_user(self):
        """A pair with gamma= must differ from a scalar run with gamma=."""
        x = field()
        pair = np.stack([x[0], np.zeros(N, dtype=complex)])
        self.assertFalse(np.allclose(link()(pair)[0], link()(x[0])))


class TestUnitaryInvariance(unittest.TestCase):
    r"""The property that makes Manakov the right averaged model.

    The nonlinear term depends only on :math:`|E_x|^2 + |E_y|^2`, which
    no constant unitary rotation changes, so propagation commutes with
    any such rotation. A wrong cross-polarization coefficient -- 1
    instead of 8/9, or a missing term -- breaks this immediately.
    """

    def test_propagation_commutes_with_a_constant_jones_rotation(self):
        x = field()
        chain = link()
        direct = chain(x)
        for theta, phi in ((0.7, 1.3), (np.pi / 4, 0.0), (1.1, -2.0)):
            with self.subTest(theta=theta, phi=phi):
                rotation = jones(theta, phi)
                rotated = rotation.conj().T @ chain(rotation @ x)
                error = (np.max(np.abs(rotated - direct))
                         / np.max(np.abs(direct)))
                self.assertLess(float(error), 1e-12)

    def test_the_total_intensity_drives_both_polarizations(self):
        """Doubling the *other* polarization must change this one."""
        x = field()
        weak = np.stack([x[0], 0.01 * x[1]])
        strong = np.stack([x[0], x[1]])
        self.assertGreater(
            float(np.max(np.abs(link()(strong)[0] - link()(weak)[0]))), 0.0)


class TestRoundTrip(unittest.TestCase):

    def test_dbp_inverts_the_dual_polarization_link(self):
        x = field()
        chain = link()
        compensator = DBP(2, L_span=SPAN_KM, StPS=STEPS, fs=FS)
        error = np.max(np.abs(compensator(chain(x)) - x)) / np.max(np.abs(x))
        self.assertLess(float(error), 1e-12)

    def test_the_span_stays_transparent_on_both_polarizations(self):
        x = field()
        y = link()(x)
        for polarization in (0, 1):
            with self.subTest(polarization=polarization):
                ratio = (np.mean(np.abs(y[polarization]) ** 2)
                         / np.mean(np.abs(x[polarization]) ** 2))
                self.assertAlmostEqual(float(ratio), 1.0, places=6)


class TestNoise(unittest.TestCase):

    def test_each_polarization_gets_its_own_ase(self):
        """One EDFA, two independent noises -- not the same one twice."""
        x = np.zeros((2, N), dtype=complex)
        y = FiberLink(4, L_span=SPAN_KM, StPS=1, fs=FS, noise_scaling=1,
                      seed=5, use_only_linear=True)(x)
        self.assertGreater(float(np.mean(np.abs(y) ** 2)), 0.0)
        correlation = np.abs(np.vdot(y[0], y[1])) / np.linalg.norm(y) ** 2
        self.assertLess(float(correlation), 0.1)

    def test_the_noise_is_reproducible_from_the_seed(self):
        x = field()
        first = FiberLink(2, L_span=SPAN_KM, StPS=2, fs=FS, seed=11)(x)
        second = FiberLink(2, L_span=SPAN_KM, StPS=2, fs=FS, seed=11)(x)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
