"""Raman gain applied inside the split-step loop of ``FiberLink`` (D45).

The second pass of the Raman work: the solver produces the gain profile
:math:`G(z)`, and the link consumes it *inside* the linear step, so the
Kerr term sees the power the fibre really carries. The two properties
that make the integration trustworthy are here -- a span stays
transparent whether or not it is pumped, and a vanishing Raman gain
reproduces the unpumped link bit for bit.
"""
import logging
import unittest

import numpy as np

from comnumpy.optical import FiberLink, solve_raman

FS = 100e9
N = 1024
SPAN_KM = 80.0
STEPS = 20


def field(power_W=1e-3, seed=0, n=N):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(power_W / 2)


def solution(pump_W=0.5, bandwidth_Hz=0.0, length_km=SPAN_KM, gain=0.4):
    return solve_raman(length_km=length_km, gain_peak_W_km=gain,
                       pump_backward_W=pump_W, bandwidth_Hz=bandwidth_Hz)


def link(**kwargs):
    kwargs.setdefault("noise_scaling", 0)
    return FiberLink(kwargs.pop("n_spans", 3), L_span=SPAN_KM, StPS=STEPS,
                     fs=FS, **kwargs)


class TestTransparency(unittest.TestCase):
    """A span must come out at the power it went in, pumped or not."""

    def test_a_pumped_span_is_transparent(self):
        x = field()
        for use_only_linear in (False, True):
            with self.subTest(use_only_linear=use_only_linear):
                y = link(raman=solution(),
                         use_only_linear=use_only_linear)(x)
                ratio = float(np.mean(np.abs(y) ** 2) / np.mean(np.abs(x) ** 2))
                self.assertAlmostEqual(ratio, 1.0, places=6)

    def test_the_linear_only_mode_applies_the_gain_too(self):
        """It has no step loop, so it needs the profile applied whole.

        Without this the EDFA is reduced for a Raman gain that is never
        applied, and the span silently loses the whole on-off gain --
        15 dB, measured, before the fix.
        """
        x = field()
        pumped = link(raman=solution(), use_only_linear=True)(x)
        bare = link(use_only_linear=True)(x)
        np.testing.assert_allclose(np.mean(np.abs(pumped) ** 2),
                                   np.mean(np.abs(bare) ** 2), rtol=1e-9)

    def test_the_edfa_makes_up_exactly_what_raman_did_not(self):
        raman = solution()
        pumped = link(raman=raman)
        pumped.prepare(field())
        bare = link()
        bare.prepare(field())
        span_loss_dB = 0.2 * SPAN_KM
        self.assertAlmostEqual(20 * np.log10(bare.edfa_gain), span_loss_dB,
                               places=9)
        self.assertAlmostEqual(20 * np.log10(pumped.edfa_gain),
                               span_loss_dB - raman.on_off_gain_dB, places=9)


class TestRegression(unittest.TestCase):

    def test_a_vanishing_raman_gain_reproduces_the_unpumped_link(self):
        """The check that no code path changed for existing chains."""
        x = field()
        bare = link()(x)
        negligible = solve_raman(length_km=SPAN_KM, gain_peak_W_km=1e-12,
                                 pump_backward_W=1e-9, bandwidth_Hz=0.0)
        pumped = link(raman=negligible)(x)
        error = np.max(np.abs(pumped - bare)) / np.max(np.abs(bare))
        self.assertLess(error, 1e-9)

    def test_raman_none_is_the_default_and_changes_nothing(self):
        x = field()
        np.testing.assert_array_equal(link()(x), link(raman=None)(x))


class TestWhereTheGainHappens(unittest.TestCase):
    """The reason a profile is used instead of a lumped gain.

    Both spans are transparent by construction, so the output power
    cannot distinguish them: what differs is *where* along the span the
    gain is delivered, and that is read off the per-step gains.
    """

    def test_the_step_gains_multiply_to_the_on_off_gain(self):
        raman = solution()
        chain = link(raman=raman)
        chain.prepare(field())
        total_dB = 20 * np.log10(float(np.prod(chain.raman_step_gain_)))
        self.assertAlmostEqual(total_dB, raman.on_off_gain_dB, places=6)

    def test_counter_pumping_loads_the_gain_at_the_end_of_the_span(self):
        chain = link(raman=solution())
        chain.prepare(field())
        gains_dB = 20 * np.log10(chain.raman_step_gain_)
        first_half = gains_dB[:STEPS // 2].sum()
        second_half = gains_dB[STEPS // 2:].sum()
        self.assertGreater(second_half, 3 * first_half)


class TestStepConvergence(unittest.TestCase):
    """The Raman gain must not degrade the order of the scheme.

    The symmetric split-step is second order, and the gain belongs to
    the *linear* operator: applied once per step instead of split around
    the Kerr term it breaks the symmetry and drops the scheme to first
    order. Measured on the SPM phase of a CW field, that cost a factor
    24 in accuracy at StPS=20 before the gain was split.
    """

    def spm_phase(self, n_steps, raman, step_type="linear"):
        x = np.ones(256, dtype=complex) * np.sqrt(2e-3)
        y = FiberLink(1, L_span=SPAN_KM, StPS=n_steps, fs=FS,
                      noise_scaling=0, raman=raman, step_type=step_type)(x)
        return float(np.angle(y[128]))

    def test_the_scheme_stays_second_order_with_raman(self):
        raman = solution()
        reference = self.spm_phase(2000, raman)
        errors = [abs(self.spm_phase(n, raman) - reference)
                  for n in (10, 20, 40)]
        for coarse, fine in zip(errors, errors[1:], strict=False):
            # second order: halving the step divides the error by ~4
            self.assertGreater(coarse / fine, 3.0)

    def test_the_half_step_gains_still_telescope_to_the_total(self):
        """Splitting must not disturb the exactness of the total gain."""
        raman = solution()
        for n_steps in (1, 3, 20, 97):
            with self.subTest(StPS=n_steps):
                chain = FiberLink(1, L_span=SPAN_KM, StPS=n_steps, fs=FS,
                                  noise_scaling=0, raman=raman)
                chain.prepare(field())
                self.assertEqual(len(chain.raman_step_gain_), 2 * n_steps)
                total_dB = 20 * np.log10(float(np.prod(chain.raman_step_gain_)))
                self.assertAlmostEqual(total_dB, raman.on_off_gain_dB, places=6)

    def test_warns_when_logarithmic_steps_meet_raman(self):
        with self.assertLogs("comnumpy.optical.links", logging.WARNING) as logs:
            chain = FiberLink(1, L_span=SPAN_KM, StPS=10, fs=FS,
                              noise_scaling=0, raman=solution(),
                              step_type="logarithmic")
            chain.prepare(field())
        self.assertIn("logarithmic steps", logs.output[0])


class TestAse(unittest.TestCase):

    def test_a_zero_bandwidth_solution_adds_no_raman_noise(self):
        chain = link(raman=solution(bandwidth_Hz=0.0))
        chain.prepare(field())
        self.assertEqual(chain.raman_sigma2_, 0.0)

    def test_raman_ase_degrades_the_output_snr(self):
        x = field()
        clean = link(raman=solution(bandwidth_Hz=0.0), noise_scaling=0)(x)
        noisy = link(raman=solution(bandwidth_Hz=12.5e9), noise_scaling=1,
                     seed=7)(x)
        added = float(np.mean(np.abs(noisy - clean) ** 2))
        self.assertGreater(added, 0.0)

    def test_the_noise_is_reproducible_from_the_seed(self):
        x = field()
        first = link(raman=solution(bandwidth_Hz=12.5e9), noise_scaling=1,
                     seed=3)(x)
        second = link(raman=solution(bandwidth_Hz=12.5e9), noise_scaling=1,
                      seed=3)(x)
        np.testing.assert_array_equal(first, second)


class TestGuards(unittest.TestCase):

    def test_rejects_a_solution_over_a_different_span(self):
        with self.assertRaises(ValueError) as ctx:
            link(raman=solution(length_km=50.0))(field())
        self.assertIn("80.0 km", str(ctx.exception))

    def test_warns_when_raman_over_compensates_the_span(self):
        """A gain above the span loss makes the EDFA an attenuator."""
        strong = solve_raman(length_km=SPAN_KM, gain_peak_W_km=0.4,
                             pump_backward_W=1.0, bandwidth_Hz=0.0)
        self.assertGreater(strong.on_off_gain_dB, 0.2 * SPAN_KM)
        with self.assertLogs("comnumpy.optical.links", logging.WARNING) as logs:
            chain = link(raman=strong)
            chain.prepare(field())
        self.assertIn("over-compensates", logs.output[0])
        self.assertLess(chain.edfa_gain, 1.0)

    def test_an_over_compensated_span_is_still_transparent(self):
        x = field()
        strong = solve_raman(length_km=SPAN_KM, gain_peak_W_km=0.4,
                             pump_backward_W=1.0, bandwidth_Hz=0.0)
        y = link(raman=strong)(x)
        self.assertAlmostEqual(
            float(np.mean(np.abs(y) ** 2) / np.mean(np.abs(x) ** 2)), 1.0,
            places=6)


if __name__ == "__main__":
    unittest.main()
