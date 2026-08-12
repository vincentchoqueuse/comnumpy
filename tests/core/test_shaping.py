r"""Probabilistic shaping: what an enumerative matcher must guarantee.

A distribution matcher is not an approximation, and that is the whole
point of the construction: it is a bijection between bit strings and a
finite set of sequences. So the first family of tests here is exact --
``decode(encode(bits)) == bits`` for thousands of random inputs, on
several compositions and energy budgets -- and the second checks the
properties the two codes are *chosen* for:

* CCDM emits the target composition in **every** block, not on average;
* ESS keeps every block inside the energy sphere, and beats CCDM at
  short blocklengths, which is the reason it exists;
* the rate loss of both goes to zero as the block grows, which is the
  reason a finite block is not free;
* shaping actually buys something -- the shaped constellation reaches a
  higher achievable rate than the uniform one at equal power, measured
  with the estimators of :mod:`comnumpy.core.information` rather than
  asserted.
"""
import math
import unittest

import numpy as np

from comnumpy.core import Sequential
from comnumpy.core.capacity import constellation_capacity
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.shaping import (AmplitudeDemapper, AmplitudeMapper,
                                   ConstantCompositionMatcher,
                                   DistributionDematcher, DistributionMatcher,
                                   SphereShaper, blahut_arimoto,
                                   composition_from_distribution,
                                   distribution_entropy, maxwell_boltzmann,
                                   shaping_gain_dB)
from comnumpy.exceptions import ShapeError

AMPLITUDES = np.array([1.0, 3.0, 5.0, 7.0])       # half of 8-PAM
ULTIMATE_GAIN_DB = 10 * np.log10(np.pi * np.e / 6)


def random_bits(rng, n):
    return rng.integers(0, 2, size=n)


class TestInvertibility(unittest.TestCase):
    """The property that makes it a code and not a heuristic."""

    def test_ccdm_round_trips_every_input(self):
        rng = np.random.default_rng(0)
        for composition in [(6, 2), (17, 10, 4, 1), (8, 8, 8, 8), (30, 2)]:
            matcher = ConstantCompositionMatcher(
                np.arange(len(composition), dtype=float) * 2 + 1,
                composition=composition)
            with self.subTest(composition=composition):
                for _ in range(200):
                    bits = random_bits(rng, matcher.n_bits)
                    np.testing.assert_array_equal(
                        matcher.decode(matcher.encode(bits)), bits)

    def test_ess_round_trips_every_input(self):
        rng = np.random.default_rng(1)
        for length, budget in [(8, 3), (12, 10), (16, 20), (20, 5)]:
            shaper = SphereShaper(AMPLITUDES, length=length,
                                  max_energy=budget)
            with self.subTest(length=length, max_energy=budget):
                for _ in range(100):
                    bits = random_bits(rng, shaper.n_bits)
                    np.testing.assert_array_equal(
                        shaper.decode(shaper.encode(bits)), bits)

    def test_the_map_is_injective_over_the_whole_input_set(self):
        """Exhaustive on a small code: 2^k distinct blocks, no collision."""
        matcher = ConstantCompositionMatcher(np.array([1.0, 3.0, 5.0]),
                                             composition=(3, 2, 1))
        seen = set()
        for index in range(2 ** matcher.n_bits):
            bits = np.array([(index >> shift) & 1
                             for shift in range(matcher.n_bits - 1, -1, -1)])
            seen.add(tuple(matcher.encode(bits)))
        self.assertEqual(len(seen), 2 ** matcher.n_bits)

    def test_the_two_codes_use_a_prefix_of_the_enumeration(self):
        """Index 0 is the first sequence in lexicographic order."""
        matcher = ConstantCompositionMatcher(np.array([1.0, 3.0]),
                                             composition=(4, 2))
        first = matcher.encode(np.zeros(matcher.n_bits, dtype=int))
        np.testing.assert_array_equal(first, [0, 0, 0, 0, 1, 1])


class TestConstantComposition(unittest.TestCase):

    def test_every_block_carries_the_composition_exactly(self):
        """Not on average: every single one."""
        rng = np.random.default_rng(2)
        composition = (17, 10, 4, 1)
        matcher = ConstantCompositionMatcher(AMPLITUDES,
                                             composition=composition)
        for _ in range(50):
            block = matcher.encode(random_bits(rng, matcher.n_bits))
            counts = np.bincount(block, minlength=4)
            np.testing.assert_array_equal(counts, composition)

    def test_the_composition_tracks_the_target_distribution(self):
        target = maxwell_boltzmann(AMPLITUDES, entropy=1.5)
        errors = []
        for length in (16, 64, 256, 1024):
            counts = np.array(composition_from_distribution(target, length))
            errors.append(float(np.max(np.abs(counts / length - target))))
        self.assertTrue(all(a >= b for a, b in zip(errors, errors[1:],
                                                   strict=False)), errors)
        self.assertLess(errors[-1], 1e-3)

    def test_the_rate_loss_vanishes_with_the_block_length(self):
        target = maxwell_boltzmann(AMPLITUDES, entropy=1.5)
        losses = [ConstantCompositionMatcher(AMPLITUDES, distribution=target,
                                             length=length).rate_loss
                  for length in (16, 64, 256, 1024)]
        self.assertTrue(all(a > b for a, b in zip(losses, losses[1:],
                                                  strict=False)), losses)
        self.assertLess(losses[-1], 0.02)
        self.assertGreater(losses[0], 0.1)

    def test_the_rate_never_exceeds_the_entropy_it_reproduces(self):
        """k/n <= H(n_i/n): a matcher cannot invent information."""
        target = maxwell_boltzmann(AMPLITUDES, entropy=1.7)
        for length in (8, 32, 128, 512):
            matcher = ConstantCompositionMatcher(AMPLITUDES,
                                                 distribution=target,
                                                 length=length)
            with self.subTest(length=length):
                composition = np.array(matcher.composition) / length
                self.assertLessEqual(matcher.rate,
                                     distribution_entropy(composition))


class TestSphereShaping(unittest.TestCase):

    def test_no_block_leaves_the_sphere(self):
        rng = np.random.default_rng(3)
        shaper = SphereShaper(AMPLITUDES, length=16, max_energy=20)
        for _ in range(100):
            block = shaper.encode(random_bits(rng, shaper.n_bits))
            energy = sum(shaper.energies[symbol] for symbol in block)
            self.assertLessEqual(energy, shaper.max_energy)

    def test_the_energies_are_the_reduced_integer_ones(self):
        """Amplitudes 1, 3, 5, 7 have squared values 1, 9, 25, 49."""
        shaper = SphereShaper(AMPLITUDES, length=4, max_energy=6)
        self.assertEqual(shaper.energies, (0, 1, 3, 6))

    def test_a_normalized_alphabet_is_refused_by_name(self):
        from comnumpy.core.utils import get_alphabet
        with self.assertRaises(ValueError) as ctx:
            SphereShaper(get_alphabet("PAM", 8)[4:], length=8, max_energy=5)
        self.assertIn("norm=False", str(ctx.exception))

    def test_asking_for_bits_finds_the_smallest_budget_that_carries_them(self):
        shaper = SphereShaper(AMPLITUDES, length=12, n_bits=15)
        self.assertGreaterEqual(shaper.n_bits, 15)
        tighter = SphereShaper(AMPLITUDES, length=12,
                               max_energy=shaper.max_energy - 1)
        self.assertLess(tighter.n_bits, 15)

    def test_it_beats_a_constant_composition_at_short_blocklength(self):
        r"""The reason ESS exists.

        At equal average energy per symbol, the sphere holds more
        sequences than the set of permutations of one composition,
        because it lets a block spend unevenly. The advantage is a
        short-block phenomenon: it shrinks as the blocklength grows.
        """
        advantages = []
        for length in (12, 24, 48):
            shaper = SphereShaper(AMPLITUDES, length=length,
                                  max_energy=length)   # 1 unit per symbol
            budget = shaper.max_energy
            # the constant composition of the same average energy, and
            # the largest rate it can offer
            best = 0.0
            for counts in _compositions(length, 4):
                energy = sum(count * cost
                             for count, cost in zip(counts, (0, 1, 3, 6),
                                                    strict=True))
                if energy <= budget:
                    best = max(best, _ccdm_rate(counts))
            with self.subTest(length=length):
                self.assertGreater(shaper.rate, best)
            advantages.append(shaper.rate - best)
        self.assertLess(advantages[-1], advantages[0])


def _ccdm_rate(counts):
    """Bits per symbol of the CCDM built on that composition."""
    total = sum(counts)
    sequences = math.factorial(total)
    for count in counts:
        sequences //= math.factorial(count)
    return (sequences.bit_length() - 1) / total


def _compositions(total, parts):
    """Every way of writing ``total`` as ``parts`` non-negative counts."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


class TestMaxwellBoltzmann(unittest.TestCase):

    def test_lambda_zero_is_the_uniform_distribution(self):
        np.testing.assert_allclose(maxwell_boltzmann(AMPLITUDES, lam=0.0),
                                   np.full(4, 0.25))

    def test_the_entropy_target_is_reached(self):
        for target in (1.1, 1.5, 1.9, 2.0):
            with self.subTest(entropy=target):
                shaped = maxwell_boltzmann(AMPLITUDES, entropy=target)
                self.assertAlmostEqual(distribution_entropy(shaped), target,
                                       places=9)

    def test_energy_decreases_as_lambda_grows(self):
        energies = [float(np.sum(maxwell_boltzmann(AMPLITUDES, lam=lam)
                                 * AMPLITUDES ** 2))
                    for lam in (0.0, 0.01, 0.05, 0.2)]
        self.assertTrue(all(a > b for a, b in zip(energies, energies[1:],
                                                  strict=False)), energies)

    def test_it_maximizes_the_entropy_at_its_own_energy(self):
        """The defining property, checked against random competitors."""
        shaped = maxwell_boltzmann(AMPLITUDES, entropy=1.5)
        energy = float(np.sum(shaped * AMPLITUDES ** 2))
        rng = np.random.default_rng(5)
        for _ in range(300):
            other = rng.dirichlet(np.ones(4))
            if float(np.sum(other * AMPLITUDES ** 2)) <= energy + 1e-12:
                self.assertLessEqual(distribution_entropy(other),
                                     distribution_entropy(shaped) + 1e-9)

    def test_a_symmetric_constellation_cannot_be_shaped_below_one_bit(self):
        """The silent-wrong-answer case: equal energies keep equal mass."""
        pam = np.arange(-3, 4, 2).astype(float)      # 4-PAM, energies 9,1,1,9
        with self.assertRaises(ValueError) as ctx:
            maxwell_boltzmann(pam, entropy=0.6)
        message = str(ctx.exception)
        self.assertIn("smallest energy", message)
        self.assertIn("2 here", message)

    def test_the_two_parameterizations_are_exclusive(self):
        with self.assertRaises(ValueError) as ctx:
            maxwell_boltzmann(AMPLITUDES, lam=0.1, entropy=1.5)
        self.assertIn("D41", str(ctx.exception))
        with self.assertRaises(ValueError):
            maxwell_boltzmann(AMPLITUDES)


class TestTheLawThatMaximizesTheRate(unittest.TestCase):
    r"""Blahut-Arimoto, and what it says about the closed form.

    :func:`maxwell_boltzmann` maximizes the *entropy* at a given energy,
    which is not the same problem as maximizing the *rate over a
    channel*. The tests below solve the second problem numerically and
    hold the first against it: the answers must agree closely, but the
    inequality only points one way, and asserting it is the only way to
    know that the closed form is the shortcut it is claimed to be.

    Everything is indexed by the **energy budget**, never by the
    multiplier. A transmitter has a power budget; the multiplier is an
    internal variable, and comparing two laws that spend different
    energies compares nothing at all.
    """

    PAM16 = np.arange(-15, 16, 2).astype(float)
    UNIFORM_ENERGY = 85.0                     # mean of the odd squares
    GRID = [(snr_dB, energy) for snr_dB in (12.0, 18.0, 24.0)
            for energy in (25.0, 45.0, 70.0, 85.0)]

    @classmethod
    def setUpClass(cls):
        """One root-find per grid point, not per test."""
        cls.solved = {
            (snr_dB, energy): blahut_arimoto(
                cls.PAM16, sigma2=cls.UNIFORM_ENERGY / 10 ** (snr_dB / 10),
                energy=energy)
            for snr_dB, energy in cls.GRID}

    @staticmethod
    def rate(law, sigma2, alphabet):
        """Exact mutual information of ``law``, by quadrature.

        ``constellation_capacity`` puts the noise on two dimensions, so
        a real channel of variance ``sigma2`` is passed as the complex
        SNR ``1 / (2 sigma2)``.
        """
        return float(constellation_capacity(alphabet, 1 / (2 * sigma2),
                                            px=law))

    def test_it_spends_the_budget_it_was_given(self):
        """The root-find has to land, or nothing below means anything."""
        for (snr_dB, target), law in self.solved.items():
            with self.subTest(snr_dB=snr_dB, energy=target):
                self.assertAlmostEqual(
                    float(np.sum(law * self.PAM16 ** 2)), target, places=6)

    def test_a_loose_budget_on_a_clean_channel_returns_the_uniform_law(self):
        law = blahut_arimoto(self.PAM16, sigma2=85.0 / 1e4,
                             energy=self.UNIFORM_ENERGY)
        np.testing.assert_allclose(law, np.full(16, 1 / 16), atol=1e-7)

    def test_a_symmetric_constellation_gives_a_symmetric_law(self):
        for law in self.solved.values():
            np.testing.assert_allclose(law, law[::-1], atol=1e-9)

    def test_it_beats_maxwell_boltzmann_at_the_same_energy(self):
        """The defining property: nothing on this budget carries more."""
        for (snr_dB, target), best in self.solved.items():
            sigma2 = self.UNIFORM_ENERGY / 10 ** (snr_dB / 10)
            with self.subTest(snr_dB=snr_dB, energy=target):
                closed = maxwell_boltzmann(
                    self.PAM16, lam=_matched_lambda(self.PAM16, target))
                self.assertAlmostEqual(
                    float(np.sum(closed * self.PAM16 ** 2)), target,
                    places=6)
                self.assertGreaterEqual(
                    self.rate(best, sigma2, self.PAM16),
                    self.rate(closed, sigma2, self.PAM16) - 1e-6)

    def test_nothing_random_on_the_same_energy_carries_more_either(self):
        """The optimality claim against competitors that are not MB.

        Beating one closed form could be luck in the shape of the
        family; beating a few hundred draws that spend the same energy
        cannot be.
        """
        rng = np.random.default_rng(11)
        sigma2 = self.UNIFORM_ENERGY / 10 ** (18.0 / 10)
        budget = 45.0
        ceiling = self.rate(self.solved[(18.0, budget)], sigma2, self.PAM16)
        energies = self.PAM16 ** 2
        # A uniform draw over the simplex spends 85 on average, so most
        # would miss the budget. Each draw is mixed with the cheapest law
        # -- all the mass on the two innermost points -- and the mixing
        # fraction is solved for: the energy is affine in it, so the
        # competitor lands on the budget exactly.
        cheapest = np.zeros(16)
        cheapest[[7, 8]] = 0.5
        for _ in range(200):
            draw = rng.dirichlet(np.ones(16))
            spent = float(draw @ energies)
            if spent <= budget:
                other = draw            # already inside the budget
            else:
                fraction = (budget - 1.0) / (spent - 1.0)
                other = fraction * draw + (1 - fraction) * cheapest
            self.assertLessEqual(float(other @ energies), budget + 1e-9)
            self.assertLessEqual(
                self.rate(other, sigma2, self.PAM16), ceiling + 1e-9)

    def test_the_closed_form_costs_a_few_hundredths_of_a_bit_at_worst(self):
        """Kschischang and Pasupathy's claim, as a number.

        This is what justifies using the closed form everywhere else in
        the module. Measured over this grid the gap runs from 0.00001
        bit (24 dB, budget 25) to 0.036 bit (12 dB, budget 85): largest
        where the budget is loosest and the channel noisiest, because
        that is where the true maximizer starts dropping points and
        Maxwell-Boltzmann, which cannot, has flattened to the uniform
        law. It collapses at high SNR, where both are uniform anyway.
        """
        worst = 0.0
        for (snr_dB, target), law in self.solved.items():
            sigma2 = self.UNIFORM_ENERGY / 10 ** (snr_dB / 10)
            closed = maxwell_boltzmann(
                self.PAM16, lam=_matched_lambda(self.PAM16, target))
            worst = max(worst, self.rate(law, sigma2, self.PAM16)
                        - self.rate(closed, sigma2, self.PAM16))
        self.assertLess(worst, 0.05, f"gap of {worst:.4f} bit")
        # and the claim is only worth something if it is tight: a bound
        # of one bit would pass too, and would say nothing
        self.assertGreater(worst, 0.001, f"gap of {worst:.4f} bit")

    def test_at_low_snr_it_drops_points_the_closed_form_cannot(self):
        """Why the two families are not the same object.

        Maxwell-Boltzmann gives every point a strictly positive
        probability, whatever lambda does. The true maximizer does not:
        on a noisy enough channel it sets some of them to exactly zero,
        because two points the receiver cannot tell apart are worth less
        than one point used more often. That is also why the iteration
        is slow there -- it is converging to the boundary of the simplex.
        """
        law = blahut_arimoto(self.PAM16, sigma2=85.0 / 10 ** 0.6,
                             energy=self.UNIFORM_ENERGY)
        self.assertGreater(int(np.sum(law < 1e-12)), 0)
        self.assertTrue(bool(np.all(maxwell_boltzmann(self.PAM16,
                                                      lam=0.5) > 0)))

    def test_an_unbinding_budget_is_refused(self):
        """The constraint has to bind, or the question is not the one asked.

        Left free, the maximizer spends *more* than the uniform law: it
        pushes mass onto the outer points to separate them further. So
        there is a ceiling above which a budget constrains nothing, and
        asking for it is a mistake worth naming rather than answering.
        """
        unconstrained = blahut_arimoto(self.PAM16, sigma2=85.0 / 10, lam=0.0)
        spent = float(np.sum(unconstrained * self.PAM16 ** 2))
        self.assertGreater(spent, self.UNIFORM_ENERGY)
        with self.assertRaises(ValueError) as ctx:
            blahut_arimoto(self.PAM16, sigma2=85.0 / 10, energy=spent * 1.5)
        self.assertIn("does not bind", str(ctx.exception))

    def test_the_refusals(self):
        with self.assertRaises(ValueError) as ctx:
            blahut_arimoto(np.array([1 + 1j, -1 - 1j]), sigma2=1.0, lam=0.0)
        self.assertIn("np.unique(np.real(alphabet))", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            blahut_arimoto(self.PAM16, sigma2=1.0, lam=0.1, energy=40.0)
        self.assertIn("D41", str(ctx.exception))
        with self.assertRaises(ValueError):
            blahut_arimoto(self.PAM16, sigma2=1.0)
        with self.assertRaises(ValueError):
            blahut_arimoto(self.PAM16, sigma2=0.0, lam=0.0)
        with self.assertRaises(ValueError):
            blahut_arimoto(self.PAM16, sigma2=1.0, lam=-0.1)

    def test_it_says_so_when_it_has_not_converged(self):
        """Non-convergence is reported, not swallowed."""
        with self.assertLogs("comnumpy.core.shaping", level="WARNING") as logs:
            blahut_arimoto(self.PAM16, sigma2=85.0 / 10 ** 0.6, lam=0.004,
                           max_iter=5)
        self.assertIn("still to gain", logs.output[0])


def _matched_lambda(points, energy):
    """The Maxwell-Boltzmann parameter spending exactly ``energy``."""
    low, high = 0.0, 1e4 / float(np.ptp(points ** 2))
    for _ in range(200):
        middle = 0.5 * (low + high)
        spent = float(np.sum(maxwell_boltzmann(points, lam=middle)
                             * points ** 2))
        if spent > energy:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


class TestShapingGain(unittest.TestCase):

    def test_a_uniform_distribution_is_the_zero_of_the_scale(self):
        for order in (4, 16, 64):
            pam = np.arange(-(order - 1), order, 2).astype(float)
            with self.subTest(order=order):
                gain = shaping_gain_dB(pam, np.full(order, 1 / order))
                self.assertAlmostEqual(
                    gain, 10 * np.log10(order ** 2 / (order ** 2 - 1)),
                    places=9)
                # 0.28 dB at M=4, 0.007 at M=16, 0.0007 at M=64: the
                # residual is the discrete grid, and it vanishes with M
                self.assertLess(gain, 0.3)

    def test_no_distribution_exceeds_the_1_53_dB_limit(self):
        """The ceiling the whole subject is named after."""
        for order in (4, 8, 16, 32, 64, 128):
            pam = np.arange(-(order - 1), order, 2).astype(float)
            for fraction in (0.3, 0.5, 0.7, 0.85, 0.95, 0.999):
                entropy = fraction * np.log2(order)
                if entropy < 1.0:
                    continue              # below the symmetric-pair floor
                with self.subTest(order=order, entropy=round(entropy, 2)):
                    shaped = maxwell_boltzmann(pam, entropy=entropy)
                    self.assertLessEqual(shaping_gain_dB(pam, shaped),
                                         ULTIMATE_GAIN_DB + 1e-9)

    def test_a_large_constellation_approaches_the_limit(self):
        pam = np.arange(-127, 128, 2).astype(float)
        shaped = maxwell_boltzmann(pam, entropy=4.0)
        self.assertGreater(shaping_gain_dB(pam, shaped),
                           ULTIMATE_GAIN_DB - 0.01)

    def test_a_complex_alphabet_is_refused_with_the_way_out(self):
        with self.assertRaises(ValueError) as ctx:
            shaping_gain_dB(np.array([1 + 1j, -1 - 1j]), np.full(2, 0.5))
        self.assertIn("np.unique(np.real(alphabet))", str(ctx.exception))


class TestShapingBuysSomething(unittest.TestCase):
    r"""The end-to-end claim, measured rather than asserted.

    At **equal average power**, a Maxwell-Boltzmann 8-PAM carries more
    information through an AWGN channel than a uniform 8-PAM. The
    comparison has to be run at the optimal parameter, because
    :math:`\lambda` is not free: shaping trades entropy for energy, so a
    fixed :math:`\lambda` that helps at 8 dB *loses* at 18 dB, where the
    constellation is close to saturating at its 3 bits and the entropy
    given away is no longer paid back.

    So what is checked is the shape of the whole curve: shaping never
    hurts, it is worth about a tenth of a bit where the constellation is
    used at two thirds of its entropy, it vanishes at saturation, and
    the optimal parameter decreases as the SNR grows.
    """

    SNR_DB = (6.0, 10.0, 14.0, 18.0)
    LAMBDAS = np.concatenate([[0.0], np.logspace(-2.5, -0.5, 8)])

    def best_shaping(self, snr):
        """Best MI over the Maxwell-Boltzmann family, at unit power."""
        pam = np.arange(-7, 8, 2).astype(float)
        best, best_lam = -np.inf, 0.0
        for lam in self.LAMBDAS:
            weights = maxwell_boltzmann(pam, lam=lam)
            # rescale so every candidate spends the same average power
            scaled = pam / np.sqrt(float(np.sum(weights * pam ** 2)))
            value = _mutual_information(scaled, weights, snr)
            if value > best:
                best, best_lam = value, float(lam)
        return best, best_lam

    def uniform_rate(self, snr):
        pam = np.arange(-7, 8, 2).astype(float)
        return float(constellation_capacity(
            pam / np.sqrt(float(np.mean(pam ** 2))), snr))

    def test_it_never_loses_and_wins_where_it_should(self):
        gains, lambdas = [], []
        for snr_dB in self.SNR_DB:
            snr = 10 ** (snr_dB / 10)
            best, lam = self.best_shaping(snr)
            gains.append(best - self.uniform_rate(snr))
            lambdas.append(lam)
            with self.subTest(snr_dB=snr_dB):
                self.assertGreater(gains[-1], -0.01)      # never hurts

        # it is worth something in the middle of the range...
        self.assertGreater(gains[1], 0.05)
        # ...and essentially nothing once 8-PAM saturates
        self.assertLess(gains[-1], gains[1] / 2)
        # the optimal parameter decreases with the SNR
        self.assertTrue(all(a >= b for a, b in zip(lambdas, lambdas[1:],
                                                   strict=False)), lambdas)

    def test_a_fixed_parameter_eventually_loses(self):
        """Why the optimization above is not decoration."""
        pam = np.arange(-7, 8, 2).astype(float)
        weights = maxwell_boltzmann(pam, lam=0.074)       # optimal at 6 dB
        scaled = pam / np.sqrt(float(np.sum(weights * pam ** 2)))
        snr = 10 ** 2.2                                   # 22 dB
        self.assertLess(_mutual_information(scaled, weights, snr),
                        self.uniform_rate(snr))

    def test_a_matcher_actually_delivers_that_distribution(self):
        """From bits to symbols: the empirical law is the target one."""
        rng = np.random.default_rng(7)
        target = maxwell_boltzmann(AMPLITUDES, entropy=1.5)
        matcher = ConstantCompositionMatcher(AMPLITUDES, distribution=target,
                                             length=256)
        bits = random_bits(rng, 40 * matcher.n_bits)
        indices = DistributionMatcher(matcher)(bits)
        empirical = np.bincount(indices, minlength=4) / indices.size
        np.testing.assert_allclose(empirical, target, atol=5e-3)

    def test_the_sphere_shaper_delivers_a_decreasing_law(self):
        """ESS does not impose it per block; it produces it on average."""
        rng = np.random.default_rng(8)
        shaper = SphereShaper(AMPLITUDES, length=24, max_energy=24)
        blocks = np.stack([shaper.encode(random_bits(rng, shaper.n_bits))
                           for _ in range(400)])
        empirical = np.bincount(blocks.ravel(), minlength=4) / blocks.size
        self.assertTrue(np.all(np.diff(empirical) < 0), empirical)
        energy = float(np.sum(empirical * AMPLITUDES ** 2))
        self.assertLess(energy, float(np.mean(AMPLITUDES ** 2)))


def _mutual_information(alphabet, weights, snr, size=100000, seed=11):
    r"""MI of a *non-uniform* input, by Monte Carlo over its own law.

    ``compute_mi`` assumes equiprobable symbols -- that is what D48
    decided, rather than approximating the prior terms -- so the shaped
    case is written out here:

    .. math::

        I(X;Y) = \mathbb{E}\left[\log_2
        \frac{p(y|x)}{\sum_i p(y|a_i) P(a_i)}\right]

    The Gaussian normalization cancels between numerator and
    denominator, so only the squared distances are needed.
    """
    rng = np.random.default_rng(seed)
    symbols = rng.choice(len(alphabet), size=size, p=weights)
    noise = np.sqrt(1 / snr / 2) * (rng.normal(size=size)
                                    + 1j * rng.normal(size=size))
    received = alphabet[symbols] + noise
    distance = np.abs(received[:, None] - alphabet[None, :]) ** 2
    log_weighted = -snr * distance + np.log(weights)[None, :]
    largest = np.max(log_weighted, axis=-1)
    denominator = largest + np.log(np.sum(
        np.exp(log_weighted - largest[:, None]), axis=-1))
    # log p(y|x), *not* log p(y|x)P(x): the prior is already inside the
    # denominator, and putting it in both makes every term non-positive
    numerator = -snr * distance[np.arange(size), symbols]
    return float(np.mean(numerator - denominator) / np.log(2))


class TestChainBlocks(unittest.TestCase):

    def test_the_pair_is_the_identity_on_a_bit_stream(self):
        rng = np.random.default_rng(9)
        for shaper in (ConstantCompositionMatcher(AMPLITUDES,
                                                  composition=(8, 5, 2, 1)),
                       SphereShaper(AMPLITUDES, length=16, max_energy=16)):
            with self.subTest(shaper=type(shaper).__name__):
                bits = random_bits(rng, 7 * shaper.n_bits)
                indices = DistributionMatcher(shaper)(bits)
                np.testing.assert_array_equal(
                    DistributionDematcher(shaper)(indices), bits)

    def test_a_stream_that_does_not_divide_is_refused_with_the_number(self):
        shaper = ConstantCompositionMatcher(AMPLITUDES,
                                            composition=(8, 5, 2, 1))
        with self.assertRaises(ShapeError) as ctx:
            DistributionMatcher(shaper)(np.zeros(shaper.n_bits + 1, dtype=int))
        self.assertIn(f"{shaper.n_bits} bits per block", str(ctx.exception))

    def test_a_block_outside_the_code_is_reported_not_decoded(self):
        """After a detector error there is no index to read, and it says so."""
        matcher = ConstantCompositionMatcher(AMPLITUDES,
                                             composition=(4, 2, 1, 1))
        block = matcher.encode(np.zeros(matcher.n_bits, dtype=int))
        block[0] = 3                                  # breaks the composition
        with self.assertRaises(ValueError) as ctx:
            matcher.decode(block)
        self.assertIn("composition", str(ctx.exception))

        shaper = SphereShaper(AMPLITUDES, length=8, max_energy=4)
        outside = np.full(8, 3)                       # the costliest amplitude
        with self.assertRaises(ValueError) as ctx:
            shaper.decode(outside)
        self.assertIn("outside the sphere", str(ctx.exception))


class TestTheSignHalf(unittest.TestCase):
    r"""What :class:`AmplitudeMapper` adds, and what it must not disturb.

    In PAS the sign is spent on parity bits, so the block that puts it
    there has one job -- add an equiprobable :math:`\pm 1` -- and one
    promise: the amplitude law it was given comes out untouched, which is
    what makes the composite constellation symmetric Maxwell-Boltzmann at
    the same energy.
    """

    def test_the_pair_is_the_identity_on_amplitude_indices(self):
        rng = np.random.default_rng(3)
        mapper = AmplitudeMapper(AMPLITUDES, seed=0)
        demapper = AmplitudeDemapper(AMPLITUDES)
        indices = rng.integers(0, AMPLITUDES.size, size=5000)
        np.testing.assert_array_equal(demapper(mapper(indices)), indices)

    def test_deciding_on_the_magnitude_is_the_maximum_likelihood_decision(self):
        """The shortcut is not one: on a symmetric constellation the
        nearest point to ``y`` has the amplitude nearest to ``|y|``."""
        rng = np.random.default_rng(4)
        pam = np.concatenate([-AMPLITUDES[::-1], AMPLITUDES])
        received = rng.normal(scale=3.0, size=20000)
        full = pam[np.argmin(np.abs(received[:, None] - pam), axis=-1)]
        by_magnitude = AMPLITUDES[AmplitudeDemapper(AMPLITUDES)(received)]
        np.testing.assert_allclose(np.abs(full), by_magnitude)

    def test_the_sign_leaves_the_amplitude_law_alone(self):
        """P(+a) = P(-a) = P(a)/2, which is why the sign is free."""
        mapper = AmplitudeMapper(AMPLITUDES, seed=1)
        law = maxwell_boltzmann(AMPLITUDES, entropy=1.25)
        rng = np.random.default_rng(5)
        indices = rng.choice(AMPLITUDES.size, size=200000, p=law)
        sent = mapper(indices)
        for index, amplitude in enumerate(AMPLITUDES):
            for sign in (-1.0, 1.0):
                with self.subTest(point=sign * amplitude):
                    self.assertAlmostEqual(
                        float(np.mean(sent == sign * amplitude)),
                        law[index] / 2, delta=0.005)

    def test_a_signed_alphabet_is_refused_by_both_blocks(self):
        for block in (AmplitudeMapper, AmplitudeDemapper):
            with self.subTest(block=block.__name__):
                with self.assertRaises(ValueError) as ctx:
                    block(np.array([-1.0, 1.0, 3.0]))
                self.assertIn("-1.0", str(ctx.exception))

    def test_the_shaped_link_recovers_its_bits_and_is_reproducible(self):
        """The whole architecture, end to end: bits in, bits out."""
        shaper = ConstantCompositionMatcher(
            AMPLITUDES, distribution=maxwell_boltzmann(AMPLITUDES,
                                                       entropy=1.25),
            length=32)
        link = Sequential([
            SymbolGenerator(2, name="bits"),
            DistributionMatcher(shaper),
            AmplitudeMapper(AMPLITUDES, name="mapper"),
            AWGN(snr_dB=30.0, name="noise"),
            AmplitudeDemapper(AMPLITUDES),
            DistributionDematcher(shaper),
        ], taps=["bits", "mapper"])
        link.seed(2)
        recovered = link(20 * shaper.n_bits)
        np.testing.assert_array_equal(recovered, link.tap("bits"))
        signed = link.tap("mapper")
        # the sign block is stochastic, so D6 must reach it: same seed,
        # same signs, not merely the same magnitudes
        link.seed(2)
        np.testing.assert_allclose(link(20 * shaper.n_bits), recovered)
        np.testing.assert_allclose(link.tap("mapper"), signed)


class TestShapedSource(unittest.TestCase):
    """``SymbolGenerator(distribution=...)``: the law as a source."""

    def test_the_empirical_law_is_the_one_asked_for(self):
        law = maxwell_boltzmann(np.arange(-7, 8, 2).astype(float),
                                entropy=2.25)
        generator = SymbolGenerator(8, distribution=law, seed=0)
        measured = np.bincount(generator(400000), minlength=8) / 400000
        np.testing.assert_allclose(measured, law, atol=0.004)

    def test_the_default_is_still_uniform(self):
        measured = np.bincount(SymbolGenerator(4, seed=0)(100000),
                               minlength=4) / 100000
        np.testing.assert_allclose(measured, 0.25, atol=0.01)

    def test_a_law_of_the_wrong_length_names_both_numbers(self):
        with self.assertRaises(ValueError) as ctx:
            SymbolGenerator(4, distribution=np.full(8, 1 / 8))
        self.assertIn("8 probabilities", str(ctx.exception))
        self.assertIn("4 symbols", str(ctx.exception))

    def test_a_law_that_does_not_sum_to_one_is_refused(self):
        with self.assertRaises(ValueError):
            SymbolGenerator(4, distribution=np.full(4, 0.2))
        with self.assertRaises(ValueError):
            SymbolGenerator(2, distribution=np.array([-0.5, 1.5]))


if __name__ == "__main__":
    unittest.main()
