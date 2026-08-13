"""The experiment engine: reproducible by construction, honest about it.

`Experiment` exists to replace the hand-written collection loop, so the
properties pinned here are the ones those loops kept getting wrong: the
seed is always known afterwards, the same seed gives the same numbers,
the arrays stay aligned with the values, and a simulate that changes
what it observes mid-run is refused rather than silently misaligned.
"""
import unittest

import numpy as np

from comnumpy import Experiment
from comnumpy.exceptions import ComnumpyError, ShapeError


def simulate(config, seed):
    """A cheap experiment: noisy estimate of the studied parameter."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=config["n"])
    return {"estimate": config["snr_dB"] + float(np.mean(noise)),
            "power": float(np.mean(noise ** 2))}


class TestExperiment(unittest.TestCase):

    def experiment(self, **overrides):
        settings = {"parameter": "snr_dB", "values": [0, 10, 20], "seed": 7}
        settings.update(overrides)
        return Experiment({"n": 100}, **settings)

    def test_arrays_are_aligned_with_the_values(self):
        result = self.experiment().run(simulate)
        self.assertEqual(list(result.data), ["estimate", "power"])
        for values in result.data.values():
            self.assertEqual(values.shape, (3,))
        np.testing.assert_array_equal(result.values, [0, 10, 20])

    def test_the_parameter_reaches_the_configuration(self):
        seen = []

        def probe(config, seed):
            seen.append(config["snr_dB"])
            return {"ok": 1.0}

        self.experiment().run(probe)
        self.assertEqual(seen, [0, 10, 20])

    def test_the_experiment_config_is_never_mutated(self):
        conditions = {"n": 100}
        Experiment(conditions, parameter="snr_dB", values=[0, 10],
                   seed=1).run(simulate)
        self.assertEqual(conditions, {"n": 100})

    def test_same_seed_same_numbers(self):
        first = self.experiment().run(simulate)
        second = self.experiment().run(simulate)
        np.testing.assert_array_equal(first.data["estimate"],
                                      second.data["estimate"])

    def test_different_seeds_differ(self):
        first = self.experiment(seed=1).run(simulate)
        second = self.experiment(seed=2).run(simulate)
        self.assertFalse(np.array_equal(first.data["estimate"],
                                        second.data["estimate"]))

    def test_points_draw_independent_noise(self):
        """Two points must not share a realization, whatever their value."""
        result = self.experiment(values=[5, 5, 5]).run(simulate)
        self.assertEqual(len(set(result.data["estimate"].tolist())), 3)

    def test_a_missing_seed_is_drawn_and_kept(self):
        experiment = self.experiment(seed=None)
        self.assertIsNotNone(experiment.seed)
        result = experiment.run(simulate)
        self.assertEqual(result.seed, experiment.seed)
        replay = self.experiment(seed=result.seed).run(simulate)
        np.testing.assert_array_equal(replay.data["estimate"],
                                      result.data["estimate"])

    def test_save_narrows_what_is_kept(self):
        for save in ("power", ["power"], {"power": True, "estimate": False}):
            with self.subTest(save=save):
                result = self.experiment(save=save).run(simulate)
                self.assertEqual(list(result.data), ["power"])

    def test_save_of_an_unproduced_name_says_what_exists(self):
        with self.assertRaises(ComnumpyError) as raised:
            self.experiment(save=["ber"]).run(simulate)
        message = str(raised.exception)
        self.assertIn("ber", message)
        self.assertIn("estimate", message)

    def test_result_carries_the_conditions(self):
        result = self.experiment().run(simulate)
        self.assertEqual(result.parameter, "snr_dB")
        self.assertEqual(result.config, {"n": 100})
        self.assertGreaterEqual(result.elapsed_, 0.0)

    def test_bad_values_are_refused(self):
        for values in ([], [[0, 1], [2, 3]]):
            with self.subTest(values=values):
                with self.assertRaises(ShapeError):
                    self.experiment(values=values)

    def test_a_non_mapping_config_is_refused(self):
        with self.assertRaises(ComnumpyError):
            Experiment([("n", 100)], parameter="snr_dB", values=[0])

    def test_a_simulate_that_returns_no_mapping_is_refused(self):
        with self.assertRaises(ComnumpyError) as raised:
            self.experiment().run(lambda config, seed: 0.5)
        self.assertIn("float", str(raised.exception))

    def test_inconsistent_observations_are_refused(self):
        """A quantity that appears mid-run cannot be aligned; refuse it."""

        def drifting(config, seed):
            if config["snr_dB"] > 0:
                return {"ser": 0.1, "extra": 1.0}
            return {"ser": 0.5}

        with self.assertRaises(ComnumpyError) as raised:
            self.experiment().run(drifting)
        self.assertIn("extra", str(raised.exception))


class TestExperimentResult(unittest.TestCase):

    def setUp(self):
        self.result = Experiment(
            {"n": 100}, parameter="snr_dB", values=[0, 10, 20],
            seed=7).run(simulate)

    def test_as_data_feeds_the_data_module(self):
        from comnumpy.data import format_data
        table = format_data(self.result.as_data(), xlabel="snr_dB")
        self.assertIn("estimate", table)
        self.assertEqual(len(table.split("\n")), 2 + 3)

    def test_print_shows_the_conditions_and_the_seed(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.result.print()
        text = buffer.getvalue()
        self.assertIn("seed      : 7", text)
        self.assertIn("n=100", text)
        self.assertIn("snr_dB", text)

    def test_plot_draws_one_curve_per_quantity(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = self.result.plot()
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(ax.get_xlabel(), "snr_dB")
        plt.close(ax.figure)


class TestGroupedObservations(unittest.TestCase):
    """A study over several methods observes one value per method.

    Collected as a group, that structure survives to the result -- no
    name mangling on the way in, no re-packing on the way out.
    """

    @staticmethod
    def compare(config, seed):
        return {"ser": {"ZF": 2.0 / config["snr_dB"],
                        "ML": 1.0 / config["snr_dB"]},
                "elapsed": 0.5}

    def setUp(self):
        self.result = Experiment(
            {}, parameter="snr_dB", values=[10, 20], seed=3).run(self.compare)

    def test_a_group_collects_into_a_dict_of_arrays(self):
        np.testing.assert_allclose(self.result.data["ser"]["ML"],
                                   [0.1, 0.05])
        np.testing.assert_allclose(self.result.data["elapsed"], [0.5, 0.5])

    def test_as_data_of_a_group_is_its_curves(self):
        curves = self.result.as_data("ser")["curves"]
        self.assertEqual(list(curves), ["ZF", "ML"])

    def test_as_data_without_a_key_is_the_ungrouped_part(self):
        self.assertEqual(list(self.result.as_data()["curves"]), ["elapsed"])

    def test_an_unknown_key_names_the_groups_that_exist(self):
        with self.assertRaises(ComnumpyError) as raised:
            self.result.as_data("bler")
        self.assertIn("ser", str(raised.exception))

    def test_print_makes_one_table_per_group(self):
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.result.print()
        text = buffer.getvalue()
        self.assertIn("ser", text)
        self.assertIn("ZF", text)
        self.assertIn("elapsed", text)

    def test_plot_takes_the_group_name(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = self.result.plot("ser", yscale="log")
        self.assertEqual(len(ax.lines), 2)
        self.assertEqual(ax.get_ylabel(), "ser")
        plt.close(ax.figure)

    def test_a_group_that_changes_members_is_refused(self):
        def drifting(config, seed):
            members = {"ZF": 0.1}
            if config["snr_dB"] > 10:
                members["ML"] = 0.05
            return {"ser": members}

        with self.assertRaises(ComnumpyError) as raised:
            Experiment({}, parameter="snr_dB", values=[10, 20],
                       seed=1).run(drifting)
        self.assertIn("ML", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
