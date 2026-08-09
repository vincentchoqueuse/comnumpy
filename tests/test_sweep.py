"""sweep(): reconfigure, reseed, run, collect (decision D35)."""
import unittest

import numpy as np

from comnumpy import AWGN, Sequential, SymbolDemapper, \
    SymbolGenerator, SymbolMapper, compute_ser, get_alphabet, sweep


class TestSweep(unittest.TestCase):

    def build(self):
        alphabet = get_alphabet("QAM", 4)
        return Sequential([
            SymbolGenerator(4, name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=0, name="noise"),
            SymbolDemapper(alphabet),
        ])

    def test_results_aligned_and_monotonic(self):
        values = np.arange(0, 14, 4)
        out = sweep(self.build(), "noise.snr_dB", values,
                    {"ser": compute_ser}, 20_000, reference="tx", seed=1)
        self.assertEqual(out["ser"].shape, values.shape)
        self.assertTrue(np.all(np.diff(out["ser"]) < 0))  # SER falls with SNR

    def test_reproducible_across_calls(self):
        values = [0, 6]
        a = sweep(self.build(), "noise.snr_dB", values,
                  {"ser": compute_ser}, 5_000, reference="tx", seed=3)
        b = sweep(self.build(), "noise.snr_dB", values,
                  {"ser": compute_ser}, 5_000, reference="tx", seed=3)
        np.testing.assert_array_equal(a["ser"], b["ser"])

    def test_multi_param_zip(self):
        chain = self.build()
        out = sweep(chain, ["noise.snr_dB", "noise.name"],
                    [(0, "noise"), (10, "noise")],
                    {"snr_set": lambda t, y: chain["noise"].snr_dB},
                    1_000, reference="tx", seed=2)
        np.testing.assert_array_equal(out["snr_set"], [0, 10])

    def test_metric_without_reference(self):
        out = sweep(self.build(), "noise.snr_dB", [0, 20],
                    {"power": lambda y: float(np.mean(np.abs(y) ** 2))},
                    2_000, seed=4)
        self.assertEqual(len(out["power"]), 2)


if __name__ == "__main__":
    unittest.main()
