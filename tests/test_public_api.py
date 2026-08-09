"""Public API surface guarantees (decision D36).

Two CI-enforced budgets: ``import comnumpy`` stays under 200 ms (best of
three runs, so OS caches are warm) and never pulls matplotlib.
"""
import subprocess
import sys
import unittest

IMPORT_CHECK = (
    "import sys, time;"
    "t0 = time.perf_counter();"
    "import comnumpy;"
    "t1 = time.perf_counter();"
    "assert 'matplotlib' not in sys.modules, 'matplotlib imported eagerly';"
    "print((t1 - t0) * 1000)"
)


class TestPublicAPI(unittest.TestCase):

    def test_bare_import_is_light(self):
        """import comnumpy: no matplotlib, < 200 ms (best of 3)."""
        times = []
        for _ in range(3):
            out = subprocess.run(
                [sys.executable, "-c", IMPORT_CHECK],
                capture_output=True, text=True, check=True,
            )
            times.append(float(out.stdout.strip()))
        self.assertLess(min(times), 200.0,
                        f"import comnumpy took {min(times):.0f} ms (budget: 200 ms)")

    def test_all_declared(self):
        import comnumpy
        import comnumpy.core
        import comnumpy.ofdm
        import comnumpy.mimo
        import comnumpy.optical
        for module in (comnumpy, comnumpy.core, comnumpy.ofdm,
                       comnumpy.mimo, comnumpy.optical):
            self.assertTrue(hasattr(module, "__all__"),
                            f"{module.__name__} does not declare __all__")

    def test_lazy_plotting_resolves(self):
        from comnumpy.core import plot_iq
        self.assertTrue(callable(plot_iq))

    def test_exceptions_exported(self):
        import comnumpy
        self.assertTrue(issubclass(comnumpy.ShapeError, ValueError))
        self.assertTrue(issubclass(comnumpy.NotFittedError, RuntimeError))
        self.assertTrue(issubclass(comnumpy.ShapeError, comnumpy.ComnumpyError))

    def test_no_in_chain_instrumentation_block(self):
        """Observation is done with taps; instrumentation blocks are gone.

        A chain must describe the communication system only, so no
        recorder/logger/scope/monitor block may come back into the public
        surface (see CONVENTIONS.md).
        """
        import comnumpy
        import comnumpy.core
        import comnumpy.mimo
        import comnumpy.ofdm
        import comnumpy.optical
        banned = ("Recorder", "MetricRecorder", "Logger", "Debugger",
                  "PowerReporter", "TimeSignalMonitor", "Scope", "TimeScope",
                  "SpectrumScope", "IQScope", "KDEScope", "WelchScope",
                  "FFTMonitor")
        for module in (comnumpy, comnumpy.core, comnumpy.ofdm,
                       comnumpy.mimo, comnumpy.optical):
            for name in banned:
                self.assertNotIn(name, module.__all__,
                                 f"{module.__name__} re-exports {name}")
                self.assertFalse(hasattr(module, name),
                                 f"{module.__name__} defines {name}")

    def test_chain_output_is_reachable_without_instrumentation(self):
        """The canonical flow: name a block, tap it, read it back."""
        from comnumpy import (AWGN, Sequential, SymbolDemapper, SymbolGenerator,
                              SymbolMapper, compute_ser, get_alphabet)
        alphabet = get_alphabet("QAM", 16)
        chain = Sequential([
            SymbolGenerator(16, seed=42, name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=15, seed=123),
            SymbolDemapper(alphabet),
        ], taps=["tx"])
        detected = chain(10_000)
        self.assertLess(compute_ser(chain.tap("tx"), detected), 0.05)


if __name__ == "__main__":
    unittest.main()
