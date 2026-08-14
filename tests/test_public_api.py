"""Public API surface guarantees (decision D36).

Two CI-enforced budgets: ``import comnumpy`` stays under 200 ms (best of
three runs, so OS caches are warm) and never pulls matplotlib. Two more
guarantees about the surface itself: every public module declares
``__all__``, and every name it declares is one the module defines.
"""
import importlib
import pathlib
import subprocess
import sys
import unittest

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "src" / "comnumpy"


def public_modules():
    """Every module a user may import, i.e. not ``__init__`` nor ``_*``."""
    return sorted(path for path in SOURCE.rglob("*.py")
                  if not path.name.startswith("_"))


def declares_all(path):
    return any(line.startswith("__all__")
               for line in path.read_text().splitlines())


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

    def test_every_public_module_declares_all(self):
        """D36 is about modules, not only about the four packages.

        A module with no ``__all__`` exports whatever it happens to have
        imported: ``from comnumpy.core.channels import np`` used to work.
        """
        missing = [str(path.relative_to(SOURCE))
                   for path in public_modules() if not declares_all(path)]
        self.assertEqual(missing, [],
                         f"modules without __all__: {missing}")

    def test_every_exported_name_exists(self):
        """The other half: a name in ``__all__`` that is not defined.

        It is silent until someone writes ``from module import *`` or
        reads the documentation, and Sphinx will not catch it either.
        """
        for path in public_modules():
            module_name = ("comnumpy."
                           + str(path.relative_to(SOURCE))[:-3].replace("/", "."))
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                for name in module.__all__:
                    self.assertTrue(
                        hasattr(module, name),
                        f"{module_name}.__all__ names {name}, which is "
                        f"not defined there")

    def test_no_module_exports_an_imported_third_party_name(self):
        """``__all__`` lists what a module *is*, not what it uses."""
        borrowed = {"np", "numpy", "scipy", "plt", "dataclass", "field",
                    "Optional", "Literal", "Processor"}
        for path in public_modules():
            module_name = ("comnumpy."
                           + str(path.relative_to(SOURCE))[:-3].replace("/", "."))
            module = importlib.import_module(module_name)
            leaked = borrowed & set(module.__all__)
            with self.subTest(module=module_name):
                # Processor is legitimately exported by the module that
                # defines it, and by no other
                if module_name == "comnumpy.core.generics":
                    leaked = leaked - {"Processor"}
                self.assertEqual(leaked, set(),
                                 f"{module_name} exports borrowed names {leaked}")

    def test_lazy_plotting_resolves(self):
        from comnumpy.core import plot_iq
        self.assertTrue(callable(plot_iq))

    def test_exceptions_exported(self):
        import comnumpy
        self.assertTrue(issubclass(comnumpy.ShapeError, ValueError))
        self.assertTrue(issubclass(comnumpy.NotFittedError, RuntimeError))
        self.assertTrue(issubclass(comnumpy.ShapeError, comnumpy.ComnumpyError))

    def test_no_in_chain_instrumentation_block(self):
        """Observation is a chain declaration; instrumentation blocks are gone.

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
        """The canonical flow: name a block, observe it, read it back."""
        from comnumpy import (AWGN, Sequential, SymbolDemapper, SymbolGenerator,
                              SymbolMapper, compute_ser, get_alphabet)
        alphabet = get_alphabet("QAM", 16)
        chain = Sequential([
            SymbolGenerator(16, seed=42, name="tx"),
            SymbolMapper(alphabet),
            AWGN(snr_dB=15, seed=123),
            SymbolDemapper(alphabet),
        ], observations=["tx"])
        detected = chain(10_000)
        self.assertLess(compute_ser(chain.observation("tx"), detected), 0.05)


if __name__ == "__main__":
    unittest.main()
