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

    def test_lazy_scope_resolves(self):
        import comnumpy
        self.assertTrue(callable(comnumpy.Scope))

    def test_exceptions_exported(self):
        import comnumpy
        self.assertTrue(issubclass(comnumpy.ShapeError, ValueError))
        self.assertTrue(issubclass(comnumpy.NotFittedError, RuntimeError))
        self.assertTrue(issubclass(comnumpy.ShapeError, comnumpy.ComnumpyError))


if __name__ == "__main__":
    unittest.main()
