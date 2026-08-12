"""Smoke test: the teaching examples must still run.

`examples/` is the layer that teaches, and nothing executed it -- so it
rotted silently. A renamed block, a dropped keyword argument or an
attribute that moved only surfaced when a reader ran the script and got
a traceback instead of a figure.

This test runs each fast example in a subprocess and checks its exit
code. It deliberately asserts nothing about the numbers: an example
teaches, it plots and prints, it does not assert. A claim of
correctness belongs in `validation/` (decision D7), a fast check
belongs in the rest of `tests/`. The only claim made here is "this
still runs".

Isolation -- why a sandbox and not a cleanup:
the examples hardcode their output directory (`img_dir =
"../../docs/tutorials/img/"`); there is no environment variable to
redirect it, so the honest options were "let them overwrite the figures
committed under `docs/` and restore afterwards" or "run them somewhere
else". Restoring afterwards is racy (a failed run leaves the tree
dirty, and a concurrent `make doc` would lose its regenerated figure),
so each run happens in a throwaway copy of `examples/` with an empty
`docs/**/img` skeleton beside it. `../../docs/...` then resolves inside
the sandbox and the working tree is never written to at all --
`test_fast_examples_run` checks that afterwards.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

# Generous: the slowest example kept here measured 20 s on an idle
# machine. The timeout only has to catch a script that hangs, on a CI
# runner that may be several times slower.
TIMEOUT_S = 120

# Examples too slow for a smoke test. Times are wall clock measured on
# 2026-08-09 (Python 3.11, MPLBACKEND=Agg, idle 4-core machine); where
# wall and CPU diverge the script is threaded through BLAS/FFT, so it
# degrades badly on a busy runner -- that is why the cut is well under
# the timeout. They are skipped, not omitted: a skipped test is visible
# in the report, a missing one is not. Re-measure before moving an entry
# in or out.
SLOW = {
    "mimo/monte_carlo_simulation_1.py":
        "Monte-Carlo BER sweep, 4 detectors -- 32 s measured 2026-08-09",
    "mimo/monte_carlo_simulation_2.py":
        "Monte-Carlo BER sweep, OSIC variants -- 143 s measured 2026-08-09",
    "mimo/run_all_scripts.py":
        "runner, not an example: re-executes the other mimo scripts -- "
        "~350 s by construction",
    "ofdm/one_shot_ofdm.py":
        "twenty zero-forcing inversions of a 1284 x 1280 convolution "
        "matrix, plus a runtime sweep -- 33 s wall measured 2026-08-12",
    "optical/CD_compensation_part1.py":
        "BER vs SNR for 3 modulations, 200 000 symbols through a 249-tap "
        "compensator -- 227 s measured 2026-08-09",
    "optical/NLI_simulation.py":
        "split-step fibre and six receivers over eight launch powers -- "
        "120 s measured 2026-08-12, which is the timeout itself",
    "optical/one_shot_NLI.py":
        "the whole chain re-run at six span counts, then once more for "
        "the profile and once for the receiver comparison -- 76 spans of "
        "split-step propagation, 73 s measured 2026-08-12",
    "simple/one_shot_srrc_awgn.py":
        "16 001-tap SRRC filter at oversampling 8 -- 113 s wall / 316 s "
        "CPU measured 2026-08-09",
    "simple/probabilistic_shaping.py":
        "a scalar optimization of lambda at fourteen SNRs, each iteration "
        "a 40-node Gauss-Hermite quadrature over a 64-QAM -- 230 s wall / "
        "299 s CPU measured 2026-08-12",
}

# Examples that are broken right now. Kept visible as expected failures
# rather than deleted or hidden in SLOW: the smoke test must not go green
# by looking away. Fix the example, then delete its entry -- the test
# then reports an unexpected success until the entry is gone.
BROKEN: dict[str, str] = {}


def discover_examples() -> list[str]:
    """Every example script, as a path relative to `examples/`."""
    return sorted(
        path.relative_to(EXAMPLES).as_posix()
        for path in EXAMPLES.glob("*/*.py")
    )


def figure_snapshot() -> dict[str, tuple[float, int]]:
    """Mtime and size of the committed figures the examples write.

    `docs/**/img` only: that is where every `savefig` in `examples/`
    points. `validation/figures` is deliberately left out -- no example
    writes there, and watching a directory that the validation scripts
    legitimately rewrite would only make this check flaky.
    """
    return {
        str(path.relative_to(REPO)): (path.stat().st_mtime, path.stat().st_size)
        for path in REPO.glob("docs/**/img/*") if path.is_file()
    }


class TestExamplesRun(unittest.TestCase):

    sandbox: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="comnumpy_examples_")
        cls.sandbox = Path(cls._tmp.name)
        shutil.copytree(EXAMPLES, cls.sandbox / "examples",
                        ignore=shutil.ignore_patterns("__pycache__"))
        # Mirror the directories the scripts write into, empty: savefig
        # does not create them.
        for img_dir in REPO.glob("docs/**/img"):
            (cls.sandbox / img_dir.relative_to(REPO)).mkdir(parents=True,
                                                            exist_ok=True)
        for mermaid_dir in REPO.glob("docs/**/mermaid"):
            (cls.sandbox / mermaid_dir.relative_to(REPO)).mkdir(parents=True,
                                                                exist_ok=True)
        (cls.sandbox / "validation" / "figures").mkdir(parents=True,
                                                       exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_example(self, relative_path: str) -> subprocess.CompletedProcess:
        """Run one example from its own directory, headless."""
        script = self.sandbox / "examples" / relative_path
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"          # no display, no blocking show()
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)
        try:
            return subprocess.run(
                [sys.executable, script.name],
                cwd=script.parent,          # the scripts use relative paths
                env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"examples/{relative_path} did not finish within "
                      f"{TIMEOUT_S} s -- it hangs, or it belongs in SLOW")

    def test_fast_examples_run(self):
        fast = [name for name in discover_examples()
                if name not in SLOW and name not in BROKEN]
        self.assertTrue(fast, "no example found under examples/*/")

        before = figure_snapshot()
        for name in fast:
            with self.subTest(example=name):
                process = self.run_example(name)
                self.assertEqual(
                    process.returncode, 0,
                    f"examples/{name} exited with {process.returncode}\n"
                    f"--- stderr ---\n{process.stderr.strip()}\n"
                    f"--- last stdout ---\n{process.stdout.strip()[-800:]}")

        self.assertEqual(figure_snapshot(), before,
                         "the examples wrote into the working tree; the "
                         "sandbox in setUpClass is not isolating them")
        self.check_diagrams_did_not_drift()

    def check_diagrams_did_not_drift(self):
        """A chain diagram in `docs/` must be what the chain draws today.

        The tutorials show `chain.to_mermaid()` rather than a hand-drawn
        picture, so the two can only agree if something compares them.
        Only the diagrams the *fast* examples write are covered: an
        example in SLOW is not re-run here, so its diagram is checked
        when someone runs it.
        The sandbox run has just written every diagram from the chains
        as they are now; the committed copies must match, character for
        character.
        """
        stale = []
        for produced in sorted(self.sandbox.glob("docs/**/mermaid/*.mmd")):
            relative = produced.relative_to(self.sandbox)
            committed = REPO / relative
            if not committed.exists():
                stale.append(f"{relative}: never committed")
            elif committed.read_text() != produced.read_text():
                stale.append(f"{relative}: the chain draws something else")
        self.assertEqual(stale, [], "\n".join(
            ["chain diagrams have drifted from the chains that draw them; "
             "re-run the example that writes them:"] + stale))

    def test_slow_and_broken_lists_are_current(self):
        """A renamed example must not silently drop out of the smoke test."""
        known = set(discover_examples())
        stale = sorted((set(SLOW) | set(BROKEN)) - known)
        self.assertEqual(stale, [], f"SLOW/BROKEN name scripts that no "
                                    f"longer exist: {stale}")


def _slow_placeholder(reason: str):
    """A visible skip, so the report names the example it did not run."""
    @unittest.skip(reason)
    def test_slow_example(self) -> None:
        raise AssertionError("unreachable")   # pragma: no cover
    return test_slow_example


def _broken_placeholder(name: str, reason: str):
    """A visible expected failure, named, with the diagnosis attached."""
    @unittest.expectedFailure
    def test_broken_example(self: TestExamplesRun) -> None:
        process = self.run_example(name)
        self.assertEqual(process.returncode, 0,
                         f"examples/{name} is known broken: {reason}\n"
                         f"--- stderr ---\n{process.stderr.strip()}")
    test_broken_example.__doc__ = f"examples/{name} -- known broken: {reason}"
    return test_broken_example


for _name, _reason in SLOW.items():
    setattr(TestExamplesRun,
            "test_slow_" + _name.replace("/", "_").removesuffix(".py"),
            _slow_placeholder(_reason))

for _name, _reason in BROKEN.items():
    setattr(TestExamplesRun,
            "test_broken_" + _name.replace("/", "_").removesuffix(".py"),
            _broken_placeholder(_name, _reason))


if __name__ == "__main__":
    unittest.main()
