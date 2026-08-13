"""Run the examples `tests/test_examples_run.py` marks as SLOW.

The smoke test skips them so that a pull request does not wait minutes
per script. Skipping is the right call there and the wrong one
everywhere: for a while nothing ran them at all, and a regression lived
in `CD_compensation_part1.py` for two commits -- it raised `TypeError:
'Constellation' object is not subscriptable` on its first pass while
every check stayed green.

The list is not repeated here. It is imported from the test module, so
adding an entry to `SLOW` is enough to have it covered, and an entry
that is removed stops being run without anyone editing a workflow.

Each script runs in its own process, from its own directory, with the
same sandbox the smoke test uses: the examples hardcode
`../../docs/tutorials/img/`, and a scheduled job has no business
rewriting figures that are committed. Every script runs even if an
earlier one failed -- one broken example must not hide the state of the
other eight.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.test_examples_run import SLOW  # noqa: E402  -- after sys.path

TIMEOUT_S = 1800


def sandbox() -> Path:
    """A throwaway copy of `examples/`, with the directories it writes to."""
    root = Path(tempfile.mkdtemp(prefix="comnumpy_slow_"))
    shutil.copytree(REPO / "examples", root / "examples",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for pattern in ("docs/**/img", "docs/**/mermaid"):
        for directory in REPO.glob(pattern):
            (root / directory.relative_to(REPO)).mkdir(parents=True,
                                                       exist_ok=True)
    (root / "validation" / "figures").mkdir(parents=True, exist_ok=True)
    return root


def main() -> int:
    only = os.environ.get("ONLY", "").strip()
    names = sorted(name for name in SLOW if only in name)
    if not names:
        print(f"no SLOW example matches {only!r}; known: {sorted(SLOW)}")
        return 1

    root = sandbox()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p)

    failures = []
    for name in names:
        script = root / "examples" / name
        print(f"::group::{name}", flush=True)
        start = time.perf_counter()
        try:
            done = subprocess.run([sys.executable, script.name],
                                  cwd=script.parent, env=env,
                                  capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
            code, output = done.returncode, done.stdout + done.stderr
        except subprocess.TimeoutExpired:
            code, output = 1, f"did not finish within {TIMEOUT_S} s"
        elapsed = time.perf_counter() - start
        print(output[-4000:])
        print("::endgroup::", flush=True)
        status = "ok" if code == 0 else "FAILED"
        print(f"{status:6s} {name:42s} {elapsed:7.1f} s", flush=True)
        if code != 0:
            failures.append((name, output[-2000:]))

    if failures:
        print(f"\n{len(failures)} of {len(names)} slow examples failed:")
        for name, output in failures:
            print(f"\n--- {name} ---\n{output}")
        return 1
    print(f"\nall {len(names)} slow examples ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
