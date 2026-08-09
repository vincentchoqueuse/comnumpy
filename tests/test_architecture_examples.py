"""The ARCHITECTURE.md examples must run (principle P3, decision D8).

The decision record forbids an unexecuted numeric example anywhere -- and
its own canonical snippet silently violated that for two versions: it
read the recorded signal and ran the chain in one call, so argument
evaluation order made the read happen first, on an empty record. This
test extracts the canonical example from the document and runs it, so the
rule now applies to the document that states it.
"""
import pathlib
import re
import unittest

ADD = pathlib.Path(__file__).resolve().parents[1] / "ARCHITECTURE.md"
ANCHOR = "Cible d'usage (D36 + D40c réunis)"


def canonical_example() -> str:
    text = ADD.read_text(encoding="utf-8")
    after = text[text.index(ANCHOR):]
    match = re.search(r"```python\n(.*?)```", after, re.S)
    assert match is not None, "no python block after the D40c anchor"
    return match.group(1)


class TestArchitectureExamples(unittest.TestCase):

    def test_canonical_example_runs(self):
        namespace: dict = {}
        exec(compile(canonical_example(), str(ADD), "exec"), namespace)
        # the example measures a SER; at 15 dB on 16-QAM it is a few percent
        self.assertLess(namespace["ser"], 0.05)

    def test_canonical_example_respects_the_line_budget(self):
        """D40c: eight lines, imports included."""
        lines = [ln for ln in canonical_example().splitlines() if ln.strip()]
        self.assertLessEqual(len(lines), 8, "\n".join(lines))

    def test_decision_numbers_are_unique_and_contiguous(self):
        text = ADD.read_text(encoding="utf-8")
        numbers = sorted({int(n) for n in re.findall(r"^\| D(\d+) \|", text,
                                                     re.M)})
        self.assertEqual(numbers, list(range(1, max(numbers) + 1)),
                         "decision numbers must be unique and contiguous")


if __name__ == "__main__":
    unittest.main()
