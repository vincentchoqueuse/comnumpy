"""The documentation must point at files that exist.

A `literalinclude` whose target is missing does not fail the Sphinx
build -- it emits a warning and renders an empty block, so the page
silently loses the code it was written around. Five of them pointed at
`one_shot_nli.py` while the file is `one_shot_NLI.py`: invisible on
macOS, broken on any case-sensitive filesystem, which is what CI and
readthedocs run on.
"""
import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
DIRECTIVE = re.compile(r"\.\.\s+(literalinclude|include)::\s+(\S+)")


class TestDocsReferences(unittest.TestCase):

    def test_included_files_exist(self):
        """Case included: the check must fail on a case-only mismatch."""
        missing = []
        for page in sorted(DOCS.rglob("*.rst")):
            for number, line in enumerate(page.read_text().splitlines(), 1):
                match = DIRECTIVE.match(line.strip())
                if not match:
                    continue
                target = (page.parent / match.group(2)).resolve()
                # `exists()` alone is case-insensitive on macOS, so
                # confirm the name as spelled is in the directory
                if not (target.exists()
                        and target.name in {entry.name for entry
                                            in target.parent.iterdir()}):
                    missing.append(f"{page.relative_to(DOCS)}:{number} "
                                   f"-> {match.group(2)}")
        self.assertEqual(missing, [],
                         "\n".join(["documentation includes a missing file:"]
                                   + missing))


if __name__ == "__main__":
    unittest.main()
