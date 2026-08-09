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
RANGED = re.compile(
    r"literalinclude::\s+(\S+)\n(?:\s+:\w+:.*\n)*?\s+:lines:\s+(\d+)-(\d+)")

# Figure furniture, not the lesson: the output directory every example
# writes into, and the matplotlib presentation calls. A tutorial that
# skipped an actual computation would still be reported.
NOT_QUOTED = re.compile(
    r"^\s*(img_dir\s*=|plt\.(show|savefig|figure|grid|legend|title"
    r"|x(lim|label|scale)|y(lim|label|scale)|tight_layout)\()")


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


    def test_line_ranges_fit_inside_the_file(self):
        """A range past the end of the file quotes nothing, silently.

        The coverage check below cannot see this: asking for lines 80-92
        of an 89-line example *over*-covers, so every line is accounted
        for and the test passes while the page shows a truncated block.
        Sphinx reports it; this test exists so the sandbox does too.
        """
        problems = []
        for page in sorted(DOCS.rglob("*.rst")):
            for target, first, last in RANGED.findall(page.read_text()):
                source = (page.parent / target).resolve()
                if not source.exists():
                    continue        # reported by test_included_files_exist
                length = len(source.read_text().splitlines())
                if int(last) > length or int(first) < 1:
                    problems.append(
                        f"{page.relative_to(DOCS)} asks for lines "
                        f"{first}-{last} of {source.name}, which has "
                        f"{length}")
        self.assertEqual(problems, [],
                         "\n".join(["literalinclude ranges out of bounds:"]
                                   + problems))

    def test_line_ranges_still_cover_the_whole_example(self):
        """A ``:lines:`` range is a silent coupling to line numbers.

        Insert one line into an example and every range below it points
        at the wrong code, with nothing to say so -- the page still
        builds and still looks right. The invariant that catches it:
        the ranges of a given page must, together, quote every
        executable line of the file they cite. Code that appears in the
        example and in no range is either drift or something the
        tutorial forgot to explain.
        """
        problems = []
        for page in sorted(DOCS.rglob("*.rst")):
            ranges: dict[Path, set[int]] = {}
            for target, first, last in RANGED.findall(page.read_text()):
                source = (page.parent / target).resolve()
                ranges.setdefault(source, set()).update(
                    range(int(first), int(last) + 1))
            for source, covered in ranges.items():
                if not source.exists():
                    continue        # reported by test_included_files_exist
                for number, line in enumerate(source.read_text().splitlines(), 1):
                    if (number not in covered and line.strip()
                            and not line.strip().startswith("#")
                            and not NOT_QUOTED.match(line)):
                        problems.append(
                            f"{page.relative_to(DOCS)} quotes "
                            f"{source.name} but never line {number}: "
                            f"{line.strip()[:60]}")
        self.assertEqual(problems, [],
                         "\n".join(["literalinclude ranges have drifted:"]
                                   + problems))


if __name__ == "__main__":
    unittest.main()
