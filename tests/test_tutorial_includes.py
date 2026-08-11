"""The tutorials must quote the code they claim to.

``literalinclude`` ranges are line numbers into files that keep changing.
Nothing else checks them: a range that has drifted still renders, still
looks like code, and simply shows the wrong lines under the wrong
heading. Sphinx is happy, the page is wrong, and only a reader following
along notices. That has already happened twice in this repository -- two
ranges published swapped, and a near miss when an example was edited.

Two properties catch a drifted range without anyone declaring what each
range is supposed to contain:

* it must **parse on its own**. A range that starts or ends mid-statement
  is showing an arbitrary window, not a unit of code;
* it must **start at column zero**. A range beginning with an indented
  line has slid into the middle of a function body.
"""
import ast
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DIRECTIVE = re.compile(
    r"\.\.\s+literalinclude::\s*(\S+)(.*?)(?=\n\S|\n\.\.|\Z)", re.S)
RANGE = re.compile(r":lines:\s*([0-9]+)\s*-\s*([0-9]+)")


def ranges():
    for rst in sorted(DOCS.rglob("*.rst")):
        for directive in DIRECTIVE.finditer(rst.read_text()):
            span = RANGE.search(directive.group(2))
            if span:
                yield (rst, (rst.parent / directive.group(1)).resolve(),
                       int(span.group(1)), int(span.group(2)))


class TestLiteralIncludeRanges(unittest.TestCase):

    def test_there_are_ranges_to_check(self):
        """Otherwise a broken parser would make this suite vacuously green."""
        self.assertGreater(len(list(ranges())), 50)

    def test_every_target_exists_and_the_range_fits_in_it(self):
        for rst, path, start, end in ranges():
            with self.subTest(rst=rst.name, lines=f"{start}-{end}"):
                self.assertTrue(path.is_file(), path)
                length = len(path.read_text().splitlines())
                self.assertLessEqual(start, end)
                self.assertGreaterEqual(start, 1)
                self.assertLessEqual(end, length)

    def test_every_range_is_a_standalone_piece_of_code(self):
        for rst, path, start, end in ranges():
            if path.suffix != ".py":
                continue
            block = path.read_text().splitlines()[start - 1:end]
            with self.subTest(rst=rst.name, lines=f"{start}-{end}"):
                self.assertTrue(any(line.strip() for line in block),
                                "the range is blank")
                first = block[0]
                self.assertFalse(
                    first.startswith((" ", "\t"))
                    and not first.lstrip().startswith("#"),
                    f"starts mid-block: {first.strip()[:60]!r}")
                try:
                    ast.parse("\n".join(block))
                except SyntaxError as error:
                    self.fail(f"does not parse on its own: {error.msg}")


if __name__ == "__main__":
    unittest.main()
