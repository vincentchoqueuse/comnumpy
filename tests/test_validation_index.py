"""Every validation script must be listed, and every listing must exist.

`validation/README.md` is the index a reader uses to find out what the
library has actually been proven against (decision D7). It had drifted:
six of eleven scripts were missing from the table, so the honest answer
to "what is validated?" was smaller than the truth -- and the reverse
drift, a row for a script that was deleted, would be worse.

This test does not run the scripts; they are slow by design. It only
checks that the index and the directory agree.
"""
import re
import unittest
from pathlib import Path

VALIDATION = Path(__file__).resolve().parents[1] / "validation"


class TestValidationIndex(unittest.TestCase):

    def setUp(self):
        self.scripts = {path.name for path in VALIDATION.glob("*.py")}
        self.listed = set(re.findall(r"`([a-z0-9_]+\.py)`",
                                     (VALIDATION / "README.md").read_text()))

    def test_every_script_is_documented(self):
        undocumented = sorted(self.scripts - self.listed)
        self.assertEqual(undocumented, [],
                         f"validation scripts missing from "
                         f"validation/README.md: {undocumented}")

    def test_every_row_names_a_real_script(self):
        phantom = sorted(self.listed - self.scripts)
        self.assertEqual(phantom, [],
                         f"validation/README.md lists scripts that do not "
                         f"exist: {phantom}")


if __name__ == "__main__":
    unittest.main()
