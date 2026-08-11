"""py.typed only means something if every module is on the strict list."""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestTheTypeMarkerIsHonest(unittest.TestCase):
    """A partial ``py.typed`` is worse than none (decision D37).

    The marker tells type checkers to trust every annotation in the
    package. Shipping it while a module sits outside the strict ratchet
    would export a promise nothing verifies -- and the failure is silent,
    because the checker believes the marker rather than looking.
    """

    def test_the_marker_exists(self):
        self.assertTrue((ROOT / "src" / "comnumpy" / "py.typed").is_file())

    def test_every_module_is_on_the_strict_list(self):
        config = json.loads(re.sub(
            r"^\s*//.*$", "",
            (ROOT / "pyrightconfig.json").read_text(), flags=re.M))
        listed = set(config["include"])
        on_disk = {str(path.relative_to(ROOT))
                   for path in (ROOT / "src" / "comnumpy").rglob("*.py")}
        self.assertEqual(sorted(on_disk - listed), [],
                         "modules shipped under py.typed but never checked "
                         "in strict mode")
        self.assertEqual(sorted(listed - on_disk), [],
                         "pyrightconfig lists modules that no longer exist")


if __name__ == "__main__":
    unittest.main()
