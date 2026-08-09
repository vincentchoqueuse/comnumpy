"""Every public module must have a documentation page, and be reachable.

The documentation described 25 of the 46 modules in ``src/``. Everything
added after the first milestone -- capacity, fading, WDM, Raman, the
whole ``fec`` package, allocation, frames, sequences, serialization,
sweep, the exception hierarchy -- existed with full course-material
docstrings that no generated page ever showed. Sphinx does not notice:
a module with no ``automodule`` directive is not an error, it is simply
absent, which is the worst way for documentation to be wrong.

Two checks, and the second matters as much as the first: a page that
exists but sits in no toctree is invisible in the built site, so it
documents nothing.
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "documentation"
SOURCE = REPO / "src" / "comnumpy"

AUTODOC = re.compile(r"^\.\.\s+auto(?:module|class|function)::\s+([\w.]+)",
                     re.MULTILINE)

# Modules with nothing to show a reader: private helpers and the CuPy
# dispatch shim, which is an implementation detail of D3.
NOT_DOCUMENTED = {
    "comnumpy._backend",
}


def public_modules():
    for path in sorted(SOURCE.rglob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        dotted = ("comnumpy."
                  + path.relative_to(SOURCE).with_suffix("").as_posix()
                  .replace("/", "."))
        if dotted not in NOT_DOCUMENTED:
            yield dotted


def documented_modules():
    for page in DOCS.rglob("*.rst"):
        for name in AUTODOC.findall(page.read_text()):
            yield name, page


class TestDocsCoverage(unittest.TestCase):

    def test_every_public_module_has_a_page(self):
        documented = {name for name, _ in documented_modules()}
        missing = sorted(set(public_modules()) - documented)
        self.assertEqual(missing, [],
                         f"modules with no documentation page: {missing}")

    def test_no_page_documents_a_module_that_is_gone(self):
        existing = set(public_modules()) | NOT_DOCUMENTED
        phantom = sorted({f"{name} ({page.relative_to(DOCS)})"
                          for name, page in documented_modules()
                          if name.split(".")[0] == "comnumpy"
                          and name not in existing})
        self.assertEqual(phantom, [],
                         f"documentation pages for modules that no longer "
                         f"exist: {phantom}")

    def test_every_page_is_reachable_from_a_toctree(self):
        """A page in no toctree is built but never linked -- invisible."""
        listed = set()
        for index in DOCS.rglob("index.rst"):
            body = index.read_text()
            block = body.split(".. toctree::", 1)[-1]
            for line in block.splitlines():
                entry = line.strip()
                if entry and not entry.startswith(":") and " " not in entry:
                    listed.add((index.parent / entry).resolve())
        orphans = sorted(
            str(page.relative_to(DOCS))
            for page in DOCS.rglob("*.rst")
            if page.name != "index.rst" and page.with_suffix("").resolve() not in listed)
        self.assertEqual(orphans, [],
                         f"documentation pages in no toctree: {orphans}")


if __name__ == "__main__":
    unittest.main()
