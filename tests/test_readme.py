"""Execute the code blocks of README.md (D8: the README example must run)."""
import re
import unittest
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


class TestReadme(unittest.TestCase):

    def test_python_blocks_execute(self):
        content = README.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)
        self.assertTrue(blocks, "no python block found in README.md")
        for block in blocks:
            exec(compile(block, str(README), "exec"), {})


if __name__ == "__main__":
    unittest.main()
