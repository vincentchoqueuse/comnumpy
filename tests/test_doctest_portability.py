"""Doctest output must not depend on the sign of a zero.

IEEE 754 has two zeros, and ``numpy`` prints them differently: an array
holding -0.0 renders as ``-0.`` where +0.0 renders as ``0.``. Which one
a rounded computation lands on depends on the last bit of the value
before rounding, and therefore on the numpy version, the platform and
the compiler flags.

That is not a hypothetical. A doctest of ``DataAidedFIRCompensator``
passed on Python 3.11 locally and failed on 3.12 in CI, printing
``[ 1.  0.5  0. -0. ]`` against an expected ``[1.  0.5 0.  0. ]`` --
same code, same inputs, different sign of a zero. Seven other doctests
displayed a rounded zero and were one numpy release away from the same
fate.

The rule this test enforces: **no expected doctest output may contain a
negative zero.** Producing one is avoided by adding ``+ 0.0`` to the
rounded array, since IEEE guarantees ``-0.0 + 0.0`` is ``+0.0``.
"""
import doctest
import importlib
import pkgutil
import re
import unittest

import comnumpy

# a standalone -0. or -0.0, not part of a longer number like -0.25
NEGATIVE_ZERO = re.compile(r"(?<![\d.])-0\.0*(?![1-9\d])")


def all_doctests():
    for module in pkgutil.walk_packages(comnumpy.__path__, "comnumpy."):
        try:
            imported = importlib.import_module(module.name)
        except Exception:          # pragma: no cover - optional deps
            continue
        for test in doctest.DocTestFinder().find(imported):
            for example in test.examples:
                yield test.name, example


class TestDoctestPortability(unittest.TestCase):

    def test_no_expected_output_contains_a_negative_zero(self):
        offenders = [f"{name}: {example.want.strip()[:70]}"
                     for name, example in all_doctests()
                     if NEGATIVE_ZERO.search(example.want or "")]
        self.assertEqual(
            offenders, [],
            "\n".join(["doctests pinned to a negative zero, which is not "
                       "portable across numpy versions -- add `+ 0.0` to "
                       "the rounded array:"] + offenders))

    def test_the_pattern_recognizes_a_negative_zero(self):
        """A guard that matched nothing would pass for the wrong reason."""
        for text in ("[ 1. -0. ]", "-0.0", "[-0.  1.]"):
            with self.subTest(text=text):
                self.assertTrue(NEGATIVE_ZERO.search(text))
        for text in ("[1. 0.]", "-0.25", "-0.5", "1e-05"):
            with self.subTest(text=text):
                self.assertIsNone(NEGATIVE_ZERO.search(text))


if __name__ == "__main__":
    unittest.main()
