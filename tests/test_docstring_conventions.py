"""Structural check of the course-material docstring template (D10, rule R5).

Sections whose presence is machine-checkable are enforced here for every
module already converted to the section-4.10 template; R1/R2 content
(the equations and the symbol-parameter bijection) stays in human
review, as the architecture document prescribes. Add a module to
CONVERTED_MODULES when you convert it -- the list is a ratchet: it only
grows.
"""
import dataclasses
import importlib
import inspect
import unittest

from comnumpy.core.generics import Processor

# ratchet list: converted to the section-4.10 template (D10)
CONVERTED_MODULES = [
    "comnumpy.core.channels",
    "comnumpy.core.generators",
    "comnumpy.core.mappers",
    "comnumpy.core.impairments",
    "comnumpy.core.monitors",
    "comnumpy.fec.convolutional",
    "comnumpy.ofdm.processors",
    "comnumpy.ofdm.compensators",
    "comnumpy.ofdm.predistorders",
    "comnumpy.optical.channels",
    "comnumpy.optical.links",
    "comnumpy.optical.dbp",
    "comnumpy.optical.devices",
    "comnumpy.mimo.channels",
    "comnumpy.mimo.detectors",
]


# identity pass-through blocks: no algorithm, hence no citation required (R3)
NO_REFERENCE_NEEDED = {
    "comnumpy.core.monitors.Recorder",
    "comnumpy.core.monitors.Logger",
    "comnumpy.core.monitors.Debugger",
    "comnumpy.core.monitors.PowerReporter",
    "comnumpy.core.monitors.TimeSignalMonitor",
    "comnumpy.optical.devices.PowerControl",
}


def public_processor_classes(module):
    for name, obj in vars(module).items():
        if (isinstance(obj, type) and issubclass(obj, Processor)
                and obj is not Processor and not name.startswith("_")
                and obj.__module__ == module.__name__
                and dataclasses.is_dataclass(obj)):
            yield name, obj


class TestDocstringTemplate(unittest.TestCase):

    def test_converted_modules_follow_the_template(self):
        problems = []
        for module_name in CONVERTED_MODULES:
            module = importlib.import_module(module_name)
            for name, cls in public_processor_classes(module):
                doc = inspect.getdoc(cls) or ""
                where = f"{module_name}.{name}"
                if "Signal Model" not in doc:
                    problems.append(f"{where}: missing 'Signal Model' section")
                if "Axes:" not in doc:
                    problems.append(f"{where}: missing 'Axes:' category line")
                if "References" not in doc and where not in NO_REFERENCE_NEEDED:
                    problems.append(f"{where}: missing 'References' section")
        self.assertEqual(problems, [],
                         "\n".join(["template violations (D10):"] + problems))


if __name__ == "__main__":
    unittest.main()
