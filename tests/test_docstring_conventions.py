"""Structural check of the course-material docstring template (D10, rule R5).

Sections whose presence is machine-checkable are enforced here for every
module already converted to the section-4.10 template; R1/R2 content
(the equations and the symbol-parameter bijection) stays in human
review, as the architecture document prescribes. Add a module to
CONVERTED_MODULES when you convert it -- the list is a ratchet: it only
grows.

One list, both checks. A converted module is checked on *everything*
public it exposes: its ``Processor`` classes and its processing
functions. Splitting that in two lists was how ``core.capacity`` and
``core.fading`` -- which are function-only -- sat in the ratchet while
nothing about them was ever verified. A module that carries neither
kind of object now fails the ratchet rather than passing it vacuously.
"""
import dataclasses
import importlib
import inspect
import unittest

from comnumpy.core.generics import Processor

# ratchet list: converted to the section-4.10 template (D10)
CONVERTED_MODULES = [
    "comnumpy.core.channels",
    "comnumpy.core.fading",
    "comnumpy.core.capacity",
    "comnumpy.core.generators",
    "comnumpy.core.mappers",
    "comnumpy.core.impairments",
    "comnumpy.core.metrics",
    "comnumpy.core.utils",
    "comnumpy.core.sequences",
    "comnumpy.fec.convolutional",
    "comnumpy.fec.ldpc",
    "comnumpy.fec.analysis",
    "comnumpy.ofdm.processors",
    "comnumpy.ofdm.compensators",
    "comnumpy.ofdm.predistorders",
    "comnumpy.ofdm.metrics",
    "comnumpy.optical.channels",
    "comnumpy.optical.links",
    "comnumpy.optical.dbp",
    "comnumpy.optical.devices",
    "comnumpy.mimo.channels",
    "comnumpy.mimo.detectors",
    "comnumpy.mimo.compensators",
    "comnumpy.mimo.utils",
    "comnumpy.ofdm.chains",
    "comnumpy.core.processors",
    "comnumpy.core.filters",
    "comnumpy.core.devices",
    "comnumpy.core.frames",
    "comnumpy.optical.compensators",
    "comnumpy.optical.wdm",
    "comnumpy.optical.raman",
    "comnumpy.optical.fiber",
    "comnumpy.core.compensators",
    "comnumpy.core.shaping",
]


# identity pass-through blocks: no algorithm, hence no citation required (R3)
NO_REFERENCE_NEEDED = {
    "comnumpy.optical.devices.PowerControl",
    # pure array remapping: no algorithm, hence nothing to cite
    "comnumpy.core.processors.Complex2Real",
    "comnumpy.core.processors.AutoConcatenator",
    "comnumpy.core.processors.SampleRemover",
    "comnumpy.core.processors.DelayRemover",
    "comnumpy.core.processors.DataAdder",
    "comnumpy.core.processors.DataExtractor",
    # format conversion and plotting helpers: no algorithm to cite
    "comnumpy.core.utils.sym_2_bin",
    "comnumpy.core.utils.plot_alphabet",
}


# Functions that carry no signal model: they manage the catalog, validate
# an argument, or draw. R1 applies to *processing* functions; forcing an
# equation onto ``available_delay_profiles`` would be ritual, not course
# material. Every entry must resolve, so a renamed helper cannot leave a
# stale exemption behind (see test_the_exemption_lists_are_not_stale).
NOT_PROCESSING_FUNCTIONS = {
    "comnumpy.core.utils.plot_alphabet",
    "comnumpy.core.fading.register_delay_profile",
    "comnumpy.core.fading.available_delay_profiles",
    "comnumpy.core.fading.get_delay_profile",
    "comnumpy.core.fading.validate_taps_fit",
    "comnumpy.optical.raman.register_gain_spectrum",
    "comnumpy.optical.raman.available_gain_spectra",
    "comnumpy.optical.raman.get_gain_spectrum",
    "comnumpy.optical.fiber.register_fiber",
    "comnumpy.optical.fiber.available_fibers",
    "comnumpy.optical.fiber.get_fiber",
}


def public_functions(module):
    for name, obj in vars(module).items():
        if (callable(obj) and not isinstance(obj, type)
                and getattr(obj, "__module__", None) == module.__name__
                and not name.startswith("_")):
            yield name, obj


def public_processor_classes(module):
    for name, obj in vars(module).items():
        if (isinstance(obj, type) and issubclass(obj, Processor)
                and obj is not Processor and not name.startswith("_")
                and obj.__module__ == module.__name__
                and dataclasses.is_dataclass(obj)):
            yield name, obj


class TestDocstringTemplate(unittest.TestCase):

    def test_converted_classes_follow_the_template(self):
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

    def test_converted_functions_follow_the_template(self):
        """R1 covers processing functions, not only Processor classes."""
        problems = []
        for module_name in CONVERTED_MODULES:
            module = importlib.import_module(module_name)
            for name, func in public_functions(module):
                doc = inspect.getdoc(func) or ""
                where = f"{module_name}.{name}"
                if not doc:
                    problems.append(f"{where}: no docstring at all")
                    continue
                if (where not in NOT_PROCESSING_FUNCTIONS
                        and "Signal Model" not in doc):
                    problems.append(f"{where}: missing 'Signal Model' section")
                if (where not in NOT_PROCESSING_FUNCTIONS
                        and where not in NO_REFERENCE_NEEDED
                        and "References" not in doc):
                    problems.append(f"{where}: missing 'References' section")
                if "Examples" not in doc:
                    problems.append(f"{where}: missing 'Examples' section")
        self.assertEqual(problems, [],
                         "\n".join(["template violations (D10, functions):"] + problems))

    def test_every_ratchet_entry_actually_checks_something(self):
        """A module with nothing public turns its ratchet line into a lie."""
        empty = []
        for module_name in CONVERTED_MODULES:
            module = importlib.import_module(module_name)
            if not (list(public_processor_classes(module))
                    or list(public_functions(module))):
                empty.append(module_name)
        self.assertEqual(empty, [],
                         "listed as converted but exposing no public "
                         f"Processor or function: {empty}")

    def test_the_exemption_lists_are_not_stale(self):
        """An exemption that no longer resolves silently weakens the check."""
        known = set()
        for module_name in CONVERTED_MODULES:
            module = importlib.import_module(module_name)
            for name, _ in public_processor_classes(module):
                known.add(f"{module_name}.{name}")
            for name, _ in public_functions(module):
                known.add(f"{module_name}.{name}")
        stale = sorted((NO_REFERENCE_NEEDED | NOT_PROCESSING_FUNCTIONS) - known)
        self.assertEqual(stale, [], f"exemptions with no target left: {stale}")


if __name__ == "__main__":
    unittest.main()
