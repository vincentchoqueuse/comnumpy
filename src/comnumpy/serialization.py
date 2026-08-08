"""Chain serialization (decisions D31/D32).

A chain exports to JSON as a list of blocks ``{"id", "type", "params",
"inputs"}``. The *intent* is serialized -- block types and constructor
parameters -- never derived state. Free arrays (an explicit alphabet, a
carrier-type mask) go to a companion ``.npz`` file; the JSON stays
readable and editable.

Known frontier (documented, not worked around): callables -- callbacks,
the ``rule`` of ``scattered_allocation`` -- are not serializable.

Examples
--------
>>> from comnumpy import Sequential, SymbolGenerator
>>> from comnumpy.core.channels import AWGN
>>> chain = Sequential([SymbolGenerator(4, seed=1), AWGN(sigma2=0.1, seed=2)])
>>> text = to_json(chain)
>>> chain2 = from_json(text)
>>> chain2.block_ids()
['generator', 'awgn']
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import pathlib
from typing import Optional

import numpy as np

from comnumpy.core.generics import Processor, Sequential

__all__ = ["to_json", "from_json", "register_block"]

FORMAT_VERSION = "1.0"

# modules scanned for Processor subclasses (name -> class registry)
_SCAN_MODULES = [
    "comnumpy.core.generators", "comnumpy.core.mappers",
    "comnumpy.core.channels", "comnumpy.core.filters",
    "comnumpy.core.processors", "comnumpy.core.monitors",
    "comnumpy.core.compensators", "comnumpy.core.impairments",
    "comnumpy.core.devices",
    "comnumpy.ofdm.processors", "comnumpy.ofdm.chains",
    "comnumpy.ofdm.compensators", "comnumpy.ofdm.predistorders",
    "comnumpy.mimo.channels", "comnumpy.mimo.detectors",
    "comnumpy.mimo.compensators",
    "comnumpy.fec.convolutional",
    "comnumpy.optical.channels", "comnumpy.optical.links",
    "comnumpy.optical.dbp", "comnumpy.optical.devices",
    "comnumpy.optical.compensators",
]

_EXTRA_BLOCKS: dict[str, type] = {}


def register_block(cls: type) -> type:
    """Class decorator adding a user block to the deserialization registry."""
    _EXTRA_BLOCKS[cls.__name__] = cls
    return cls


def _registry() -> dict[str, type]:
    registry: dict[str, type] = {}
    for module_name in _SCAN_MODULES:
        module = importlib.import_module(module_name)
        for attr in vars(module).values():
            if (isinstance(attr, type) and issubclass(attr, Processor)
                    and attr is not Processor):
                registry.setdefault(attr.__name__, attr)
    registry.update(_EXTRA_BLOCKS)
    return registry


def _encode_value(value, arrays: dict, key: str):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        arrays[key] = value
        return {"__ndarray__": key}
    if isinstance(value, (list, tuple)):
        return [_encode_value(v, arrays, f"{key}.{i}") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"__block__": _encode_block(value, arrays, key)}
    raise TypeError(
        f"cannot serialize parameter {key!r} of type {type(value).__name__}: "
        f"only scalars, strings, arrays, lists and dataclass blocks are "
        f"supported (callables are a documented frontier of decision D31)")


def _encode_block(module, arrays: dict, key: str) -> dict:
    if not dataclasses.is_dataclass(module):
        raise TypeError(
            f"block {key!r} ({type(module).__name__}) is not a dataclass and "
            f"cannot be serialized")
    params = {}
    for f in dataclasses.fields(module):
        if not f.init:
            continue  # internal state is derived, never serialized (D31)
        value = getattr(module, f.name)
        default = None
        if f.default is not dataclasses.MISSING:
            default = f.default
            if _is_trivially_equal(value, default):
                continue  # keep the JSON minimal: defaults are implied
        params[f.name] = _encode_value(value, arrays, f"{key}.{f.name}")
    return {"type": type(module).__name__, "params": params}


def _is_trivially_equal(value, default):
    try:
        return bool(value == default) and type(value) is type(default)
    except Exception:
        return False


def to_json(chain: Sequential, path=None, npz_path=None, indent: int = 2) -> str:
    """Serialize a Sequential chain to JSON (+ optional array sidecar).

    Parameters
    ----------
    chain : Sequential
        The chain to serialize.
    path : str or Path, optional
        Where to write the JSON. When omitted, the text is only returned.
    npz_path : str or Path, optional
        Companion file for array-valued parameters. Required if the
        chain contains any (an explicit alphabet, a mask...).
    indent : int, optional
        JSON indentation. Default 2.

    Returns
    -------
    str
        The JSON document.
    """
    arrays: dict[str, np.ndarray] = {}
    blocks = []
    ids = chain.block_ids()
    for index, (block_id, module) in enumerate(zip(ids, chain.module_list, strict=True)):
        entry = {"id": block_id}
        entry.update(_encode_block(module, arrays, block_id))
        # explicit inputs field from day one (decision D31): implicit chain
        entry["inputs"] = [ids[index - 1]] if index > 0 else []
        blocks.append(entry)

    document = {"comnumpy": FORMAT_VERSION, "blocks": blocks}
    if arrays:
        if npz_path is None:
            raise ValueError(
                f"the chain contains array parameters ({sorted(arrays)}); "
                f"pass npz_path= to store them in a companion .npz file")
        np.savez(npz_path, **arrays)
        document["arrays"] = pathlib.Path(npz_path).name

    text = json.dumps(document, indent=indent)
    if path is not None:
        pathlib.Path(path).write_text(text, encoding="utf-8")
    return text


def _decode_value(value, arrays):
    if isinstance(value, dict):
        if "__ndarray__" in value:
            if arrays is None:
                raise ValueError(
                    f"the document references array {value['__ndarray__']!r}; "
                    f"pass npz_path= to from_json")
            return arrays[value["__ndarray__"]]
        if "__block__" in value:
            return _instantiate(value["__block__"], arrays)
        raise ValueError(f"unknown JSON object {sorted(value)}")
    if isinstance(value, list):
        return [_decode_value(v, arrays) for v in value]
    return value


def _instantiate(entry: dict, arrays):
    registry = _registry()
    type_name = entry["type"]
    if type_name not in registry:
        raise KeyError(
            f"unknown block type {type_name!r}; register user blocks with "
            f"@register_block")
    params = {name: _decode_value(value, arrays)
              for name, value in entry.get("params", {}).items()}
    return registry[type_name](**params)


def from_json(source, npz_path: Optional[str] = None) -> Sequential:
    """Rebuild a Sequential chain from :func:`to_json` output.

    Parameters
    ----------
    source : str or Path
        JSON text, or path to a JSON file.
    npz_path : str or Path, optional
        Companion array file, required when the document references one.
    """
    text = source
    candidate = pathlib.Path(source) if isinstance(source, (str, pathlib.Path)) else None
    if candidate is not None and str(source).lstrip()[:1] != "{" and candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
    document = json.loads(text)

    if document.get("comnumpy") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported format version {document.get('comnumpy')!r}; "
            f"this comnumpy reads version {FORMAT_VERSION!r}")

    arrays = None
    if npz_path is not None:
        arrays = np.load(npz_path)

    blocks = document["blocks"]
    # only implicit linear chains are executable today; the inputs field
    # keeps the door open to a DAG without a format break (D31)
    ids = [entry["id"] for entry in blocks]
    for index, entry in enumerate(blocks):
        expected = [ids[index - 1]] if index > 0 else []
        if entry.get("inputs", expected) != expected:
            raise NotImplementedError(
                f"block {entry['id']!r} has non-linear inputs "
                f"{entry['inputs']}; only linear chains are supported")

    return Sequential([_instantiate(entry, arrays) for entry in blocks])
