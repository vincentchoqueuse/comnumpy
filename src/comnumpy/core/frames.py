"""Frame structure objects (decision D28).

A frame is described by a :class:`FrameStructure`: an ordered list of
typed fields (:class:`FrameField` with a :class:`FieldRole`), shared by
the :class:`Framer` and the :class:`Deframer` exactly like a
``CarrierAllocation`` is shared by the carrier allocator and extractor.
``FieldRole`` types the time axis the way ``CarrierType`` types the
frequency axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from comnumpy.core.generics import Processor
from comnumpy.exceptions import ShapeError

__all__ = ["FieldRole", "FrameField", "FrameStructure", "Framer", "Deframer"]


class FieldRole(IntEnum):
    SYNC = 0         # detection, AGC, timing, coarse CFO
    TRAINING = 1     # channel estimation
    HEADER = 2
    PAYLOAD = 3
    TAIL = 4
    PAD = 5


_ROLE_GLYPH = {
    FieldRole.SYNC: "sync",
    FieldRole.TRAINING: "training",
    FieldRole.HEADER: "header",
    FieldRole.PAYLOAD: "unknown at TX",
    FieldRole.TAIL: "tail",
    FieldRole.PAD: "pad",
}


@dataclass(frozen=True, slots=True)
class FrameField:
    """One typed field of a frame.

    Parameters
    ----------
    name : str
        Field name ("STF", "LTF", "SIG", "PAYLOAD"...).
    role : FieldRole
        What the field is for.
    values : np.ndarray, optional
        Transmitted samples. ``None`` means unknown at transmit time --
        only allowed for the PAYLOAD field, whose length is then given
        by ``length``.
    length : int, optional
        Field length in samples; inferred from ``values`` when provided.
    """
    name: str
    role: FieldRole
    values: Optional[np.ndarray] = None
    length: Optional[int] = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.values is not None:
            values = np.asarray(self.values)
            values.setflags(write=False)
            object.__setattr__(self, "values", values)
            if self.length is not None and self.length != len(values):
                raise ValueError(
                    f"field {self.name!r}: length={self.length} contradicts "
                    f"values of length {len(values)}")
            object.__setattr__(self, "length", len(values))
        elif self.length is None:
            raise ValueError(
                f"field {self.name!r} has neither values nor length")


@dataclass(frozen=True, slots=True)
class FrameStructure:
    """Ordered list of typed frame fields (decision D28).

    Invariants: exactly one PAYLOAD field, and every other field carries
    its transmitted values.

    Examples
    --------
    >>> from comnumpy.core.sequences import zadoff_chu
    >>> frame = FrameStructure((
    ...     FrameField("SYNC", FieldRole.SYNC, zadoff_chu(1, 63)),
    ...     FrameField("PAYLOAD", FieldRole.PAYLOAD, length=1000),
    ... ), standard="example")
    >>> frame.frame_length, frame.payload_length
    (1063, 1000)
    """
    fields: tuple
    standard: str = field(default="custom", kw_only=True)
    reference: str = field(default="", kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names}")
        payloads = [f for f in self.fields if f.role == FieldRole.PAYLOAD]
        if len(payloads) != 1:
            raise ValueError(
                f"a frame needs exactly one PAYLOAD field, got {len(payloads)}")
        for f in self.fields:
            if f.role != FieldRole.PAYLOAD and f.values is None:
                raise ValueError(
                    f"field {f.name!r} ({f.role.name}) must carry its "
                    f"transmitted values; only the PAYLOAD is unknown at TX")

    # -- properties -----------------------------------------------------
    @property
    def frame_length(self) -> int:
        return int(sum(f.length for f in self.fields))

    @property
    def payload_length(self) -> int:
        return int(next(f.length for f in self.fields
                        if f.role == FieldRole.PAYLOAD))

    def slice_of(self, name: str) -> slice:
        """Position of a field inside the frame."""
        start = 0
        for f in self.fields:
            if f.name == name:
                return slice(start, start + f.length)
            start += f.length
        raise KeyError(f"unknown field {name!r}; "
                       f"known: {[f.name for f in self.fields]}")

    def field_by_name(self, name: str) -> FrameField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"unknown field {name!r}; "
                       f"known: {[f.name for f in self.fields]}")

    def fields_by_role(self, role: FieldRole) -> tuple:
        return tuple(f for f in self.fields if f.role == role)

    # -- rendering (P7, same mechanism as the D21 spectral map) ---------
    def __repr__(self) -> str:
        header = self.standard
        if self.reference:
            header += f"{' ' * max(1, 36 - len(self.standard))}[{self.reference}]"
        cells = "|" + "|".join(f"--{f.name}--" if f.role != FieldRole.PAYLOAD
                               else f" {f.name} " for f in self.fields) + "|"
        widths = [len(f"--{f.name}--") if f.role != FieldRole.PAYLOAD
                  else len(f" {f.name} ") for f in self.fields]
        lengths = " " + " ".join(str(f.length).center(w)
                                 for f, w in zip(self.fields, widths, strict=True))
        roles = " " + " ".join(_ROLE_GLYPH[f.role].center(w)
                               for f, w in zip(self.fields, widths, strict=True))
        return "\n".join([header, cells, lengths, roles])


@dataclass(slots=True)
class Framer(Processor):
    r"""Insert the frame fields around the payload (decision D28).

    Signal Model
    ------------
    .. math::

        \mathbf{y} = [\, \mathbf{s}_1 \,|\, \cdots \,|\, \mathbf{x} \,|\,
        \cdots \,|\, \mathbf{s}_M \,]

    where the :math:`\mathbf{s}_i` are the known field values of the
    shared :class:`FrameStructure` and :math:`\mathbf{x}` is the payload.

    Axes: *declared axis* -- Block layout ``(..., T, N_payload)`` with
    the frame index on ``T``; fields are added on the content axis -1.
    The S/P block size is not a free parameter: it equals
    ``frame.payload_length`` and ``prepare()`` enforces it.
    """
    frame: FrameStructure
    name: str = field(default="framer", kw_only=True)

    def prepare(self, X: np.ndarray):
        N = self.frame.payload_length
        if X.shape[-1] != N:
            raise ShapeError(
                f"Framer expects (..., T, {N}) for frame "
                f"{self.frame.standard!r} (payload field length {N}), got "
                f"{X.shape}. Set Serial2Parallel(N_sub={N}) or use "
                f"frame.payload_length.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        out_shape = X.shape[:-1] + (self.frame.frame_length,)
        Y = np.zeros(out_shape, dtype=np.result_type(X.dtype, complex))
        for f in self.frame.fields:
            sl = self.frame.slice_of(f.name)
            if f.role == FieldRole.PAYLOAD:
                Y[..., sl] = X
            else:
                Y[..., sl] = f.values
        return Y


@dataclass(slots=True)
class Deframer(Processor):
    """Extract the payload (and expose every field) from received frames.

    Axes: *declared axis* -- Block layout ``(..., T, frame_length)``.
    After ``forward``, the received samples of each field are available
    through :meth:`get_field`.
    """
    frame: FrameStructure
    name: str = field(default="deframer", kw_only=True)
    # internal state (declared for slots, D40a)
    received_fields: dict = field(init=False, repr=False, default_factory=dict)

    def prepare(self, X: np.ndarray):
        N = self.frame.frame_length
        if X.shape[-1] != N:
            raise ShapeError(
                f"Deframer expects (..., T, {N}) for frame "
                f"{self.frame.standard!r} (frame length {N}), got {X.shape} "
                f"-- share the same FrameStructure as the transmitter's "
                f"Framer.")

    def get_field(self, name: str) -> np.ndarray:
        """Received samples of a field, recorded by the last ``forward``."""
        if name not in self.received_fields:
            raise KeyError(
                f"field {name!r} not recorded yet; run the deframer first "
                f"(known fields: {[f.name for f in self.frame.fields]})")
        return self.received_fields[name]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.received_fields = {
            f.name: X[..., self.frame.slice_of(f.name)]
            for f in self.frame.fields}
        payload_name = self.frame.fields_by_role(FieldRole.PAYLOAD)[0].name
        return self.received_fields[payload_name]
