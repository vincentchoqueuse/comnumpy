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

    def __post_init__(self) -> None:
        # accept the raw int (JSON round trip, user code) but always store
        # the enum, so `role.name` works and an invalid role fails here
        object.__setattr__(self, "role", FieldRole(self.role))
        if self.values is not None:
            # np.array, not asarray: freezing must apply to a copy this
            # field owns, never to the caller's own buffer
            values = np.array(self.values)
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

    @property
    def size(self) -> int:
        """Field length in samples (guaranteed set after construction)."""
        assert self.length is not None  # internal invariant, see __post_init__
        return self.length


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
    fields: tuple[FrameField, ...]
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
        return int(sum(f.size for f in self.fields))

    @property
    def payload_length(self) -> int:
        return next(f.size for f in self.fields
                    if f.role == FieldRole.PAYLOAD)

    def slice_of(self, name: str) -> slice:
        """Position of a field inside the frame."""
        start = 0
        for f in self.fields:
            if f.name == name:
                return slice(start, start + f.size)
            start += f.size
        raise KeyError(f"unknown field {name!r}; "
                       f"known: {[f.name for f in self.fields]}")

    def field_by_name(self, name: str) -> FrameField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(f"unknown field {name!r}; "
                       f"known: {[f.name for f in self.fields]}")

    def fields_by_role(self, role: FieldRole) -> tuple[FrameField, ...]:
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

    The model is *structural* rather than analytical: no arithmetic is
    performed on the samples. Writing the frame as the ordered list of
    its :math:`M` typed fields, of lengths :math:`L_1, \dots, L_M`, the
    output row of index :math:`t` is the concatenation

    .. math::

        y_t[n] = \begin{cases}
        s_i\left[n - \sum_{j<i} L_j\right] &
        \text{if field } i \text{ is known at TX} \\
        x_t\left[n - \sum_{j<i} L_j\right] &
        \text{if field } i \text{ is the PAYLOAD}
        \end{cases}

    for :math:`0 \le n < N_{\mathrm{frame}} = \sum_i L_i`. Exactly one
    field carries the role ``PAYLOAD`` and receives the input; every
    other field (``SYNC``, ``TRAINING``, ``HEADER``, ``TAIL``, ``PAD``)
    carries values fixed at construction time. The output is complex,
    since the known fields generally are.

    Axes: *declared axis* -- Block layout ``(..., T, N_payload)`` with
    the frame index on ``T``; fields are added on the content axis -1.
    The S/P block size is not a free parameter: it equals
    ``frame.payload_length`` and ``prepare()`` enforces it.

    Parameters
    ----------
    frame : FrameStructure
        Frame description shared with the receiver's :class:`Deframer`.
        Its ``payload_length`` fixes the accepted input size
        :math:`N_{\mathrm{payload}}` and its ``frame_length`` the output
        size :math:`N_{\mathrm{frame}}`.
    name : str, optional, keyword-only
        Name of the framer instance. Default is ``"framer"``.

    Raises
    ------
    ShapeError
        If the last axis of the input does not equal
        ``frame.payload_length``.

    References
    ----------
    IEEE Std 802.11-2020, Clause 17 (Orthogonal frequency division
    multiplexing PHY) -- example of a real typed frame structure: short
    training field, long training field, SIGNAL field, then DATA.

    Examples
    --------
    >>> frame = FrameStructure((
    ...     FrameField("SYNC", FieldRole.SYNC, np.array([1.0, -1.0])),
    ...     FrameField("PAYLOAD", FieldRole.PAYLOAD, length=3),
    ... ), standard="example")
    >>> framer = Framer(frame)
    >>> print(framer(np.array([[1.0, 2.0, 3.0]])))
    [[ 1.+0.j -1.+0.j  1.+0.j  2.+0.j  3.+0.j]]
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
    r"""Extract the payload (and expose every field) from received frames.

    Signal Model
    ------------
    The exact inverse of :class:`Framer`, and equally structural: the
    received frame is cut back along the field boundaries of the shared
    :class:`FrameStructure`. Field :math:`i` occupies the slice
    :math:`[\sum_{j<i} L_j, \; \sum_{j \le i} L_j)` of the frame axis,

    .. math::

        \hat{\mathbf{s}}_i = \mathbf{y}\!\left[\, \sum_{j<i} L_j : \,
        \sum_{j \le i} L_j \,\right], \qquad i = 1, \dots, M

    and ``forward`` returns the single field whose role is ``PAYLOAD``:

    .. math::

        \hat{\mathbf{x}} = \hat{\mathbf{s}}_{i_\mathrm{payload}}

    The other fields are not discarded: the received
    :math:`\hat{\mathbf{s}}_i` are recorded and read back with
    :meth:`get_field` -- that is how a receiver gets the ``SYNC`` field
    for timing or the ``TRAINING`` field for channel estimation without
    adding a block to the chain (D42).

    Axes: *declared axis* -- Block layout ``(..., T, frame_length)``.
    After ``forward``, the received samples of each field are available
    through :meth:`get_field`.

    Parameters
    ----------
    frame : FrameStructure
        Frame description, the very same object as the transmitter's
        :class:`Framer`. Its ``frame_length`` fixes the accepted input
        size :math:`N_{\mathrm{frame}}`.
    name : str, optional, keyword-only
        Name of the deframer instance. Default is ``"deframer"``.

    Raises
    ------
    ShapeError
        If the last axis of the input does not equal
        ``frame.frame_length``.

    References
    ----------
    IEEE Std 802.11-2020, Clause 17 (Orthogonal frequency division
    multiplexing PHY) -- example of a real typed frame structure: short
    training field, long training field, SIGNAL field, then DATA.

    Examples
    --------
    >>> frame = FrameStructure((
    ...     FrameField("SYNC", FieldRole.SYNC, np.array([1.0, -1.0])),
    ...     FrameField("PAYLOAD", FieldRole.PAYLOAD, length=3),
    ... ), standard="example")
    >>> deframer = Deframer(frame)
    >>> print(deframer(np.array([[1.0, -1.0, 1.0, 2.0, 3.0]])))
    [[1. 2. 3.]]
    >>> print(deframer.get_field("SYNC"))
    [[ 1. -1.]]
    """
    frame: FrameStructure
    name: str = field(default="deframer", kw_only=True)
    # internal state (declared for slots, D40a)
    received_fields: dict[str, np.ndarray] = field(init=False, repr=False, default_factory=dict)

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
        # copies, not views: the recorded fields are state that outlives
        # the pass, and must not change if the caller reuses the buffer
        self.received_fields = {
            f.name: X[..., self.frame.slice_of(f.name)].copy()
            for f in self.frame.fields}
        payload_name = self.frame.fields_by_role(FieldRole.PAYLOAD)[0].name
        return self.received_fields[payload_name]
