"""Exception hierarchy for comnumpy (decision D38).

Three members, no more. The double inheritance preserves existing
``except ValueError`` / ``except RuntimeError`` user code.

Error messages follow the normative template: *observed, expected, action*
("got X, expected Y, do Z").
"""

__all__ = ["ComnumpyError", "ShapeError", "NotFittedError"]


class ComnumpyError(Exception):
    """Base class for every comnumpy-raised error.

    ``except ComnumpyError`` separates "my chain is misconfigured" from an
    underlying numpy/scipy bug, a distinction a bare ``ValueError`` cannot make.
    """


class ShapeError(ComnumpyError, ValueError):
    """Raised by ``prepare()`` when the input does not match the block's declared axes.

    The message states the observed shape, the expected shape, and the action
    to fix it, e.g.::

        ShapeError: expected shape (..., ant, N), got (N,) -- add an antenna
        axis or use a SISO block.
    """


class NotFittedError(ComnumpyError, RuntimeError):
    """Raised when ``forward()`` needs an estimated quantity but ``fit()`` was never called."""
