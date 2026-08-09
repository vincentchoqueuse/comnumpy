"""Figure style (decision D27b).

The style sheet ships with the package and is **never applied at
import time** -- importing a library must not mutate the user's
matplotlib state. Activate it explicitly::

    import matplotlib.pyplot as plt
    import comnumpy.style

    plt.style.use(comnumpy.style.PATH)      # global, explicit
    with comnumpy.style.context():          # or scoped
        ...

Colors come from the Okabe-Ito colorblind-safe palette, and no
information is carried by color alone (decision D27c): semantic tables
such as :data:`comnumpy.ofdm.allocation.CARRIER_STYLE` pair each color
with a glyph and a hatch.
"""
from __future__ import annotations

import pathlib

__all__ = ["PATH", "context"]

PATH: pathlib.Path = pathlib.Path(__file__).parent / "comnumpy.mplstyle"


def context():
    """Context manager applying the comnumpy style locally.

    Returns
    -------
    contextlib.AbstractContextManager
        The matplotlib style context for ``with comnumpy.style.context():``.
    """
    import matplotlib.pyplot as plt  # local import (D36)
    return plt.style.context(str(PATH))
