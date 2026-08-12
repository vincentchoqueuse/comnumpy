import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

__all__ = [
    "Constellation", "get_alphabet", "plot_alphabet", "sym_2_bin",
    "hard_projector", "soft_projector", "esn0_to_snr_dB", "ebn0_to_snr_dB",
    "zf_estimator", "mmse_estimator",
]

if TYPE_CHECKING:  # matplotlib stays out of the import path (D36)
    from matplotlib.axes import Axes


def get_alphabet(modulation: str, order: int, type: str = "gray",
                 norm: bool = True) -> np.ndarray:
    r"""
    Return the symbol alphabet of a memoryless modulation of order :math:`M`.

    Signal Model
    ------------
    A memoryless modulation is described by its alphabet
    :math:`\mathcal{A} = \{a_0, \ldots, a_{M-1}\}`: the mapper turns the
    symbol index :math:`s[n] \in \{0, \ldots, M-1\}` into the transmitted
    sample

    .. math::

        x[n] = a_{s[n]}

    The raw constellations are tabulated on the usual grids -- odd
    integers :math:`\pm 1, \pm 3, \ldots` for PAM and QAM, the unit
    circle for PSK -- and the row order of the file *is* the bit-to-symbol
    mapping (Gray or natural binary).

    With ``norm=True`` (the default) the alphabet is rescaled to **unit
    average symbol energy**, symbols being assumed equiprobable:

    .. math::

        a_m \leftarrow \frac{a_m}{\sqrt{E_s}}, \qquad
        E_s = \frac{1}{M}\sum_{m=0}^{M-1} \left|a_m\right|^2

    so that :math:`\mathbb{E}\left[|x[n]|^2\right] = 1`. This is the
    convention the rest of the library assumes: with :math:`E_s = 1` an
    absolute noise variance :math:`\sigma^2` *is* :math:`N_0`, and the
    MMSE regularizer :math:`\sigma^2 \mathbf{I}` of
    :func:`mmse_estimator` needs no rescaling. With ``norm=False`` the
    raw grid is returned instead (:math:`E_s = 10` for 16-QAM, for
    instance) and those expressions must be rescaled by :math:`E_s`.

    Parameters
    ----------
    modulation : str
        Modulation family: ``"PAM"``, ``"PSK"`` or ``"QAM"``.
    order : int
        Modulation order :math:`M`, i.e. :math:`M = 2^k` symbols carrying
        :math:`k = \log_2(M)` bits each. Tabulated values are 4, 16, 32,
        64, 128 and 256.
    type : str, optional
        Bit-to-symbol mapping: ``"gray"`` or ``"bin"`` (natural binary).
        Default is ``"gray"``.
    norm : bool, optional
        If True, normalize the alphabet to unit average symbol energy
        :math:`E_s = 1`. Default is True.

    Returns
    -------
    np.ndarray
        Complex alphabet :math:`\mathcal{A}` of length :math:`M`, indexed
        by the symbol index :math:`s`.

    Notes
    -----
    The tables are CSV files of the ``data`` subdirectory of this module,
    named ``<modulation>_<order>_<type>.csv``, with one row per symbol
    (index, real part, imaginary part) sorted by increasing index.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 3 (memoryless modulation methods and Gray
    mapping).

    Examples
    --------
    >>> alphabet = get_alphabet("QAM", 4)
    >>> print(np.round(alphabet, 4))
    [-0.7071+0.7071j -0.7071-0.7071j  0.7071+0.7071j  0.7071-0.7071j]
    >>> print(round(float(np.mean(np.abs(alphabet)**2)), 6))
    1.0
    >>> print(np.round(get_alphabet("QAM", 16, norm=False)[:4], 1))
    [-3.+3.j -3.+1.j -3.-3.j -3.-1.j]
    """
    alphabet = _construct_alphabet(modulation, order, type)

    if norm:
        alphabet = alphabet/np.sqrt(np.mean(np.abs(alphabet)**2))

    return alphabet


def _gray_labels(n_bits: int) -> np.ndarray:
    """Reflected binary code on ``n_bits`` bits, in index order."""
    index = np.arange(2 ** n_bits)
    return index ^ (index >> 1)


def _pam_levels(order: int, mapping: str) -> np.ndarray:
    """Odd-integer levels of a PAM axis, indexed by label."""
    n_bits = int(np.log2(order))
    labels = _gray_labels(n_bits) if mapping == "gray" else np.arange(order)
    # position m carries label labels[m]; index the levels by label
    return (2 * np.arange(order) - (order - 1))[np.argsort(labels)]


def _cross_qam(order: int, mapping: str) -> np.ndarray:
    """Read a cross constellation from its table.

    32-QAM and 128-QAM are not square, so they are not the product of
    two PAM axes and there is no one-line construction to check. They
    are the only alphabets still tabulated.
    """
    directory = pathlib.Path(__file__).resolve().parent / "data"
    filename = directory / f"QAM_{order}_{mapping}.csv"
    if not filename.exists():
        raise ValueError(
            f"no QAM-{order} alphabet: square orders (4, 16, 64, 256, 1024, "
            f"...) are constructed, and the cross constellations 32 and 128 "
            f"are tabulated -- {order} is neither")
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return data[:, 1] + 1j*data[:, 2]


def _construct_alphabet(modulation: str, order: int, mapping: str) -> np.ndarray:
    r"""Build the raw (unnormalized) alphabet of a memoryless modulation.

    PSK is the unit circle, PAM the odd integers, and square QAM the
    product of two PAM axes with the quadrature axis **negated** -- the
    convention that puts the first symbol at the top left of the
    constellation. In every case the labelling is applied by indexing
    the geometric positions by their label, so ``mapping="gray"`` gives
    a genuine Gray code and ``"bin"`` natural binary.

    These formulas replace the CSV tables the library used to ship:
    they reproduce all thirty-two of them entry by entry, to the six
    decimals the files stored, and are exact where the files were
    rounded (``tests/core/test_alphabet.py`` pins that against the
    tables, which are kept in git history). Only the two cross
    constellations, 32-QAM and 128-QAM, remain tabulated.
    """
    if order < 2 or order & (order - 1):
        raise ValueError(
            f"expected a power-of-two modulation order of at least 2, got "
            f"{order}")
    if mapping not in ("gray", "bin"):
        raise ValueError(f"expected mapping 'gray' or 'bin', got {mapping!r}")

    if modulation == "PSK":
        n_bits = int(np.log2(order))
        labels = _gray_labels(n_bits) if mapping == "gray" else np.arange(order)
        return np.exp(2j * np.pi * np.arange(order) / order)[np.argsort(labels)]
    if modulation == "PAM":
        return _pam_levels(order, mapping).astype(complex)
    if modulation == "QAM":
        root = int(round(np.sqrt(order)))
        if root * root != order:
            return _cross_qam(order, mapping)
        levels = _pam_levels(root, mapping)
        # high bits select the in-phase level, low bits the quadrature
        # one, on a *negated* axis
        return (np.repeat(levels, root) - 1j * np.tile(levels, root)).astype(complex)
    raise ValueError(
        f"unknown modulation {modulation!r}; expected 'PSK', 'PAM' or 'QAM'")


def plot_alphabet(alphabet: np.ndarray, ax: Optional["Axes"] = None,
                  label: str = "alphabet", title: str = "Constellation",
                  **kwargs: Any) -> "Axes":
    r"""
    Plot the constellation diagram of a symbol alphabet.

    Each symbol :math:`a_m` of the alphabet
    :math:`\mathcal{A} = \{a_0, \ldots, a_{M-1}\}` is drawn as a point of
    the complex plane, its real part on the horizontal axis and its
    imaginary part on the vertical one. This is a plotting helper, not a
    processing block: it transforms no signal.

    Parameters
    ----------
    alphabet : np.ndarray
        Complex-valued symbol alphabet :math:`\mathcal{A}` to plot,
        typically returned by :func:`get_alphabet`.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If None, a new figure and axis are created.
    label : str, optional
        Label for the scatter plot. Default is ``"alphabet"``.
    title : str, optional
        Title of the plot. Default is ``"Constellation"``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot (decision D25).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> ax = plot_alphabet(get_alphabet("PSK", 4))
    >>> ax.get_xlabel()
    'real part'
    """
    import matplotlib.pyplot as plt  # local import (D36)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.real(alphabet), np.imag(alphabet), "o", label=label, **kwargs)
    ax.set_xlabel("real part")
    ax.set_ylabel("imag part")
    ax.set_title(title)
    return ax


@dataclass(frozen=True, slots=True)
class Constellation:
    r"""A modulation alphabet, and everything that is a property of it.

    Signal Model
    ------------
    The alphabet :func:`get_alphabet` returns, plus the facts that always
    travel with it. A memoryless modulation is a set
    :math:`\mathcal{A} = \{a_0, \ldots, a_{M-1}\}`, and everything a
    study asks of it -- how many bits it carries, what its average energy
    is, how close its nearest neighbours are, what error rate it reaches
    at a given SNR, what rate it can carry -- is determined by the family
    and the order :math:`M`. Passing those two around separately is how a
    page ends up drawing the closed form of a 16-QAM under the
    measurement of a 64-QAM, which nothing checks.

    The object is a drop-in for the array it wraps: ``np.asarray``
    returns the alphabet, so every block and every function that takes
    one keeps working.

    Parameters
    ----------
    family : str
        Modulation family: ``"PAM"``, ``"PSK"`` or ``"QAM"``.
    order : int
        Modulation order :math:`M`.
    labelling : str, optional, keyword-only
        Bit-to-symbol mapping, ``"gray"`` (default) or ``"binary"``.
    norm : bool, optional, keyword-only
        Rescale to unit average symbol energy. Default True, which is the
        convention the rest of the library assumes.

    Attributes
    ----------
    alphabet : np.ndarray
        The symbols :math:`a_m`, in labelling order.
    bits_per_symbol : int
        :math:`k = \log_2 M`.
    energy : float
        Average symbol energy :math:`E_s`, over an equiprobable input.
    min_distance : float
        Smallest distance between two symbols, which sets the high-SNR
        error rate.

    Raises
    ------
    ValueError
        If the family, the order or the labelling is unknown. The check
        is :func:`get_alphabet`'s, made once here instead of at every
        call site.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 4.

    Examples
    --------
    >>> qam = Constellation("QAM", 16)
    >>> qam.bits_per_symbol, round(qam.energy, 6), round(qam.min_distance, 4)
    (4, 1.0, 0.6325)
    >>> np.asarray(qam).shape            # a drop-in for the array
    (16,)
    >>> print(f"{qam.metrics(10.0)['ser']:.4e}")      # at Eb/N0 = 10 dB
    7.0043e-03
    """

    family: str
    order: int
    labelling: str = field(default="gray", kw_only=True)
    norm: bool = field(default=True, kw_only=True)
    # derived at construction: the point of the object is that these
    # cannot disagree with the alphabet they describe
    alphabet: np.ndarray = field(init=False, repr=False)
    bits_per_symbol: int = field(init=False)
    energy: float = field(init=False, repr=False)
    min_distance: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        alphabet = get_alphabet(self.family, self.order, self.labelling,
                                self.norm)
        distances = np.abs(alphabet[:, None] - alphabet[None, :])
        np.fill_diagonal(distances, np.inf)
        # frozen dataclass: the derived fields are set once, here
        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "bits_per_symbol", int(np.log2(self.order)))
        object.__setattr__(self, "energy",
                           float(np.mean(np.abs(alphabet) ** 2)))
        object.__setattr__(self, "min_distance", float(np.min(distances)))

    def __array__(self, dtype: Optional[npt.DTypeLike] = None,
                  copy: Optional[bool] = None) -> np.ndarray:
        """Let ``np.asarray`` see the alphabet, so blocks take the object."""
        if dtype is None:
            return self.alphabet.copy() if copy else self.alphabet
        return self.alphabet.astype(dtype, copy=bool(copy))

    def __len__(self) -> int:
        return self.order

    def info(self) -> dict[str, Any]:
        """What the constellation is, as a dictionary ready to print.

        Returns
        -------
        dict
            ``family``, ``order``, ``labelling``, ``bits_per_symbol``,
            ``energy``, ``min_distance``, and ``papr_dB`` -- the last
            being the peak-to-average power ratio of the *constellation*,
            a lower bound on what the shaped waveform will show.

        Examples
        --------
        >>> print(Constellation("PSK", 8).info()["papr_dB"])
        0.0
        """
        peak = float(np.max(np.abs(self.alphabet) ** 2))
        return {"family": self.family,
                "order": self.order,
                "labelling": self.labelling,
                "bits_per_symbol": self.bits_per_symbol,
                "energy": self.energy,
                "min_distance": self.min_distance,
                "papr_dB": round(float(10 * np.log10(peak / self.energy)), 6)}

    def plot(self, ax: Optional["Axes"] = None, **kwargs: Any) -> "Axes":
        """Draw the constellation diagram (decision D25).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis to draw on. If None, a new figure and axis are created.
        **kwargs
            Forwarded to :func:`plot_alphabet`.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the plot.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> Constellation("QAM", 16).plot().get_title()
        '16-QAM'
        """
        kwargs.setdefault("title", f"{self.order}-{self.family}")
        return plot_alphabet(self.alphabet, ax=ax, **kwargs)

    def metrics(self, snr_dB: npt.ArrayLike, *, per: str = "bit",
                channel: str = "awgn", diversity: int = 1,
                metrics: Tuple[str, ...] = ("ser", "ber"),
                px: Optional[np.ndarray] = None,
                ) -> "dict[str, np.ndarray | float]":
        r"""What this constellation is worth at a given SNR.

        Signal Model
        ------------
        A front-end to the four quantities a page draws against a
        measurement, all of them functions of the same family, order and
        SNR:

        * ``"ser"`` and ``"ber"``, the closed-form error rates of
          :func:`~comnumpy.core.metrics.compute_metric_awgn_theo` or
          :func:`~comnumpy.core.metrics.compute_metric_rayleigh_theo`;
        * ``"mi"`` and ``"gmi"``, the mutual information and the
          bit-interleaved rate of
          :func:`~comnumpy.core.capacity.constellation_capacity` and
          :func:`~comnumpy.core.capacity.bicm_capacity`.

        The two families are parameterized differently by convention: the
        error rates against :math:`E_b/N_0`, the rates against the SNR
        per symbol, which is :math:`k` times larger. The object knows
        :math:`k`, so ``per=`` says which of the two is being passed and
        the conversion happens here rather than at the call site -- get
        it wrong and the curve moves by :math:`10\log_{10} k` dB, with
        nothing to signal it.

        ``"mi"`` and ``"gmi"`` are quadratures over :math:`M^2` points,
        not closed forms, so they are not in the default and cost seconds
        rather than microseconds on a large constellation.

        Parameters
        ----------
        snr_dB : float or np.ndarray
            Signal-to-noise ratio in dB -- the quantity a chain is swept
            over.
        per : str, optional, keyword-only
            ``"bit"`` (default) if ``snr_dB`` is :math:`E_b/N_0`,
            ``"symbol"`` if it is :math:`E_s/N_0`.
        channel : str, optional, keyword-only
            ``"awgn"`` (default) or ``"rayleigh"``, for the error rates.
            The rates below are AWGN only.
        diversity : int, optional, keyword-only
            Number of combined branches, for ``channel="rayleigh"``.
        metrics : tuple of str, optional, keyword-only
            Which quantities to return, among ``"ser"``, ``"ber"``,
            ``"mi"`` and ``"gmi"``. Default ``("ser", "ber")``.
        px : np.ndarray, optional, keyword-only
            Symbol distribution, for a shaped input. The constellation is
            rescaled to unit energy *under this law* before the rates are
            computed -- a shaped input compared as it stands is simply a
            quieter one. Not available for the error rates, whose closed
            forms assume an equiprobable input.

        Returns
        -------
        dict of str to float or np.ndarray
            One entry per requested metric, in the order requested, each
            with the shape of ``snr_dB``.

        Raises
        ------
        ValueError
            If ``per``, ``channel`` or a metric name is unknown, if a rate
            is asked of a fading channel, or if ``px`` is given together
            with an error rate.

        References
        ----------
        J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
        McGraw-Hill, 2008, Section 4.3; G. Ungerboeck, "Channel coding
        with multilevel/phase signals", IEEE Trans. Inf. Theory 28(1),
        1982 (the rate a constellation carries).

        Examples
        --------
        >>> qpsk = Constellation("PSK", 4)
        >>> theory = qpsk.metrics(np.array([4.0, 8.0]))
        >>> print(np.array2string(theory["ber"], precision=6))
        [0.012423 0.000191]
        >>> rate = qpsk.metrics(20.0, per="symbol", metrics=("mi",))
        >>> print(round(float(rate["mi"]), 3))
        2.0
        """
        # local imports: comnumpy.core.metrics imports this module
        from .capacity import bicm_capacity, constellation_capacity
        from .metrics import (compute_metric_awgn_theo,
                              compute_metric_rayleigh_theo)

        unknown = []
        for name in metrics:
            if name not in ("ser", "ber", "mi", "gmi"):
                unknown.append(name)
        if unknown:
            raise ValueError(
                f"Constellation.metrics: unknown metric(s) {unknown}; "
                f"expected any of 'ser', 'ber', 'mi', 'gmi'.")
        if per not in ("bit", "symbol"):
            raise ValueError(
                f"Constellation.metrics: per={per!r}; expected 'bit' if "
                f"snr_dB is Eb/N0, or 'symbol' if it is Es/N0. The two "
                f"differ by {10 * np.log10(self.bits_per_symbol):.2f} dB "
                f"on this constellation, so the choice is not cosmetic.")
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(
                f"Constellation.metrics: channel={channel!r}; expected "
                f"'awgn' or 'rayleigh'.")

        linear = 10 ** (np.asarray(snr_dB, dtype=float) / 10)
        if per == "bit":
            per_bit = linear
            per_symbol = linear * self.bits_per_symbol
        else:
            per_bit = linear / self.bits_per_symbol
            per_symbol = linear

        wants_rates = ("mi" in metrics) or ("gmi" in metrics)
        wants_errors = ("ser" in metrics) or ("ber" in metrics)
        if wants_rates and channel == "rayleigh":
            raise ValueError(
                "Constellation.metrics: 'mi' and 'gmi' are rates of the "
                "AWGN channel; over fading they would have to be averaged "
                "over the fading law, which this front-end does not do. "
                "Ask for them with channel='awgn'.")
        if wants_errors and px is not None:
            raise ValueError(
                "Constellation.metrics: the closed-form error rates assume "
                "an equiprobable input, so px= cannot apply to 'ser' or "
                "'ber'. Ask for the rates alone, or drop px.")

        found: "dict[str, np.ndarray | float]" = {}
        if wants_errors:
            if channel == "awgn":
                found.update(compute_metric_awgn_theo(
                    self.family, self.order, per_bit))
            else:
                found.update(compute_metric_rayleigh_theo(
                    self.family, self.order, per_bit, diversity=diversity))
        if wants_rates:
            alphabet = self.alphabet
            if px is not None:
                law = np.asarray(px, dtype=float)
                alphabet = alphabet / np.sqrt(
                    float(law @ np.abs(alphabet) ** 2))
            if "mi" in metrics:
                found["mi"] = constellation_capacity(alphabet, per_symbol,
                                                     px=px)
            if "gmi" in metrics:
                found["gmi"] = bicm_capacity(alphabet, per_symbol, px=px)

        ordered: "dict[str, np.ndarray | float]" = {}
        for name in metrics:
            ordered[name] = found[name]
        return ordered



def sym_2_bin(sym: np.ndarray, width: int = 4) -> np.ndarray:
    r"""
    Expand symbol indices into their binary representation, MSB first.

    Signal Model
    ------------
    Each symbol index :math:`s[n] \in \{0, \ldots, M-1\}` is written in
    base 2 on :math:`k` bits (:math:`k =` ``width``, with
    :math:`M \leq 2^k`):

    .. math::

        s[n] = \sum_{i=0}^{k-1} b[nk + i] \, 2^{\,k-1-i},
        \qquad b[\cdot] \in \{0, 1\}

    The bits of a symbol are emitted most-significant first and the
    symbols are concatenated, so a sequence of :math:`N` indices produces
    :math:`kN` bits. This is the same ordering as the bit table of
    :class:`~comnumpy.core.mappers.SymbolDemapper`, which is what makes
    the BER of :mod:`comnumpy.core.metrics` comparable across blocks.

    Axes: *element-wise* -- each index is expanded independently; the
    input is read as a flat sequence and the output is 1-D of length
    :math:`kN`.

    Parameters
    ----------
    sym : array-like
        Sequence of :math:`N` integer symbol indices :math:`s[n]`.
    width : int, optional
        Number of bits per symbol :math:`k`. Default is 4.

    Returns
    -------
    np.ndarray
        1-D integer array of :math:`kN` bits (0s and 1s).

    Notes
    -----
    Each symbol is represented by exactly ``width`` bits: a shorter
    binary representation is left-padded with zeros. An index larger than
    :math:`2^k - 1` is *not* rejected -- ``numpy.binary_repr`` then emits
    more than ``width`` bits and the output loses its block structure.

    Examples
    --------
    >>> print(sym_2_bin([0, 5, 9], width=4))
    [0 0 0 0 0 1 0 1 1 0 0 1]
    >>> print(sym_2_bin([0, 3], width=2))
    [0 0 1 1]
    """

    sym = np.asarray(sym)
    weights = np.arange(width - 1, -1, -1)          # MSB first
    bits = (sym[..., None] >> weights) & 1
    return bits.reshape(sym.shape[:-1] + (sym.shape[-1] * width,)) \
        if sym.ndim else bits.reshape(width)


def hard_projector(z: np.ndarray,
                   alphabet: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Project symbols onto the nearest constellation point (hard decision).

    Signal Model
    ------------
    Given an equalized observation :math:`z[n]` and the alphabet
    :math:`\mathcal{A} = \{a_0, \ldots, a_{M-1}\}`, the decision is the
    minimum-distance (nearest-neighbour) rule

    .. math::

        \hat{s}[n] = \arg\min_{m \in \{0, \ldots, M-1\}}
        \left| z[n] - a_m \right|^2,
        \qquad \hat{x}[n] = a_{\hat{s}[n]}

    For the AWGN model :math:`z[n] = x[n] + b[n]` with
    :math:`b[n] \sim \mathcal{CN}(0, \sigma^2)` and equiprobable symbols,
    this rule is the maximum-likelihood (hence MAP) detector: the
    likelihood decreases monotonically with the Euclidean distance, so
    :math:`\sigma^2` does not enter the decision.

    Axes: *element-wise* -- the search is carried out over the alphabet
    on an axis appended internally, so both outputs keep the shape of
    ``z``. Ties are broken by ``numpy.argmin`` (smallest index wins).

    Parameters
    ----------
    z : np.ndarray
        Observation :math:`z[n]`, of any shape.
    alphabet : np.ndarray
        1-D constellation alphabet :math:`\mathcal{A}` of length
        :math:`M`.

    Returns
    -------
    s : np.ndarray
        Integer indices :math:`\hat{s}[n]` of the nearest constellation
        points, with the shape of ``z``.
    x : np.ndarray
        Nearest constellation symbols :math:`\hat{x}[n] = a_{\hat{s}[n]}`,
        with the shape of ``z``.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 4 (optimum receivers for the AWGN
    channel).

    Examples
    --------
    >>> alphabet = get_alphabet("QAM", 4)
    >>> z = np.array([0.9+1.1j, -0.2-0.8j])
    >>> s, x = hard_projector(z, alphabet)
    >>> print(s)
    [2 1]
    >>> print(np.round(x, 4))
    [ 0.7071+0.7071j -0.7071-0.7071j]
    """
    # accept anything array-like, a Constellation included
    alphabet = np.asarray(alphabet)
    error = np.abs(z[..., np.newaxis] - alphabet)**2
    index = np.argmin(error, axis=-1)
    s = index.astype(int)
    x = alphabet[s]
    return s, x


def soft_projector(z: np.ndarray, alphabet: np.ndarray, sigma2: float,
                   kernel: Optional[np.ndarray] = None) -> np.ndarray:
    r"""
    Compute the soft (MMSE) symbol estimate under a Gaussian noise model.

    Signal Model
    ------------
    For the scalar observation :math:`z[n] = x[n] + b[n]` with
    :math:`b[n] \sim \mathcal{CN}(0, \sigma^2)` and equiprobable symbols
    drawn from :math:`\mathcal{A} = \{a_0, \ldots, a_{M-1}\}`, the MMSE
    estimate is the posterior mean

    .. math::

        \hat{x}[n] = \mathbb{E}\left[x[n] \mid z[n]\right] =
        \frac{\displaystyle\sum_{m=0}^{M-1} a_m \,
        e^{-|z[n]-a_m|^2/\sigma^2}}
        {\displaystyle\sum_{m=0}^{M-1} e^{-|z[n]-a_m|^2/\sigma^2}}

    The exponent carries :math:`\sigma^2` and not :math:`2\sigma^2`
    because the noise is circularly symmetric complex: its density is
    :math:`(\pi\sigma^2)^{-1} e^{-|b|^2/\sigma^2}`, :math:`\sigma^2`
    being the *total* variance of the two quadratures.

    Passing a ``kernel`` :math:`g_{m}[n]` replaces :math:`a_m` in the
    numerator, which turns the same weights into any posterior average

    .. math::

        \frac{\sum_m g_{m}[n] \, e^{-|z[n]-a_m|^2/\sigma^2}}
             {\sum_m e^{-|z[n]-a_m|^2/\sigma^2}}
        = \mathbb{E}\left[g \mid z[n]\right]

    The AMP detectors of :mod:`comnumpy.mimo.detectors` use
    :math:`g_{m}[n] = |a_m - \hat{x}[n]|^2` to obtain the posterior
    variance that drives their state evolution.

    Axes: *element-wise* -- each observation is processed independently;
    the input is flattened, so the output is 1-D of size ``z.size``.

    Parameters
    ----------
    z : np.ndarray
        Observation :math:`z[n]`, flattened internally.
    alphabet : np.ndarray
        1-D constellation alphabet :math:`\mathcal{A}` of length
        :math:`M`.
    sigma2 : float
        Noise variance :math:`\sigma^2` (total, both quadratures). Only
        its real part is used. Large values flatten the weights towards
        the alphabet mean, small values recover the hard decision of
        :func:`hard_projector`.
    kernel : np.ndarray or None, optional
        Values :math:`g_{m}[n]` averaged in place of :math:`a_m`, of
        shape ``(z.size, M)`` (or broadcastable to it). If None, the
        alphabet itself is used and the posterior mean is returned.

    Returns
    -------
    np.ndarray
        Soft estimates :math:`\hat{x}[n]`, 1-D of size ``z.size``.

    References
    ----------
    * S. M. Kay, *Fundamentals of Statistical Signal Processing:
      Estimation Theory*, Prentice Hall, 1993 (the MMSE estimator is the
      posterior mean).
    * C. Jeon, R. Ghods, A. Maleki and C. Studer, "Optimality of large
      MIMO detection via approximate message passing," Proc. IEEE Int.
      Symp. Information Theory (ISIT), 2015, pp. 1227-1231 (the ``F``
      function and its variance kernel).

    Examples
    --------
    >>> alphabet = get_alphabet("QAM", 4)
    >>> z = np.array([0.9+1.1j, -0.2-0.8j])
    >>> print(np.round(soft_projector(z, alphabet, 0.1), 4))
    [ 0.7071+0.7071j -0.7022-0.7071j]
    >>> print(np.round(soft_projector(z, alphabet, 100.0), 4))
    [ 0.009+0.011j -0.002-0.008j]
    """
    # accept anything array-like, a Constellation included
    alphabet = np.asarray(alphabet).reshape(1, -1)
    z = z.reshape(-1, 1)

    term1 = np.exp(-(1 / np.real(sigma2)) * np.abs(alphabet - z) ** 2)

    if kernel is None:
        kernel = alphabet

    num = np.sum(kernel * term1, axis=1)
    den = np.sum(term1, axis=1)
    return num / den


def esn0_to_snr_dB(esn0_dB: npt.ArrayLike,
                   oversampling: int = 1) -> npt.NDArray[np.float64]:
    r"""
    Convert a symbol-energy-to-noise ratio :math:`E_s/N_0` (dB) into the SNR (dB) expected by ``AWGN``.

    Signal Model
    ------------
    Three quantities must not be confused. :math:`E_s` is the energy of
    one constellation symbol, :math:`E_b` the energy of one information
    bit and :math:`N_0` the noise power spectral density; the SNR of
    :class:`~comnumpy.core.channels.AWGN` is a *per-sample* power ratio

    .. math::

        \mathrm{SNR} = \frac{P_x}{\sigma^2}, \qquad
        P_x = \mathbb{E}\left[|x[n]|^2\right]

    At one sample per symbol :math:`P_x = E_s` and :math:`\sigma^2 = N_0`,
    so the two quantities coincide. Oversampling by :math:`L` samples per
    symbol multiplies the noise bandwidth -- hence the per-sample noise
    variance :math:`\sigma^2 = L N_0` -- while an energy-preserving pulse
    shaping leaves the measured signal power at :math:`E_s`, so

    .. math::

        \mathrm{SNR_{dB}} = E_s/N_0\big|_{dB} - 10\log_{10}(L)

    Axes: *element-wise* -- a pure scalar conversion, applied pointwise
    to an array of operating points when sweeping.

    Parameters
    ----------
    esn0_dB : float or np.ndarray
        Symbol-energy-to-noise-spectral-density ratio
        :math:`E_s/N_0\big|_{dB}`, in dB.
    oversampling : int, optional
        Number of samples per symbol :math:`L`. Default is 1.

    Returns
    -------
    float
        The SNR in dB, relative to the measured signal power.

    See Also
    --------
    ebn0_to_snr_dB : same conversion from the energy per information bit.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 4 (energy-per-symbol and
    energy-per-bit normalizations).

    Examples
    --------
    >>> print(float(esn0_to_snr_dB(10)))
    10.0
    >>> print(round(float(esn0_to_snr_dB(10, oversampling=4)), 4))
    3.9794
    """
    return esn0_dB - 10 * np.log10(oversampling)


def ebn0_to_snr_dB(ebn0_dB: npt.ArrayLike, bits_per_symbol: int,
                   code_rate: float = 1.0,
                   oversampling: int = 1) -> npt.NDArray[np.float64]:
    r"""
    Convert a bit-energy-to-noise ratio :math:`E_b/N_0` (dB) into the SNR (dB) expected by ``AWGN``.

    Signal Model
    ------------
    One constellation symbol carries :math:`k = \log_2(M)` channel bits,
    of which :math:`kR` are information bits when a code of rate
    :math:`R` is used. The energy budget of a symbol is therefore shared
    by :math:`kR` information bits:

    .. math::

        E_s = k R \, E_b

    Combined with the per-sample SNR of
    :class:`~comnumpy.core.channels.AWGN` (see
    :func:`esn0_to_snr_dB`, where :math:`\sigma^2 = L N_0` for
    :math:`L` samples per symbol), this gives

    .. math::

        \mathrm{SNR_{dB}} = E_b/N_0\big|_{dB} + 10\log_{10}(k R)
        - 10\log_{10}(L)

    The conversion requires chain-level knowledge (:math:`k`, :math:`R`,
    :math:`L`), which is why it lives here and not inside the ``AWGN``
    block (decision D41). Comparing modulations at equal
    :math:`E_b/N_0` is the fair comparison -- at equal :math:`E_s/N_0`,
    a dense constellation is unduly favoured because it spends the same
    energy on more bits.

    Axes: *element-wise* -- a pure scalar conversion, applied pointwise
    to an array of operating points when sweeping.

    Parameters
    ----------
    ebn0_dB : float or np.ndarray
        Information-bit-energy-to-noise-spectral-density ratio
        :math:`E_b/N_0\big|_{dB}`, in dB.
    bits_per_symbol : int
        Number of bits carried by one constellation symbol
        :math:`k = \log_2(M)` (e.g. 4 for 16-QAM).
    code_rate : float, optional
        FEC code rate :math:`R \in (0, 1]`. Default is 1.0 (uncoded,
        :math:`E_s = k E_b`).
    oversampling : int, optional
        Number of samples per symbol :math:`L`. Default is 1.

    Returns
    -------
    float
        The SNR in dB, relative to the measured signal power.

    See Also
    --------
    esn0_to_snr_dB : same conversion from the energy per symbol.

    References
    ----------
    J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
    McGraw-Hill, 2008, Chapter 4 (energy-per-symbol and
    energy-per-bit normalizations).

    Examples
    --------
    >>> print(round(float(ebn0_to_snr_dB(10, bits_per_symbol=4)), 4))
    16.0206
    >>> print(round(float(ebn0_to_snr_dB(10, bits_per_symbol=2, code_rate=0.5)), 4))
    10.0
    """
    return ebn0_dB + 10 * np.log10(bits_per_symbol * code_rate) - 10 * np.log10(oversampling)


def zf_estimator(Y: np.ndarray, H: np.ndarray) -> np.ndarray:
    r"""
    Perform Zero Forcing (ZF) linear equalization using the channel matrix pseudoinverse.

    Signal Model
    ------------
    For the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H}\mathbf{x}[n] + \mathbf{b}[n],
        \qquad \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    with :math:`\mathbf{H}` of size :math:`N_r \times N_t`, the ZF
    equalizer applies the Moore-Penrose pseudoinverse:

    .. math::

        \mathbf{z}[n] = \mathbf{H}^{\dagger}\mathbf{y}[n]
        = \left(\mathbf{H}^H\mathbf{H}\right)^{-1}\mathbf{H}^H
        \mathbf{y}[n]
        = \mathbf{x}[n] + \mathbf{H}^{\dagger}\mathbf{b}[n]

    the closed form on the right holding when :math:`\mathbf{H}` has full
    column rank (:math:`N_r \geq N_t`). Interference between streams is
    cancelled exactly, at the price of noise enhancement: the residual
    noise has covariance
    :math:`\sigma^2\left(\mathbf{H}^H\mathbf{H}\right)^{-1}`, which blows
    up when :math:`\mathbf{H}` is ill-conditioned. :func:`mmse_estimator`
    trades a residual bias for a bounded noise gain.

    Axes: *declared axis* -- expects the MIMO layout ``(..., ant, N)``:
    :math:`N_r` receive antennas on axis -2 of ``Y``, time on axis -1,
    with ``H`` a 2-D ``(N_r, N_t)`` matrix. The output carries
    :math:`N_t` streams on axis -2.

    Parameters
    ----------
    Y : np.ndarray
        Received samples :math:`\mathbf{y}[n]` stacked column-wise, of
        shape ``(..., N_r, N)``.
    H : np.ndarray
        Channel matrix :math:`\mathbf{H}` of shape ``(N_r, N_t)``.

    Returns
    -------
    np.ndarray
        Equalized signal :math:`\mathbf{z}[n]` of shape
        ``(..., N_t, N)``, still to be mapped onto the alphabet by
        :func:`hard_projector`.

    References
    ----------
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.
    * J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
      McGraw-Hill, 2008, Chapter 9 (linear equalization).

    Examples
    --------
    >>> H = np.array([[1.0, 0.5], [0.0, 2.0]])
    >>> X = np.array([[1.0, -1.0], [1.0, 1.0]])
    >>> Y = np.matmul(H, X)
    >>> print(np.round(zf_estimator(Y, H), 4))
    [[ 1. -1.]
     [ 1.  1.]]
    """
    A = LA.pinv(H)
    Z_est = np.matmul(A, Y)
    return Z_est


def mmse_estimator(Y: np.ndarray, H: np.ndarray, sigma2: float) -> np.ndarray:
    r"""
    Perform Minimum Mean Square Error (MMSE) linear equalization.

    Signal Model
    ------------
    For the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H}\mathbf{x}[n] + \mathbf{b}[n],
        \qquad \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    the linear estimator minimizing
    :math:`\mathbb{E}\left[\|\mathbf{z}[n]-\mathbf{x}[n]\|^2\right]` is

    .. math::

        \mathbf{z}[n] = \left(\mathbf{H}^H\mathbf{H}
        + \sigma^2 \mathbf{I}_{N_t}\right)^{-1}
        \mathbf{H}^H\mathbf{y}[n]

    The regularization term is the noise-to-signal ratio
    :math:`(\sigma^2/E_s)\,\mathbf{I}_{N_t}`; it reduces to
    :math:`\sigma^2 \mathbf{I}_{N_t}`, as implemented, because the
    library normalizes the constellation to unit symbol energy
    :math:`E_s = 1` (see :func:`get_alphabet`) and the streams are
    assumed uncorrelated,
    :math:`\mathbb{E}[\mathbf{x}[n]\mathbf{x}^H[n]] = \mathbf{I}_{N_t}`.
    Two limits are worth remembering: :math:`\sigma^2 \to 0` gives back
    the zero-forcing solution of :func:`zf_estimator`, while a large
    :math:`\sigma^2` shrinks the estimate towards zero rather than
    amplifying the noise. The MMSE output is biased, which is why it is
    followed by a decision on the alphabet, not used as an unbiased
    estimate.

    Axes: *declared axis* -- expects the MIMO layout ``(..., ant, N)``:
    :math:`N_r` receive antennas on axis -2 of ``Y``, time on axis -1,
    with ``H`` a 2-D ``(N_r, N_t)`` matrix. The output carries
    :math:`N_t` streams on axis -2.

    Parameters
    ----------
    Y : np.ndarray
        Received samples :math:`\mathbf{y}[n]` stacked column-wise, of
        shape ``(..., N_r, N)``.
    H : np.ndarray
        Channel matrix :math:`\mathbf{H}` of shape ``(N_r, N_t)``.
    sigma2 : float
        Noise variance :math:`\sigma^2` per receive antenna, in the same
        power units as the unit-energy constellation.

    Returns
    -------
    np.ndarray
        Equalized signal :math:`\mathbf{z}[n]` of shape
        ``(..., N_t, N)``, still to be mapped onto the alphabet by
        :func:`hard_projector`.

    References
    ----------
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.
    * J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed.,
      McGraw-Hill, 2008, Chapter 9 (MMSE linear equalization).

    Examples
    --------
    >>> H = np.array([[1.0, 0.5], [0.0, 2.0]])
    >>> X = np.array([[1.0, -1.0], [1.0, 1.0]])
    >>> Y = np.matmul(H, X)
    >>> print(np.round(mmse_estimator(Y, H, 0.1), 4))
    [[ 0.9151 -0.8931]
     [ 0.9868  0.9647]]
    """
    _, N_t = H.shape
    H_H = np.conjugate(np.transpose(H))
    A = np.matmul(H_H, H) + sigma2 * np.eye(N_t)
    Z_est = LA.solve(A, np.matmul(H_H, Y))
    return Z_est
