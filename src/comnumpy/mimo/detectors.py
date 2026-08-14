import numpy as np
import numpy.linalg as LA
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional
from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector, soft_projector, zf_estimator, mmse_estimator

__all__ = [
    "MaximumLikelihoodDetector", "SphereDecoder", "LinearDetector",
    "OrderedSuccessiveInterferenceCancellationDetector",
    "ApproximateMessagePassingDetector",
    "OrthogonalApproximateMessagePassingDetector",
]


# The exhaustive search is blocked on both axes so that its cost matrix
# stays a few megabytes whatever the constellation: 1024 samples by 2048
# candidates is 16 MB of float64, against 34 GB for 16-QAM on four
# antennas taken in one piece.
_ML_BLOCK = 1024
_ML_CANDIDATES = 2048


def _required_channel(H: Optional[np.ndarray], block: str) -> np.ndarray:
    """The channel matrix, or a message saying how to provide it.

    Returning the value rather than only checking it is what lets the
    caller -- and the type checker -- treat it as an array from there on.
    """
    if H is None:
        raise ValueError(
            f"{block} needs the channel matrix H and none was set. Got "
            f"H=None, expected an (N_r, N_t) array. Pass it at "
            f"construction ({block}(alphabet, H=H)) or assign "
            f"detector.H once the channel is known.")
    return np.asarray(H)


def _detect_stacked(detect: "Callable[[np.ndarray, np.ndarray], np.ndarray]",
                    H: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """One detection per leading index of a stacked channel (D51).

    The search algorithms are per-draw by nature -- an enumeration, a
    tree, a cancellation order all depend on the one matrix in front of
    them -- so the batch is an explicit loop: channel ``k`` decides
    frame ``k``, and no state leaks between draws. Convenience and
    correctness, not a speedup.
    """
    from comnumpy.exceptions import ShapeError
    if Y.ndim < 2 or Y.shape[:-2] != H.shape[:-2]:
        raise ShapeError(
            f"a stacked channel with leading shape {H.shape[:-2]} "
            f"decides one frame per draw and needs observations "
            f"(..., N_r, N) with the same leading shape, got {Y.shape}")
    S = np.empty(H.shape[:-2] + (H.shape[-1], Y.shape[-1]), dtype=int)
    for index in np.ndindex(*H.shape[:-2]):
        S[index] = detect(H[index], Y[index])
    return S


def _required_noise_variance(sigma2: Optional[float], block: str) -> float:
    """Same, for the noise variance the MMSE-flavoured detectors need."""
    if sigma2 is None:
        raise ValueError(
            f"{block} needs the noise variance sigma2 and none was set. "
            f"Got sigma2=None, expected a positive number. Pass it at "
            f"construction ({block}(alphabet, H=H, sigma2=...)); the "
            f"variance applied by AWGN is available as its sigma2_.")
    if sigma2 <= 0:
        raise ValueError(
            f"{block} needs a positive noise variance, got sigma2="
            f"{sigma2}. A zero or negative variance has no MMSE "
            f"solution; use method='zf' for the noiseless limit.")
    return float(sigma2)


@dataclass(slots=True)
class MaximumLikelihoodDetector(Processor):
    r"""Maximum-likelihood (ML) MIMO detector for white Gaussian noise.

    Signal Model
    ------------
    The detector assumes the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n]

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t`, :math:`\mathbf{x}[n] \in \mathcal{M}^{N_t}`
    the transmitted vector drawn from the constellation
    :math:`\mathcal{M}` and :math:`\mathbf{b}[n]` a circular white
    Gaussian noise. The ML decision under this model is the exhaustive
    search

    .. math::

        \widehat{\mathbf{x}}[n] = \arg \min_{\mathbf{x} \in \mathcal{M}^{N_t}}
        \|\mathbf{y}[n] - \mathbf{H}\mathbf{x}\|^2

    and the detector returns the integer indices of
    :math:`\widehat{\mathbf{x}}[n]` in the alphabet
    (:math:`\widehat{\mathbf{x}}[n]` = ``alphabet[S_[:, n]]``).

    .. WARNING::

        The exhaustive search evaluates :math:`|\mathcal{M}|^{N_t}`
        candidates and becomes expensive for a large number of transmit
        antennas or a high-order constellation.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2. ``H`` may be a stack ``(K, N_r, N_t)`` against ``Y`` of
    shape ``(K, N_r, N)``: channel k decides frame k, through an
    internal per-draw loop -- convenience, not a speedup (D51).

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    name : str, optional, keyword-only
        Name of the detector. Default is ``"ML Detector"``.

    Attributes
    ----------
    S_ : np.ndarray
        Decisions of the last call, as alphabet indices
        (data-dependent, hence the trailing underscore, D23).

    Raises
    ------
    ValueError
        If ``H`` is not set when the detector is called.

    References
    ----------
    * E. G. Larsson, P. Stoica and G. Ganesan, *Space-Time Block Coding
      for Wireless Communications*, Cambridge University Press, 2003.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> H = np.eye(2)
    >>> Y = np.array([[0.9, -1.1], [-0.8, 1.2]])
    >>> MaximumLikelihoodDetector(alphabet, H=H)(Y)
    array([[1, 0],
           [0, 1]])
    """

    alphabet: np.ndarray
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    name: str = field(default="ML Detector", kw_only=True)
    # estimated state (D23), declared for slots (D40a)
    S_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    def __post_init__(self):
        # accept anything array-like, a Constellation included
        self.alphabet = np.asarray(self.alphabet)

    def get_nb_candidates(self) -> int:
        H = _required_channel(self.H, type(self).__name__)
        return len(self.alphabet) ** H.shape[-1]

    def get_candidates(self, alphabet: np.ndarray,
                       N_t: int) -> tuple[np.ndarray, np.ndarray]:
        """Every vector of :math:`\\mathcal{M}^{N_t}`, as indices and symbols.

        The enumeration is mixed-radix counting -- the order
        ``itertools.product(range(M), repeat=N_t)`` produces -- written as
        arithmetic on an index grid, because the loop that built it one
        candidate at a time was the dominant cost for a large alphabet.
        """
        order = len(alphabet)
        count = order ** N_t
        # digit k of every integer in [0, order**N_t), most significant
        # first: that is exactly product(range(order), repeat=N_t)
        powers = order ** np.arange(N_t - 1, -1, -1)
        S = (np.arange(count)[None, :] // powers[:, None]) % order
        return S.astype(float), alphabet[S]

    def forward(self, Y: np.ndarray) -> np.ndarray:
        H = _required_channel(self.H, type(self).__name__)
        if H.ndim > 2:
            S = _detect_stacked(self._detect, H, Y)
            self.S_ = S
            return S
        S = self._detect(H, Y)
        self.S_ = S
        return S

    def _detect(self, H: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """The exhaustive search against one channel matrix."""
        _, N_t = H.shape
        _, N = Y.shape
        alphabet = self.alphabet

        S_candidates, X_candidates = self.get_candidates(alphabet, N_t)
        Y_candidates = np.matmul(H, X_candidates)      # (N_r, C)

        # ||y - Hx||^2 = ||y||^2 - 2 Re(y^H Hx) + ||Hx||^2, and the first
        # term does not depend on the candidate: the search is one matrix
        # product against the whole block instead of one pass per sample.
        energy = np.sum(np.abs(Y_candidates) ** 2, axis=0)          # (C,)
        S = np.empty((N_t, N), dtype=int)
        for start in range(0, N, _ML_BLOCK):
            block = np.conjugate(Y[:, start:start + _ML_BLOCK]).T    # (n, N_r)
            n_block = block.shape[0]
            best = np.full(n_block, np.inf)
            winner = np.zeros(n_block, dtype=int)
            # blocked on both axes: the cost matrix is (n, C) and would
            # be hundreds of megabytes for a large constellation. The
            # running minimum keeps it bounded, and `<` keeps the
            # first-minimum tie-break of a single argmin.
            for first in range(0, energy.size, _ML_CANDIDATES):
                columns = slice(first, first + _ML_CANDIDATES)
                cost = energy[None, columns] - 2 * np.real(
                    np.matmul(block, Y_candidates[:, columns]))
                local = np.argmin(cost, axis=1)
                values = cost[np.arange(n_block), local]
                better = values < best
                winner[better] = local[better] + first
                best[better] = values[better]
            S[:, start:start + n_block] = S_candidates[:, winner]

        return S


@dataclass(slots=True)
class SphereDecoder(Processor):
    r"""Maximum likelihood by tree search instead of exhaustion.

    Signal Model
    ------------
    The decision is the one of :class:`MaximumLikelihoodDetector` --
    exactly, not approximately -- reached without evaluating every
    candidate. Write the thin QR factorization of the channel,
    :math:`\mathbf{H} = \mathbf{Q}\mathbf{R}` with
    :math:`\mathbf{Q}^H\mathbf{Q} = \mathbf{I}_{N_t}` and
    :math:`\mathbf{R}` upper triangular, and project the observation on
    :math:`\mathbf{z} = \mathbf{Q}^H \mathbf{y}`:

    .. math::

        \left\|\mathbf{y} - \mathbf{H}\mathbf{x}\right\|^2 =
        \underbrace{\left\|\mathbf{z} - \mathbf{R}\mathbf{x}\right\|^2}
        _{\text{depends on } \mathbf{x}}
        + \left\|\left(\mathbf{I} - \mathbf{Q}\mathbf{Q}^H\right)
          \mathbf{y}\right\|^2

    The second term is a constant, so minimizing the first *is* maximum
    likelihood. Its value is what makes the search possible, because
    :math:`\mathbf{R}` is triangular:

    .. math::

        \left\|\mathbf{z} - \mathbf{R}\mathbf{x}\right\|^2 =
        \sum_{k=N_t-1}^{0} \left|R_{kk}\right|^2
        \left|c_k(x_{k+1}, \ldots, x_{N_t-1}) - x_k\right|^2,
        \qquad
        c_k = \frac{z_k - \sum_{i>k} R_{ki} x_i}{R_{kk}}

    Every term is non-negative and layer :math:`k` depends only on the
    layers already decided, so a partial sum can only grow: the moment
    it exceeds the best full metric found so far, the whole subtree
    below it can be discarded. That is the sphere: the search visits
    only the lattice points inside a ball whose radius shrinks each time
    a better solution is found.

    The enumeration order within a layer is the Schnorr-Euchner one --
    alphabet points by increasing :math:`|c_k - a|` -- for two reasons.
    The first leaf reached is then the successive-interference-cancelling
    (Babai) solution, which gives a finite radius immediately, so no
    initial radius has to be guessed; and since the terms are visited in
    increasing order, the first one that exceeds the bound ends the
    layer, no ``continue`` needed.

    **What it costs.** The visited-node count is data dependent -- that
    is the whole point -- and is reported as ``nodes_``, the average per
    detected vector. At high SNR it approaches :math:`N_t` (the Babai
    point is already ML and nothing else survives the first bound);
    at low SNR, or on an ill-conditioned channel, it grows towards the
    exhaustive :math:`|\mathcal{M}|^{N_t}`. In Python the per-node
    overhead is large, so this block is *slower* than the vectorized
    exhaustive search on small problems and becomes the only option when
    :math:`|\mathcal{M}|^{N_t}` stops fitting in memory -- 16-QAM on
    four antennas is 65 536 candidates per symbol, 64-QAM on four is
    16.7 million.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2. ``H`` may be a stack ``(K, N_r, N_t)`` against ``Y`` of
    shape ``(K, N_r, N)``: channel k decides frame k, through an
    internal per-draw loop -- convenience, not a speedup (D51) -- and returns the
    alphabet indices, like every other detector here.

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`,
        with :math:`N_r \geq N_t`. Must be set before the call.
    name : str, optional, keyword-only
        Name of the detector. Default is ``"Sphere Decoder"``.

    Attributes
    ----------
    nodes_ : float
        Average number of tree nodes visited per detected vector in the
        last call -- data-dependent, hence the trailing underscore (D23).
        Compare it with :math:`|\mathcal{M}|^{N_t}`.
    S_ : np.ndarray
        Decisions of the last call, as alphabet indices (D23).

    Raises
    ------
    ValueError
        If ``H`` is not set, if the channel has fewer receive than
        transmit antennas (the decomposition needs a full-column-rank
        :math:`\mathbf{R}`), or if that rank is deficient.

    References
    ----------
    * U. Fincke, M. Pohst, "Improved methods for calculating vectors of
      short length in a lattice", Mathematics of Computation, vol. 44,
      no. 170, pp. 463-471, 1985.
    * C. P. Schnorr, M. Euchner, "Lattice basis reduction: improved
      practical algorithms and solving subset sum problems",
      Mathematical Programming, vol. 66, pp. 181-199, 1994.
    * E. Agrell, T. Eriksson, A. Vardy, K. Zeger, "Closest point search
      in lattices", IEEE Trans. Inf. Theory, vol. 48, no. 8,
      pp. 2201-2214, 2002.
    * B. Hassibi, H. Vikalo, "On the sphere-decoding algorithm I.
      Expected complexity", IEEE Trans. Signal Process., vol. 53, no. 8,
      pp. 2806-2818, 2005.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> H = np.eye(2)
    >>> Y = np.array([[0.9, -1.1], [-0.8, 1.2]])
    >>> SphereDecoder(alphabet, H=H)(Y)
    array([[1, 0],
           [0, 1]])

    The decision is the exhaustive one, on far fewer candidates:

    >>> from comnumpy.core.utils import get_alphabet
    >>> from comnumpy.mimo.utils import rayleigh_channel
    >>> alphabet = get_alphabet("QAM", 16)
    >>> H = rayleigh_channel(4, 4, seed=0)
    >>> rng = np.random.default_rng(1)
    >>> sent = rng.integers(0, 16, size=(4, 200))
    >>> Y = H @ alphabet[sent] + 0.05 * (rng.standard_normal((4, 200))
    ...                                  + 1j * rng.standard_normal((4, 200)))
    >>> decoder = SphereDecoder(alphabet, H=H)
    >>> exhaustive = MaximumLikelihoodDetector(alphabet, H=H)
    >>> bool(np.array_equal(decoder(Y), exhaustive(Y)))
    True
    >>> print(f"{decoder.nodes_:.1f} nodes visited, against {16 ** 4}")
    4.5 nodes visited, against 65536
    """

    alphabet: np.ndarray
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    name: str = field(default="Sphere Decoder", kw_only=True)
    # internal state (declared for slots, D40a)
    S_: Optional[np.ndarray] = field(init=False, repr=False,
                                     default_factory=lambda: None)
    nodes_: float = field(init=False, repr=False, default=0.0)
    # running counters behind nodes_, declared for slots (D40a)
    _visited: int = field(init=False, repr=False, default=0)
    _samples: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        # accept anything array-like, a Constellation included
        self.alphabet = np.asarray(self.alphabet)

    def _triangularize(self, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The thin QR of one channel matrix, with the guards it needs."""
        n_rx, n_tx = H.shape
        if n_rx < n_tx:
            raise ValueError(
                f"{type(self).__name__} needs at least as many receive as "
                f"transmit antennas, got H of shape {H.shape}. An "
                f"underdetermined channel has no unique closest lattice "
                f"point; use ApproximateMessagePassingDetector instead.")
        Q, R = LA.qr(H)
        diagonal = np.abs(np.diag(R))
        if np.min(diagonal) <= 1e-12 * np.max(diagonal):
            raise ValueError(
                f"{type(self).__name__} got a rank-deficient channel: the "
                f"triangular factor has a diagonal entry of "
                f"{np.min(diagonal):.3g} against {np.max(diagonal):.3g}. "
                f"The tree search divides by it, so no sphere is defined.")
        return Q, R

    def _closest(self, R: np.ndarray, gains: np.ndarray, z: np.ndarray,
                 alphabet: np.ndarray) -> tuple[np.ndarray, int]:
        """The closest lattice point to one observation, and its cost.

        Depth first from the last layer, Schnorr-Euchner order within a
        layer, and a bound that tightens every time a leaf is reached.
        """
        n_tx = R.shape[0]
        symbols = np.empty(n_tx, dtype=complex)
        indices = np.empty(n_tx, dtype=int)
        bound = np.inf
        best = np.zeros(n_tx, dtype=int)
        nodes = 0

        def search(level: int, metric: float) -> None:
            nonlocal bound, best, nodes
            centre = ((z[level] - np.dot(R[level, level + 1:],
                                         symbols[level + 1:]))
                      / R[level, level])
            distance = np.abs(alphabet - centre) ** 2
            for choice in np.argsort(distance):
                step = metric + gains[level] * distance[choice]
                if step >= bound:
                    break            # sorted: the rest of the layer too
                nodes += 1
                symbols[level] = alphabet[choice]
                indices[level] = choice
                if level == 0:
                    bound = step
                    best = indices.copy()
                else:
                    search(level - 1, step)

        search(n_tx - 1, 0.0)
        return best, nodes

    def forward(self, Y: np.ndarray) -> np.ndarray:
        H = _required_channel(self.H, type(self).__name__)
        if H.ndim > 2:
            # nodes_ averages over every sample of every draw: the
            # per-draw counts accumulate through _detect below
            self._visited, self._samples = 0, 0
            S = _detect_stacked(self._detect, H, Y)
            self.nodes_ = self._visited / max(1, self._samples)
            self.S_ = S
            return S
        self._visited, self._samples = 0, 0
        S = self._detect(H, Y)
        self.nodes_ = self._visited / max(1, self._samples)
        self.S_ = S
        return S

    def _detect(self, H: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """The tree search against one channel matrix."""
        Q, R = self._triangularize(H)
        Z = np.matmul(np.conjugate(Q).T, Y)          # (N_t, N)
        alphabet = np.asarray(self.alphabet).ravel()
        gains = np.abs(np.diag(R)) ** 2

        n_samples = Y.shape[-1]
        S = np.empty((R.shape[0], n_samples), dtype=int)
        for sample in range(n_samples):
            best, nodes = self._closest(R, gains, Z[:, sample], alphabet)
            S[:, sample] = best
            self._visited += nodes
        self._samples += n_samples
        return S


@dataclass(slots=True)
class LinearDetector(Processor):
    r"""Linear MIMO detector (equalization followed by hard decision).

    Signal Model
    ------------
    The detector assumes the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n], \qquad
        \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t`. Detection proceeds in two steps: linear
    equalization

    .. math::

        \mathbf{z}[n] = \mathbf{W} \mathbf{y}[n], \qquad
        \mathbf{W}_{zf} = \mathbf{H}^{\dagger}, \qquad
        \mathbf{W}_{mmse} = \left(\mathbf{H}^H \mathbf{H}
        + \sigma^2 \mathbf{I}_{N_t}\right)^{-1} \mathbf{H}^H

    then per-antenna projection of :math:`\mathbf{z}[n]` onto the nearest
    point of the constellation :math:`\mathcal{M}`. The detector returns
    the integer indices of the decisions in the alphabet.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2. ``H`` may be a stack ``(K, N_r, N_t)`` against
    ``Y`` of shape ``(K, N_r, N)`` -- one draw per trial, decided
    through numpy's stacked linear algebra in one call (D51).

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    method : {"zf", "mmse"}, keyword-only
        Linear equalizer :math:`\mathbf{W}`. Default is ``"zf"``.
    sigma2 : float, optional, keyword-only
        Noise variance :math:`\sigma^2` (required by the MMSE equalizer
        only).
    name : str, optional, keyword-only
        Name of the detector. Default is ``"ZF Detector"``.

    Raises
    ------
    ValueError
        If ``H`` is not set when the detector is called.

    References
    ----------
    * E. G. Larsson, P. Stoica and G. Ganesan, *Space-Time Block Coding
      for Wireless Communications*, Cambridge University Press, 2003.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> Y = np.array([[0.9, -1.1], [-0.8, 1.2]])
    >>> LinearDetector(alphabet, H=np.eye(2))(Y)
    array([[1, 0],
           [0, 1]])
    """
    alphabet: np.ndarray
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    method: Literal["zf", "mmse"] = field(default="zf", kw_only=True)
    sigma2: Optional[float] = field(default=None, kw_only=True)
    name: str = field(default="ZF Detector", kw_only=True)

    def __post_init__(self):
        # accept anything array-like, a Constellation included
        self.alphabet = np.asarray(self.alphabet)

    def linear_estimator(self, Y: np.ndarray) -> np.ndarray:
        r"""
        Perform Zero Forcing or MMSE linear equalization
        """
        block = type(self).__name__
        H = _required_channel(self.H, block)
        match self.method:
            case "zf":
                output = zf_estimator(Y, H)
            case "mmse":
                output = mmse_estimator(
                    Y, H, _required_noise_variance(self.sigma2, block))
        return output

    def forward(self, Y: np.ndarray) -> np.ndarray:
        Z = self.linear_estimator(Y)
        S, _ = hard_projector(Z, self.alphabet)
        return S


@dataclass(slots=True)
class OrderedSuccessiveInterferenceCancellationDetector(Processor):
    r"""Ordered Successive Interference Cancellation (OSIC) MIMO detector.

    Signal Model
    ------------
    The detector assumes the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n], \qquad
        \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t` and :math:`\mathbf{x}[n] \in \mathcal{M}^{N_t}`.
    At each of the :math:`N_t` stages, the stream selected by the
    ordering rule is equalized (ZF or MMSE), hard-detected on the
    constellation :math:`\mathcal{M}`, and its estimated contribution is
    cancelled:

    .. math::

        \mathbf{y}[n] \leftarrow \mathbf{y}[n]
        - \mathbf{h}_{k} \, \widehat{x}_{k}[n]

    where :math:`\mathbf{h}_{k}` is the column of :math:`\mathbf{H}`
    associated with the detected stream :math:`k`. The ordering rule is
    ``"sinr"`` (max post-equalization SINR, MMSE), ``"colnorm"`` (max
    column norm of :math:`\mathbf{H}`, ZF) or ``"snr"`` (max
    post-equalization SNR, ZF). The detector returns the integer indices
    of the decisions in the alphabet.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2. ``H`` may be a stack ``(K, N_r, N_t)`` against ``Y`` of
    shape ``(K, N_r, N)``: channel k decides frame k, through an
    internal per-draw loop -- convenience, not a speedup (D51).

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    osic_type : {"sinr", "colnorm", "snr"}, keyword-only
        Ordering strategy. Default is ``"sinr"``.
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    method : {"zf", "mmse"}, keyword-only
        Per-stage equalizer; overridden in ``__post_init__`` by the
        ordering rule (``"sinr"`` forces MMSE, ``"colnorm"`` and
        ``"snr"`` force ZF).
    sigma2 : float, optional, keyword-only
        Noise variance :math:`\sigma^2` (required for the ``"sinr"``
        ordering).
    name : str, optional, keyword-only
        Name of the detector. Default is ``"OSIC Detector"``.

    Raises
    ------
    ValueError
        If ``osic_type`` is unknown, or if ``H`` (or ``sigma2`` for the
        ``"sinr"`` ordering) is not set when the detector is called.

    References
    ----------
    * Y. S. Cho, J. Kim, W. Y. Yang and C. G. Kang, *MIMO-OFDM Wireless
      Communications with MATLAB*, John Wiley & Sons, 2010.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> H = np.array([[1.0, 0.5], [0.0, 1.0]])
    >>> Y = H @ alphabet[[1, 0]][:, None]
    >>> OrderedSuccessiveInterferenceCancellationDetector(alphabet, osic_type="snr", H=H)(Y)
    array([[1],
           [0]])
    """
    alphabet: np.ndarray
    osic_type: str = field(default="sinr", kw_only=True)  # 'sinr', 'colnorm', or 'snr'
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    method: Literal["zf", "mmse"] = field(default="zf", kw_only=True)
    sigma2: Optional[float] = field(default=None, kw_only=True)
    name: str = field(default="OSIC Detector", kw_only=True)

    def __post_init__(self):
        self.alphabet = np.asarray(self.alphabet)
        if self.osic_type == "sinr":
            self.method = "mmse"
        elif self.osic_type in ("colnorm", "snr"):
            self.method = "zf"
        else:
            raise ValueError("osic_type must be 'sinr', 'colnorm', or 'snr'")

    def ordering(self, H: np.ndarray) -> int:
        NT = H.shape[1]

        match self.osic_type:

            case "sinr":
                sigma2 = _required_noise_variance(self.sigma2,
                                                  type(self).__name__)
                W = LA.inv(H.conj().T @ H + sigma2 * np.eye(NT)) @ H.conj().T
                WH = W @ H

                diag_WH2 = np.abs(np.diag(WH))**2                         # |WH[i,i]|^2
                WH2 = np.abs(WH)**2                                       # |WH[i,j]|^2
                interference = np.sum(WH2, axis=1) - diag_WH2             # somme des interférences
                noise_term = sigma2 * np.sum(np.abs(W)**2, axis=1)        # bruit résiduel
                denominator = interference + noise_term
                sinr = diag_WH2 / denominator
                return int(np.argmax(sinr))

            case "colnorm":
                return int(np.argmax(np.linalg.norm(H, axis=0)))

            case "snr":
                G = LA.inv(H.conj().T @ H) @ H.conj().T
                return int(np.argmin(LA.norm(G, axis=1)))

            case _:
                raise ValueError("osic_type must be 'sinr', 'colnorm', or 'snr'.")

    def forward(self, Y: np.ndarray) -> np.ndarray:
        H = _required_channel(self.H, type(self).__name__)
        if H.ndim > 2:
            return _detect_stacked(self._detect, H, Y)
        return self._detect(H, Y)

    def _detect(self, H_full: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """The ordered cancellation against one channel matrix."""
        block = type(self).__name__

        Y_temp = Y.copy()
        NT = H_full.shape[1]
        S_hat = np.zeros((NT, Y.shape[1]), dtype=int)
        order = []
        remaining_idx = list(range(NT))

        for _ in range(NT):
            H_temp = H_full[:, remaining_idx]
            idx_local = self.ordering(H_temp)
            best_current_idx = remaining_idx[idx_local]
            order.append(best_current_idx)

            # perform estimation
            match self.method:
                case "zf":
                    Z = zf_estimator(Y_temp, H_temp)
                case "mmse":
                    Z = mmse_estimator(
                        Y_temp, H_temp,
                        _required_noise_variance(self.sigma2, block))

            # perform detection
            S, _ = hard_projector(Z, self.alphabet)
            s_est = S[idx_local, :]
            x_est = self.alphabet[s_est]

            # update Y, S and remaining_ix
            Y_temp = Y_temp - H_temp[:, idx_local][:, np.newaxis] * x_est
            S_hat[best_current_idx, :] = s_est
            del remaining_idx[idx_local]

        return S_hat


@dataclass(slots=True)
class ApproximateMessagePassingDetector(Processor):
    r"""Approximate Message Passing (AMP) MIMO detector.

    Signal Model
    ------------
    The detector assumes the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n], \qquad
        \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t` and :math:`\mathbf{x}[n] \in \mathcal{M}^{N_t}`.
    For each received vector :math:`\mathbf{y}`, the AMP iteration
    (:math:`t = 1, \dots, N_{it}`) is

    .. math::

        \mathbf{z}^{(t)} = \mathbf{x}^{(t)} + \mathbf{H}^H \mathbf{r}^{(t)},
        \qquad
        \mathbf{x}^{(t+1)} = F\left(\mathbf{z}^{(t)},
        \sigma^2 (1 + \tau_t^2)\right)

    .. math::

        \mathbf{r}^{(t+1)} = \mathbf{y} - \mathbf{H}\mathbf{x}^{(t+1)}
        + \frac{\tau_{t+1}^2}{1 + \tau_t^2} \, \mathbf{r}^{(t)}

    where :math:`F` is the posterior-mean (soft) denoiser over the
    constellation :math:`\mathcal{M}` and the state :math:`\tau_t^2` is
    tracked from the denoiser output variance with the system ratio
    :math:`\beta = N_t / N_r`. The final estimate is hard-projected on
    :math:`\mathcal{M}` and the detector returns the integer indices of
    the decisions in the alphabet.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2.

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    sigma2 : float, optional, keyword-only
        Noise variance :math:`\sigma^2`. Must be set before calling the
        detector.
    alpha : float, keyword-only
        Damping factor (reserved; unused by the current iteration).
        Default is 1.
    N_it : int, keyword-only
        Number of iterations :math:`N_{it}`. Default is 100.
    name : str, optional, keyword-only
        Name of the detector. Default is ``"AMP Detector"``.

    Raises
    ------
    ValueError
        If ``H`` or ``sigma2`` is not set when the detector is called.

    References
    ----------
    * C. Jeon, R. Ghods, A. Maleki and C. Studer, "Optimality of large
      MIMO detection via approximate message passing," Proc. IEEE Int.
      Symp. Information Theory (ISIT), 2015, pp. 1227-1231.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> Y = np.array([[0.9], [-1.1]])
    >>> ApproximateMessagePassingDetector(alphabet, H=np.eye(2), sigma2=0.1)(Y)
    array([[1],
           [0]])
    """
    alphabet: np.ndarray
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    sigma2: Optional[float] = field(default=None, kw_only=True)
    alpha: float = field(default=1, kw_only=True)
    N_it: int = field(default=100, kw_only=True)
    name: str = field(default="AMP Detector", kw_only=True)

    def __post_init__(self):
        # accept anything array-like, a Constellation included
        self.alphabet = np.asarray(self.alphabet)

    def fit(self, y: np.ndarray) -> np.ndarray:
        # see Algorithm 2
        block = type(self).__name__
        H = _required_channel(self.H, block)
        sigma2 = _required_noise_variance(self.sigma2, block)
        N_r, N_t = H.shape
        x_t = np.zeros(N_t)
        r_t = y - np.matmul(H, x_t)
        H_H = np.transpose(np.conjugate(H))
        beta = N_t / N_r  # system ratio (below equation 1)
        tau_2 = beta * 1 / sigma2

        for _ in range(self.N_it):
            z_t = x_t + np.matmul(H_H, r_t)
            sigma2_t = sigma2 * (1 + tau_2)
            x_t = soft_projector(
                z_t, self.alphabet, sigma2_t
            )  # F function in the original publication
            kernel = np.abs(self.alphabet.reshape(1, -1) - x_t.reshape(-1, 1)) ** 2
            # same effective variance as F: G is the posterior variance of
            # the *same* distribution (Jeon et al., Algorithm 2); tau_2
            # alone made the weights collapse and the state go NaN
            G = soft_projector(z_t, self.alphabet, sigma2_t, kernel)
            tau_2_old = tau_2
            tau_2 = (beta / sigma2) * np.mean(G)
            term1 = tau_2 / (1 + tau_2_old)
            r_t = y - np.matmul(H, x_t) + term1 * r_t

        return x_t

    def forward(self, Y: np.ndarray) -> np.ndarray:
        H = _required_channel(self.H, type(self).__name__)
        _, N = Y.shape
        _, N_t = H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:, n] = self.fit(Y[:, n])

        S, _ = hard_projector(X, self.alphabet)
        return S


@dataclass(slots=True)
class OrthogonalApproximateMessagePassingDetector(Processor):
    r"""Orthogonal Approximate Message Passing (OAMP) MIMO detector.

    Signal Model
    ------------
    The detector assumes the flat MIMO observation model

    .. math::

        \mathbf{y}[n] = \mathbf{H} \, \mathbf{x}[n] + \mathbf{b}[n], \qquad
        \mathbf{b}[n] \sim \mathcal{CN}\left(\mathbf{0},
        \sigma^2 \mathbf{I}_{N_r}\right)

    where :math:`\mathbf{H}` is the channel matrix of size
    :math:`N_r \times N_t` and :math:`\mathbf{x}[n] \in \mathcal{M}^{N_t}`.
    For each received vector :math:`\mathbf{y}`, the OAMP iteration
    (:math:`t = 1, \dots, N_{it}`) alternates a linear estimator and a
    posterior-mean denoiser :math:`F` over the constellation
    :math:`\mathcal{M}`:

    .. math::

        \mathbf{z}^{(t)} = \mathbf{x}^{(t)} + \mathbf{W}_t
        \left(\mathbf{y} - \mathbf{H}\mathbf{x}^{(t)}\right), \qquad
        \mathbf{x}^{(t+1)} = F\left(\mathbf{z}^{(t)}, \tau_t^2\right)

    where :math:`\mathbf{W}_t` is the matched filter
    :math:`\mathbf{H}^H` (``type="H"``), the pseudo-inverse
    :math:`\mathbf{H}^{\dagger}` (``type="pinv"``) or the MMSE matrix
    :math:`v_t^2 \mathbf{H}^H (v_t^2 \mathbf{H}\mathbf{H}^H +
    \sigma^2 \mathbf{I}_{N_r})^{-1}` (``type="MMSE"``). The error
    variances :math:`v_t^2` and :math:`\tau_t^2` are tracked from the
    residual and from :math:`\mathbf{B}_t = \mathbf{I}_{N_t} -
    \mathbf{W}_t \mathbf{H}`. The final estimate is hard-projected on
    :math:`\mathcal{M}` and the detector returns the integer indices of
    the decisions in the alphabet.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2.

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    sigma2 : float, optional, keyword-only
        Noise variance :math:`\sigma^2`. Must be set before calling the
        detector.
    alpha : float, keyword-only
        Damping factor (reserved; unused by the current iteration).
        Default is 1.
    N_it : int, keyword-only
        Number of iterations :math:`N_{it}`. Default is 100.
    type : {"H", "pinv", "MMSE"}, keyword-only
        Linear estimator :math:`\mathbf{W}_t`. Default is ``"MMSE"``.
    name : str, optional, keyword-only
        Name of the detector. Default is ``"OAMP Detector"``.

    Raises
    ------
    ValueError
        If ``H`` or ``sigma2`` is not set when the detector is called,
        or if ``type`` is unknown.

    References
    ----------
    * J. Ma and L. Ping, "Orthogonal AMP," IEEE Access, vol. 5,
      pp. 2020-2033, 2017.
    * D. Tse and P. Viswanath, *Fundamentals of Wireless Communication*,
      Cambridge University Press, 2005, Chapter 8.

    Examples
    --------
    >>> alphabet = np.array([-1.0 + 0j, 1.0 + 0j])
    >>> Y = np.array([[0.9], [-1.1]])
    >>> OrthogonalApproximateMessagePassingDetector(alphabet, H=np.eye(2), sigma2=0.1)(Y)
    array([[1],
           [0]])
    """

    alphabet: np.ndarray
    H: Optional[np.ndarray] = field(default=None, kw_only=True)
    sigma2: Optional[float] = field(default=None, kw_only=True)
    alpha: float = field(default=1, kw_only=True)
    N_it: int = field(default=100, kw_only=True)
    type: Literal["H", "pinv", "MMSE"] = field(default="MMSE", kw_only=True)
    name: str = field(default="OAMP Detector", kw_only=True)

    def __post_init__(self):
        # accept anything array-like, a Constellation included
        self.alphabet = np.asarray(self.alphabet)

    def get_W(self, vt_2: float = 0.0) -> np.ndarray:
        block = type(self).__name__
        H = _required_channel(self.H, block)

        match self.type:

            case "H":
                H_H = np.transpose(np.conjugate(H))
                W = H_H

            case "pinv":
                W = LA.pinv(H)

            case "MMSE":
                N_r, _ = H.shape
                H_H = np.transpose(np.conjugate(H))
                sigma2 = _required_noise_variance(self.sigma2, block)
                term1 = vt_2 * np.matmul(H, H_H) + sigma2 * np.eye(N_r)
                W = vt_2 * np.matmul(H_H, LA.inv(term1))

            case _:
                raise ValueError(f"Unknown type: {self.type}")

        return W

    def get_vt_2(self, error: np.ndarray, epsilon: float = 0.001) -> float:
        block = type(self).__name__
        H = _required_channel(self.H, block)
        N_r, _ = H.shape
        R = _required_noise_variance(self.sigma2, block) * np.eye(N_r)
        H_H = np.conjugate(np.transpose(H))
        num = np.sum(np.abs(error) ** 2) - np.trace(R)
        # trace(H^H H) is real analytically but complex-typed: take the
        # real part rather than lean on numpy's complex ordering
        den = np.real(np.trace(np.matmul(H_H, H)))
        return float(max(num / den, epsilon))

    def get_tau_2(self, B: np.ndarray, W: np.ndarray, vt_2: float) -> float:
        block = type(self).__name__
        N_r, N_t = _required_channel(self.H, block).shape
        R = _required_noise_variance(self.sigma2, block) * np.eye(N_r)
        W_H = np.conjugate(np.transpose(W))
        B_H = np.conjugate(np.transpose(B))
        term1 = (vt_2 / N_t) * np.trace(np.matmul(B, B_H))
        term2 = (1 / N_t) * np.trace(np.matmul(W, np.matmul(R, W_H)))
        tau_2 = term1 + term2
        return float(np.real(tau_2))

    def fit(self, y: np.ndarray) -> np.ndarray:
        tau_2, vt_2 = 1.0, 1.0
        H = _required_channel(self.H, type(self).__name__)
        _, N_t = H.shape
        x_t = np.zeros(N_t)

        for _ in range(self.N_it):
            W = self.get_W(vt_2)
            B = np.eye(N_t) - np.matmul(W, H)
            error = y - np.matmul(H, x_t)
            z_t = x_t + np.matmul(W, error)
            x_t = soft_projector(z_t, self.alphabet, tau_2)
            vt_2 = self.get_vt_2(error)
            tau_2 = self.get_tau_2(B, W, vt_2)

        return x_t

    def forward(self, Y: np.ndarray) -> np.ndarray:
        block = type(self).__name__
        _required_noise_variance(self.sigma2, block)
        _, N = Y.shape
        _, N_t = _required_channel(self.H, block).shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:, n] = self.fit(Y[:, n])

        S, _ = hard_projector(X, self.alphabet)
        return S
