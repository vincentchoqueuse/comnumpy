import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core import Processor
from comnumpy.core.utils import hard_projector

__all__ = ["BlindDualMIMOCompensator"]


@dataclass(slots=True)
class BlindDualMIMOCompensator(Processor):
    r"""Blind adaptive 2x2 MIMO equalizer (CMA, RDE or DD stochastic gradient).

    Signal Model
    ------------
    At each step :math:`n`, the two output polarizations are computed
    from the estimated equalizer matrix :math:`\mathbf{H}` of size
    :math:`2 \times 2(2L+1)`:

    .. math::

        y_i[n] = \mathbf{h}_i^H \tilde{\mathbf{x}}[n], \qquad
        \tilde{\mathbf{x}}[n] = \begin{bmatrix}
        x_0[n] & \cdots & x_0[n-2L] & x_1[n] & \cdots & x_1[n-2L]
        \end{bmatrix}^T

    where :math:`\mathbf{h}_i^T` is the :math:`i`-th row of
    :math:`\mathbf{H}` and :math:`x_0`, :math:`x_1` are the two received
    polarizations. The equalizer is updated by stochastic gradient
    descent, :math:`\mathbf{H} \leftarrow \mathbf{H} + \mu \mathbf{g}[n]`,
    minimizing one of the following blind losses:

    * Constant Modulus Algorithm (CMA):
      :math:`\mathcal{L}_{CMA}(y[n]) = \left| R - |y[n]|^2 \right|^2`,
      where :math:`R = \mathbb{E}[|s|^4] / \mathbb{E}[|s|^2]` is derived
      from the alphabet;
    * Radius Directed Equalization (RDE):
      :math:`\mathcal{L}_{RDE}(y[n]) = \left| \mathcal{P}_{rad}^2(|y[n]|) - |y[n]|^2 \right|^2`,
      where :math:`\mathcal{P}_{rad}` projects onto the list of alphabet radii;
    * Decision Directed (DD):
      :math:`\mathcal{L}_{DD}(y[n]) = \left| \mathcal{P}_{\mathcal{M}}(y[n]) - y[n] \right|^2`,
      where :math:`\mathcal{P}_{\mathcal{M}}` projects onto the alphabet.

    The equalizer is adaptive (decision D22): every ``forward`` call
    updates :math:`\mathbf{H}` sample by sample, and ``partial_fit(X)``
    runs one adaptation pass over ``X`` from the current state.

    Staged adaptation
    -----------------
    **CMA is the only one of the three that converges from a cold
    start.** RDE and DD both need decisions that are already roughly
    right: started from the identity equalizer on 16-QAM, RDE stalls --
    3.5 dB where the noise floor is 24. The usual recipe is to run CMA
    first and switch once it has opened the eye, which is what a
    ``['cma', 'rde']`` schedule means in the literature. Two ways to do
    it here:

    * between passes, with :meth:`partial_fit` and :meth:`set_params`
      (or a plain assignment) to change ``mode`` and ``mu``;
    * inside a single pass, by overriding
      :meth:`process_after_iteration`, which is called after every
      output sample with the most recent outputs and exists for exactly
      this (see ``examples/mimo/one_shot_CMA.py``).

    Measured on a static Jones rotation with a 24 dB noise floor, the
    three stages give 21.0, 23.9 and 23.98 dB: CMA opens the eye, RDE
    takes it to within a tenth of the floor, and DD closes what is
    left.

    Axes: *declared axis* -- expects the dual-polarization layout
    ``(2, N)``, produces ``(2, N // oversampling)`` (one output sample
    per ``oversampling`` input samples: fractionally spaced equalizer).

    **Delay.** The centre tap of the initial equalizer sits :math:`L`
    input samples back, so output sample :math:`k` is aligned with
    input sample :math:`k \cdot os - L`: the block has a group delay of
    :math:`L` input samples, and comparing its output with the
    transmitted symbols without accounting for it reads as pure noise.
    The first :math:`(2L+1)/os` output samples are never written -- the
    filter has no history yet -- and are left at zero.

    Parameters
    ----------
    L : int, optional
        Half-length of the filter (each row of :math:`\mathbf{H}` holds
        :math:`2L+1` taps per polarization). Default is 10.
    alphabet : np.ndarray, keyword-only
        Modulation alphabet, used to derive the CMA radius :math:`R`,
        the RDE radius list and the DD decisions.
    mu : float, optional, keyword-only
        Step size :math:`\mu` of the gradient update. Default is 1e-4.
    oversampling : int, optional, keyword-only
        Oversampling factor of the input. When greater than one, the
        algorithm implements a fractionally spaced equalizer. Default is 1.
    mode : {"cma", "rde", "dd"}, optional, keyword-only
        Blind loss used for the update. Default is ``"cma"``. See
        *Staged adaptation* above: only ``"cma"`` converges from a cold
        start.
    sub_block_length : int, optional, keyword-only
        Number of recent output samples handed to
        :meth:`process_after_iteration`. Default is 20.
    name : str, optional, keyword-only
        Name of the processor instance. Default is ``"mimo filter"``.

    Attributes
    ----------
    H_ : np.ndarray
        Estimated equalizer matrix of shape ``(2, 2*(2*L+1))`` (decision
        D23: the trailing underscore distinguishes it from the
        *configured* channel ``H`` of ``FlatMIMOChannel``). Initialized
        to the identity equalizer (center tap) and updated at each call.

    Raises
    ------
    ValueError
        If the input is not a dual-polarization signal of shape ``(2, N)``.

    References
    ----------
    * D. N. Godard, "Self-recovering equalization and carrier tracking in
      two-dimensional data communication systems," IEEE Transactions on
      Communications, vol. 28, no. 11, pp. 1867-1875, 1980.
    * M. S. Faruk, S. J. Savory, "Digital signal processing for coherent
      transceivers employing multilevel formats," Journal of Lightwave
      Technology, vol. 35, no. 5, pp. 1125-1141, 2017.

    Examples
    --------
    >>> alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    >>> compensator = BlindDualMIMOCompensator(L=2, alphabet=alphabet)
    >>> rng = np.random.default_rng(0)
    >>> X = alphabet[rng.integers(0, 4, size=(2, 200))]
    >>> Y = compensator(X)
    >>> print(Y.shape, compensator.H_.shape)
    (2, 200) (2, 10)
    """
    L: int = 10
    # keyword-only but *required*: every mode needs it, and the None
    # default it used to carry crashed in __post_init__ on
    # abs(None) -- a default that cannot be used is not a default
    alphabet: np.ndarray = field(kw_only=True)
    mu: float = field(default=1e-4, kw_only=True)
    oversampling: int = field(default=1, kw_only=True)
    mode: Literal["cma", "rde", "dd"] = field(default="cma", kw_only=True)
    # a field, not a bare class attribute: unannotated it was excluded
    # from __slots__, so `compensator.sub_block_length = 30` raised
    # "read-only" and set_params (D34) could not reach it either
    sub_block_length: int = field(default=20, kw_only=True)
    name: str = field(default="mimo filter", kw_only=True)
    # estimated equalizer matrix (D23: underscore distinguishes it from the
    # *configured* channel H of FlatMIMOChannel), declared for slots (D40a)
    H_: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    # both are derived from the (required) alphabet in __post_init__, so
    # they are never None and the callers need no defensive check
    radius_cma: float = field(init=False, repr=False, default=0.0)
    radius_list: np.ndarray = field(init=False, repr=False,
                                    default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        """
        Prepare the filter coefficients.
        """
        self.initialize_H()
        # float(), not np.float64: the field is declared float, and the
        # Godard radius is a scalar, not a zero-dimensional array
        self.radius_cma = float(np.mean(np.abs(self.alphabet)**4)
                                / np.mean(np.abs(self.alphabet)**2))
        self.radius_list = np.unique(np.abs(self.alphabet))

    def initialize_H(self) -> None:
        H = np.zeros((2, 2*(2*self.L+1)), dtype=complex)
        H[0, self.L] = 1
        H[1, (2*self.L+1)+self.L] = 1
        self.H_ = H

    def grad(self, input: np.ndarray, output: np.ndarray,
             target: Optional[np.ndarray] = None) -> np.ndarray:

        if self.mode == "cma":
            # see equation 19/20
            error = self.radius_cma - np.abs(output)**2
            term1 = error * np.conjugate(output)
        elif self.mode == "rde":
            _, radius_est = hard_projector(np.abs(output), self.radius_list)
            error = radius_est**2 - np.abs(output)**2
            term1 = error * np.conjugate(output)
        elif self.mode == "dd":
            _, output_est = hard_projector(output, self.alphabet)
            term1 = np.conjugate(output_est - output)
        else:
            # three independent `if`s left `grad` unbound here, so an
            # unexpected mode raised UnboundLocalError instead of saying
            # what was wrong (D38)
            raise ValueError(
                f"expected mode 'cma', 'rde' or 'dd', got {self.mode!r}")

        return term1.reshape(-1, 1) * input

    def process_after_iteration(self, n: int, Y_sub: np.ndarray) -> None:
        """Hook called after each output sample; does nothing by default.

        Override it to schedule the stages of the adaptation (switch
        ``mode`` once the eye is open) or to correct a residual phase.

        Parameters
        ----------
        n : int
            Index of the output sample just produced.
        Y_sub : np.ndarray
            The last ``sub_block_length`` outputs, ``(2, K)``, oldest
            first.
        """

    def partial_fit(self, X: np.ndarray):
        """
        One adaptation pass over ``X`` (adaptive regime of decision D22):
        the estimate ``H_`` keeps evolving from its current state.
        """
        self.forward(X)
        return self

    def forward(self, X: np.ndarray) -> np.ndarray:

        if X.shape[0] != 2:
            raise ValueError(f"Blind Dual MIMO Compensator only works for dual polarization signals (X shape={X.shape})")

        L = self.L
        os = self.oversampling
        M, N = X.shape
        Y = np.zeros((M, N//os), dtype=complex)

        assert self.H_ is not None      # initialize_H ran in __post_init__
        for n in range(2*L + 1, N, os):
            x_sub = np.ravel(X[:, n:n-(2*L+1):-1])
            y_sub = np.matmul(np.conjugate(self.H_), x_sub)  # filter output
            grad = self.grad(x_sub, y_sub)
            self.H_ += self.mu*grad  # implement equation in matrix form directly
            index = n//os
            Y[:, index] = y_sub

            # the last sub_block_length outputs. This used to be
            # Y[:, index-1::-100], a strided view of the *whole* history:
            # it grew with n, so the hook cost O(n) per sample and the
            # pass was quadratic -- and sub_block_length, which names
            # exactly this, went unused.
            start = max(0, index + 1 - self.sub_block_length)
            self.process_after_iteration(index, Y[:, start:index + 1])

        return Y
