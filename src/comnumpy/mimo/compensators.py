import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core import Processor
from comnumpy.core.utils import hard_projector


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

    Axes: *declared axis* -- expects the dual-polarization layout
    ``(2, N)``, produces ``(2, N // oversampling)`` (one output sample
    per ``oversampling`` input samples: fractionally spaced equalizer).

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
    norm : bool, optional, keyword-only
        Flag to normalize the filter weights. Default is True.
    mode : {"cma", "rde", "dd"}, optional, keyword-only
        Blind loss used for the update. Default is ``"cma"``.
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
    norm: bool = field(default=True, kw_only=True)
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
        pass

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
            Y[:, n//os] = y_sub

            # perform process after_iteration
            self.process_after_iteration(n//os, Y[:, n//os-1::-100])

        return Y
