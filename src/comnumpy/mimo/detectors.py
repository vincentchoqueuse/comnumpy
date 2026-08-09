import numpy as np
import itertools
import numpy.linalg as LA
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core.generics import Processor
from comnumpy.core.utils import hard_projector, soft_projector, zf_estimator, mmse_estimator


def _validate_H(H):
    if H is None:
        raise ValueError("Channel H is not set.")
    elif not isinstance(H, np.ndarray):
        raise TypeError("Channel H must be a NumPy array.")


def _validate_sigma2(sigma2):
    if sigma2 is None:
        raise ValueError("Noise variance sigma2 is not set.")
    elif sigma2 < 0:
        raise TypeError("Noise variance sigma2 should be greater than 0.")


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
    (:math:`\widehat{\mathbf{x}}[n]` = ``alphabet[S[:, n]]``).

    .. WARNING::

        The exhaustive search evaluates :math:`|\mathcal{M}|^{N_t}`
        candidates and becomes expensive for a large number of transmit
        antennas or a high-order constellation.

    Axes: *declared axis* -- expects ``(ant, N)`` with antennas on
    axis -2.

    Parameters
    ----------
    alphabet : np.ndarray
        Symbol constellation :math:`\mathcal{M}` (1D array).
    H : np.ndarray, optional, keyword-only
        Channel matrix :math:`\mathbf{H}` of size :math:`N_r \times N_t`.
        Must be set before calling the detector.
    name : str, optional, keyword-only
        Name of the detector. Default is ``"ML Detector"``.

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
    # internal state (declared for slots, D40a)
    S: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)

    def get_nb_candidates(self):
        _, N_t = self.H.shape
        return len(self.alphabet) ** N_t

    def get_candidates(self, alphabet, N_t):
        symbols = np.arange(len(alphabet))
        input_list = [p for p in itertools.product(symbols, repeat=N_t)]

        # preallocation of memory
        X = np.zeros((N_t, len(input_list)), dtype=complex)
        S = np.zeros((N_t, len(input_list)))

        for indice in range(len(input_list)):
            input = np.array(input_list[indice])
            x = self.alphabet[input]
            X[:, indice] = x
            S[:, indice] = input

        return S, X

    def forward(self, Y):

        _validate_H(self.H)

        H = self.H
        _, N_t = H.shape
        _, N = Y.shape
        S = np.zeros((N_t, N), dtype=int)
        alphabet = self.alphabet

        S_candidates, X_candidates = self.get_candidates(alphabet, N_t)
        Y_candidates = np.matmul(
            H, X_candidates
        )  # compute all combinaison of received data

        for n in range(N):
            y = np.transpose(np.atleast_2d(Y[:, n]))
            index_min = np.argmin(np.sum(np.abs(y - Y_candidates) ** 2, axis=0))
            S[:, n] = S_candidates[:, index_min]

        self.S = S
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
    axis -2.

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

    def linear_estimator(self, Y):
        r"""
        Perform Zero Forcing or MMSE linear equalization
        """
        match self.method:
            case "zf":
                output = zf_estimator(Y, self.H)
            case "mmse":
                output = mmse_estimator(Y, self.H, self.sigma2 )
        return output

    def forward(self, Y):
        _validate_H(self.H)
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
    axis -2.

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
                W = LA.inv(H.conj().T @ H + self.sigma2 * np.eye(NT)) @ H.conj().T
                WH = W @ H

                diag_WH2 = np.abs(np.diag(WH))**2                         # |WH[i,i]|^2
                WH2 = np.abs(WH)**2                                       # |WH[i,j]|^2
                interference = np.sum(WH2, axis=1) - diag_WH2             # somme des interférences
                noise_term = self.sigma2 * np.sum(np.abs(W)**2, axis=1)   # bruit résiduel
                denominator = interference + noise_term
                sinr = diag_WH2 / denominator
                return int(np.argmax(sinr))

            case "colnorm":
                return np.argmax(np.linalg.norm(H, axis=0))

            case "snr":
                G = LA.inv(H.conj().T @ H) @ H.conj().T
                return int(np.argmin(LA.norm(G, axis=1)))

            case _:
                raise ValueError("osic_type must be 'sinr', 'colnorm', or 'snr'.")

    def forward(self, Y: np.ndarray) -> np.ndarray:
        if self.H is None or (self.sigma2 is None and self.osic_type == "sinr"):
            raise ValueError("H and sigma2 must be set before calling forward().")

        Y_temp = Y.copy()
        NT = self.H.shape[1]
        S_hat = np.zeros((NT, Y.shape[1]), dtype=int)
        order = []
        remaining_idx = list(range(NT))

        for _ in range(NT):
            H_temp = self.H[:, remaining_idx]
            idx_local = self.ordering(H_temp)
            best_current_idx = remaining_idx[idx_local]
            order.append(best_current_idx)

            # perform estimation
            match self.method:
                case "zf":
                    Z = zf_estimator(Y_temp, H_temp)
                case "mmse":
                    Z = mmse_estimator(Y_temp, H_temp, self.sigma2 )

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

    def fit(self, y):
        # see Algorithm 2
        H = self.H
        N_r, N_t = H.shape
        x_t = np.zeros(N_t)
        r_t = y - np.matmul(H, x_t)
        H_H = np.transpose(np.conjugate(H))
        beta = N_t / N_r  # system ratio (below equation 1)
        tau_2 = beta * 1 / self.sigma2

        for _ in range(self.N_it):
            z_t = x_t + np.matmul(H_H, r_t)
            sigma2_t = self.sigma2 * (1 + tau_2)
            x_t = soft_projector(
                z_t, self.alphabet, sigma2_t
            )  # F function in the original publication
            kernel = np.abs(self.alphabet.reshape(1, -1) - x_t.reshape(-1, 1)) ** 2
            G = soft_projector(z_t, self.alphabet, tau_2, kernel)
            tau_2_old = tau_2
            tau_2 = (beta / (self.sigma2)) * np.mean(G)
            term1 = tau_2 / (1 + tau_2_old)
            r_t = y - np.matmul(self.H, x_t) + term1 * r_t

        return x_t

    def forward(self, Y):
        _validate_H(self.H)
        _validate_sigma2(self.sigma2)

        H = self.H
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

    def get_W(self, vt_2=0):
        H = self.H

        match self.type:

            case "H":
                H_H = np.transpose(np.conjugate(H))
                W = H_H

            case "pinv":
                W = LA.pinv(H)

            case "MMSE":
                N_r, _ = H.shape
                H_H = np.transpose(np.conjugate(H))
                term1 = vt_2 * np.matmul(H, H_H) + self.sigma2 * np.eye(N_r)
                W = vt_2 * np.matmul(H_H, LA.inv(term1))

            case _:
                raise ValueError(f"Unknown type: {self.type}")

        return W

    def get_vt_2(self, error, epsilon=0.001):
        H = self.H
        N_r, _ = H.shape
        R = self.sigma2 * np.eye(N_r)
        H_H = np.conjugate(np.transpose(H))
        num = np.sum(np.abs(error) ** 2) - np.trace(R)
        den = np.trace(np.matmul(H_H, H))
        return max(num / den, epsilon)

    def get_tau_2(self, B, W, vt_2):
        N_r, N_t = self.H.shape
        R = self.sigma2 * np.eye(N_r)
        W_H = np.conjugate(np.transpose(W))
        B_H = np.conjugate(np.transpose(B))
        term1 = (vt_2 / N_t) * np.trace(np.matmul(B, B_H))
        term2 = (1 / N_t) * np.trace(np.matmul(W, np.matmul(R, W_H)))
        tau_2 = term1 + term2
        return tau_2

    def fit(self, y):
        tau_2, vt_2 = 1, 1
        _, N_t = self.H.shape
        x_t = np.zeros(N_t)

        for _ in range(self.N_it):
            W = self.get_W(vt_2)
            B = np.eye(N_t) - np.matmul(W, self.H)
            error = y - np.matmul(self.H, x_t)
            z_t = x_t + np.matmul(W, error)
            x_t = soft_projector(z_t, self.alphabet, tau_2)
            vt_2 = self.get_vt_2(error)
            tau_2 = self.get_tau_2(B, W, vt_2)

        return x_t

    def forward(self, Y):
        _validate_H(self.H)
        _validate_sigma2(self.sigma2)


        _, N = Y.shape
        _, N_t = self.H.shape
        X = np.zeros((N_t, N), dtype=complex)
        for n in range(N):
            X[:, n] = self.fit(Y[:, n])

        S, _ = hard_projector(X, self.alphabet)
        return S
