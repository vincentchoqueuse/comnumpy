import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core import Processor
from .utils import apply_chromatic_dispersion, apply_kerr_nonlinearity, compute_beta2
from .constants import CD_COEFFICIENT, SPEED_OF_LIGHT, WAVELENGTH, KERR_COEFFICIENT

__all__ = ["PhaseNoise", "ChromaticDispersion", "KerrNonLinearity",
           "PMDEmulator"]


@dataclass(slots=True)
class PhaseNoise(Processor):
    r"""Wiener (random-walk) phase noise channel.

    Signal Model
    ------------
    .. math::

        y[n] = x[n] \, e^{j \phi[n]}, \qquad
        \phi[n] = \sum_{k=0}^{n} \delta\phi_k, \qquad
        \delta\phi_k \sim \mathcal{N}\left(0, \sigma^2\right)

    where :math:`\sigma^2` is the variance of the independent phase
    increments :math:`\delta\phi_k` (a Wiener process accumulated along
    the samples). The magnitude :math:`|y[n]| = |x[n]|` is preserved.

    Axes: *axis -1* -- the phase random walk accumulates along the
    samples. What shares one walk is declared by ``per``, which names
    the *event* -- the slice of the array that sees one laser:

    - ``"pair"`` (default): one walk per trailing ``(2, N)`` block --
      the two rows of a polarization pair come from one laser, and
      every leading axis is batch: independent draws per trial. A 2-D
      or deeper input whose axis -2 is not 2 is refused, because it
      cannot be read both ways: use ``"row"`` for independent rows or
      ``"signal"`` for one shared walk.
    - ``"row"``: one independent walk per 1-D row.
    - ``"signal"``: one walk broadcast over the whole array.

    A 1-D input sees one walk under every setting.

    Parameters
    ----------
    sigma2 : float
        Variance :math:`\sigma^2` of the per-sample phase increments, in
        rad^2.
    per : {"pair", "row", "signal"}, optional, keyword-only
        The event that shares one laser. Default is ``"pair"``.
    seed : int, optional, keyword-only
        Local RNG seed.
    name : str, optional, keyword-only
        Name of the channel instance. Default is ``"phase noise"``.

    References
    ----------
    T. Pfau, S. Hoffmann, R. Noe, "Hardware-Efficient Coherent Digital
    Receiver Concept With Feedforward Carrier Recovery for M-QAM
    Constellations," Journal of Lightwave Technology, vol. 27, no. 8,
    pp. 989-999, 2009.

    Examples
    --------
    >>> x = np.ones(3, dtype=complex)
    >>> y = PhaseNoise(0.01, seed=0)(x)
    >>> print(np.round(np.abs(y), 6))
    [1. 1. 1.]
    """
    sigma2: float
    per: Literal["pair", "row", "signal"] = field(default="pair",
                                                  kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="phase noise", kw_only=True)
    # internal state (declared for slots, D40a)
    rng: Optional[np.random.Generator] = field(init=False, repr=False, default=None)
    _b: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def prepare(self, x: np.ndarray) -> None:
        from comnumpy.exceptions import ShapeError  # local import (D36)
        if self.per == "pair" and np.ndim(x) >= 2 and np.shape(x)[-2] != 2:
            raise ShapeError(
                f"PhaseNoise(per='pair') expects a polarization pair "
                f"(..., 2, N), got shape {np.shape(x)} -- an axis -2 of "
                f"size {np.shape(x)[-2]} cannot be read as a pair. Use "
                f"per='row' for one independent laser per row, or "
                f"per='signal' for one walk shared by the whole array.")

    def noise_rvs(self, X: np.ndarray) -> np.ndarray:
        """Draw one Wiener walk per event declared by ``per``."""
        assert self.rng is not None      # set in __post_init__
        n = np.shape(X)[-1]
        if self.per == "signal" or np.ndim(X) <= 1:
            shape: tuple[int, ...] = (n,)
        elif self.per == "pair":
            # one walk per trailing (2, N) block, broadcast over the pair
            shape = np.shape(X)[:-2] + (1, n)
        else:                            # "row"
            shape = np.shape(X)
        noise = self.rng.normal(loc=0, scale=np.sqrt(self.sigma2),
                                size=shape)
        self._b = np.cumsum(noise, axis=-1)
        return self._b

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.exp(1j * self.noise_rvs(x))


@dataclass(slots=True)
class ChromaticDispersion(Processor):
    r"""Chromatic dispersion (CD) of a fiber span, applied exactly in the frequency domain.

    Signal Model
    ------------
    The block solves the linear part of the nonlinear Schroedinger
    equation (NLSE) for the complex field envelope :math:`A(z, t)`:

    .. math::

        \frac{\partial A(z, t)}{\partial z} =
        -j \frac{\beta_2}{2} \frac{\partial^2 A(z, t)}{\partial t^2}
        - \frac{\alpha}{2} A(z, t)

    whose exact solution over a fiber of length :math:`z` is applied in
    the frequency domain:

    .. math::

        \hat{A}(z, \omega) = \hat{A}(0, \omega) \,
        e^{j \frac{\beta_2}{2} z \omega^2} \, e^{-\frac{\alpha}{2} z}

    with the group velocity dispersion :math:`\beta_2` (ps^2/km) derived
    from the dispersion coefficient :math:`D`, the wavelength
    :math:`\lambda` and the speed of light :math:`c`, and the attenuation
    :math:`\alpha = \frac{\ln 10}{10} \, \alpha_{dB}` (Np/km):

    .. math::

        \beta_2 = -\frac{10^3 \, D \lambda^2}{2 \pi c}

    With ``direction=-1`` the phase shift and the attenuation are both
    inverted (backward propagation, used for CD compensation).

    Axes: *declared axis* -- requires a full-field 1D signal (N,).

    Parameters
    ----------
    z : float
        Fiber length :math:`z` in km.
    fs : float, keyword-only
        Sampling frequency in Hz; sets the discrete pulsation grid
        :math:`\omega`. Default is 1.
    alpha_dB : float, keyword-only
        Attenuation :math:`\alpha_{dB}` in dB/km. Default is 0 (lossless).
    direction : int, keyword-only
        Propagation direction, 1 for forward and -1 for backward.
        Default is 1.
    lamb : float, keyword-only
        Wavelength :math:`\lambda` in nm. Default is ``WAVELENGTH``.
    D : float, keyword-only
        Dispersion coefficient :math:`D` in ps/nm/km. Default is
        ``CD_COEFFICIENT``.
    c : float, keyword-only
        Speed of light :math:`c` in m/s. Default is ``SPEED_OF_LIGHT``.
    name : str, optional, keyword-only
        Identifier for the dispersion instance. Default is ``"cd"``.

    References
    ----------
    * G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
      2013, Section 3.2.
    * S. J. Savory, "Digital filters for coherent optical receivers,"
      Optics Express, vol. 16, no. 2, pp. 804-817, 2008.
    * A. Shahkarami, "Complexity reduction over bi-RNN-based Kerr nonlinearity equalization
      in dual-polarization fiber-optic communications via a CRNN-based approach,"
      Dissertation, Institut polytechnique de Paris, 2022.
      URL: https://www.theses.fr/2022IPPAT034.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=64) + 1j * rng.normal(size=64)
    >>> cd = ChromaticDispersion(80.0, fs=10e9)
    >>> cd_back = ChromaticDispersion(80.0, fs=10e9, direction=-1)
    >>> bool(np.allclose(cd_back(cd(x)), x))
    True
    """
    z: float
    fs: float = field(default=1, kw_only=True)
    alpha_dB: float = field(default=0, kw_only=True)
    direction: int = field(default=1, kw_only=True)
    name: str = field(default="cd", kw_only=True)
    lamb: float = field(default=WAVELENGTH, kw_only=True)
    D: float = field(default=CD_COEFFICIENT, kw_only=True)
    c: float = field(default=SPEED_OF_LIGHT, kw_only=True)

    @property
    def beta2(self):
        return compute_beta2(self.lamb, self.D, self.c)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = apply_chromatic_dispersion(x, self.z, self.beta2, alpha_dB=self.alpha_dB, fs=self.fs, direction=self.direction)
        return y

@dataclass(slots=True)
class KerrNonLinearity(Processor):
    r"""Kerr nonlinearity (self-phase modulation) of a fiber segment.

    Signal Model
    ------------
    The block solves the nonlinear part of the nonlinear Schroedinger
    equation (NLSE) for the complex field envelope :math:`A(z, t)`:

    .. math::

        \frac{\partial A(z, t)}{\partial z} = j \gamma |A(z, t)|^2 A(z, t)

    whose exact solution over a segment of length :math:`z` is the
    intensity-dependent phase rotation

    .. math::

        A(z, t) = g \, A(0, t) \, e^{j \gamma z |A(0, t)|^2}

    where :math:`\gamma` is the Kerr coefficient (rad/W/km) and :math:`g`
    an optional amplitude gain. With ``direction=-1`` the phase rotation
    is conjugated (backward propagation, used in DBP).

    Axes: *declared axis* -- requires a full-field 1D signal (N,). A
    pointwise Kerr step applied to separate channels would model SPM only
    (no XPM, no FWM).

    Parameters
    ----------
    z : float
        Segment length :math:`z` in km.
    direction : int, keyword-only
        Propagation direction, 1 for forward and -1 for backward.
        Default is 1.
    gamma : float, keyword-only
        Kerr coefficient :math:`\gamma` in rad/W/km. Default is
        ``KERR_COEFFICIENT``.
    gain : float, keyword-only
        Amplitude gain :math:`g` (linear). Default is 1.
    name : str, optional, keyword-only
        Identifier for the nonlinearity instance. Default is ``"nl"``.

    References
    ----------
    * G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press,
      2013, Section 4.1.
    * C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
      communication systems," IEEE Journal on Selected Areas in Communications,
      vol. 39, no. 1, pp. 280-294, 2021.

    Examples
    --------
    >>> x = np.array([1.0 + 0j])
    >>> y = KerrNonLinearity(10, gamma=1.3)(x)
    >>> print(round(float(np.angle(y[0])), 4))
    0.4336
    """

    z: float
    direction: int = field(default=1, kw_only=True)
    name: str = field(default="nl", kw_only=True)
    gamma: float = field(default=KERR_COEFFICIENT, kw_only=True)
    gain: float = field(default=1, kw_only=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = apply_kerr_nonlinearity(x, self.z, self.gamma, gain=self.gain, direction=self.direction)
        return y




@dataclass(slots=True)
class PMDEmulator(Processor):
    r"""Polarization rotation and first-order PMD, as a section emulator.

    Signal Model
    ------------
    The fibre's random birefringence is emulated by :math:`K` sections,
    each a random unitary Jones rotation followed by a differential
    group delay (DGD) of :math:`\tau / \sqrt{K}` between the two
    principal states, applied exactly in the frequency domain:

    .. math::

        \mathbf{Y}(\omega) = \prod_{k=1}^{K}
        \mathbf{D}_k(\omega) \, \mathbf{R}_k \; \mathbf{X}(\omega),
        \qquad
        \mathbf{D}_k(\omega) = \mathrm{diag}\!\left(
            e^{+j \omega \tau / 2\sqrt{K}}, \;
            e^{-j \omega \tau / 2\sqrt{K}} \right)

    Each :math:`\mathbf{R}_k` is drawn Haar-uniform from the seeded
    local generator, so the emulator is unitary -- it conserves energy
    exactly -- and reproducible. With ``n_sections=1`` and ``dgd=0`` it
    reduces to one random rotation of the state of polarization, which
    is the part every polarization demultiplexer must undo even before
    PMD enters.

    Randomly oriented sections add their DGDs in quadrature, hence the
    :math:`1/\sqrt{K}` per section: the declared ``dgd`` is the **RMS**
    DGD of the ensemble. Over section draws the DGD is Maxwellian, with
    mean :math:`\sqrt{8/3\pi}\,\tau \approx 0.921\,\tau` (Poole &
    Wagner). With ``n_sections=1`` the DGD is deterministic and equals
    :math:`\tau` exactly. The concatenation of sections also makes the
    DGD frequency-dependent, which is what distinguishes PMD from a
    wavelength-flat rotation.

    Axes: *polarization pair* -- expects ``(..., 2, N)``; the Jones
    matrices act on the polarization axis, the delays on the last one.

    Parameters
    ----------
    dgd : float
        RMS differential group delay :math:`\tau` in seconds.
    n_sections : int, optional, keyword-only
        Number of concatenated sections :math:`K`. Default is 8.
    fs : float, optional, keyword-only
        Sampling frequency in Hz. Default is 1.0.
    seed : int, optional, keyword-only
        Local RNG seed for the section rotations. Same seed, same fibre.
    name : str, optional, keyword-only
        Block name. Default ``"pmd"``.

    Raises
    ------
    ShapeError
        If the input does not carry a polarization pair ``(..., 2, N)``.

    References
    ----------
    C. D. Poole and R. E. Wagner, "Phenomenological approach to
    polarisation dispersion in long single-mode fibres," Electronics
    Letters, vol. 22, no. 19, pp. 1029-1030, 1986.
    S. J. Savory, "Digital filters for coherent optical receivers,"
    Optics Express, vol. 16, no. 2, pp. 804-817, 2008.

    Examples
    --------
    >>> x = np.zeros((2, 8), dtype=complex); x[0] = 1.0
    >>> y = PMDEmulator(0.0, n_sections=1, seed=1)(x)
    >>> bool(np.abs(np.sum(np.abs(y) ** 2) - 8.0) < 1e-9)   # unitary
    True
    >>> bool(np.abs(y[1, 0]) > 0)      # the rotation mixes the two rows
    True
    """
    dgd: float
    n_sections: int = field(default=8, kw_only=True)
    fs: float = field(default=1.0, kw_only=True)
    seed: Optional[int] = field(default=None, kw_only=True)
    name: str = field(default="pmd", kw_only=True)
    rotations_: Optional[np.ndarray] = field(init=False, repr=False,
                                             default=None)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        sections = []
        for _ in range(self.n_sections):
            # Haar-uniform 2x2 unitary: QR of a complex Gaussian matrix,
            # with the phase convention fixed by the R diagonal
            gaussian = (rng.standard_normal((2, 2))
                        + 1j * rng.standard_normal((2, 2)))
            q, r = np.linalg.qr(gaussian)
            sections.append(q * (np.diag(r) / np.abs(np.diag(r))))
        self.rotations_ = np.stack(sections)

    def prepare(self, x: np.ndarray) -> None:
        from comnumpy.exceptions import ShapeError  # local import (D36)
        from .utils import is_polarization_pair     # local import (D36)
        if not is_polarization_pair(x, "PMDEmulator"):
            raise ShapeError(
                f"PMDEmulator expects a polarization pair (..., 2, N), "
                f"got shape {x.shape} -- PMD is a two-polarization "
                f"effect, there is nothing to delay against on one.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert self.rotations_ is not None   # set in __post_init__
        omega = 2 * np.pi * self.fs * np.fft.fftfreq(x.shape[-1])
        delay = np.exp(1j * omega * self.dgd
                       / (2 * np.sqrt(self.n_sections)))
        spectrum = np.fft.fft(x, axis=-1)
        for rotation in self.rotations_:
            spectrum = np.einsum("ij,...jn->...in", rotation, spectrum)
            spectrum = spectrum * np.stack([delay, np.conj(delay)])
        return np.fft.ifft(spectrum, axis=-1)
