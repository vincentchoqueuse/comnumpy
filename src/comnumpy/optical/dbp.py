import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from comnumpy.core import Processor
from .fiber import FiberSpec
from .utils import (get_linear_step_size, get_logarithmic_step_size, compute_erbium_doped_fiber_amplifier_gain,
                    step_transfers, apply_frequency_response, TransferKey,
                    apply_kerr_nonlinearity,
                    is_polarization_pair, manakov_kerr)

__all__ = ["DBP"]


@dataclass(slots=True)
class DBP(Processor):
    r"""Digital back-propagation (DBP) compensator for a multi-span fiber link.

    Signal Model
    ------------
    DBP inverts the deterministic part of the fiber propagation by
    integrating the nonlinear Schroedinger equation (NLSE) backwards
    (:math:`z \to -z`) for the complex field envelope :math:`A(z, t)`:

    .. math::

        \frac{\partial A(z, t)}{\partial z} =
        +\frac{\alpha}{2} A(z, t)
        + j \frac{\beta_2}{2} \frac{\partial^2 A(z, t)}{\partial t^2}
        - j \gamma |A(z, t)|^2 A(z, t)

    where :math:`\alpha = \frac{\ln 10}{10} \alpha_{dB}` is the
    attenuation (Np/km), :math:`\beta_2 = -10^3 D \lambda^2 / (2 \pi c)`
    the group velocity dispersion (ps^2/km) and :math:`\gamma` the Kerr
    coefficient (rad/W/km). For each of the ``N_spans`` spans, the signal
    is first divided by the EDFA amplitude gain
    :math:`\sqrt{G} = 10^{\alpha_{dB} L_{\mathrm{span}} / 20}`, then
    propagated backwards with ``StPS`` split steps taken in reverse
    order. With matching parameters and ASE noise disabled, DBP is the
    exact numerical inverse of :class:`~comnumpy.optical.links.FiberLink`.

    **Two polarizations.** Like :class:`~comnumpy.optical.links.FiberLink`,
    a field shaped ``(..., 2, N)`` is back-propagated with the Manakov
    equation -- the two polarizations share the total intensity and the
    coefficient carries the :math:`8/9` factor. The two blocks read the
    model off the shape the same way, so a link and its compensator
    cannot end up integrating different equations.

    Axes: *declared axis* -- a full-field signal ``(N,)`` or ``(..., P, N)``
    with ``P`` in ``{1, 2}`` polarizations. The leading axes are a
    **batch**: ``(B, P, N)`` propagates ``B`` realizations in one call,
    which is 1.8x faster than the same ``B`` calls at 12288 samples and
    3.1x at 1024 -- the split-step loop, the per-span amplifier and the
    parameter precomputation are paid once instead of ``B`` times. A
    batch of single-polarization fields is ``(B, 1, N)``; ``(B, N)``
    reads ``B`` as polarizations and is refused.

    Parameters
    ----------
    N_spans : int
        Number of spans to back-propagate.
    L_span : float, keyword-only
        Span length :math:`L_{\mathrm{span}}` in km. Default is 80.
    StPS : int, keyword-only
        Number of split steps :math:`\delta z` per span. Default is 1.
    fs : float, keyword-only
        Sampling frequency in Hz. Default is 1.
    step_type : {"linear", "logarithmic"}, keyword-only
        Step-size distribution within a span. Default is ``"linear"``.
    step_method : {"symmetric", "asymetric"}, keyword-only
        Split-step scheme. Default is ``"symmetric"``.
    use_only_linear : bool, keyword-only
        If True, only the linear effects (CD and attenuation) are
        compensated, in a single exact step per span. Default is False.
    c : float, keyword-only
        Speed of light :math:`c` in m/s. Default is ``SPEED_OF_LIGHT``.
    h : float, keyword-only
        Planck constant :math:`h` in J.s (kept for symmetry with
        :class:`FiberLink`; unused by the deterministic inverse).
    gamma : float, keyword-only
        Kerr coefficient :math:`\gamma` in rad/W/km. Default is
        ``KERR_COEFFICIENT``.
    lamb : float, keyword-only
        Wavelength :math:`\lambda` in nm. Default is ``WAVELENGTH``.
    alpha_dB : float, keyword-only
        Fiber loss :math:`\alpha_{dB}` in dB/km. Default is ``FIBER_LOSS``.
    cd_coefficient : float, keyword-only
        Dispersion coefficient :math:`D` in ps/nm/km. Default is
        ``CD_COEFFICIENT``.
    nu : float, keyword-only
        Optical carrier frequency :math:`\nu` in Hz (kept for symmetry
        with :class:`FiberLink`; unused by the deterministic inverse).
    step_log_factor : float, keyword-only
        Adjustment factor of the logarithmic step-size distribution.
        Default is 0.4.
    name : str, optional, keyword-only
        Name of the compensator instance. Default is ``"dbp"``.

    Raises
    ------
    ShapeError
        If the input is neither a full-field 1D signal ``(N,)`` nor a
        polarization pair ``(..., P, N)`` with ``P`` in ``{1, 2}``
        (validated in ``prepare()``): a pointwise Kerr step on a
        multi-channel array would silently produce SPM only (no XPM, no
        FWM).

    References
    ----------
    * O. V. Sinkin, R. Holzlohner, J. Zweck and C. R. Menyuk, "Optimization of the split-step Fourier method in modeling optical-fiber
      communications systems," Journal of Lightwave Technology, vol. 21, no. 1, pp. 61-68, Jan. 2003, doi: 10.1109/JLT.2003.808628.
    * E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital backpropagation,"
      Journal of Lightwave Technology, vol. 26, no. 20, pp. 3416-3425, 2008.
    * C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
      communication systems," IEEE Journal on Selected Areas in Communications,
      vol. 39, no. 1, pp. 280-294, 2021.

    Examples
    --------
    >>> from comnumpy.optical.links import FiberLink
    >>> rng = np.random.default_rng(0)
    >>> x = 1e-3 * (rng.normal(size=128) + 1j * rng.normal(size=128))
    >>> link = FiberLink(2, L_span=80.0, StPS=4, fs=10e9, noise_scaling=0)
    >>> dbp = DBP(2, L_span=80.0, StPS=4, fs=10e9)
    >>> bool(np.allclose(dbp(link(x)), x))
    True
    """

    N_spans: int = 1
    L_span: float = field(default=80, kw_only=True)
    StPS: int = field(default=1, kw_only=True)
    fs: float = field(default=1, kw_only=True)
    fiber: FiberSpec = field(default_factory=FiberSpec, kw_only=True)
    step_type: Literal["linear", "logarithmic"] = field(default="linear", kw_only=True)
    step_method: Literal["symmetric", "asymetric"] = field(default="symmetric", kw_only=True)
    use_only_linear: bool = field(default=False, kw_only=True)
    step_log_factor: float = field(default=0.4, kw_only=True)
    name: str = field(default="dbp", kw_only=True)
    # internal state (declared for slots, D40a)
    step_size: Optional[np.ndarray] = field(init=False, repr=False, default_factory=lambda: None)
    beta2: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    gain: Optional[float] = field(init=False, repr=False, default_factory=lambda: None)
    manakov_: bool = field(init=False, repr=False, default=False)
    transfer_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    transfer_index_: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    transfer_key_: Optional[TransferKey] = field(init=False, repr=False, default=None)

    def prepare(self, x: np.ndarray) -> None:
        self.manakov_ = is_polarization_pair(x, type(self).__name__)
        match self.step_type:
            case "linear":
                step_size = get_linear_step_size(self.L_span, self.StPS)
            case "logarithmic":
                step_size = get_logarithmic_step_size(self.L_span, self.StPS, alpha_dB=self.fiber.alpha_dB, step_log_factor=self.step_log_factor)
            case _:
                raise NotImplementedError(f"Step type {self.step_type} is not implemented")

        edfa_gain = compute_erbium_doped_fiber_amplifier_gain(self.fiber.alpha_dB, self.L_span)
        self.beta2 = self.fiber.beta2
        self.gain = 1/edfa_gain
        self.step_size = step_size[::-1]  # reverse order
        self._build_transfer(x.shape[-1])

    def _apply_kerr(self, y: np.ndarray, dz: float) -> np.ndarray:
        """Backward Kerr step: scalar NLSE, or Manakov for a pair."""
        gamma, intensity = manakov_kerr(y, self.fiber.gamma, self.manakov_)
        return apply_kerr_nonlinearity(y, dz, gamma, direction=-1,
                                       intensity=intensity)

    def _build_transfer(self, n_samples: int) -> None:
        """Cache one linear-step transfer function per distinct length."""
        assert self.step_size is not None and self.beta2 is not None
        if self.use_only_linear:
            lengths = np.array([self.L_span], dtype=float)
        elif self.step_method == "symmetric":
            lengths = np.asarray(self.step_size, dtype=float) / 2
        else:
            lengths = np.asarray(self.step_size, dtype=float)
        previous = None if self.transfer_key_ is None else (
            self.transfer_, self.transfer_index_, self.transfer_key_)
        self.transfer_, self.transfer_index_, self.transfer_key_ = step_transfers(
            n_samples, lengths, beta2=self.beta2, fs=self.fs,
            alpha_dB=self.fiber.alpha_dB, direction=-1, previous=previous)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x

        # set by prepare(), which the Processor base always runs first
        assert (self.gain is not None and self.beta2 is not None
                and self.step_size is not None
                and self.transfer_ is not None
                and self.transfer_index_ is not None)
        for _ in range(self.N_spans):
            y = self.gain * y  # correct for edfa gain
            if self.use_only_linear:
                y = apply_frequency_response(y, self.transfer_[0])
            else:
                for num_step in range(self.StPS):
                    dz = self.step_size[num_step]
                    H = self.transfer_[self.transfer_index_[num_step]]
                    if self.step_method == "symmetric":
                        y = apply_frequency_response(y, H)
                        y = self._apply_kerr(y, dz)
                        y = apply_frequency_response(y, H)

                    if self.step_method == "asymetric":
                        y = apply_frequency_response(y, H)
                        y = self._apply_kerr(y, dz)

        return y
