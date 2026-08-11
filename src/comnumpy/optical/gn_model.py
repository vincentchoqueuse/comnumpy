r"""The Gaussian Noise model: what a fibre link costs, in closed form.

A split-step simulation answers "what does this link do?" one launch
power at a time, and each answer costs minutes. The **GN model** answers
"how much nonlinear noise does this link produce?" in closed form, in
microseconds, and it is the reference every fibre simulator is checked
against.

Its assumption is in its name. After enough uncompensated dispersion the
propagating signal looks Gaussian, so the nonlinear interference (NLI)
it generates on itself behaves like additive Gaussian noise whose power
grows with the **cube** of the launch power. The link is then an
additive-noise channel with two terms:

.. math::

    \mathrm{SNR} = \frac{P}{P_{\mathrm{ASE}} + \eta P^3}

and everything an engineer wants -- the optimal launch power, the peak
SNR, the reach -- follows from that one expression (see
:func:`optimal_launch_power`).

**Which equation the fibre is integrating matters.** The 16/27 the
model is built on is a *Manakov* coefficient: it assumes a
dual-polarization signal in a fibre whose birefringence averages the
nonlinearity, which is what a real coherent link does and what
:class:`~comnumpy.optical.links.FiberLink` integrates when the field is
shaped ``(..., 2, N)``. Hand a one-dimensional field to the same block
and it integrates the *scalar* NLSE instead, which produces
**3.375 times** more nonlinear interference at equal total power. That
is not a small print: it is 5.3 dB, and it is the difference between a
simulation that confirms this module and one that appears to refute it.
Pass ``polarizations=1`` to :func:`gn_model_nli_power` in that case --
the weights become 2 and 4 rather than 16/27 and 32/27, which is
Table I of Serena and Bononi (2015) with the 8/9 removed because the
scalar equation uses :math:`\gamma` directly.

**What it is not.** The Gaussian assumption is also the model's known
bias: a real modulation format is not Gaussian, its fourth moment is
smaller, and the true NLI is *lower* than the GN model predicts. The
correction is the **EGN model**, and it is deliberately **not
implemented here** -- see the note at the end of
``validation/optical_gn_model.py``, which measures the effect instead
of predicting it, and reports it against the published values below.

References
----------
P. Poggiolini, "The GN model of non-linear propagation in uncompensated
coherent optical systems", J. Lightwave Technol., vol. 30, no. 24,
pp. 3857-3879, 2012 (eq. 120 and 123 of the extended version,
arXiv:1209.0394, are the two implemented below);
P. Poggiolini, G. Bosco, A. Carena, V. Curri, Y. Jiang, F. Forghieri,
"The GN-model of fiber non-linear propagation and its applications",
J. Lightwave Technol., vol. 32, no. 4, pp. 694-721, 2014;
P. Serena and A. Bononi, "A time-domain extended Gaussian noise model",
J. Lightwave Technol., vol. 33, no. 7, pp. 1459-1472, 2015 -- Table I
for the polarization weights, and Section III for the published
benchmark this module is pinned to;
A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini, F. Forghieri,
"EGN model of non-linear fiber propagation", Opt. Express, vol. 22,
no. 13, pp. 16335-16362, 2014 (the modulation-format correction this
module does not implement).

**The published number this module reproduces.** Serena and Bononi
(2015), Section III, simulate 15 channels at 32 GBd on a 37.5 GHz grid
over 5 x 100 km of SMF and report the normalized NLI coefficient
:math:`a_{\mathrm{NL}}`, defined by
:math:`\sigma^2_{\mathrm{NLI}} = a_{\mathrm{NL}} P^3`, as **-23.5 dB**
(mW^-2) for a Gaussian-modulated signal -- the case the GN model is
written for. This module returns -23.30 dB for that link, an agreement
of 0.2 dB against an independent split-step Monte Carlo simulation
published by a different group; the check is
``tests/optical/test_gn_model.py::TestAgainstPublishedBenchmark``.

The two implemented equations were cross-checked line by line against
the reference open-source implementation of the same paper, GNPy
(Telecom Infra Project, ``gnpy/core/science_utils.py``,
``NliSolver._gn_analytic`` and ``NliSolver._psi``), which fixes the same
conventions this library already uses: :math:`\alpha` is the **power**
attenuation coefficient, so :math:`P(z) = P(0)e^{-\alpha z}` and
:math:`L_{\mathrm{eff}} = (1 - e^{-\alpha L})/\alpha`. GNPy works in
metres and s^2/m where this library works in kilometres and ps^2/km, so
the agreement is not a shared typo:
``tests/optical/test_gn_model.py`` re-transcribes both GNPy routines in
SI units, from scratch, and requires the two to agree to twelve digits.
"""
from __future__ import annotations

import math

import numpy as np

from comnumpy.optical.fiber import FiberSpec

__all__ = ["gn_model_psi", "gn_model_nli_power", "gn_model_snr",
           "optimal_launch_power"]

# Eq. 120 weighs the channel on itself and the channels on it
# differently: self-channel interference carries 16/27, cross-channel
# twice that. GNPy names them SPM_WEIGHT and XPM_WEIGHT.
#
# The single-polarization row is Table I of Serena and Bononi (2015),
# whose scalar case reads (8/9)^2 {2 S_SGN + 4 S_XGN + ...}. Our scalar
# NLSE applies gamma rather than 8*gamma/9, so the (8/9)^2 divides out
# and the weights are exactly 2 and 4 -- 3.375 times the Manakov pair.
_SPM_WEIGHT = {2: 16.0 / 27.0, 1: 2.0}
_XPM_WEIGHT = {2: 32.0 / 27.0, 1: 4.0}


def gn_model_psi(delta_f_Hz: np.ndarray | float, baud_cut_Hz: float,
                 baud_pump_Hz: np.ndarray | float, beta2_ps2_km: float,
                 effective_length_km: float,
                 asymptotic_length_km: float) -> np.ndarray:
    r"""The :math:`\psi` factor of the GN model, eq. 123.

    Signal Model
    ------------
    The nonlinear interference a *pump* channel deposits on a *cut*
    channel is an integral over the two spectra of the phase-matching
    condition. When both are approximated by rectangles, that integral
    is available in closed form:

    .. math::

        \psi = \frac{L_{\mathrm{eff}}^2}
                    {4 \pi \left|\beta_2\right| L_a}
        \left[
        \mathrm{asinh}\left(\pi^2 L_a \left|\beta_2\right| R_{\mathrm{cut}}
                            \left(\Delta f + \tfrac{R_{\mathrm{pump}}}{2}\right)\right)
        - \mathrm{asinh}\left(\pi^2 L_a \left|\beta_2\right| R_{\mathrm{cut}}
                              \left(\Delta f - \tfrac{R_{\mathrm{pump}}}{2}\right)\right)
        \right]

    with :math:`L_a = 1/\alpha` the asymptotic length and
    :math:`\Delta f` the spacing between the two carriers. The
    :math:`\mathrm{asinh}` is the signature of the model: it is why the
    NLI grows only *logarithmically* with the bandwidth, and why doubling
    the number of channels costs far less than doubling their power.

    Axes: *element-wise* -- ``delta_f_Hz`` may be an array; everything
    else is scalar.

    Parameters
    ----------
    delta_f_Hz : float or np.ndarray
        Carrier spacing :math:`\Delta f` in Hz (0 for the channel on
        itself).
    baud_cut_Hz : float
        Symbol rate of the channel the NLI lands on, in Bd.
    baud_pump_Hz : np.ndarray or float
        Symbol rate of the channel producing it, in Bd.
    beta2_ps2_km : float
        Group-velocity dispersion in ps^2/km (the library's unit; see
        :attr:`~comnumpy.optical.fiber.FiberSpec.beta2`).
    effective_length_km : float
        :math:`L_{\mathrm{eff}} = (1 - e^{-\alpha L})/\alpha`, in km.
    asymptotic_length_km : float
        :math:`L_a = 1/\alpha`, in km.

    Returns
    -------
    np.ndarray
        :math:`\psi`, in km^2 s^-2 -- the bracket is dimensionless and
        the prefactor carries the units, chosen so that dividing by a
        squared symbol rate and multiplying by :math:`\gamma^2` in
        :func:`gn_model_nli_power` leaves W^-2.

    Raises
    ------
    ValueError
        If the dispersion is zero: the model divides by it, and a
        dispersionless fibre has no phase mismatch to integrate over.

    References
    ----------
    Poggiolini, JLT 30(24), 2012, eq. 123 (arXiv:1209.0394); GNPy
    ``NliSolver._psi``.

    Examples
    --------
    >>> from comnumpy.optical.fiber import get_fiber
    >>> fiber = get_fiber("SMF")
    >>> alpha = fiber.alpha_per_km
    >>> psi = gn_model_psi(0.0, 32e9, 32e9, fiber.beta2,
    ...                    fiber.effective_length_km(80.0), 1 / alpha)
    >>> print(f"{float(psi):.4e}")
    2.4259e+23
    """
    if beta2_ps2_km == 0:
        raise ValueError(
            "the GN model divides by the dispersion, and got beta2 = 0. A "
            "dispersionless fibre phase-matches every mixing product, so "
            "this closed form does not describe it -- use the split-step "
            "simulation instead.")
    beta2 = abs(beta2_ps2_km) * 1e-24            # ps^2/km -> s^2/km
    delta_f = np.asarray(delta_f_Hz, dtype=float)
    scale = math.pi ** 2 * asymptotic_length_km * beta2 * baud_cut_Hz
    psi = 0.5 * (np.arcsinh(scale * (delta_f + baud_pump_Hz / 2))
                 - np.arcsinh(scale * (delta_f - baud_pump_Hz / 2)))
    return psi * effective_length_km ** 2 / (
        2 * math.pi * beta2 * asymptotic_length_km)


def gn_model_nli_power(fiber: FiberSpec, *, span_length_km: float,
                       n_spans: int, powers_W: np.ndarray,
                       frequencies_Hz: np.ndarray, baud_rates_Hz: np.ndarray,
                       coherence_exponent: float = 0.0,
                       polarizations: int = 2) -> np.ndarray:
    r"""Nonlinear interference power of every channel of a comb, eq. 120.

    Signal Model
    ------------
    Each pair of channels contributes, and the contribution of the pump
    :math:`j` on the cut :math:`i` is cubic in the powers:

    .. math::

        P_{\mathrm{NLI},i} = \sum_j \eta_{ij} \, P_i P_j^2,
        \qquad
        \eta_{ij} = \frac{\gamma^2 \, w_{ij} \, \psi_{ij}}{R_j^2}

    with :math:`w_{ii} = 16/27` for the channel on itself and
    :math:`w_{ij} = 32/27` for a neighbour -- the factor two being the
    two ways a pair of pump photons can mix. :math:`\psi_{ij}` is
    :func:`gn_model_psi`. Those two numbers assume the **Manakov**
    equation, i.e. a dual-polarization signal; for a scalar simulation
    see ``polarizations`` below.

    Over :math:`N_s` identical spans the standard approximation is
    *incoherent* accumulation, each span adding its own noise:

    .. math::

        P_{\mathrm{NLI}}^{\mathrm{link}} = N_s^{1+\varepsilon} \,
        P_{\mathrm{NLI}}^{\mathrm{span}}

    :math:`\varepsilon = 0` being the incoherent case and a small
    positive value the empirical correction for the partial coherence a
    long link develops.

    Axes: *reduces the pair axis* -- the comb is described by three
    arrays of one entry per channel, and one NLI power per channel comes
    back.

    Parameters
    ----------
    fiber : FiberSpec
        Fibre of one span; only ``alpha_per_km``, ``beta2`` and
        ``gamma`` are read.
    span_length_km : float, keyword-only
        Length :math:`L` of one span in km.
    n_spans : int, keyword-only
        Number of identical spans :math:`N_s`.
    powers_W : np.ndarray, keyword-only
        Launch power of each channel, in W.
    frequencies_Hz : np.ndarray, keyword-only
        Carrier frequency of each channel, in Hz (only differences
        matter).
    baud_rates_Hz : np.ndarray, keyword-only
        Symbol rate of each channel, in Bd.
    coherence_exponent : float, optional, keyword-only
        :math:`\varepsilon` above. Default 0, the incoherent
        accumulation.
    polarizations : int, optional, keyword-only
        2 (default) for the dual-polarization Manakov link the model is
        written for, which is also what
        :class:`~comnumpy.optical.links.FiberLink` integrates when the
        field is shaped ``(..., 2, N)``. Pass 1 to compare against a
        *scalar* NLSE simulation -- a one-dimensional field -- whose
        weights are 2 and 4 instead, giving 3.375 times (5.3 dB) more
        interference at the same total power.

    Returns
    -------
    np.ndarray
        NLI power of each channel, in W, referred to the input of a
        span (the convention of the reference implementation: the link
        is lossless end to end once the amplifiers are counted).

    Raises
    ------
    ValueError
        If the three arrays do not have the same length, if the number
        of spans is not positive, or if ``polarizations`` is neither 1
        nor 2.

    References
    ----------
    Poggiolini, JLT 30(24), 2012, eq. 120; GNPy
    ``NliSolver._gn_analytic``; Serena and Bononi, JLT 33(7), 2015,
    Table I for the two sets of weights.

    Examples
    --------
    A single 32 GBd channel over one 80 km span of standard fibre, at
    0 dBm:

    >>> from comnumpy.optical.fiber import get_fiber
    >>> import numpy as np
    >>> power = np.array([1e-3])
    >>> nli = gn_model_nli_power(get_fiber("SMF"), span_length_km=80.0,
    ...                          n_spans=1, powers_W=power,
    ...                          frequencies_Hz=np.array([193.4e12]),
    ...                          baud_rates_Hz=np.array([32e9]))
    >>> print(f"{10 * np.log10(nli[0] / 1e-3):.2f} dBm")
    -36.25 dBm

    Eight neighbours on a 50 GHz grid make it worse, but far less than
    proportionally. If each neighbour hurt the centre channel as much as
    the channel hurts itself, the eight of them would add
    :math:`10\log_{10}(1 + 2 \times 8) = 12.3` dB; the
    :math:`\mathrm{asinh}` turns that into 4.3 dB:

    >>> comb = 193.4e12 + 50e9 * np.arange(-4, 5)
    >>> nli_comb = gn_model_nli_power(get_fiber("SMF"), span_length_km=80.0,
    ...                               n_spans=1, powers_W=np.full(9, 1e-3),
    ...                               frequencies_Hz=comb,
    ...                               baud_rates_Hz=np.full(9, 32e9))
    >>> print(f"{10 * np.log10(nli_comb[4] / 1e-3):.2f} dBm  "
    ...       f"({10 * np.log10(nli_comb[4] / nli[0]):.2f} dB worse)")
    -31.95 dBm  (4.30 dB worse)

    The same span integrated with the scalar equation instead, which is
    what a one-dimensional field gets, is 5.28 dB worse again:

    >>> scalar = gn_model_nli_power(get_fiber("SMF"), span_length_km=80.0,
    ...                             n_spans=1, powers_W=power,
    ...                             frequencies_Hz=np.array([193.4e12]),
    ...                             baud_rates_Hz=np.array([32e9]),
    ...                             polarizations=1)
    >>> print(f"{10 * np.log10(scalar[0] / nli[0]):.2f} dB")
    5.28 dB
    """
    powers = np.asarray(powers_W, dtype=float).ravel()
    frequencies = np.asarray(frequencies_Hz, dtype=float).ravel()
    bauds = np.asarray(baud_rates_Hz, dtype=float).ravel()
    if not (powers.size == frequencies.size == bauds.size):
        raise ValueError(
            f"one power, one frequency and one symbol rate per channel are "
            f"expected, got {powers.size}, {frequencies.size} and "
            f"{bauds.size}.")
    if n_spans <= 0:
        raise ValueError(f"a link has at least one span, got {n_spans}.")
    if polarizations not in _SPM_WEIGHT:
        raise ValueError(
            f"a fibre carries one or two polarizations, got "
            f"{polarizations}. Pass 2 (the default) for the Manakov "
            f"equation a coherent link uses, 1 to compare against a "
            f"scalar simulation.")

    alpha = fiber.alpha_per_km
    effective_length = fiber.effective_length_km(span_length_km)
    asymptotic_length = 1.0 / alpha

    # (cut, pump) grids: the cut is the channel the NLI lands on
    spacing = np.abs(frequencies[:, None] - frequencies[None, :])
    weight = np.where(np.eye(powers.size, dtype=bool),
                      _SPM_WEIGHT[polarizations], _XPM_WEIGHT[polarizations])
    psi = np.empty_like(spacing)
    for cut in range(powers.size):
        psi[cut] = gn_model_psi(spacing[cut], bauds[cut], bauds,
                                fiber.beta2, effective_length,
                                asymptotic_length)
    eta = fiber.gamma ** 2 * weight * psi / bauds[None, :] ** 2
    per_span = powers * np.sum(eta * powers[None, :] ** 2, axis=1)
    return per_span * n_spans ** (1.0 + coherence_exponent)


def gn_model_snr(ase_power_W: float, nli_coefficient: float,
                 power_W: np.ndarray | float) -> np.ndarray:
    r"""The two-noise SNR of a fibre link.

    Signal Model
    ------------
    .. math::

        \mathrm{SNR}(P) = \frac{P}{P_{\mathrm{ASE}} + \eta P^3}

    The amplifiers contribute a power-independent term, the fibre a
    cubic one, and the tension between them is the whole design problem:
    turning the power up buys signal linearly and noise cubically.

    Axes: *element-wise* in ``power_W``.

    Parameters
    ----------
    ase_power_W : float
        Amplified spontaneous emission power in the channel bandwidth,
        in W.
    nli_coefficient : float
        :math:`\eta` such that the NLI power is
        :math:`\eta P^3`, in 1/W^2. It is
        :func:`gn_model_nli_power` divided by :math:`P^3`.
    power_W : float or np.ndarray
        Launch power per channel, in W.

    Returns
    -------
    np.ndarray
        Signal-to-noise ratio, linear.

    References
    ----------
    Poggiolini et al., JLT 32(4), 2014, Section IV.

    Examples
    --------
    >>> import numpy as np
    >>> snr = gn_model_snr(1e-6, 5e3, np.array([1e-4, 1e-3]))
    >>> print(np.round(10 * np.log10(snr), 2))
    [19.98 22.22]
    """
    power = np.asarray(power_W, dtype=float)
    return power / (ase_power_W + nli_coefficient * power ** 3)


def optimal_launch_power(ase_power_W: float,
                         nli_coefficient: float) -> tuple[float, float]:
    r"""The launch power that maximizes the SNR, and that SNR.

    Signal Model
    ------------
    Setting the derivative of :math:`P / (P_{\mathrm{ASE}} + \eta P^3)`
    to zero gives a condition worth remembering in words -- **the
    optimum is where the nonlinear noise is half the ASE**:

    .. math::

        \eta P_{\mathrm{opt}}^3 = \frac{P_{\mathrm{ASE}}}{2},
        \qquad
        P_{\mathrm{opt}} = \left(\frac{P_{\mathrm{ASE}}}{2\eta}\right)^{1/3},
        \qquad
        \mathrm{SNR}_{\mathrm{max}} = \frac{2}{3}\,
        \frac{P_{\mathrm{opt}}}{P_{\mathrm{ASE}}}

    Two consequences follow immediately, and both are checked in
    ``tests/optical/test_gn_model.py`` rather than asserted here.

    First, the peak is **asymmetric**, because one side of it is
    governed by a linear term and the other by a cubic one. Missing the
    optimum by 1 dB costs 0.24 dB of SNR if the error is upwards but
    only 0.21 dB if it is downwards; by 3 dB, 2.20 dB against 1.50 dB.
    A link is therefore run slightly *under* its optimum, where a power
    error is the cheaper kind.

    Second, since :math:`P_{\mathrm{opt}} \propto P_{\mathrm{ASE}}^{1/3}`,
    ten times the accumulated ASE -- ten times the spans, near enough --
    moves the optimum by 10/3 dB, not by 10.

    Parameters
    ----------
    ase_power_W : float
        ASE power in the channel bandwidth, in W.
    nli_coefficient : float
        :math:`\eta` in 1/W^2, as in :func:`gn_model_snr`.

    Returns
    -------
    tuple of float
        Optimal launch power in W, and the SNR reached there (linear).

    Raises
    ------
    ValueError
        If either argument is not positive: without ASE the optimum is
        zero power, without nonlinearity it is infinite, and neither is
        a design point.

    References
    ----------
    Poggiolini et al., JLT 32(4), 2014, eq. 26-27.

    Examples
    --------
    >>> import numpy as np
    >>> power, snr = optimal_launch_power(1e-6, 5e3)
    >>> print(f"{10 * np.log10(power / 1e-3):.2f} dBm, "
    ...       f"{10 * np.log10(snr):.2f} dB")
    -3.33 dBm, 24.91 dB

    The condition, checked rather than asserted:

    >>> print(round(5e3 * power ** 3 / (1e-6 / 2), 12))
    1.0
    """
    if ase_power_W <= 0 or nli_coefficient <= 0:
        raise ValueError(
            f"the optimum balances two noises against each other and needs "
            f"both, got ase_power_W={ase_power_W} and "
            f"nli_coefficient={nli_coefficient}. Without ASE the best "
            f"power is zero; without nonlinearity it is unbounded.")
    power = (ase_power_W / (2 * nli_coefficient)) ** (1 / 3)
    return power, float(gn_model_snr(ase_power_W, nli_coefficient, power))
