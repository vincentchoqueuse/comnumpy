Digital Back-Propagation
========================

The previous tutorial *budgeted* for the fibre nonlinearity: it predicted how
much nonlinear noise a link makes and chose the launch power that minimizes
the total. This tutorial asks the other question -- whether the receiver can
**undo** the nonlinearity instead, and what that is worth.

.. note::

   **Before you start.** :doc:`gn_model` introduced the fibre, the
   split-step propagation and the launch-power trade-off. They are used here
   without being re-explained. Everything below is single-polarization.

**What you'll learn:**

- How a signal degrades span after span, and why a linear receiver cannot
  follow it.
- What digital back-propagation is, in one equation.
- What it buys in effective SNR and in symbol error rate, and what it costs
  in computation.

This tutorial is suited for engineers and students interested in optical
communications and nonlinear fiber effects.


Channel Model
^^^^^^^^^^^^^

Propagation along a single-mode fibre is governed by the nonlinear
Schrödinger equation, which for one polarization reads

.. math::

   \frac{\partial u}{\partial z} =
   \underbrace{-\frac{\alpha}{2}\, u
   - j\frac{\beta_2}{2}\frac{\partial^2 u}{\partial t^2}}_{\text{linear}}
   + \underbrace{j \gamma \left|u\right|^2 u}_{\text{nonlinear}}

where :math:`\alpha` is the attenuation, :math:`\beta_2` the group-velocity
dispersion and :math:`\gamma` the Kerr coefficient. The two terms are the
whole difficulty: the linear one is diagonal in frequency, the nonlinear one
is diagonal in time, and no basis diagonalizes both.

The **split-step Fourier method** exploits exactly that. Over a step short
enough, the two act almost independently, so each is applied where it is
diagonal:

.. math::

   u(z + \delta) \simeq
   \mathcal{F}^{-1}\!\left\{ e^{\left(-\frac{\alpha}{2}
   + j\frac{\beta_2}{2}\omega^2\right)\delta}\,
   \mathcal{F}\!\left\{ e^{j\gamma\left|u(z)\right|^2\delta}\, u(z)\right\}\right\}

one FFT pair per step. A link is :math:`N_{sp}` spans of that, each followed
by an amplifier which restores the power and adds spontaneous-emission noise
of variance :math:`N_{ase} = (G-1)h\nu\,n_{sp}`, with
:math:`n_{sp} = \mathrm{NF}/\left(2(1 - 1/G)\right)`.

.. note::

   The factor two in :math:`n_{sp}` is the definition of the optical noise
   figure, :math:`\mathrm{NF} = 2 n_{sp}(1 - 1/G)`, not a modelling choice.
   A fully inverted amplifier has :math:`n_{sp} = 1` and therefore
   :math:`\mathrm{NF} = 3` dB: that is the quantum limit of a
   phase-insensitive amplifier, and no EDFA does better.

Implementation
""""""""""""""

We reproduce the setup of Häger and Pfister: 25 spans of 80 km, 16-QAM at
10.7 GBd, root-raised-cosine pulses, and 500 split steps per span so that the
propagation itself is not in question.

.. literalinclude:: ../../examples/optical/one_shot_NLI.py
   :language: python
   :lines: 1-44

The link is one block. Its ``post_span`` callback is called after each
amplifier, which is how the degradation figure below is measured without
running the propagation twenty-five times:

.. literalinclude:: ../../examples/optical/one_shot_NLI.py
   :language: python
   :lines: 46-76

.. code::

   25 spans at 500 steps per span: 40.6 s

.. image:: img/one_shot_nli_fig1.png
   :width: 100%
   :align: center
   :alt: Received signal at the end of the link

At the output of the fibre nothing is recognizable: the dispersion has spread
each symbol over hundreds of neighbours, so the field looks Gaussian whatever
was transmitted.


Degradation, span by span
^^^^^^^^^^^^^^^^^^^^^^^^^

The receiver first has to undo the dispersion, which is a linear all-pass and
costs one FFT pair. Applying it over exactly the spans travelled gives what a
linear receiver would see if the link stopped there:

.. literalinclude:: ../../examples/optical/one_shot_NLI.py
   :language: python
   :lines: 78-142

.. image:: img/one_shot_nli_fig2.png
   :width: 100%
   :align: center
   :alt: Constellations after 1, 12 and 25 spans

.. code::

   span   effective SNR
      1          22.47 dB
      5          21.15 dB
     10          18.89 dB
     15          17.22 dB
     20          15.78 dB
     25          14.61 dB

.. image:: img/one_shot_nli_fig3.png
   :width: 100%
   :align: center
   :alt: Effective SNR against the number of spans

**Nearly 8 dB lost over the link**, and the loss is not the amplifier noise
alone. Two things accumulate span after span: the ASE, which adds one
amplifier's worth of noise each time, and the nonlinear interference, which
the dispersion converts into a scatter that no linear filter can undo. The
constellations show the difference in kind -- after one span the clusters are
tight and merely rotated, after twenty-five they are rotated *and* smeared.


Digital Back-Propagation
^^^^^^^^^^^^^^^^^^^^^^^^

The nonlinear Schrödinger equation is deterministic and invertible. In the
absence of noise, propagating the received field back through the same
equation with the signs reversed,

.. math::

   \frac{\partial u}{\partial z} =
   \frac{\alpha}{2}\, u + j\frac{\beta_2}{2}\frac{\partial^2 u}{\partial t^2}
   - j \gamma \left|u\right|^2 u

returns exactly what was transmitted. That is **digital back-propagation**:
the same split-step method the channel was simulated with, run backwards. Its
only approximation is the step size, and its only genuine limit is the ASE
noise, which was added *along* the link and is amplified rather than removed
by the backward pass.

Dispersion compensation alone is the same algorithm with the nonlinear term
switched off -- one step per span instead of :math:`\mathrm{StPS}` -- so the
two strategies of this tutorial differ by one argument.

Results
"""""""

.. literalinclude:: ../../examples/optical/one_shot_NLI.py
   :language: python
   :lines: 144-165

.. code::

   dispersion compensation    SNR=14.61 dB  SER=0.0322  residual phase= -39.3 deg    10 ms
   digital back-propagation   SNR=19.50 dB  SER=0.0003  residual phase=  -1.0 deg  2101 ms

.. image:: img/one_shot_nli_fig4.png
   :width: 100%
   :align: center
   :alt: Constellations after linear compensation and after DBP

The residual phase names the culprit. Dispersion compensation undoes the
dispersion and the loss and leaves a **39 degree** rotation of the whole
constellation: that is self-phase modulation, the Kerr effect turning average
power into phase. Removing that rotation is not enough either -- the
remaining scatter still costs 3.2 % of the symbols, because the nonlinearity
acts *along* the fibre, interleaved with the dispersion, and not as one
rotation at the end.

Back-propagation inverts that interleaving. The residual rotation falls to one
degree, the effective SNR rises by **4.9 dB**, and the symbol error rate drops
by two orders of magnitude.


What it costs
^^^^^^^^^^^^^

The last column of the table is the reason DBP is not simply switched on
everywhere. Dispersion compensation is one FFT pair for the whole link; DBP at
:math:`\mathrm{StPS}` steps per span is :math:`N_{sp} \times \mathrm{StPS}`
FFT pairs plus as many pointwise phase rotations. Here that is 10 ms against
2101 ms -- **210 times** -- for 4.9 dB.

That ratio is what the literature on low-complexity back-propagation exists to
improve, and it is also why the useful question is not "DBP or not" but *how
many steps*.


Monte Carlo Evaluation
^^^^^^^^^^^^^^^^^^^^^^

We therefore sweep the launch power, with one receiver per step count, on the
second link of Häger and Pfister: ten spans of 100 km, 16-QAM at 32 GBd. The
reference curve is the same link with its nonlinearity switched off, i.e. what
the amplifier noise alone would allow.

.. literalinclude:: ../../examples/optical/NLI_simulation.py
   :language: python
   :lines: 1-78

.. literalinclude:: ../../examples/optical/NLI_simulation.py
   :language: python
   :lines: 80-132

.. code::

   launch power [dBm]   -6.0   -4.5   -3.0   -1.5    0.0    1.5    3.0    4.5
   amplifier noise only     15.3   16.4   17.6   18.6   19.3   20.1   21.1   21.2
   dispersion compensation  15.2   16.2   17.1   17.6   17.2   16.3   14.0   11.2
   DBP, 1 step/span         15.2   16.3   17.3   17.9   17.8   17.2   15.4   12.7
   DBP, 2 steps/span        15.2   16.3   17.4   18.3   18.6   18.5   17.8   15.8
   DBP, 4 steps/span        15.2   16.4   17.5   18.5   19.1   19.6   20.2   19.3
   DBP, 50 steps/span       15.2   16.4   17.6   18.5   19.2   19.7   20.5   19.9

.. image:: img/nli_simulation_fig1.png
   :width: 100%
   :align: center
   :alt: Effective SNR against launch power, one curve per step count

This is the figure the whole subject is about. Every curve rises at low power,
where the amplifier noise dominates and turning the laser up helps; every
curve except the reference then turns over, where the nonlinear interference
grows faster than the signal. The maximum of each curve is the operating point
of that receiver, and back-propagation **moves it to the right**:

.. code::

   receiver                  best SNR   at power    total time
   amplifier noise only      21.18 dB    4.5 dBm       0.1 s
   dispersion compensation   17.61 dB   -1.5 dBm       0.1 s
   DBP, 1 step/span          17.93 dB   -1.5 dBm       0.2 s
   DBP, 2 steps/span         18.56 dB    0.0 dBm       0.4 s
   DBP, 4 steps/span         20.21 dB    3.0 dBm       0.8 s
   DBP, 50 steps/span        20.53 dB    3.0 dBm       8.7 s

Read the last two columns together. Going from one step per span to four buys
**2.3 dB**; going from four to fifty buys **0.3 dB more** and costs eleven
times the computation. The returns collapse because the step-size error falls
with the number of steps while the ASE noise does not: at fifty steps the
receiver is still 0.65 dB from the noise-only bound, and no number of steps
will close that last gap.

Against dispersion compensation alone, four steps per span are worth 2.6 dB
of effective SNR **and** 4.5 dB of launch power, which is the number a link
budget actually spends.

.. image:: img/nli_simulation_fig2.png
   :width: 100%
   :align: center
   :alt: The same sweep, in symbol error rate

The error rate says the same thing in the units a link is specified in, and
adds the shape the SNR curve hides: a bathtub, whose left wall is the
amplifier noise and whose right wall is the nonlinearity. Back-propagation
does not move the left wall -- it cannot -- and pushes the right one out.

The flat bottom at :math:`1.2 \times 10^{-4}` is the estimator, not the link:
8192 symbols per point cannot resolve fewer than one error, so the two
best receivers are floor-limited from 0 dBm on. That is why the comparison
above is made in effective SNR, which uses every symbol rather than only the
wrong ones.


Conclusion
^^^^^^^^^^

This tutorial highlighted:

- How to propagate a signal through a nonlinear fibre with the split-step
  method, and how to watch it degrade span by span with a callback.
- Why the damage is of two kinds, and why only one of them is linear.
- What digital back-propagation is, and that it is the channel model run
  backwards.
- What it buys -- 4.9 dB and two decades of error rate -- and what it costs.

Key takeaway:
**The fibre nonlinearity is deterministic, so it can be inverted; what cannot
be inverted is the amplifier noise that was added along the way. Digital
back-propagation buys back most of the nonlinear penalty, at a computational
cost proportional to the number of steps it takes.**

References
^^^^^^^^^^

- C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
  communication systems", *IEEE J. Sel. Areas Commun.*, vol. 39, no. 1,
  pp. 280-294, 2021 (arXiv:2010.14258) -- the link parameters, the noise
  model and the step-count comparison used here.
- E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
  using digital backpropagation", *J. Lightwave Technol.*, vol. 26, no. 20,
  pp. 3416-3425, 2008.
- G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013,
  Chapter 2 -- the split-step method and its step-size error.
