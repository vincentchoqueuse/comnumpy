Probabilistic Shaping
=====================

A uniform QAM does not reach the capacity of the AWGN channel, and the reason
is not the receiver: it is the *distribution* of the transmitted symbols.
Probabilistic shaping changes that distribution, and this tutorial follows the
argument from the limit down to the block that produces it.

.. note::

   **Before you start.** :doc:`awgn` introduced the chain and the error rate.
   Here nothing is simulated for most of the page -- the quantities are
   information-theoretic and computed in closed form.

**What you'll learn:**

- Why the AWGN capacity is reached by a Gaussian input, and what that costs a
  finite constellation.
- How the Maxwell-Boltzmann law closes part of that gap, and how much.
- How a distribution matcher produces a non-uniform law from uniform bits,
  and what a finite block costs in rate.
- Where the whole thing sits in a real chain, and why the metric changes name
  when it gets there.


The AWGN Channel and its Capacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The channel is

.. math::

   Y = X + N, \qquad N \sim \mathcal{CN}(0, \sigma^2)

and under a power constraint :math:`\mathbb{E}[|X|^2] \leq P` its capacity is

.. math::

   C = \log_2\left(1 + \mathrm{SNR}\right), \qquad
   \mathrm{SNR} = \frac{P}{\sigma^2}

The input that reaches it is **Gaussian**. The short reason is that
:math:`I(X;Y) = h(Y) - h(Y|X) = h(Y) - h(N)`, so maximizing the mutual
information means maximizing the differential entropy of :math:`Y` at a fixed
variance -- and the Gaussian is the maximum-entropy law at fixed variance.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 1-29


Why a Uniform QAM Is Not Optimal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A real transmitter does not send a Gaussian. It sends one of :math:`M`
constellation points, and by default it sends them equiprobably. The rate is
then the mutual information :math:`I(X;Y)` of that discrete input, which
:func:`~comnumpy.core.capacity.constellation_capacity` evaluates:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 31-59

.. image:: img/probabilistic_shaping_fig1.png
   :width: 100%
   :align: center
   :alt: AWGN capacity against the mutual information of uniform QAM

.. code::

   SNR   capacity    4-QAM   16-QAM   64-QAM  256-QAM
     0 dB     1.00      0.97      0.99      0.99      0.99
     8 dB     2.87      1.95      2.68      2.73      2.74
    16 dB     5.35      2.00      3.97      4.96      5.03
    24 dB     7.98      2.00      4.00      6.00      7.40

Two things are visible, and they are different.

A finite constellation **saturates**: :math:`I(X;Y) \leq H(X) \leq \log_2 M`,
so 4-QAM stops at 2 bits whatever the SNR, and the capacity keeps climbing.
That part is fixed by using a bigger constellation.

The second is the one shaping is about. Even below saturation, the uniform
curve sits **below** the capacity -- 256-QAM reaches 5.03 bits at 16 dB where
the channel offers 5.35, and it does so with sixteen times the points of the
4-QAM that is already saturated there. Enlarging the constellation buys the
first part and not the second, because a uniform law on a square grid is not
the Gaussian the channel wants.


Probabilistic Shaping
^^^^^^^^^^^^^^^^^^^^^

Instead of moving the points, change how often each is sent: the inner ones
more, the outer ones less. The natural law is **Maxwell-Boltzmann**,

.. math::

   P_X(x_i) = \frac{e^{-\lambda \left|x_i\right|^2}}
                   {\sum_j e^{-\lambda \left|x_j\right|^2}}

which is the maximum-entropy law on the constellation at a given average
energy -- the discrete counterpart of the argument that made the Gaussian
optimal. The parameter :math:`\lambda` sets the strength: :math:`\lambda = 0`
is uniform, and larger values concentrate the probability at the centre.

How to choose it? Either solve for a target average power
:math:`\sum_i P_X(x_i)|x_i|^2 = P`, a scalar problem, or optimize it directly
for the rate, :math:`\lambda^{*} = \arg\max_\lambda I(X;Y)`. The second is
what the figure below does, at every SNR.

One precaution matters more than it looks. Shaping *lowers* the average
energy, so a shaped constellation compared as it stands would simply be a
quieter one. The constellation is therefore rescaled back to unit power
before the mutual information is computed: the comparison is at **equal
transmit power**, which is the only comparison that means anything.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 61-116

.. image:: img/probabilistic_shaping_fig2.png
   :width: 100%
   :align: center
   :alt: capacity, uniform 64-QAM and shaped 64-QAM

.. code::

   SNR    lambda    H(P_X)   uniform    shaped   capacity   gap closed
     0 dB    6.022    3.896     0.992     1.000      1.000       100 %
     4 dB    4.779    4.228     1.765     1.812      1.812       100 %
     8 dB    3.350    4.721     2.735     2.868      2.870        99 %
    12 dB    2.072    5.287     3.825     4.047      4.075        89 %
    16 dB    1.048    5.757     4.961     5.142      5.351        46 %
    20 dB    0.318    5.974     5.801     5.830      6.658         3 %
    24 dB    0.016    6.000     5.996     5.996      7.978         0 %

.. code::

   to carry 4 bit/symbol: uniform 12.61 dB, shaped 11.84 dB, saving 0.77 dB

Read the ``lambda`` column against the last one. Where the constellation is
comfortably larger than the rate the channel can carry, shaping closes almost
the whole gap: at 8 dB the shaped 64-QAM reaches 2.868 bits against a capacity
of 2.870. Where the constellation is saturated, :math:`\lambda^{*}` falls to
zero and shaping has nothing left to give -- at 24 dB the best law *is* the
uniform one, and the remaining 2 bits of gap need a bigger constellation, not
a better distribution.

In between is where a link actually operates, and there the saving is real:
0.77 dB of SNR to carry 4 bits per symbol. The asymptotic value for a dense
constellation is 1.53 dB, the shaping gain of a sphere over a cube.

The rest of the page works at one operating point, 18 dB, and on the law
that lives there:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 118-130

.. code::

   at 18 dB: lambda = 0.640, H(amplitudes) = 1.830 bits


Distribution Matching
^^^^^^^^^^^^^^^^^^^^^

The law is one thing. A transmitter is fed an **equiprobable** bit stream, so
something has to turn uniform bits into symbols with that law, invertibly:

.. math::

   \text{uniform bits} \;\xrightarrow{\ \mathrm{DM}\ }\;
   \text{shaped symbols}

**Constant composition distribution matching** does it by fixing the
composition. On a block of length :math:`n`, symbol :math:`i` appears exactly
:math:`n_i` times, so every block has the same empirical law by construction,
and the matcher is an enumeration of the arrangements. There are

.. math::

   N = \binom{n}{n_1, \ldots, n_M} = \frac{n!}{n_1!\,n_2!\cdots n_M!}

of them, so the block carries :math:`k = \lfloor \log_2 N \rfloor` bits and
the rate is :math:`k/n`. As :math:`n` grows, :math:`k/n \to H(X)`; at finite
:math:`n` it falls short, and that shortfall is the **rate loss**.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 132-145

.. code::

      n   composition                            k/n     H(X)  rate loss
     16   (6, 5, 3, 2)                          1.500    1.830      0.383
     64   (26, 20, 12, 6)                       1.688    1.830      0.138
    256   (103, 80, 49, 24)                     1.781    1.830      0.048
   1024   (410, 321, 198, 95)                   1.815    1.830      0.015

The rate loss falls roughly as :math:`\log_2(n)/n`, so it is a matter of
block length: 0.38 bit per symbol at :math:`n = 16`, 0.015 at
:math:`n = 1024`. Long blocks cost latency and arithmetic instead, which is
what the other matchers -- enumerative sphere shaping, multiset-partition DM
-- exist to trade differently. The library ships
:class:`~comnumpy.core.shaping.SphereShaper` for the first.


From Theory to a Chain: PAS and GMI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matcher shapes *amplitudes*, and a QAM symbol is an amplitude and a sign.
**Probabilistic amplitude shaping** puts the two together: the matcher
produces the shaped amplitudes, the signs stay uniform, and the parity bits of
a systematic FEC code supply them -- which is why the shaping and the coding
compose instead of fighting.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 147-159

.. mermaid:: mermaid/shaping_pas.mmd

Uniform bits go in, shaped symbols come out, so the check is to run it and
count what comes out. The signs are equiprobable, so the law the signed
8-PAM carries is

.. math::

   P_Y(\pm a_i) = \tfrac{1}{2} P_A(a_i)

which is the Maxwell-Boltzmann law on the full constellation at the same
:math:`\lambda` -- the curve the measured frequencies are laid over:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 161-188

.. image:: img/probabilistic_shaping_fig3.png
   :width: 100%
   :align: center
   :alt: measured symbol frequencies against the Maxwell-Boltzmann law

.. code::

   largest deviation from the target law: 0.0014

The bars sit on the curve, and the residual 0.0014 is not sampling noise:
a constant-composition matcher produces the *quantized* law
:math:`n_i/n` exactly, in every block. At :math:`n = 256` the composition
(103, 80, 49, 24) gives 0.2012 where the law asks for 0.2003, and that
0.001 is what the figure shows. The same quantization is what the rate loss
of the previous section measures.

Once the chain is real, the metric changes name. The mutual information is
what the *symbol-wise* channel offers. A chain that demaps to per-bit
log-likelihood ratios and hands them to a binary decoder does not get it: the
parallel bit channels ignore what the bits of a symbol share. What it gets is
the generalized mutual information,
:func:`~comnumpy.core.capacity.bicm_capacity`, which with a shaped law is
:math:`H(X) - \sum_i H(B_i \mid Y)`.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 190-202

.. code::

   SNR    MI uniform  GMI uniform   MI shaped  GMI shaped
     0 dB       0.992        0.843       0.996       0.815
     8 dB       2.735        2.581       2.805       2.674
    16 dB       4.961        4.960       5.116       5.116
    24 dB       5.996        5.996       5.900       5.900

The GMI sits below the MI, and the gap is the price of bit-wise demapping. It
is largest at low SNR, where a symbol's bits are most dependent, and closes at
high SNR where each bit is decided on its own. In optical systems the same
quantity normalized by the entropy -- the NGMI -- is what a soft-decision FEC
threshold is quoted against.

Note the last row: the shaped 64-QAM carries *less* than the uniform one at
24 dB, because its entropy is below 6 bits. Shaping is not free, and it is not
always the right choice -- which the ``lambda`` column of the previous section
already said.


Conclusion
^^^^^^^^^^

The argument, in one line:

.. code::

   Gaussian optimal -> discrete QAM -> gap to capacity
                    -> probabilistic shaping -> distribution matching

This tutorial highlighted:

- Why the AWGN capacity is reached by a Gaussian input, and why a uniform QAM
  is not one.
- How the Maxwell-Boltzmann law closes most of that gap where a link operates,
  and none of it where the constellation is saturated.
- How a constant-composition matcher produces the law from uniform bits, and
  what a finite block costs.
- Where it sits in a chain, and why the achievable rate there is the GMI.

Key takeaway:
**The AWGN capacity is reached by a Gaussian input. A uniform QAM does not have
that distribution, so it leaves a gap that a larger constellation does not
close. Probabilistic shaping changes the probabilities instead of the points,
and a distribution matcher produces those probabilities from uniform bits.**


References
^^^^^^^^^^

- G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and
  rate-matched low-density parity-check coded modulation", *IEEE Trans.
  Commun.*, vol. 63, no. 12, pp. 4651-4665, 2015 -- probabilistic amplitude
  shaping, and the rate it achieves.
- P. Schulte and G. Böcherer, "Constant composition distribution matching",
  *IEEE Trans. Inf. Theory*, vol. 62, no. 1, pp. 430-434, 2016.
- J. Cho and P. J. Winzer, "Probabilistic constellation shaping for optical
  fiber communications", *J. Lightwave Technol.*, vol. 37, no. 6,
  pp. 1590-1607, 2019.
- G. D. Forney and L.-F. Wei, "Multidimensional constellations -- Part I",
  *IEEE J. Sel. Areas Commun.*, vol. 7, no. 6, pp. 877-892, 1989 -- the
  1.53 dB.
- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed.,
  Wiley, 2006, Chapters 8 and 9.
