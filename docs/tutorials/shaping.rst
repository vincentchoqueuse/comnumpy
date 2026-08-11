Probabilistic Shaping
=====================

A constellation is two things: a set of points, and a law saying how often
each one is sent. Every earlier tutorial chose the points carefully and then
took the law for granted -- ``SymbolGenerator`` draws uniformly, and nobody
asked why.

This one asks. It turns out the uniform law is never the right answer on a
Gaussian channel, that the right answer has a name and a closed form, and
that using it costs a piece of machinery -- a *distribution matcher* --
because the data we have to send is uniform whether we like it or not.

.. note::

   **Before you start.** This tutorial reads rates rather than error rates,
   so it assumes you are comfortable with a chain
   (:doc:`../getting_started/first_simulation`) and have met mutual
   information somewhere. It sits naturally after :doc:`coding`: shaping and
   coding are the two halves of the same transmitter, which is what the PAS
   architecture in Part 4 is about.

**What you'll learn:**

- What a constellation actually carries over a noisy channel -- entropy,
  mutual information, and the generalized mutual information a *real*
  receiver is limited by.
- Which law maximizes that rate, computed rather than asserted, and how
  close the closed-form Maxwell-Boltzmann law comes to it.
- How to draw from a law and watch the histogram converge to it.
- Why a distribution matcher is needed at all, and what a finite block costs.
- How all of this carries over to a complex constellation, and why shaping
  is deployed on 256-QAM and not on QPSK.


Introduction
^^^^^^^^^^^^

Prerequisites
"""""""""""""

Make sure you have the following Python libraries installed:

.. code::

   numpy
   matplotlib
   comnumpy

Import Libraries
""""""""""""""""

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 1-17

Define Parameters
"""""""""""""""""

We work on **16-PAM**, written on its natural odd-integer grid
:math:`\{\pm 1, \pm 3, \ldots, \pm 15\}`. Shaping is a one-dimensional
operation -- a square QAM constellation is the product of two PAM axes, so
shaping the axis shapes the QAM, which is what Part 5 makes explicit.

The constellation comes from :func:`~comnumpy.core.utils.get_alphabet`
rather than from ``np.arange``, and that matters twice over. Its order is
the **Gray labelling**, which the bit-wise rate of Part 1 depends on; and in
that order the most significant bit is the *sign* while the three others are
the *amplitude*, which is exactly the decomposition PAS needs in Part 4.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 19-27


Part 1: what a constellation carries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start with the law nobody chose: the uniform one. Its entropy

.. math::

   H(P) = -\sum_i p_i \log_2 p_i = 4 \ \text{bit/symbol}

is the number of bits a symbol carries when the channel is perfect. It is an
upper bound and nothing more -- add noise and the receiver stops being able
to tell some points apart.

What survives the noise is the **mutual information**

.. math::

   I(X;Y) = \sum_{x} P_X(x) \int f_{Y|X}(y|x)
            \log_2 \frac{f_{Y|X}(y|x)}{f_Y(y)}\, \mathrm{d}y

the rate a decoder working on *symbols* can be driven to. But no practical
receiver works on symbols: a soft-decision system wraps a binary code around
a demapper that emits one L-value per bit, so the decoder sees :math:`m`
parallel binary channels. The rate *that* structure reaches is the
**generalized mutual information**,

.. math::

   \mathrm{GMI} = \sum_{k=1}^{m} I(B_k; Y) \leq I(X;Y)

and the gap between the two is the price of the bit-wise interface (Alvarado
*et al.*, 2015). It is small for a Gray labelling and large for a bad one,
which is the whole reason Gray labelling is used.

Both are available two ways in the library, and the tutorial uses both on
purpose: :func:`~comnumpy.core.capacity.constellation_capacity` and
:func:`~comnumpy.core.capacity.bicm_capacity` integrate them by quadrature,
while :func:`~comnumpy.core.information.compute_mi` and
:func:`~comnumpy.core.information.compute_gmi` estimate them from received
samples.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 34-56

.. warning::

   **The factor of two between a real channel and a complex one.** The
   estimators and the quadrature both use the complex convention, where the
   noise variance :math:`\sigma^2` is split over two dimensions. A *real*
   PAM channel of noise variance :math:`\sigma^2` is therefore passed as
   :math:`\rho = 1/(2\sigma^2)`, and an SNR :math:`s` on a constellation of
   energy :math:`E` as :math:`\rho = s/(2E)`. Get it wrong and every curve
   on the page moves by 3 dB while still looking perfectly plausible.

   Dividing by :math:`E` is not bookkeeping either. It is what makes the
   comparison in Part 3 fair: a shaped law spends less energy, so *at equal
   power* its constellation is wider -- and that extra width is where the
   entire shaping gain comes from.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 59-62

.. code::

   16-PAM, uniform law: H = 4.000 bit/symbol, energy 85, shaping gain 0.017 dB

The 0.017 dB is the calibration of :func:`~comnumpy.core.shaping.shaping_gain_dB`:
a uniform law must score essentially zero on any scale of shaping gain worth
having.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 64-73

The measurement uses the chain the rest of the page reuses -- a source, a
mapper, a channel -- with the transmitted symbols tapped so every estimator
has its reference:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 75-94

.. mermaid:: mermaid/shaping_study.mmd

``noise.sigma2_`` is the variance the run that has just finished actually
applied -- a data-dependent attribute, hence the trailing underscore (D24) --
so the estimator is told the same channel the chain used rather than the one
it was asked for.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 97-114

.. image:: img/probabilistic_shaping_fig1.png
   :width: 100%
   :align: center

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 116-122

.. code::

    SNR      MI       GMI    MI-GMI    measured MI   measured GMI
       4   0.8840   0.7695   0.1145        0.8826         0.7669
      10   1.6419   1.5157   0.1262        1.6444         1.5179
      16   2.5151   2.4542   0.0609        2.5180         2.4560
      22   3.4329   3.4324   0.0005        3.4315         3.4309
      28   3.9755   3.9755   0.0000        3.9749         3.9749

Three things to take from this table.

The **markers land on the lines**: an integral over a Gaussian weight and a
mean over 120 000 noisy samples agree to a few thousandths of a bit, which
is the Monte-Carlo error. They share no code path, so this is a real check
of both.

The **bit-wise interface costs about 0.12 bit** at low SNR and nothing at
all above 20 dB. With a Gray labelling the loss vanishes as soon as the
symbol decisions become reliable, because a symbol error then almost always
flips a single bit.

And the curve **saturates at 4 bit/symbol**, the entropy, however clean the
channel gets. That ceiling is the first hint that the law matters: a
constellation cannot carry more than its own entropy, so a law of lower
entropy buys nothing at high SNR -- and, as the next part shows, quite a lot
in the middle.


Part 2: which law maximizes the rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now the design question. Among all laws on this constellation spending at
most a given energy, which one carries the most bits?

.. math::

   \max_{P_X} \; I(X;Y)
   \quad \text{subject to} \quad \sum_i p_i |a_i|^2 \leq E

This has no closed form. It is a concave program, and the classical way to
solve it is the **Blahut-Arimoto** algorithm (Blahut, 1972; Arimoto, 1972):
alternate between the posterior :math:`q(i \mid y)` implied by the current
law and the law implied by that posterior,

.. math::

   q(i \mid y) = \frac{p_i f(y|a_i)}{\sum_j p_j f(y|a_j)},
   \qquad
   p_i \;\leftarrow\; \frac{e^{D_i - \lambda |a_i|^2}}
                           {\sum_j e^{D_j - \lambda |a_j|^2}},
   \qquad
   D_i = \int f(y|a_i) \log q(i \mid y)\, \mathrm{d}y

with :math:`\lambda` the Lagrange multiplier of the energy constraint. This
is :func:`~comnumpy.core.shaping.blahut_arimoto`, and the Gaussian integral
is done by Gauss-Hermite quadrature rather than sampled, so the answer is
deterministic.

Against it stands the law this library actually uses everywhere else. Among
all laws of a given energy, the one of maximum **entropy** is the
Maxwell-Boltzmann family

.. math::

   p_i = \frac{e^{-\lambda |a_i|^2}}{\sum_j e^{-\lambda |a_j|^2}},
   \qquad \lambda \geq 0

which :func:`~comnumpy.core.shaping.maxwell_boltzmann` computes in closed
form. Note carefully that this is the answer to a *different question*:
maximum entropy at a given energy, not maximum rate over a given channel.
The two coincide only in the limit of a Gaussian channel with a Gaussian
input. How much the difference costs is a number, and the point of this part
is to measure it rather than cite it (Kschischang and Pasupathy, 1993).

Comparing them means comparing at equal energy, and it is the closed form
that gets matched -- bisecting :func:`~comnumpy.core.shaping.blahut_arimoto`
for a target energy would cost a full alternating maximization per step,
while matching Maxwell-Boltzmann to an energy costs nothing:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 129-153

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 156-175

.. code::

   At 18 dB on the uniform law, sigma^2 = 1.347

     lambda   energy   H(P)   MI(best)   MI(Maxwell-Boltzmann)     gap
      0.000    96.63  3.918     2.8475       (no MB law spends that much)
      0.002    82.63  3.956     2.8277               2.8148   0.0129
      0.006    59.80  3.895     2.6995               2.6941   0.0053
      0.020    23.52  3.323     2.1025               2.1024   0.0001
      0.060     6.99  2.449     1.3145               1.3145   0.0000

**The closed form is worth using.** At every energy where the comparison is
defined the gap is at most 0.013 bit, and it collapses to nothing as the
constraint tightens. That is the result Kschischang and Pasupathy proved and
the reason the rest of this module never mentions Blahut-Arimoto again: one
line of closed form buys 99.5 % of an iterative solver.

.. warning::

   **Look at the first row.** With :math:`\lambda = 0` -- no energy
   constraint at all -- the maximizer spends **96.63**, more than the
   uniform law's 85. It pushes mass *outwards*, because with energy free the
   best thing to do with the outer points is to use them more, not less.

   Probabilistic shaping is a statement about a *constrained* problem. The
   bell curve everyone pictures is the answer to "most bits for this much
   energy", never to "most bits". Remove the constraint and the answer
   inverts.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 177-181

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 183-202

.. image:: img/probabilistic_shaping_fig2.png
   :width: 100%
   :align: center

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 204-213

.. code::

   At 6 dB the maximizer sets 6 of the 16 probabilities to exactly zero, and
   spends 86.4 doing it -- more than the uniform 85.
   No Maxwell-Boltzmann law does either. At lambda = 0.5 the outermost point
   still keeps 1.1e-49, and asking the family for that energy is refused
   outright:
     no Maxwell-Boltzmann law on this constellation spends 86.35: the family
     tops out at the uniform law's 85.00, at lambda = 0.

The right-hand panel is the one worth staring at. On a noisy enough channel
the optimal law does something no Maxwell-Boltzmann law can: it **switches
points off entirely**. Two points the receiver cannot tell apart are worth
less than one point used twice as often, so the maximizer thins the
constellation out and spends the freed energy separating what is left --
which is why it ends up above the uniform energy, not below it.
Maxwell-Boltzmann gives every point a strictly positive probability whatever
:math:`\lambda` does: at :math:`\lambda = 0.5` the outermost point is down to
:math:`10^{-49}`, which is negligible but is not zero.

The refusal in that output is deliberate. The family spends *at most* what
the uniform law spends, so a bisection asked for more would run to
:math:`\lambda = 0` and return the uniform law as though it had succeeded --
the comparison would then silently be against the wrong thing. Writing the
guard is what turned that paragraph from a plausible sentence into a
measured one.

It is also why the iteration is slow there -- it is converging to the
boundary of the simplex -- and why ``blahut_arimoto`` logs the distance it
may still be from the maximum instead of returning a half-converged answer
in silence.


Part 3: drawing from the law, and what it buys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take the Maxwell-Boltzmann law at :math:`H = 3.5` bit/symbol -- half a bit
of backoff from the uniform 4 -- and simply *draw* from it.
``SymbolGenerator(16, distribution=...)`` is that draw: an i.i.d. source, the
idealization a real matcher approaches.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 220-223

.. code::

   target law: H = 3.500 bit/symbol, energy 30.20 against 85, shaping gain 1.501 dB

Half a bit of entropy bought a factor 85/30.2 = 2.8 in energy -- **4.5 dB**
of raw power, of which 1.5 dB survives the equal-rate accounting that
:func:`~comnumpy.core.shaping.shaping_gain_dB` performs. The rest pays for
the half bit given up.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 225-242

.. image:: img/probabilistic_shaping_fig3.png
   :width: 100%
   :align: center

.. code::

     symbols   empirical H   empirical energy   total variation
         200        3.4172             29.280            0.0790
       20000        3.4958             30.043            0.0094
     2000000        3.5005             30.223            0.0011

The histogram converges to the law, and the last column says how fast: the
total variation distance falls as :math:`1/\sqrt{n}`, a factor ten for every
hundredfold in symbols. Two hundred symbols already look like the right
shape and are 8 % away from it; two million are a part in a thousand.

Keep that number in mind for Part 4. A distribution matcher does **not**
produce this: it maps uniform bits onto a finite set of sequences, so its
output is neither independent nor exactly :math:`P_X` on a finite block. The
i.i.d. draw is what a matcher is trying to imitate, which is why it is the
right thing to use here -- it isolates what the *law* is worth from what a
finite block costs.

And what the law is worth is this:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 244-270

.. image:: img/probabilistic_shaping_fig4.png
   :width: 100%
   :align: center

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 272-276

.. code::

   rate 1.5 bit/symbol: uniform needs  8.96 dB, shaped  8.46 dB -- +0.50 dB saved
   rate 2.0 bit/symbol: uniform needs 12.53 dB, shaped 11.78 dB -- +0.75 dB saved
   rate 2.5 bit/symbol: uniform needs 15.90 dB, shaped 14.94 dB -- +0.96 dB saved
   rate 3.0 bit/symbol: uniform needs 19.17 dB, shaped 18.12 dB -- +1.05 dB saved

The gain is read **horizontally**, and that is the only reading an engineer
can budget: at a fixed rate, how much less SNR does the shaped system need?
Up to 1.05 dB here, growing with the rate, and short of the 1.53 dB
:math:`10\log_{10}(\pi e/6)` that Forney and Wei (1989) showed is the
supremum over all one-dimensional distributions. Sixteen points cannot get
there; Part 5 shows what does.

The gain also **vanishes at both ends**, which the right-hand panel makes
plain. At low rate there is little to gain, and as the rate approaches
4 bit/symbol the only law carrying it is the uniform one, so the shaped
curve must come back to meet it.


Part 4: data is uniform, so the law has to be built
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything above assumed a source that emits symbols with probabilities
:math:`p_i`. No such source exists. Data is compressed, encrypted, or simply
arbitrary: it is a stream of **uniform, independent bits**, and a code
downstream will assume exactly that. Something has to convert one into the
other, invertibly, and that something is a **distribution matcher**.

The conversion is not statistical, it is combinatorial. Both constructions
below are *enumerative codes*: they rank and unrank a finite set of blocks
with integer arithmetic, so ``decode(encode(bits)) == bits`` holds by
construction rather than by tolerance.

First, the split PAS is built on. The sign of a PAM symbol is equiprobable,
so it costs nothing in shaping:

.. math::

   P_Y(\pm a_i) = \tfrac{1}{2} P_A(a_i)

The matcher therefore shapes the eight **amplitudes** only, and the sign is
left for a systematic FEC encoder to fill with its parity bits -- which are
uniform, so the composite constellation keeps the symmetric law at the same
energy and gains exactly one bit per symbol (Böcherer, Steiner and Schulte,
2015). Our target of 3.5 bit/symbol is a target of 2.5 bit/amplitude:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 283-285

**CCDM** fixes the composition. Every output block holds exactly :math:`n_i`
copies of amplitude :math:`i`, and there are

.. math::

   N = \binom{n}{n_1, \ldots, n_M} = \frac{n!}{n_1! \, n_2! \cdots n_M!}

such blocks, so the matcher carries :math:`k = \lfloor \log_2 N \rfloor`
bits (Schulte and Böcherer, 2016). Every block has the target empirical
distribution *exactly*, at any blocklength. What a finite block costs is
rate:

.. math::

   R_{\mathrm{loss}} = H\!\left(\frac{n_i}{n}\right) - \frac{k}{n}
   \;\xrightarrow[n \to \infty]{}\; 0

**ESS** fixes an energy budget instead (Gültekin *et al.*, 2020). With an
integer energy :math:`e_i` per amplitude, the code is

.. math::

   \mathcal{C} = \Big\{ (s_1, \ldots, s_n) :
   \sum_{j=1}^{n} e_{s_j} \leq E_{\max} \Big\}

and counting it is a one-dimensional recursion over the remaining budget,

.. math::

   N_t(E) = \sum_{i} N_{t-1}\!\left(E - e_i\right), \qquad
   N_0(E) = 1 \;\; \text{for } E \geq 0

from which unranking follows as for CCDM.

The comparison between them is not a benchmark, it is an inclusion. Every
CCDM block costs exactly :math:`E = \sum_i n_i e_i`, so given
:math:`E_{\max} = E` the sphere **contains every CCDM block** and more
besides -- the ones that spend less. It therefore carries at least as many
bits at the same energy, always.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 290-301

.. code::

   n =   16   composition (5, 4, 3, 2, 1, 1, 0, 0)   rate 1.8125 vs 2.1875 bit/amplitude
   n =   32   composition (9, 8, 6, 4, 3, 1, 1, 0)   rate 2.0938 vs 2.3750 bit/amplitude
   n =   64   composition (18, 16, 12, 8, 5, 3, 1, 1)   rate 2.2656 vs 2.4688 bit/amplitude
   n =  128   composition (36, 32, 25, 17, 10, 5, 2, 1)   rate 2.3281 vs 2.4453 bit/amplitude

At :math:`n = 16` the sphere carries 21 % more bits than the constant
composition; by :math:`n = 128` both are closing on the entropy of the law,
2.5 bit/amplitude. That is the whole reason ESS exists: CCDM needs thousands
of symbols to be efficient, ESS is already good at a few dozen, which is the
regime of a short packet. What ESS gives up is the exact per-block
distribution -- it reproduces the law only on average.

Note also the two zeros in the composition at :math:`n = 16`: rounding a law
onto sixteen slots simply cannot represent an amplitude of probability
below :math:`1/32`, so the two outermost points are dropped. A finite block
loses rate twice -- once quantizing the law, once taking the floor of
:math:`\log_2 N`.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 303-315

.. image:: img/probabilistic_shaping_fig5.png
   :width: 100%
   :align: center

The transmitter
"""""""""""""""

As a chain, PAS is six blocks, and the receiver is the transmitter read
backwards:

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 317-334

.. mermaid:: mermaid/shaping_pas.mmd

.. code::

   145 bits -> 64 amplitudes per block, recovered exactly: True
   #    block                        id                   output shape       dtype         time ms
   -----------------------------------------------------------------------------------------------
   0    SymbolGenerator              bits                 (29000,)           int64            0.16
   1    DistributionMatcher          matcher              (12800,)           int64           47.00
   2    AmplitudeMapper              mapper               (12800,)           float64          0.18
   3    AWGN                         channel              (12800,)           float64          0.31
   4    AmplitudeDemapper            amplitude_demapper   (12800,)           int64            0.94
   5    DistributionDematcher        distribution_dematcher (29000,)           int64           34.85

``summary()`` shows where the rate conversion happens -- 29 000 bits in,
12 800 amplitudes out -- and where the time goes. The enumerative code costs
some hundred times the channel it feeds, because ranking and unranking are
exact big-integer arithmetic done one symbol at a time. That is the real
price of a matcher, and it is why hardware implementations of CCDM are a
research topic of their own.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 336-348

.. image:: img/probabilistic_shaping_fig6.png
   :width: 100%
   :align: center

Compare this histogram with the one in Part 3. Both approximate the same
law, but for different reasons: there it was sampling error, shrinking as
:math:`1/\sqrt{n}`; here it is the *quantization* of the law onto a
composition of 64 slots, which does not shrink at all as more blocks are
sent -- every block carries the same rounded composition. More symbols make
this histogram converge to the composition, not to :math:`P_X`.

Where the FEC decoder goes
""""""""""""""""""""""""""

A matcher is a code, and that has a consequence worth meeting head on. Lower
the SNR and a symbol error takes the received block **outside** the code: it
is no longer a permutation of the composition, so there is no index to read
back, and the dematcher says so rather than returning silent nonsense.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 350-354

.. code::

   at 20 dB -- the block has composition (16, 21, 10, 6, 6, 3, 0, 2) but this
   matcher enumerates (18, 16, 12, 8, 5, 3, 1, 1). A detector error can
   produce such a block: it is not in the code, so there is no index to read.

This is not a limitation of the implementation, it is the reason PAS is
built the way it is. In a real receiver the FEC decoder sits **between** the
demapper and the dematcher, and the dematcher only ever sees corrected
amplitudes. It also explains why shaping and coding cannot be designed
separately: an uncorrected error does not cost one symbol, it costs the
whole block.


Part 5: complex constellations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Everything so far has been one-dimensional, and that was not a
simplification. A square QAM constellation *is* the product of two
independent PAM axes -- the in-phase and quadrature components are shaped
separately and their probabilities multiply:

.. math::

   P_{X}(a + jb) = P_{I}(a)\, P_{Q}(b),
   \qquad
   H(X) = H(I) + H(Q),
   \qquad
   E[|X|^2] = E[I^2] + E[Q^2]

So shaping a 256-QAM is shaping a 16-PAM axis and taking the product. The
entropy doubles, the energy doubles, and the gain in **dB** -- being a ratio
of energies -- is unchanged.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 361-369

.. code::

   256-QAM as a product of two shaped 16-PAM axes: H = 7.000 bit/symbol, 7.000 expected

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 371-379

.. image:: img/probabilistic_shaping_fig7.png
   :width: 70%
   :align: center

That figure is the picture everyone has seen of probabilistic shaping: a
square grid whose points fade towards the corners. It is worth knowing that
it is nothing more than an outer product of the one-dimensional law of
Part 3 with itself.

Why high-order formats
""""""""""""""""""""""

The last question is why shaping is deployed on 64-QAM and 256-QAM and never
on QPSK. The answer is not that the gain formula changes -- it is that a
small constellation has nothing to redistribute.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 381-398

.. code::

      M   best SNR saved   at rate   still short of 1.53 dB
      4          0.218 dB      1.00                 1.315 dB
      8          0.601 dB      1.71                 0.932 dB
     16          0.896 dB      2.45                 0.637 dB
     32          1.112 dB      3.12                 0.421 dB
     64          1.260 dB      3.83                 0.273 dB

Each row shapes its own PAM to a fixed backoff of a quarter of its entropy
and reports the best SNR it saves. **4-PAM saves 0.22 dB, 64-PAM saves
1.26 dB**, and the trend towards 1.53 dB is clear and slow. A 4-PAM has two
amplitudes to play with; a 64-PAM has thirty-two, and can approximate the
Gaussian envelope that the bound assumes.

A square QAM is two of these axes, so it inherits its axis's row directly:
256-QAM the 16-PAM one at 0.90 dB, 4096-QAM the 64-PAM one at 1.26 dB. The
rate doubles, the gain in dB does not -- it is a ratio of energies, and both
the energy and the rate double together. That is why every recent optical or
DOCSIS system reaching for a high-order format reaches for shaping at the
same time, and why nobody bothers on QPSK.

.. literalinclude:: ../../examples/simple/probabilistic_shaping.py
   :language: python
   :lines: 400-403


Conclusion
^^^^^^^^^^

You have chosen a constellation's *law* rather than accepting the uniform
one, and measured what that is worth at every step.

You have learned how to:

- Read what a constellation carries with
  :func:`~comnumpy.core.capacity.constellation_capacity` and
  :func:`~comnumpy.core.capacity.bicm_capacity`, and check it against
  :func:`~comnumpy.core.information.compute_mi` and
  :func:`~comnumpy.core.information.compute_gmi` on real samples.
- Compute the rate-maximizing law with
  :func:`~comnumpy.core.shaping.blahut_arimoto`, and see how little the
  closed-form :func:`~comnumpy.core.shaping.maxwell_boltzmann` gives up
  against it.
- Draw from a law with ``SymbolGenerator(distribution=...)`` and watch the
  histogram converge.
- Build the law from uniform bits with
  :class:`~comnumpy.core.shaping.ConstantCompositionMatcher` or
  :class:`~comnumpy.core.shaping.SphereShaper`, assemble a PAS transmitter,
  and place the FEC decoder where it belongs.
- Carry all of it to a square QAM, and say why the format has to be
  high-order for any of it to pay.

From here, you can:

- Put a real code in the loop: :mod:`comnumpy.fec` provides the systematic
  encoder whose parity bits PAS spends on the signs.
- Replace the i.i.d. source of Part 3 by the matcher itself and measure how
  much of the shaping gain a finite blocklength gives back.
- Move the whole thing onto a fibre: :doc:`optical_fiber_nonlinearity` and
  :doc:`gn_model` describe a channel where the *power* is what hurts, which
  is exactly what shaping reduces.

References
""""""""""

- C. E. Shannon, "A mathematical theory of communication", *Bell Syst. Tech.
  J.*, vol. 27, pp. 379-423, 1948.
- R. E. Blahut, "Computation of channel capacity and rate-distortion
  functions", *IEEE Trans. Inf. Theory*, vol. 18, no. 4, pp. 460-473, 1972;
  S. Arimoto, "An algorithm for computing the capacity of arbitrary discrete
  memoryless channels", *IEEE Trans. Inf. Theory*, vol. 18, no. 1,
  pp. 14-20, 1972.
- G. D. Forney and L.-F. Wei, "Multidimensional constellations -- Part I",
  *IEEE J. Sel. Areas Commun.*, vol. 7, no. 6, pp. 877-892, 1989 -- the
  1.53 dB ultimate shaping gain.
- F. R. Kschischang and S. Pasupathy, "Optimal nonuniform signaling for
  Gaussian channels", *IEEE Trans. Inf. Theory*, vol. 39, no. 3,
  pp. 913-929, 1993 -- that Maxwell-Boltzmann is essentially optimal.
- A. Alvarado, E. Agrell, D. Lavery, R. Maher and P. Bayvel, "Replacing the
  soft-decision FEC limit paradigm in the design of optical communication
  systems", *J. Lightwave Technol.*, vol. 33, no. 20, pp. 4338-4352, 2015 --
  MI and GMI as the quantities to report.
- G. Böcherer, F. Steiner and P. Schulte, "Bandwidth efficient and
  rate-matched low-density parity-check coded modulation", *IEEE Trans.
  Commun.*, vol. 63, no. 12, pp. 4651-4665, 2015 -- the PAS architecture.
- P. Schulte and G. Böcherer, "Constant composition distribution matching",
  *IEEE Trans. Inf. Theory*, vol. 62, no. 1, pp. 430-434, 2016.
- Y. C. Gültekin, W. J. van Houtum, A. G. C. Koonen and F. M. J. Willems,
  "Enumerative sphere shaping for wireless communications with short
  packets", *IEEE Trans. Wireless Commun.*, vol. 19, no. 2, pp. 1098-1112,
  2020.
- J. Cho and P. J. Winzer, "Probabilistic constellation shaping for optical
  fiber communications", *J. Lightwave Technol.*, vol. 37, no. 6,
  pp. 1590-1607, 2019 -- a tutorial covering the whole subject.
- T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed.,
  Wiley, 2006, Chapters 2, 9 and 10.
