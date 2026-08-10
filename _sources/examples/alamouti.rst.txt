Alamouti Space-Time Coding Tutorial
===================================

This tutorial explains what a space-time code buys you, on the simplest and most widely deployed one: the Alamouti scheme. You will learn how to:

- Understand why a fading channel is limited by its *deep fades*, not by its average.
- Build a two-antenna Alamouti transmitter and its receiver with ``comnumpy``.
- See why the receiver needs only a matched filter, and why that is optimal.
- Measure the diversity gain, and the 3 dB it costs against receive diversity.

This tutorial is suited for engineers and students learning about MIMO systems, combining practical examples with theoretical background.

Introduction
^^^^^^^^^^^^

Why diversity
"""""""""""""

Over a flat Rayleigh channel :math:`y[n] = h\,x[n] + b[n]`, the instantaneous signal-to-noise ratio is proportional to :math:`|h|^2`, which is exponentially distributed. The receiver is therefore not limited by the *average* channel: it is limited by the probability that the channel is nearly zero. That probability behaves as

.. math ::

   \mathbb{P}\left[|h|^2 < \epsilon\right] \simeq \epsilon

so the error rate decays only as :math:`1/\mathrm{SNR}` -- one decade per decade. Ten more decibels buy one decade of error rate, which is a poor exchange.

If instead the receiver observes the symbol through :math:`d` **independent** fading coefficients, all of them must be small at once for the link to fail, and

.. math ::

   \mathbb{P}\left[\text{all } d \text{ paths weak}\right] \simeq \epsilon^{d}
   \quad \Longrightarrow \quad
   P_e \propto \mathrm{SNR}^{-d}

The exponent :math:`d` is the **diversity order**, and it is the slope of the error curve on a log-log plot. Getting :math:`d = 2` with two receive antennas is easy -- that is maximum ratio combining (MRC). Getting it with two *transmit* antennas and a single receive antenna is the problem Alamouti solved, and it is the useful one: antennas are expensive on a handset and cheap on a base station.

The difficulty is that a transmitter has no channel knowledge. Sending the same symbol from both antennas gives :math:`y = (h_1 + h_2)x + b`, and :math:`h_1 + h_2` is one Gaussian coefficient: no diversity at all.

The Alamouti scheme
"""""""""""""""""""

The idea is to spread two symbols over two antennas **and** two time slots, in a pattern that makes the two observations orthogonal. Over one codeword the two antennas transmit

.. math ::

   \mathbf{G}\left(s_1, s_2\right) =
   \begin{bmatrix} s_1 & -s_2^{*} \\ s_2 & s_1^{*} \end{bmatrix}

the first column being the first time slot and the second column the second. With one receive antenna and a channel :math:`\mathbf{h} = \left[h_1, h_2\right]` constant over the two slots, the receiver observes

.. math ::

   y_1 &= h_1 s_1 + h_2 s_2 + b_1\\
   y_2 &= -h_1 s_2^{*} + h_2 s_1^{*} + b_2

Conjugating the second equation makes the pair *linear* in :math:`\left(s_1, s_2\right)`:

.. math ::

   \begin{bmatrix} y_1 \\ y_2^{*} \end{bmatrix}
   =
   \underbrace{\begin{bmatrix} h_1 & h_2 \\ h_2^{*} & -h_1^{*}\end{bmatrix}}_{\mathbf{H}_{\mathrm{eq}}}
   \begin{bmatrix} s_1 \\ s_2 \end{bmatrix}
   + \begin{bmatrix} b_1 \\ b_2^{*}\end{bmatrix}

and the matrix :math:`\mathbf{H}_{\mathrm{eq}}` is **orthogonal**:

.. math ::

   \mathbf{H}_{\mathrm{eq}}^{H}\mathbf{H}_{\mathrm{eq}}
   = \left(\left|h_1\right|^2 + \left|h_2\right|^2\right)\mathbf{I}_2

That single identity is the whole scheme. It means the two symbols do not interfere at all, so the maximum-likelihood detector is not a search but a **matched filter**,

.. math ::

   \begin{bmatrix} \widehat{s}_1 \\ \widehat{s}_2 \end{bmatrix}
   = \frac{\mathbf{H}_{\mathrm{eq}}^{H}}{\left|h_1\right|^2 + \left|h_2\right|^2}
     \begin{bmatrix} y_1 \\ y_2^{*} \end{bmatrix}

and each symbol comes out with its SNR multiplied by :math:`\left|h_1\right|^2 + \left|h_2\right|^2` -- the sum of two independent fadings, hence diversity order 2. The code carries two symbols in two slots, so its rate is 1 symbol per channel use: the diversity is free in bandwidth.

.. note ::

   ``comnumpy`` stores every space-time code by its *linear dispersion*
   matrices, :math:`\mathbf{G}(\mathbf{s}) = \sum_k \mathbf{A}_k s_k +
   \mathbf{B}_k s_k^{*}`, and builds the real equivalent channel from
   them. The orthogonality identity above is checked when the code
   object is created, so a code that declares itself orthogonal and is
   not cannot be used at all. The constant it is orthogonal *up to* is
   measured at the same time and reused by the decoder.

Prerequisites
"""""""""""""

Ensure you have the following Python libraries installed:

.. code::

   numpy
   matplotlib
   comnumpy

Simulation Setup
^^^^^^^^^^^^^^^^

Import Libraries
""""""""""""""""

We start by importing the required libraries and ``comnumpy`` components:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 1-6

Define System Parameters
""""""""""""""""""""""""

The code is taken from the registry by name, exactly as the constellation is taken from ``get_alphabet``:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 10-16

The scaling by :math:`1/\sqrt{N_t}` matters for the comparison that follows. Two antennas each transmitting :math:`|s|^2` would spend twice the power of a single antenna, which is a 3 dB advantage that has nothing to do with coding. Splitting the power keeps the comparison about diversity alone.

One-Shot Simulation
^^^^^^^^^^^^^^^^^^^

Transmit and receive one block
""""""""""""""""""""""""""""""

We draw one channel realization, encode 500 codewords, add noise and decode:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 18-29

``SpaceTimeEncoder`` turns a stream of :math:`N` symbols into an array of shape :math:`\left(N_t, N T / K\right)` -- antennas on the second-to-last axis, as everywhere else in the library -- so it plugs straight into a MIMO channel. ``SpaceTimeDecoder`` performs the matched filter above; it needs the channel matrix, which is assumed known at the receiver.

Visualize the combining
"""""""""""""""""""""""

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 31-44

.. image:: img/one_shot_alamouti_fig1.png
   :width: 100%
   :align: center

The left panel is what the single receive antenna actually sees: two superimposed streams plus noise, with no constellation visible at all. The right panel is the same data after combining, and the four QPSK points are back. Nothing was estimated to get there -- only the two conjugations and the matched filter of the equations above.

Monte Carlo Evaluation
^^^^^^^^^^^^^^^^^^^^^^

Three schemes at equal transmit power
"""""""""""""""""""""""""""""""""""""

To see the diversity we compare three links at the same total transmit power. The Alamouti scheme is simulated with the library blocks, over a channel drawn once per block of 25 codewords (quasi-static fading):

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 47-64

The reference without diversity is a single antenna each side:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 67-77

and the reference *with* diversity is maximum ratio combining on two receive antennas, which is the best a two-path link can do:

.. math ::

   \widehat{s} = \frac{\mathbf{h}^{H}\mathbf{y}}{\left\|\mathbf{h}\right\|^2},
   \qquad
   \mathrm{SNR}_{\mathrm{out}} = \frac{\left\|\mathbf{h}\right\|^2}{\sigma^2}

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 80-93

We sweep the SNR:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 96-102

Plot SER vs SNR
"""""""""""""""

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 104-114

.. image:: img/one_shot_alamouti_fig2.png
   :width: 100%
   :align: center

Two things are visible, and they are the two things to remember.

**The slope.** The single-antenna curve falls by one decade per decade of SNR; the two diversity schemes fall twice as fast. Since the diversity order is an *asymptotic* slope, the script prints the local slope of each curve rather than a single fitted number, so that the convergence is visible:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 116-125

.. code::

   1 Tx, 1 Rx (no diversity)    local slope -0.72 -0.85 -0.93 -0.97 -1.01 -1.02
   Alamouti, 2 Tx, 1 Rx         local slope -1.11 -1.50 -1.79 -1.97
   MRC, 1 Tx, 2 Rx              local slope -1.40 -1.72 -1.88

The first curve converges to 1, the two others to 2. Intervals whose upper end rests on a handful of errors are dropped: their slope would be Monte-Carlo noise, not physics.

**The 3 dB.** Alamouti is parallel to MRC but shifted right. The script reads the SNR each scheme needs to reach a symbol error rate of :math:`10^{-3}`:

.. literalinclude:: ../../examples/mimo/one_shot_alamouti.py
   :language: python
   :lines: 127-133

.. code::

   SNR needed for SER = 0.001: 1 Tx, 1 Rx (no diversity) 28.0 dB,
   Alamouti, 2 Tx, 1 Rx 18.4 dB, MRC, 1 Tx, 2 Rx 15.6 dBThe gap to MRC is the price of transmitting *blind*: the receiver knows the channel and can weight its two branches by :math:`h_i^{*}`, while the transmitter cannot, and splits its power evenly instead. Alamouti buys the full diversity order anyway -- it only pays for it in array gain, not in slope.

.. note ::

   The same 3 dB is why the comparison had to be made at equal total
   transmit power. Had each antenna transmitted the full power, the
   Alamouti curve would have landed on top of the MRC one and the plot
   would have proved nothing.

Beyond two antennas
^^^^^^^^^^^^^^^^^^^

Alamouti is the only complex orthogonal design of rate 1. Beyond two transmit antennas, orthogonality survives only at a reduced rate -- ``comnumpy`` ships the designs of Tarokh, Jafarkhani and Calderbank at rates 1/2 and 3/4 for three and four antennas -- and above rate 1 nothing is orthogonal at all, so linear decoding is no longer optimal:

.. code:: python

   from comnumpy.mimo.coding import available_codes, get_code

   for name in available_codes():
       code = get_code(name)
       print(f"{name:22s} Nt={code.n_tx} rate={code.rate:4.2f} "
             f"orthogonal={code.is_orthogonal}")

The Golden code sits at the other end of the same trade: rate 2 on two antennas *and* full diversity, at the price of a decoder that has to search. ``SpaceTimeDecoder`` refuses it rather than returning a zero-forcing answer under a name that promises optimality; ``code.equivalent_channel(H)`` builds the equivalent channel to hand to one of the detectors of :doc:`../documentation/mimo/detectors`.

Conclusion
^^^^^^^^^^

This tutorial highlighted:

- Why a fading link is limited by its deep fades, and what diversity does about it.
- How the Alamouti codeword turns two transmit antennas into two independent observations, without any channel knowledge at the transmitter.
- Why the resulting receiver is a matched filter and why that is exactly maximum likelihood.
- That the measured slope doubles, and that Alamouti pays 3 dB against receive diversity for transmitting blind.

With ``comnumpy``, a space-time code is an object taken from a registry, its orthogonality is verified rather than assumed, and the encoder and decoder are ordinary chain blocks.
