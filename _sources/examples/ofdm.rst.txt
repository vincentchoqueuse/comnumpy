OFDM Tutorial
=============

In this tutorial, we compare the performance of a **Single Carrier (SC)** system
and an **OFDM** system over a **frequency-selective multipath channel** using the ``comnumpy`` library.

**What you'll learn:**

- How to define and simulate a frequency-selective channel.
- How to evaluate performance using the Symbol Error Rate (SER) and the runtime.
- What OFDM actually buys against single-carrier equalization -- and what it costs.

This tutorial is suitable for engineers and students interested in digital communications,
combining practical examples with theoretical insights.


Introduction
^^^^^^^^^^^^

Import Libraries
""""""""""""""""

We start by importing the necessary libraries:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 1-11

Simulation Parameters
"""""""""""""""""""""

Next, we define the parameters of the communication chain,
including the modulation order and the channel impulse response for a frequency-selective channel:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 15-33

Here, ``h`` represents the channel impulse response, drawn from an
**exponential power delay profile** -- the standard multipath model -- and
seeded, because the whole comparison below is about one realization. Its
frequency response spans a factor of 31 between its strongest and weakest
subcarrier: that notch is what "frequency selective" means, and it is what
the two receivers will disagree about.


Frequency-Selective Channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input-output relation of a frequency-selective channel is:

.. math::

   z[n] = \sum_{l=0}^{L-1} h[l]\,x[n-l] + b[n]

where :math:`h[l]` are the channel taps and :math:`b[n]` is the noise.  
This is also called a Finite Impulse Response (FIR) channel.

Stacking :math:`N` samples into a vector form:

.. math::

   \mathbf{z} = \mathbf{H}\mathbf{x} + \mathbf{b}

where :math:`\mathbf{H}` is a Toeplitz convolution matrix constructed from the taps.  
This formulation highlights that **ISI (Inter-Symbol Interference)** is unavoidable in SC systems.  


Single-Carrier Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SC chain is defined as:

.. mermaid:: mermaid/ofdm_single_carrier.mmd

That diagram is not drawn by hand: it is ``simple_chain.to_mermaid()``,
exported by the script at the end of its run (decision D33c). The block
names are the ones the code uses, and a dashed outline marks a tapped
block -- so the picture cannot say something the chain does not.

At the receiver, we apply a **Zero-Forcing (ZF) equalizer**:

.. math::

   \widehat{\mathbf{x}} = \mathbf{H}^{\dagger}\mathbf{z}

where :math:`\mathbf{H}^{\dagger}` is the pseudo-inverse of :math:`\mathbf{H}`.

Implementation
""""""""""""""

The chain is implemented in **comnumpy** as follows:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 35-42

Results
"""""""

We evaluate the performance by computing the SER and the execution time,
then plot the constellation before and after equalization:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 44-68

For the SC chain, we obtain:

.. code::

   SER: 0.12109375
   elapsed time: 1.70 s

Twelve percent of the symbols are wrong, and it took more than a second of
computation for 1280 symbols. Both numbers matter. The error rate is the
notch: zero forcing inverts the channel, so where :math:`|H(f)|` is small it
multiplies the noise by :math:`1/|H(f)|`, and that amplified noise is spread
over the whole block. The second number is the price of doing it this way --
the equalizer builds the :math:`(N + L - 1) \times N` convolution matrix and
pseudo-inverts it, which costs :math:`O(N^3)`.

.. image:: img/one_shot_ofdm_fig1.png
   :width: 100%
   :align: center
   :alt: Constellations before and after equalization (Single Carrier)


OFDM Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^

In SC systems, equalization requires matrix inversion, which is computationally expensive.
OFDM transforms the channel into a set of parallel flat-fading subchannels,
each equalized with a **simple one-tap filter**.
This drastically reduces computational complexity and improves performance.

The OFDM chain can be visualized as:

.. mermaid:: mermaid/ofdm_chain.mmd

``OFDMTransmitter`` and ``OFDMReceiver`` are themselves chains, and they
are drawn from their own ``chain`` attribute rather than described:

* Transmitter (TX)

.. mermaid:: mermaid/ofdm_transmitter.mmd

* Receiver (RX)

.. mermaid:: mermaid/ofdm_receiver.mmd

The diagram above is not drawn by hand. It is what the chain says about
itself -- ``chain.to_mermaid()`` (decision D33c) -- exported by the
script, so the block names are the ones the code uses and a dashed
outline marks a tapped block:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 114-123

Key blocks, under the names the diagrams show:

- ``Serial2Parallel`` / ``Parallel2Serial``: reshape between the serial
  stream and the parallel blocks OFDM works on.
- ``IFFTProcessor`` / ``FFTProcessor``: transform between the frequency and
  the time domain.
- ``CyclicPrefixer`` / ``CyclicPrefixRemover``: add and remove the cyclic
  prefix that turns the linear convolution into a circular one.
- ``CarrierAllocator`` / ``CarrierExtractor``: place the data and pilot
  symbols on their subcarriers, and take them back.
- ``FrequencyDomainEqualizer``: one complex division per subcarrier.

Mathematically, the received vector is:

.. math::

   \mathbf{z} = \mathbf{D}\mathbf{x} + \mathbf{n}

with :math:`\mathbf{D} = \mathrm{diag}(H[0], H[1], \dots, H[N-1])`,  
where :math:`H[k]` is the channel frequency response.  
Thus, OFDM reduces equalization to a **diagonal system**.

Implementation
""""""""""""""

The OFDM chain in **comnumpy** is implemented as:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 69-81

Results
"""""""

We compute the SER and runtime, then plot the received constellation:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 82-98

For the OFDM chain, we obtain:

.. code::

   SER: 0.04140625
   elapsed time: 0.0015 s

Three times fewer errors, in **a thousandth** of the time: 1.5 ms against
1.70 s for the same 1280 symbols. And the time ratio is not a constant -- the
single-carrier equalizer grows as :math:`N^3` while OFDM grows as
:math:`N \log N` -- so the comparison only gets more lopsided with the block
size. That is the reason wideband receivers are built this way.

Where the ranking comes from, and where it flips


One operating point is not a conclusion, so the script runs both chains again
over a range of noise variances:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 100-112

.. code::

   sigma2     single carrier      OFDM     |H| spans 31
    0.015             0.1211    0.0414
    0.008             0.0242    0.0242
    0.004             0.0008    0.0148
    0.002             0.0000    0.0094

The ranking **crosses over**, and the reason is structural rather than
numerical. Both receivers are zero forcing, so both amplify the noise by
:math:`1/|H|` where the channel is weak; what differs is *where that
amplified noise goes*.

The single-carrier equalizer inverts a **linear** convolution: :math:`N + L -
1` observations for :math:`N` unknowns, an overdetermined system whose
least-squares solution never divides by exactly zero, and whose residual noise
is spread over every symbol of the block. OFDM inverts a **circular** one: the
cyclic prefix makes the system exactly determined, :math:`N` equations for
:math:`N` unknowns, one per subcarrier -- so a subcarrier that falls in the
notch is divided by a small number and nothing else can help it.

Concentrated damage wins at low SNR and loses at high SNR. At
:math:`\sigma^2 = 0.015` everything is marginal and spreading the enhanced
noise over all 1280 symbols ruins them all, while OFDM only ruins the handful
of subcarriers in the notch. At :math:`\sigma^2 = 0.002` the spread noise has
fallen below the decision threshold everywhere and the single carrier makes no
error at all, while OFDM keeps an **error floor**: those subcarriers are dead
whatever the SNR.

That floor is not a defect of the model, it is the reason no real OFDM system
is uncoded. A code spread across the subcarriers repairs exactly the carriers
the notch killed -- and the FFT is what makes the receiver affordable in the
first place.

.. image:: img/one_shot_ofdm_fig2.png
   :width: 100%
   :align: center
   :alt: Constellation at OFDM receiver


Conclusion
^^^^^^^^^^

You have compared **Single Carrier** and **OFDM** systems over a multipath channel.

You have learned how to:

- Model a frequency-selective FIR channel in ``comnumpy``.
- Simulate both SC and OFDM systems on the same channel realization.
- Apply ZF equalization (SC) vs. one-tap equalization (OFDM).
- Compare the two in symbol error rate *and* in computational cost, and see
  that the second is where the difference lies.

Key takeaway:
**OFDM turns a frequency-selective channel into a set of flat subchannels, so
equalization becomes one complex division per subcarrier instead of an**
:math:`N \times N` **matrix inversion. That is what makes wideband receivers
affordable; the per-subcarrier error rate it gives up is bought back by coding
across the subcarriers.**

