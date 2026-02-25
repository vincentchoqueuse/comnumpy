OFDM Tutorial
=============

In this tutorial, we compare the performance of a **Single Carrier (SC)** system
and an **OFDM** system over a **frequency-selective multipath channel** using the ``comnumpy`` library.
You will learn how to:

- Define and simulate a frequency-selective channel.
- Evaluate performance using the Symbol Error Rate (SER).
- Understand why OFDM outperforms SC in multipath environments.

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
   :lines: 14-24

Here, ``h`` represents the channel impulse response.
The first tap is normalized to 1 to preserve the overall channel energy.


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

.. mermaid::

   graph LR;
      A[Generator] --> B[Mapper];
      B --> C[Channel];
      C --> D[AWGN];
      D --> E[Equalizer];
      E --> F[Demapper];

At the receiver, we apply a **Zero-Forcing (ZF) equalizer**:

.. math::

   \widehat{\mathbf{x}} = \mathbf{H}^{\dagger}\mathbf{z}

where :math:`\mathbf{H}^{\dagger}` is the pseudo-inverse of :math:`\mathbf{H}`.

Implementation
""""""""""""""

The chain is implemented in **comnumpy** as follows:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 26-41

Results
"""""""

We evaluate the performance by computing the SER and the execution time,
then plot the constellation before and after equalization:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 42-58

For the SC chain, we obtain:

- **SER** = 0.007  
- **Execution time** = 0.562 s  

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

.. mermaid::

   graph LR;
      A[Generator] --> B[Mapper];
      B --> C[OFDM Tx];
      C --> D[Channel];
      D --> E[AWGN];
      E --> F[OFDM Rx];
      F --> G[Demapper];

* Transmitter (TX)

.. mermaid::

   graph LR;
      A[Mapper] --> B[S2P];
      B --> C[IDFT];
      C --> D[CP add];
      D --> E[P2S];

* Receiver (RX)

.. mermaid::

   graph LR;
      A[P2S] --> B[CP del];
      B --> C[DFT];
      C --> D[Equalizer];
      D --> E[P2S];

Key blocks:

- **S2P / P2S**: Serial-to-Parallel and Parallel-to-Serial converters.  
- **IDFT / DFT**: Transform between frequency and time domains.  
- **CP add / CP del**: Insert/remove Cyclic Prefix to handle ISI.  
- **Equalizer**: One-tap equalization per subcarrier.  

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
   :lines: 60-79

Results
"""""""

We compute the SER and runtime, then plot the received constellation:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 81-92

For the OFDM chain, we obtain:

- **SER** = 0.004 (improved vs SC)  
- **Execution time** = 0.007 s (much lower than SC)  

.. image:: img/one_shot_ofdm_fig2.png
   :width: 100%
   :align: center
   :alt: Constellation at OFDM receiver


Conclusion
^^^^^^^^^^

You have compared **Single Carrier** and **OFDM** systems over a multipath channel.

You have learned how to:

- Model a frequency-selective FIR channel in ``comnumpy``.
- Simulate both SC and OFDM systems.
- Apply ZF equalization (SC) vs. one-tap equalization (OFDM).
- Compare performance in terms of SER and computational cost.

Key takeaway:
**OFDM transforms a frequency-selective channel into flat-fading subchannels,
enabling simple per-subcarrier equalization and superior performance in realistic multipath environments.**

