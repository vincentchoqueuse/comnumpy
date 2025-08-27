OFDM Chain Tutorial
===================

This tutorial guides you through simulating a selective channel and comparing the performance of a Single Carrier (SC) system and an OFDM system using the  ``comnumpy`` library. You'll learn how to:

- Define and simulate realistic communication channels.
- Evaluate performance using Symbol Error Rate (SER).
- Understand why OFDM performs better in multipath environments.

This tutorial is suitable for both engineers and students interested in digital communications, and serves - as both a practical example and a theoretical insight.

Prerequisites
^^^^^^^^^^^^^

Ensure you have the following Python libraries installed:

- ``numpy``
- ``matplotlib``
- ``comnumpy``

Simulation Setup
^^^^^^^^^^^^^^^^

1. Import Libraries
^^^^^^^^^^^^^^^^^^^

Begin by importing the necessary libraries

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 1-11

2. Define Simulation Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We simulate a multipath channel by generating a random impulse response. The first tap is normalized to 1 to preserve energy.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 14-24

This kind of channel causes inter-symbol interference (ISI) in single carrier systems, which can significantly degrade performance.

3. Single Carrier Chain
^^^^^^^^^^^^^^^^^^^^^^^

Let's build a basic SC simulation to illustrate the performance in a selective channel:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 26-41

We have used a linear Zero-Forcing equalizer to compensate for the channel. This equalizer is based on the computation of a matrix pseudo-inverse, which can drastically increase the computation complexity of the receiver.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 42-58

For the single carrier chain, we obtain :

- A Symbol Error Rate (SER) equals to : 0.007
- A total computational time equals to: 0.562s

The following figures present the constellation of the the received signal before and after linear equalization.

.. image:: img/one_shot_ofdm_fig1.png

4. OFDM Chain
^^^^^^^^^^^^^

In frequency-selective channels, the conventional single-carrier chain suffers from suboptimal detection performance and high computational complexity, mainly due to the need to invert a large matrix. A common solution to overcome this limitation is to employ an OFDM chain.

We now simulate the same scenario using OFDM, which divides the bandwidth into orthogonal subcarriers. This converts a frequency-selective channel into many flat channels, allowing simple optimal equalization.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 60-79

For the OFDM chain, we obtain:

- Symbol Error Rate (SER): 0.004 ( < 0.007)
- Total computational time: 0.007s ( < 0.562s)

Note that this substantial reduction in computational complexity stems from the fact that channel equalization reduces to the inversion of a diagonal matrix, while the OFDM transmitter and receiver rely on DFT/IDFT operations that can be efficiently implemented using Fast Fourier Transform (FFT) algorithms. Concerning the Symbol Error Rate (SER), the improvement arises from the fact that, under Gaussian noise, the OFDM detector is equivalent to the Maximum Likelihood detector.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 81-92

Let's visualize the received signal :

.. image:: img/one_shot_ofdm_fig2.png

Why is OFDM more robust?
^^^^^^^^^^^^^^^^^^^^^^^^

- The FIR channel introduces frequency selectivity (some frequencies are attenuated more than others).
- Single carrier systems suffer from ISI and frequency nulls.
- OFDM spreads the information over multiple narrowband subcarriers. Each subcarrier experiences a flat channel (or close to flat), equalization becomes trivial (1 tap per subcarrier), and the cyclic prefix prevents ISI, as long as the prefix is longer than the channel.


Conclusion
^^^^^^^^^^

This tutorial highlighted:

- How to use comnumpy to simulate both SC and OFDM systems.
- Why OFDM is well-suited for real-world channels.
- How to assess system performance using SER and plots.

By building modular processing chains and using the built-in blocks of comnumpy, you can quickly prototype and validate your ideas — whether you're teaching, learning, or experimenting.