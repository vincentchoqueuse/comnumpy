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
   :lines: 13-22

This kind of channel causes inter-symbol interference (ISI) in single carrier systems, which can significantly degrade performance.

3. Single Carrier Chain
^^^^^^^^^^^^^^^^^^^^^^^

Let's build a basic SC simulation to illustrate poor performance in a selective channel:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 24-34

You will likely observe a very poor Symbol Error Rate (SER), despite using a relatively strong SNR. This is due to ISI caused by the selectivity of the channel — single-carrier systems suffer unless equalization is used.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 36-45

Let's visualize the received signal and compute the SER :

.. image:: img/one_shot_ofdm_fig1.png

4. OFDM Chain
^^^^^^^^^^^^^

We now simulate the same scenario using OFDM, which divides the bandwidth into orthogonal subcarriers. This converts a frequency-selective channel into many flat channels, allowing simple equalization.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 48-61

OFDM handles the multipath channel far better. You should observe a much lower SER, showing the benefit of dividing the signal across orthogonal subcarriers and using a cyclic prefix.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 63-72

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