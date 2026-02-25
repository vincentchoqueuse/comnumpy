First Simulation
================

This tutorial guides you through a simple communication simulation using **comnumpy**.
You will build a basic QAM communication chain, transmit symbols through an AWGN channel,
and evaluate the Symbol Error Rate (SER).

**What you'll learn:**

- Creating a communication chain using ``Sequential`` and built-in processors.
- Transmitting QAM symbols through an AWGN channel.
- Measuring the Symbol Error Rate (SER) and comparing it with theory.
- Using ``Recorder`` objects to capture and visualize signals.


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

We start by importing the necessary libraries:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 1-8

Define Parameters
"""""""""""""""""

Next, we define the key simulation parameters: the modulation order,
the number of transmitted symbols, and the signal-to-noise ratio (SNR):

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 12-19


AWGN Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^

Define the Chain
""""""""""""""""

We define the communication chain using the ``Sequential`` object, which takes a list of
processors as input. The **comnumpy** library provides a wide range of built-in processors
for modulation, coding, channel modeling, and more.

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 20-28


In this simulation, the communication chain is composed of **six processor objects**:

.. mermaid::

   graph LR;
      A[Generator] --> B[Recorder];
      B --> C[Mapper];
      C --> D[AWGN];
      D --> E[Recorder];
      E --> F[Demapper];

- ``SymbolGenerator``
  Generates a sequence of random integers in the range :math:`[0, M-1]`, where each integer represents a symbol to transmit.

- ``Recorder``
  Captures the signal at a specific point in the chain for later analysis (e.g., comparing transmitted vs. received data).

- ``SymbolMapper``
  Maps each integer symbol to a point in the complex QAM constellation.

- ``AWGN``
  Simulates an **Additive White Gaussian Noise** channel, modeling the effect of thermal noise on the transmitted signal.

- ``SymbolDemapper``
  Performs hard-decision demapping by associating each received point with the nearest constellation symbol.

- ``Recorder`` (second instance)
  A second recorder placed after the channel captures the noisy received symbols for visualization (e.g., constellation diagrams).


Simulate the Chain
""""""""""""""""""

To run the simulation, simply call the ``Sequential`` object with the desired number of symbols:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 30-31

Evaluate Performance
""""""""""""""""""""

We evaluate the performance of the communication system by computing the **Symbol Error Rate (SER)**
from the transmitted and detected symbols, then comparing it with the theoretical value.

To retrieve the transmitted symbols, we call the ``get_data()`` method on the ``Recorder`` object.
Recorders can be placed at any point in the chain, depending on which signal you want to inspect.

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 33-43

For this simulation, typical output looks like:

.. code::

   SER (simu) = 0.0013
   SER (theo) = 0.0015647896369451741

Note: for small SER values, increasing ``N`` (the number of transmitted symbols)
improves the estimation accuracy.

Plot the Constellation
""""""""""""""""""""""

Visualizing the received constellation is a useful way to assess signal quality.
We use a ``Recorder`` object to capture the received symbols and plot them
with ``matplotlib``:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 45-52


.. image:: img/first_simulation_fig1.png
   :width: 100%
   :align: center

Conclusion
^^^^^^^^^^

You have successfully built and simulated your first communication chain with **comnumpy**.

From here, you can explore:

- The **OFDM and MIMO tutorials** for more advanced communication techniques.
- The **API reference** for a complete list of available processors.
- Different modulation orders, SNR values, and channel models to deepen your understanding.
