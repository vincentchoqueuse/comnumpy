First Simulation
================

This tutorial guides you through running a simple communication simulation using comnumpy.
You will create a basic QAM communication chain, transmit symbols through an AWGN channel, 
and evaluate the Symbol Error Rate (SER).

Prerequisites
^^^^^^^^^^^^^

Make sure you have installed:

- ``numpy``
- ``matplotlib``
- ``comnumpy``

Step 1: Import Libraries and Define Parameters
----------------------------------------------

First, import the necessary libraries and define simulation parameters:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 1-16

Step 2: Define Communication Chain
----------------------------------

Define the communication chain using the Sequential class, including symbol generation, mapping, and channel:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 18-27

Step 3: Run the Chain and Evaluate Performance
----------------------------------------------

Run the chain and compute the SER to assess system performance:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 29-32

Step 4: Visualize the Received Constellation
---------------------------------------------

Finally, plot the received symbols to visualize the effect of noise on the constellation:


.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 34-40

.. image:: img/first_simulation_fig1.png
   :alt: Received constellation diagram
   :align: center

Next Steps
----------

Now that you've completed your first simulation, consider exploring:

- OFDM and MIMO tutorials for advanced communication techniques.
- The full comnumpy documentation for detailed API reference.
- Experimenting by changing modulation orders, SNR, and channel models.

