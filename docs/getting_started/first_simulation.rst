First Simulation
================

A communication system is a series of operations applied to a signal, one
after the other: generate symbols, map them onto a constellation, send them
through a channel, decide what was sent. In **comnumpy** you write that series
down as an object -- a ``Sequential`` chain -- and then you run it, measure
it, and look inside it.

This first tutorial builds the smallest such chain, runs it once, and reads
one number off it. Everything else in the series is this same object with more
blocks in it.

**What you'll learn:**

- How to assemble a chain from built-in processors with ``Sequential``.
- How to run it, and what the call actually returns.
- How to look *inside* it with a tap, instead of only at its output.
- How to compare the error rate you measured with the one theory predicts.


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
   :lines: 12-18


AWGN Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^

Define the Chain
""""""""""""""""

A chain is a list of **processors**, and ``Sequential`` is what turns that
list into a single object you can call. Each processor does one thing and
hands its output to the next, so reading the list from top to bottom *is*
reading the signal path -- which is why the examples in this series are kept
linear rather than factored into functions.

The chain, as the chain itself describes it:

.. mermaid:: mermaid/first_simulation.mmd

The diagram above is not drawn by hand. It is what the chain says about
itself -- ``chain.to_mermaid()`` (decision D33c) -- exported by the
script, so the block names are the ones the code uses and a dashed
outline marks a tapped block:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 42-45

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 19-24


Four processors, and each one is worth a sentence -- these four come back in
every tutorial that follows:

- ``SymbolGenerator``
  Generates a sequence of random integers in the range :math:`[0, M-1]`, where each integer represents a symbol to transmit.

- ``SymbolMapper``
  Maps each integer symbol to a point in the complex QAM constellation.

- ``AWGN``
  Simulates an **Additive White Gaussian Noise** channel, modeling the effect of thermal noise on the transmitted signal.

- ``SymbolDemapper``
  Performs hard-decision demapping by associating each received point with the nearest constellation symbol.

Notice what is *not* in the list: nothing to record, display or measure. A
chain contains communication blocks only. When you need to see a signal in
the middle of it, you give the block a name and declare that name as a **tap**
-- ``taps=["tx", "awgn"]`` above -- and the chain keeps a copy of what that
block produced. This is how every measurement in this series is made, so it is
worth getting used to now.


Simulate the Chain
""""""""""""""""""

The chain is an object, and running it is calling it. Its first block is a
*source*, so the argument is not a signal but the number of symbols to
produce -- and what comes back is what the last block returned, here the
detected symbol indices:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 26-26

Evaluate Performance
""""""""""""""""""""

An error rate compares what was sent with what was decided, and the chain
returned only the second one. The first is at the ``"tx"`` tap:
``chain.tap("tx")`` returns what that block produced during the last run.

One habit to take right away: **run the chain first, read the tap after**. A
tap holds the last run, so reading it before calling the chain raises an
error rather than returning stale data -- which is the behaviour you want,
but it does mean the order of the two lines matters.

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 28-35

which prints:

.. code::

   SER (simu) = 0.0013
   SER (theo) = 0.0015647896369451741

The two numbers agree to about 15 %, and that gap is not a modelling error:
it is the measurement itself. An error rate estimated from :math:`N` symbols
has a standard deviation of roughly :math:`\sqrt{P_e/N}`, so with 10 000
symbols and :math:`P_e \approx 1.6 \times 10^{-3}` a spread of that size is
expected. Increasing ``N`` narrows it -- and *how many symbols a claim needs*
is the subject of the next tutorial.

Plot the Constellation
""""""""""""""""""""""

Visualizing the received constellation is a useful way to assess signal quality.
We read the symbols recorded at the ``"awgn"`` tap and hand them to
``plot_iq``, which draws on an axis and returns it:

.. literalinclude:: ../../examples/simple/one_shot_awgn.py
   :language: python
   :lines: 37-41


.. image:: img/first_simulation_fig1.png
   :width: 100%
   :align: center

Conclusion
^^^^^^^^^^

You have built a chain, run it, looked inside it, and compared one
measurement with theory. That is the whole vocabulary the rest of the series
uses.

The natural next step is :doc:`../tutorials/awgn`, which asks the question this
tutorial left open: one run gives one number at one SNR, so how do you obtain
a *curve* -- and how many symbols does each of its points deserve?
