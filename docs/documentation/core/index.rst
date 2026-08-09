Core
====





.. toctree::
   :maxdepth: 1
   :caption: Contents:

   generics
   generators
   mappers
   processors
   devices
   filters
   channels
   fading
   impairments
   compensators
   frames
   sequences
   capacity
   information
   metrics
   utils
   validators
   visualizers

Observing signals inside a chain
--------------------------------

A chain describes the communication system and nothing else: there are no
recorder, logger or scope blocks to insert between the processors. To
observe a signal, name the block and declare it as a *tap*::

   chain = Sequential([SymbolGenerator(16, name="tx"), SymbolMapper(alphabet),
                       AWGN(snr_dB=15, name="awgn")], taps=["tx", "awgn"])
   y = chain(1000)
   plot_iq(chain.tap("awgn"))

``taps`` is chain metadata: each tapped block costs one dictionary store of
a reference (no copy), and :meth:`~comnumpy.core.generics.Sequential.tap`
returns the recorded array afterwards. Plotting and reporting are plain
functions applied to the extracted arrays -- see :doc:`visualizers` and
``comnumpy.core.metrics.signal_report``.


Processor Vs Compensator
------------------------

The comnumpy library distinguishes between two types of signal processing components: Processors and Compensators. 

This separation is based on the nature of their operations and how they interact with the input signals.

- **Processors**: These components apply fixed transformations to the input signals, regardless of the signal's content. Examples include amplifiers and clippers. 

- **Compensators**: These components, on the other hand, adapt their behavior based on the input signal to achieve a desired output characteristic. Examples include normalizers and DC offset correctors. 

   

