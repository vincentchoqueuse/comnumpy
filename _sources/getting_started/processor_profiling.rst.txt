Profiling a Communication Chain
===============================

In this tutorial, we will learn how to **profile** a communication chain using **comnumpy**.  
Profiling allows us to evaluate the computational cost of each processor, helping to identify performance bottlenecks in complex simulations.

We will build an **OFDM communication chain** with channel effects, run the simulation, and visualize the profiling results.


Introduction
^^^^^^^^^^^^

Import Libraries
""""""""""""""""

We start by importing the necessary libraries:

.. literalinclude:: ../../examples/simple/profiling_awgn_ofdm.py
   :language: python
   :lines: 1-13

Define Parameters
"""""""""""""""""

Next, we define the communication and channel parameters:

.. literalinclude:: ../../examples/simple/profiling_awgn_ofdm.py
   :language: python
   :lines: 16-35


OFDM Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^

Define the Chain
""""""""""""""""

We now build a complete OFDM chain using the ``Sequential`` object.  
This chain includes mapping, carrier allocation, IFFT/FFT processing, cyclic prefix insertion, channel effects, equalization, and demapping.

.. literalinclude:: ../../examples/simple/profiling_awgn_ofdm.py
   :language: python
   :lines: 38-56

The chain is composed of the following processors:

- ``SymbolGenerator``  
  Generates a sequence of integer-valued symbols to transmit.

- ``SymbolMapper``  
  Maps integers to QAM constellation points.

- ``Serial2Parallel`` / ``Parallel2Serial``  
  Reshape data between serial and parallel streams, useful for OFDM.

- ``CarrierAllocator``  
  Assigns data and pilot symbols to their designated subcarriers.

- ``IFFTProcessor`` / ``FFTProcessor``  
  Perform the Inverse Fast Fourier Transform and Fast Fourier Transform operations, respectively.

- ``CyclicPrefixer`` / ``CyclicPrefixRemover``  
  Add and remove the cyclic prefix to mitigate inter-symbol interference.

- ``FIRChannel``  
  Models a frequency-selective multipath channel.

- ``AWGN``  
  Adds white Gaussian noise.

- ``FrequencyDomainEqualizer``  
  Compensates for channel distortions in the frequency domain.

- ``CarrierExtractor``  
  Extracts data and pilot carriers after equalization.

- ``SymbolDemapper``  
  Maps received constellation points back to integer symbols.


Profiling the Chain
^^^^^^^^^^^^^^^^^^^

To profile the chain, we use the ``plot_chain_profiling`` function.  
This function measures the execution time of each processor for a given input size 
and produces a bar chart of the results.


.. literalinclude:: ../../examples/simple/profiling_awgn_ofdm.py
   :language: python
   :lines: 57-58

The profiling figure shows the time spent in each block, allowing you to quickly identify which stages of the chain dominate the computation.

.. image:: img/profiling_chain_fig1.png
   :width: 100%
   :align: center


Conclusion
^^^^^^^^^^

Congratulations 🎉 You have successfully profiled an **OFDM communication chain** with **comnumpy**.  

Profiling is a powerful tool to:

- Detect computational bottlenecks in complex simulations.
- Compare the efficiency of different processors or chain configurations.
- Optimize large-scale communication scenarios.

From here, you may want to explore:

- Profiling different modulation schemes or OFDM sizes.
- Comparing different equalization techniques.
- Combining profiling with performance metrics such as SER or BER.
