AWGN Chain Tutorial
===================

This tutorial guides you through setting up and running a communication chain simulation using the ``comnumpy`` library. We will simulate a communication chain with a specified modulation scheme and evaluate its performance in terms of Symbol Error Rate (SER) over various Signal-to-Noise Ratios (SNRs).

Prerequisites
^^^^^^^^^^^^^

Ensure you have the following Python libraries installed:
- ``numpy``
- ``matplotlib``
- ``comnumpy``

You can install any missing libraries using pip:

.. code-block:: bash

    pip install numpy matplotlib

Simulation Setup
^^^^^^^^^^^^^^^^

1. Import Libraries
^^^^^^^^^^^^^^^^^^^

Begin by importing the necessary libraries.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from comnumpy.core import Sequential, Recorder
    from comnumpy.core.generators import SymbolGenerator
    from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
    from comnumpy.core.utils import get_alphabet
    from comnumpy.core.channels import AWGN
    from comnumpy.core.metrics import compute_ser, compute_metric_awgn_theo

2. Define Parameters
^^^^^^^^^^^^^^^^^^^^

Set the parameters for the simulation.

.. code-block:: python

    M = 16  # Modulation order
    N = 1000000  # Number of symbols
    modulation = "QAM"  # Modulation scheme
    alphabet = get_alphabet(modulation, M)  # Get alphabet for the modulation scheme
    snr_dB_list = np.arange(0, 22)  # SNR range in dB

3. Create Communication Chain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define the communication chain using the ``Sequential`` class.

.. code-block:: python

    chain = Sequential([
        SymbolGenerator(M),
        Recorder(name="recorder_tx"),
        SymbolMapper(alphabet),
        AWGN(unit="snr_dB", name="awgn_channel"),
        SymbolDemapper(alphabet),
    ])

Note that the unit parameters of the AWGN processor specifies the unit for the noise value.

4. Monte Carlo Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

Perform the simulation over the defined SNR range.

.. code-block:: python

    ser_array = np.zeros(len(snr_dB_list))

    print("* Simulation results")
    for index, snr_dB in enumerate(snr_dB_list):
        chain["awgn_channel"].value = snr_dB
        y = chain(N)
        data_tx = chain["recorder_tx"].get_data()
        ser = compute_ser(data_tx, y)
        ser_array[index] = ser
        print(f"SNR={snr_dB}: ser={ser} (exp)")

5. Compute Theoretical SER
^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the theoretical SER for comparison.

.. code-block:: python

    snr_per_bit = (10**(snr_dB_list/10))/np.log2(M)
    ser_theo_array = compute_metric_awgn_theo(modulation, M, snr_per_bit, "ser")

6. Plot Results
^^^^^^^^^^^^^^^

Visualize the SER performance.

.. code-block:: python

    plt.semilogy(snr_dB_list, ser_array, label="exp")
    plt.semilogy(snr_dB_list, ser_theo_array, "--", label="theo")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title(f"SER performance for {M}-{modulation}")
    plt.legend()
    plt.grid()
    plt.show()

.. image:: img/monte_carlo_awgn.png

Conclusion
----------

This tutorial demonstrated how to set up and run a communication chain simulation using the ``comnumpy`` library. You learned how to define the simulation parameters, create the communication chain, perform Monte Carlo simulations, compute theoretical SER, and plot the results.

Complete Code
-------------

.. literalinclude:: ../../examples/simple/monte_carlo_awgn.py
   :language: python
   :caption: Performance in AWGN Channel
