OFDM
====

.. currentmodule:: src

Basic Usage
-----------

.. code ::

    import numpy as np
    from core import Sequential
    from generators.data import Symbol_Generator
    from dsp.modem import get_alphabet, Modulator
    from dsp.utils import Serial_2_Parallel, Parallel_2_Serial
    from dsp.ofdm import get_standard_carrier_allocation, Carrier_Allocator, IFFT_Processor, Cyclic_Prefix_Adder

    # parameters
    M = 16
    N_h = 5
    N_cp = 10

    carrier_type = get_standard_carrier_allocation("802.11ah_128")
    nb_carriers = len(carrier_type)
    nb_carrier_data = len(np.where(carrier_type==1)[0])
    nb_carrier_pilots = len(np.where(carrier_type==2)[0])

    pilots = 10*np.ones(nb_carrier_pilots)
    alphabet = get_alphabet("QAM", M)

    # create sequential
    transmitter = Sequential(
        [
            Symbol_Generator(np.arange(M))
            Modulator(alphabet),
            Serial_2_Parallel(nb_carrier_data),
            Carrier_Allocator(carrier_type, pilots=pilots),
            IFFT_Processor(),
            recorder_metric,
            Cyclic_Prefix_Adder(N_cp),
            Parallel_2_Serial()
        ]
    )



Classes
-------

.. automodule:: dsp.ofdm
   :members: