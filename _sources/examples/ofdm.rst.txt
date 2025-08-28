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

Let us define some communication parameters


.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 14-24

The two last lines define the taps of a frequency-selective multipath channel.  
The first tap is normalized to 1 in order to preserve the overall channel energy.  

Mathematically, the input–output relationship of such a channel is given by

.. math::

   z[n] = h * x[n] + b[n] = \sum_{l=0}^{L_{\text{tap}}-1} h[l]\,x[n-l] + b[n]

where :math:`*` denotes the convolution operator.  
This type of channel introduces inter-symbol interference (ISI) in single-carrier systems since the received sample :math:`z[n]` depends on multiple transmitted symbols :math:`x[n]`.  

If we stack :math:`N` consecutive received samples into a column vector :math:`\mathbf{z}`, we obtain the following compact matrix formulation:

.. math::

   \mathbf{z} = \mathbf{H}\mathbf{x} + \mathbf{b}

where :math:`\mathbf{H}` is a **Toeplitz convolution matrix** constructed from the channel taps. For example, if :math:`\mathbf{x} = [x[0], x[1], \dots, x[N-1]]^T`, then :math:`\mathbf{H}` takes the form

.. math::

   \mathbf{H}=
   \begin{bmatrix}
   h[0]   & 0      & \cdots & 0 \\
   h[1]   & h[0]   & \ddots & \vdots \\
   h[2]   & h[1]   & \ddots & \vdots \\
   \vdots & \vdots & \ddots & \vdots \\
   0      & 0      & \cdots & h[L-1]
   \end{bmatrix}


3. Single Carrier Chain
^^^^^^^^^^^^^^^^^^^^^^^

Signal Model
""""""""""""

Let's build a basic SC simulation to illustrate the performance in a selective channel:

.. mermaid::

   graph LR;
      A[Generator] --> C[Mapper];
      C --> D[Channel];
      D --> E[AWGN];
      E --> G[Equalizer];
      G --> H[Demapper];

At the receiver, the channel is compensated using a simple Zero-Forcing (ZF) equalizer:

.. math::

   \widehat{\mathbf{x}} = \mathbf{H}^{\dagger}\mathbf{z}

where :math:`\mathbf{H}^{\dagger}` denotes the pseudo-inverse of the channel matrix.  
Note that, due to the large number of transmitted samples, the implementation of more sophisticated detectors—such as the Maximum Likelihood (ML) detector—becomes computationally prohibitive.

Implementation
""""""""""""""

In **comnumpy**, this communication chain can be conveniently implemented as follows:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 26-41

Results
"""""""

To evaluate this communication chain, we can evaluate some detection metric (such as SER) and the computational complexity measured in terms of computational time. We can also plot the signal constellation before and after the equalizer to highlight the benefit of the ZF equalizer.

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 42-58

For the single-carrier chain, we obtain:

- A Symbol Error Rate (SER) equals to : 0.007
- A total computational time equals to: 0.562s

The following figures present the constellation of the the received signal before and after linear equalization.

.. image:: img/one_shot_ofdm_fig1.png

4. OFDM Chain
^^^^^^^^^^^^^

In frequency-selective channels, the conventional single-carrier chain suffers from suboptimal detection performance and high computational complexity, mainly due to the need to invert a large matrix. A common solution to overcome this limitation is to employ an OFDM chain.


Signal Model
""""""""""""

.. mermaid::

   graph LR;
      A[Generator] --> B[Mapper];
      B --> C[OFDM Tx];
      C --> D[Channel];
      D --> E[AWGN];
      E --> G[OFDM Rx];
      G --> H[Demapper];

The OFDM transmitter and receiver can be decomposed as follows:

* OFDM TX:

.. mermaid::

   graph LR;
      A[Mapper] --> B[S2P];
      B --> C[IDFT];
      C --> D[CP add];
      D --> E[P2S];
      E --> G[Channel];

* OFDM RX:

.. mermaid::

   graph LR;
      A[Channel] --> B[P2S];
      B --> C[CP del];
      C --> D[DFT];
      D --> E[Equalizer];
      E --> F[P2S];


where 

- S2P / P2S: Serial-to-Parallel and Parallel-to-Serial converters. S2P takes the incoming serial bitstream and splits it into :math:N_{sc} parallel symbol streams, while P2S performs the reverse operation.
- IDFT / DFT: The Inverse Discrete Fourier Transform (TX) generates the time-domain OFDM signal from frequency-domain symbols. At the RX, the DFT transforms the received time-domain samples back into the frequency domain.
- CP add / CP del: The Cyclic Prefix is appended at the transmitter to mitigate intersymbol interference caused by multipath. At the receiver, this prefix is removed before further processing.
- Equalizer: Compensates for the distortions introduced by the channel (e.g. frequency-selective fading), restoring the original symbol constellation as closely as possible.

The main advantage of this structure is that, before equalization, one can verify that the received vector can be written as


.. math::

   \mathbf{z} = \mathbf{D}\mathbf{x} + \mathbf{n}

where 

.. math::

   \mathbf{D} = \textrm{diag}\big(H[0],H[1],\dots,H[N-1]\big)

is a diagonal matrix, and 

.. math::

   H[k]=\sum_{l=0}^{L-1} h[l]e^{-j\frac{2\pi}{N}kl}
   
corresponds to the DFT of the channel impulse response.

In other words, an OFDM system transforms a frequency-selective channel into a bank of parallel flat-fading subchannels, each of which can be equalized independently with a simple one-tap detector (e.g., ML, ZF, or MMSE). This property follows from the cyclic prefix (CP): inserting and removing the CP turns linear convolution into circular convolution, which is diagonalized by the DFT/IDFT matrices. Note that the multiplication with DFT/IDFT matrices can be computed efficiently via the FFT/IFFT, allowing to drastically reduce the computational complexity. 


Implementation
""""""""""""""

We can easily simulate the OFDM communication chain using **comnumpy** as follows:

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 60-79


Results
"""""""

.. literalinclude:: ../../examples/ofdm/one_shot_ofdm.py
   :language: python
   :lines: 81-92

For the OFDM chain, we obtain the following results:

- Symbol Error Rate (SER): 0.004 ( < 0.007)
- Total computational time: 0.007s ( < 0.562s)

Note that this substantial reduction in computational complexity stems from the fact that channel equalization reduces to the inversion of a diagonal matrix, while the OFDM transmitter and receiver rely on Fast Fourier Transform (FFT) algorithms. Concerning the Symbol Error Rate (SER), the improvement arises from the fact that, under Gaussian noise, the receiver uses a Maximum Likelihood detector under flat fading channels.


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