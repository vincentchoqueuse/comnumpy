MIMO Chain Tutorial
===================

This tutorial demonstrates how to simulate a MIMO (Multiple-Input Multiple-Output) communication system using the ``comnumpy`` library.

**What you'll learn:**

- How to build a MIMO simulation chain with Rayleigh fading.
- How to visualize received and equalized signals.
- How to compare detection algorithms (ZF, MMSE, OSIC, ML) on one chain.
- How to run a Monte Carlo evaluation of the Symbol Error Rate (SER) with ``sweep``.

This tutorial is suited for engineers and students learning about MIMO systems, combining practical examples with theoretical background.

Introduction
^^^^^^^^^^^^

Prerequisites
"""""""""""""

Ensure you have the following Python libraries installed:

.. code::

   numpy
   matplotlib
   comnumpy

Simulation Setup
^^^^^^^^^^^^^^^^

Import Libraries
""""""""""""""""

We start by importing the required libraries and ``comnumpy`` components:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 1-15

Define System Parameters
""""""""""""""""""""""""

We define the number of transmit/receive antennas, the modulation order (PSK), and the noise variance. The channel is drawn once, with a seed: the whole first half of the tutorial is about **one** realization, so it must be the same one on every run.

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 17-25

Build the MIMO Chain
""""""""""""""""""""

The link is one ``Sequential``: symbol generator, mapper, flat MIMO channel, noise, and a detector. The detector is the **last block of the chain**, not something applied to its output, so comparing four detectors is comparing four chains that differ by one block:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 28-56

This simulates a MIMO transmission over a flat-fading channel with additive Gaussian noise. The received signal is described by:

.. math ::

   \mathbf{y}[n] = \mathbf{H}\mathbf{x}[n] + \mathbf{b}[n]

The chain, as the chain itself describes it:

.. mermaid:: mermaid/mimo_zf.mmd

The diagram above is not drawn by hand. It is what the chain says about
itself -- ``chain.to_mermaid()`` (decision D33c) -- exported by the
script, so the block names are the ones the code uses and a dashed
outline marks a tapped block:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 174-180

One-Shot Simulation
^^^^^^^^^^^^^^^^^^^

Run the four chains
"""""""""""""""""""

Each chain is given the same seed before running, so the four numbers below differ by the detector alone -- same symbols, same noise, same channel:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 58-64

.. code::

   * detector ZF   : ser=0.0025
   * detector MMSE : ser=0.0025
   * detector OSIC : ser=0.0010
   * detector ML   : ser=0.0005
   * detector SD   : ser=0.0005

Visualize the Received Signal
"""""""""""""""""""""""""""""

Let's inspect what each receive antenna sees, read from the ``"noise"`` tap:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 66-76

.. image:: img/monte_carlo_mimo_fig1.png
   :width: 100%
   :align: center

Each antenna receives a noisy superposition of the two transmitted streams, and no constellation is visible on any of them.

Zero-Forcing Equalization
"""""""""""""""""""""""""

Zero forcing separates the streams with the pseudo-inverse of the channel matrix,

.. math ::

   \mathbf{z}[n] = \mathbf{H}^{\dagger}\mathbf{y}[n]

assuming perfect channel knowledge and ignoring the noise enhancement this causes. That equalization is the first step of ``LinearDetector``, which then decides; ``linear_estimator`` is that step alone, which is what a constellation plot needs:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 78-90

.. image:: img/monte_carlo_mimo_fig2.png
   :width: 100%
   :align: center

The estimated points cluster around the ideal constellation points (black crosses), although residual noise remains visible -- and it is *larger* than the channel noise, because inverting a badly conditioned matrix amplifies it.

Detection Comparison
^^^^^^^^^^^^^^^^^^^^

The four detection strategies compared here are:

- **ML**: Maximum Likelihood

.. math ::

   \widehat{\mathbf{x}}_{ML}[n] = \arg \min_{\mathbf{x}\in \mathcal{M}^{N_t}}\|\mathbf{y}[n] - \mathbf{H}\mathbf{x}\|^2_2

- **ZF**: Zero-Forcing

.. math ::
   \widehat{\mathbf{x}}_{ZF}[n] &= \boldsymbol \Pi_{\mathcal{M}}(\mathbf{z}[n])\\
   \mathbf{z}[n] &= \mathbf{H}^{\dagger}\mathbf{y}[n]

- **MMSE**: Minimum Mean Square Error

.. math ::
   \widehat{\mathbf{x}}_{MMSE}[n] &= \boldsymbol \Pi_{\mathcal{M}}(\mathbf{z}[n])\\
   \mathbf{z}[n] &= \left(\mathbf{H}^H\mathbf{H} + \sigma^2 \mathbf{I}_{N_t}\right)^{-1}\mathbf{H}^H\mathbf{y}[n]

- **OSIC**: Ordered Successive Interference Cancellation -- detect the strongest stream, subtract it, repeat on what is left.

- **SD**: Sphere Decoding -- the ML decision again, reached by searching a tree instead of scoring every candidate. See the section below.

Sphere decoding: the same decision, not the same cost
"""""""""""""""""""""""""""""""""""""""""""""""""""""

The ML detector is exhaustive: :math:`|\mathcal{M}|^{N_t}` candidates per symbol, 16 here and 65 536 for 16-QAM on four streams. The sphere decoder returns the *same* vector without visiting them all, and the reason is a factorization.

Write the thin QR decomposition of the channel, :math:`\mathbf{H} = \mathbf{Q}\mathbf{R}` with :math:`\mathbf{Q}^H\mathbf{Q} = \mathbf{I}_{N_t}` and :math:`\mathbf{R}` upper triangular, and project the observation on :math:`\mathbf{z} = \mathbf{Q}^H\mathbf{y}`:

.. math ::

   \left\|\mathbf{y} - \mathbf{H}\mathbf{x}\right\|^2 =
   \left\|\mathbf{z} - \mathbf{R}\mathbf{x}\right\|^2
   + \left\|\left(\mathbf{I} - \mathbf{Q}\mathbf{Q}^H\right)\mathbf{y}\right\|^2

The second term does not depend on :math:`\mathbf{x}`, so minimizing the first **is** maximum likelihood -- nothing has been approximated. And because :math:`\mathbf{R}` is triangular, that first term is a sum whose :math:`k`-th layer only involves the layers already decided:

.. math ::

   \left\|\mathbf{z} - \mathbf{R}\mathbf{x}\right\|^2 =
   \sum_{k=N_t-1}^{0} \left|R_{kk}\right|^2
   \left|c_k - x_k\right|^2,
   \qquad
   c_k = \frac{z_k - \sum_{i>k} R_{ki}\, x_i}{R_{kk}}

Every term is non-negative, so a partial sum can only grow. The moment it exceeds the best complete metric found so far, **the entire subtree below it can be discarded** -- it cannot contain the minimum. That is the sphere: only the lattice points inside a ball are ever visited, and the ball shrinks each time a better solution is found.

One detail makes it work without tuning. Within a layer, the alphabet is enumerated by increasing :math:`|c_k - a|` (the Schnorr-Euchner order), so the first leaf the search reaches is the successive-cancellation solution: a finite radius is available immediately, and no initial radius has to be guessed. It also means the first candidate that exceeds the bound ends the layer -- everything after it is worse.

Monte Carlo Evaluation
^^^^^^^^^^^^^^^^^^^^^^

A single channel realization proves nothing: over fading, the error rate is an *average*, and it is dominated by the rare draws where the matrix is nearly singular. Averaging therefore means running the chain once per realization -- which is a sweep whose parameter is the channel. :func:`~comnumpy.sweep.sweep` takes several dotted parameter names at once and zips them, so one sweep point sets the matrix the signal goes through **and** the one the detector inverts:

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 92-125

.. code::

   ZF    0.3313 0.2106 0.1065 0.0428 0.0148 0.0047 0.0020
   MMSE  0.2871 0.1762 0.0876 0.0345 0.0116 0.0036 0.0014
   OSIC  0.2826 0.1610 0.0668 0.0192 0.0037 0.0007 0.0003
   ML    0.2807 0.1560 0.0620 0.0159 0.0027 0.0006 0.0003

Note which detectors are told the noise variance and which are not: MMSE and OSIC weight by :math:`\sigma^2`, ZF ignores the noise by construction, and ML only compares distances -- so neither of the last two has that parameter at all, and ``set_params`` says so if you try.

Plot SER vs SNR
"""""""""""""""

``plot_error_rate`` is the library's figure for error-rate curves: hollow markers, logarithmic ordinate, a grid on both decades. Here it carries measurements only -- ZF over Rayleigh has a closed form (see the :doc:`Alamouti tutorial <alamouti>`), the three others do not.

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 127-135

.. image:: img/monte_carlo_mimo_fig3.png
   :width: 100%
   :align: center

The ordering is the textbook one and it holds at every SNR: ML is the best, OSIC follows it closely, MMSE beats ZF everywhere, and the gap widens with the SNR -- at 18 dB, ML is an order of magnitude below ZF. The reason is in the slope: with :math:`N_r = 3` receive antennas and :math:`N_t = 2` streams, ML enjoys the full receive diversity while a linear detector spends part of it cancelling the other stream.

And the sphere decoder's row is **the ML row, digit for digit** -- ``0.2807 0.1560 0.0620 0.0159 0.0027 0.0006 0.0003`` in both. That is the claim it has to make: a pruning that changed a single decision would not be maximum likelihood any more.

What the pruning removes
""""""""""""""""""""""""

The 4-PSK link above has 16 candidates, and an exhaustive search scores them in one matrix product -- there the sphere decoder is *slower*, because a tree walked in Python costs more per node than a BLAS call costs per candidate. The regime it exists for is the one where that product no longer fits, so the last section moves to 16-QAM on four streams: 65 536 candidates per symbol.

.. literalinclude:: ../../examples/mimo/one_shot_mimo.py
   :language: python
   :lines: 137-172

.. code::

   visited nodes per detected vector (16-QAM, 4x4)
      0 dB       78.0 nodes   1.19e-03 of the exhaustive search   SER 0.7831
      3 dB       46.7 nodes   7.13e-04 of the exhaustive search   SER 0.7156
      6 dB       30.0 nodes   4.58e-04 of the exhaustive search   SER 0.6356
      9 dB       19.3 nodes   2.94e-04 of the exhaustive search   SER 0.5206
     12 dB       12.9 nodes   1.97e-04 of the exhaustive search   SER 0.3544
     15 dB        7.8 nodes   1.19e-04 of the exhaustive search   SER 0.1588
     18 dB        5.9 nodes   8.94e-05 of the exhaustive search   SER 0.0231

.. image:: img/monte_carlo_mimo_fig4.png
   :width: 100%
   :align: center

Read the last column against the first. At 18 dB the search visits **5.9 nodes** where the exhaustive one scores 65 536 candidates -- and 4 of those nodes are the single path down the tree that successive cancellation would have taken, so barely two branches are ever explored. At 0 dB it visits 78, still four orders of magnitude below the exhaustive count, but thirteen times more than at 18 dB: the noisier the observation, the wider the sphere has to stay.

That data dependence is the whole character of the algorithm. Its worst case *is* the exhaustive search -- there is no bound to hide behind -- but its expected complexity is polynomial over the range of SNRs where a link is actually operated (Hassibi and Vikalo, 2005). Which is why 64-QAM on four streams, 16.7 million candidates, is out of reach for the exhaustive detector and takes this one about 5 nodes per vector at 25 dB.

Conclusion
^^^^^^^^^^

This tutorial highlighted:

- How to simulate a MIMO transmission with ``comnumpy``, as a chain whose last block is the receiver.
- How ZF equalization recovers the streams from a multi-antenna mixture, and what it costs in noise.
- How the four detectors compare, on one realization and on average.
- Why ML and OSIC outperform linear detection, and why that difference grows with the SNR.

With ``comnumpy``, you can rapidly prototype, test, and visualize MIMO systems for research, teaching, or self-study.
