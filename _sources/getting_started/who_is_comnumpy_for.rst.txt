Who is this library for?
========================

The ``comnumpy`` library is designed for researchers, engineers, and students working in digital communications. It is particularly useful for anyone who wants to simulate and analyze communication systems without implementing standard algorithms from scratch.

Prerequisites
-------------

``comnumpy`` relies only on standard numerical computing libraries such as ``numpy`` and ``scipy``. There are no complex or domain-specific dependencies to install. With a standard Python environment, you can install ``comnumpy`` and start using it right away.

Comparison with Other Solutions
-------------------------------

1. **Numpy from Scratch**:

   - **Advantages**: Using ``numpy`` directly offers maximum flexibility and control.
   - **Disadvantages**: Implementing and testing communication algorithms from scratch can be time-consuming and error-prone. ``comnumpy`` provides ready-to-use, tested implementations of common algorithms, letting you focus on research and analysis.

2. **MATLAB**:

   - **Advantages**: MATLAB is widely used in industry and academia for communication simulations and offers many specialized toolboxes.
   - **Disadvantages**: MATLAB requires a paid license, which can be a barrier for students and small teams. It is also less flexible when integrating with other programming languages and tools. ``comnumpy``, being Python-based, integrates seamlessly with the broader Python ecosystem, including machine learning frameworks and visualization libraries.

3. **Sionna**:

   - **Advantages**: Sionna is a Python library that leverages TensorFlow for end-to-end learning in communication systems. It is particularly powerful for applications that require deep learning and neural network integration.
   - **Disadvantages**: Sionna is highly specialized and may be more complex than necessary for users who need straightforward communication simulations. ``comnumpy`` aims to provide a simpler, more accessible alternative, with an emphasis on ease of use and modularity.

4. **OptiCommPy**:

   - **Advantages**: `OptiCommPy <https://github.com/edsonportosilva/OptiCommPy>`_ is an excellent library for **fibre-optic** communication systems, and it tracks recent developments in the field closely. Its coherent optical stack goes further than ``comnumpy``'s on several points that matter in practice: a Manakov split-step solver with **adaptive** step size (bounded by the nonlinear phase rotation per step, each step iterated to a tolerance) and optional GPU execution, a complete coherent receiver chain -- adaptive MIMO equalizer (CMA / RDE / DA-RDE), blind phase search, frequency-offset estimation, symbol synchronization -- and device models for lasers with phase noise and RIN, Mach-Zehnder modulators, coherent frontends and photodiodes. If your work is specifically about coherent optical transmission and its DSP, it is the more complete tool today.
   - **Disadvantages**: Its scope is optical. ``comnumpy`` is a *general* digital-communications library -- OFDM, MIMO, forward error correction, fading channels and optical fibre share one set of conventions and one chain abstraction -- so a study that spans several of these does not mean assembling several toolboxes.
   - **Cross-validated against it**: ``validation/optical_wdm_opticommpy.py`` reproduces the published 11 |times| 32 GBd PDM-16QAM, 700 km example of OptiCommPy. The transmitted comb power matches their printed figure to 0.004 dB, the achievable rates (MI, GMI, NGMI) match theirs exactly, and the end-to-end SNR lands within about 1 dB -- on the side the difference in receiver DSP predicts. The same script measures where they are ahead: on a *fixed* split-step grid, their nominal 0.5 km step is 0.85 dB short of converged, and it is their adaptive stepping that makes it enough.

.. |times| unicode:: U+000D7

Why Choose ``comnumpy``?
------------------------

- **Modularity**: Build custom communication chains by combining reusable processor blocks.
- **A familiar way to assemble a chain**: the ``Processor`` / ``Sequential`` pair is deliberately modelled on PyTorch's ``nn.Module`` / ``nn.Sequential``. If you have built a neural network in PyTorch, you already know how to build a communication chain here -- and the same flexibility follows: blocks are reconfigured after construction, chains are nested, seeded, swept and serialized without rewriting them.
- **One set of conventions across domains**: the same axis conventions, the same estimator conventions and the same chain services apply whether the chain is an OFDM link, a MIMO detector, a coded system or an optical fibre.
- **Ease of Use**: Get started quickly with clear examples and comprehensive documentation.
- **Checked against references, not against itself**: every physical model in the library is confronted with something it cannot fake -- a closed form, a conservation law, a published table, or another implementation. The scripts that do it live in ``validation/`` and run in continuous integration.
- **Open Source**: As a community-driven project, ``comnumpy`` encourages collaboration and continuous improvement.

Core Concepts
-------------

Before diving into the tutorials, it helps to understand two key abstractions in ``comnumpy``:

- **Processor**: The basic building block. Each ``Processor`` represents a single signal-processing operation (e.g., modulation, channel, equalization). It takes an input signal and returns an output signal. It is the counterpart of PyTorch's ``nn.Module``.
- **Sequential**: A container that chains multiple ``Processor`` objects together. When called, a ``Sequential`` passes data through each processor in order, forming a complete communication chain -- the counterpart of ``nn.Sequential``.

This composable design lets you build complex simulations by snapping together simple, reusable components. What the container adds beyond composition is what makes a *study* convenient rather than just a run: naming and addressing blocks, reconfiguring their parameters after construction, seeding every stochastic block reproducibly, observing signals inside the chain without inserting anything into it, and exporting the whole thing to JSON.
