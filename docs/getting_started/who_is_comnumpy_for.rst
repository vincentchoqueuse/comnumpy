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

   - **Advantages**: `OptiCommPy <https://doi.org/10.21105/joss.06600>`_ (da Silva and Herbster, *JOSS* **9**\ (98), 6600, 2024) is an excellent library for **fibre-optic** communication systems, and it tracks recent developments in the field closely. Its coherent optical stack goes further than ``comnumpy``'s on several points that matter in practice: a Manakov split-step solver with **adaptive** step size (bounded by the nonlinear phase rotation per step, each step iterated to a tolerance) and optional GPU execution, a complete coherent receiver chain -- adaptive MIMO equalizer (CMA / RDE / DA-RDE), blind phase search, frequency-offset estimation, symbol synchronization -- and device models for lasers with phase noise and RIN, Mach-Zehnder modulators, coherent frontends and photodiodes. If your work is specifically about coherent optical transmission and its DSP, it is the more complete tool today.
   - **Where ``comnumpy`` differs**: not in breadth -- breadth is not an advantage to someone who only does optics, and it is partly a liability, since focus is exactly what lets OptiCommPy go deeper. The difference is that OptiCommPy is a library of **functions** you thread arrays through by hand, while in ``comnumpy`` the *system* is an object: named, addressable, re-seeded, swept, serialized to JSON and drawn. See :ref:`what-makes-it-different` below.
   - **Cross-validated against it**: ``validation/optical_wdm_opticommpy.py`` reproduces the published 11 |times| 32 GBd PDM-16QAM, 700 km example of OptiCommPy. The transmitted comb power matches their printed figure to 0.004 dB, the achievable rates (MI, GMI, NGMI) match theirs exactly, and the end-to-end SNR lands within about 1 dB -- on the side the difference in receiver DSP predicts. The same script measures where they are ahead: on a *fixed* split-step grid, their nominal 0.5 km step is 0.85 dB short of converged, and it is their adaptive stepping that makes it enough.

.. |times| unicode:: U+000D7

.. _what-makes-it-different:

What makes ``comnumpy`` different
---------------------------------

Being general is not, by itself, a reason to choose a library: it only
pays for someone who actually crosses domains. Three things are.

**1. The chain is an object, not a sequence of calls.** The
``Processor`` / ``Sequential`` pair is deliberately modelled on
PyTorch's ``nn.Module`` / ``nn.Sequential``, and the consequences are
the ones that matter in daily research work. A chain can be::

    chain = Sequential([SymbolGenerator(16, name="tx"), SymbolMapper(alphabet),
                        AWGN(snr_dB=15, name="noise")], taps=["tx"])

    chain.seed(42)                            # every stochastic block, reproducibly
    chain.set_params(**{"noise.snr_dB": 12})  # reconfigured after construction
    results = sweep(chain, {"noise.snr_dB": range(0, 20, 2)})
    text = to_json(chain)                     # the experiment, as a file
    print(chain.to_mermaid())                 # the experiment, as a picture

None of that requires rewriting the chain, and none of it is
instrumentation inserted between the blocks: taps, wiring and seeding
are chain *metadata*, so the block list keeps describing the
communication system and nothing else. A study made of a hundred
parameter points and a figure is then a few lines, and the exact
experiment that produced the figure is a JSON file you can archive next
to it.

**2. Every docstring is teaching material.** Each block documents the
signal model it implements in mathematics, with a one-to-one
correspondence between the symbols of the equations and the parameters
of the code, the references the model comes from, and a worked example
that is executed on every commit. Reading the source is meant to be a
way to learn the subject, not only to use the function.

**3. Physical models are confronted with something they cannot fake.**
Not with their own past output: with a closed form, a conservation law,
a published table, or another implementation. Those confrontations live
in ``validation/`` as readable scripts, they print the numbers they
obtain, and they run in continuous integration -- so a model that
drifts away from its reference breaks the build. The comparison with
OptiCommPy above is one of them.

Other reasons
-------------

- **Modularity**: Build custom communication chains by combining reusable processor blocks.
- **One set of conventions across domains**: the same axis conventions, the same estimator conventions and the same chain services apply whether the chain is an OFDM link, a MIMO detector, a coded system or an optical fibre.
- **Ease of Use**: Get started quickly with clear examples and comprehensive documentation.
- **Open Source**: As a community-driven project, ``comnumpy`` encourages collaboration and continuous improvement.

Core Concepts
-------------

Before diving into the tutorials, it helps to understand two key abstractions in ``comnumpy``:

- **Processor**: The basic building block. Each ``Processor`` represents a single signal-processing operation (e.g., modulation, channel, equalization). It takes an input signal and returns an output signal. It is the counterpart of PyTorch's ``nn.Module``.
- **Sequential**: A container that chains multiple ``Processor`` objects together. When called, a ``Sequential`` passes data through each processor in order, forming a complete communication chain -- the counterpart of ``nn.Sequential``.

This composable design lets you build complex simulations by snapping together simple, reusable components. What the container adds beyond composition is what makes a *study* convenient rather than just a run: naming and addressing blocks, reconfiguring their parameters after construction, seeding every stochastic block reproducibly, observing signals inside the chain without inserting anything into it, and exporting the whole thing to JSON.
