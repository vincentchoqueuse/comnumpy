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

Why Choose ``comnumpy``?
------------------------

- **Modularity**: Build custom communication chains by combining reusable processor blocks.
- **Ease of Use**: Get started quickly with clear examples and comprehensive documentation.
- **Open Source**: As a community-driven project, ``comnumpy`` encourages collaboration and continuous improvement.

Core Concepts
-------------

Before diving into the tutorials, it helps to understand two key abstractions in ``comnumpy``:

- **Processor**: The basic building block. Each ``Processor`` represents a single signal-processing operation (e.g., modulation, channel, equalization). It takes an input signal and returns an output signal.
- **Sequential**: A container that chains multiple ``Processor`` objects together. When called, a ``Sequential`` passes data through each processor in order, forming a complete communication chain.

This composable design lets you build complex simulations by snapping together simple, reusable components.
