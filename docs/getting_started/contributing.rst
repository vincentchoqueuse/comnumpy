Contributing to Comnumpy
=========================

We welcome contributions! Whether you're fixing a bug, improving the documentation, or developing a new submodule, your input is valuable.

Setting Up for Development
--------------------------

First, fork the repository and clone your fork:

.. code-block:: bash

   git clone https://github.com/vincentchoqueuse/comnumpy.git
   cd comnumpy

Create a new virtual environment and activate it:

.. code-block:: bash

   conda create -n comnumpy-dev python=3.11
   conda activate comnumpy-dev

Then install the package in editable mode with development dependencies:

.. code-block:: bash

   pip install -e .[dev]

This will install the local package and any extra requirements listed for development and testing.

Guidelines
----------

- Follow PEP8 formatting.
- Add tests for new features or fixes (see the `tests/` directory).
- Document your code with docstrings following the NumPy/SciPy style.
- Keep pull requests focused and concise.
- Write meaningful commit messages.

Extending comnumpy
------------------

We strongly encourage contributors to develop self-contained submodules for adding new communication system models or signal processing tools. A submodule typically includes:

- A dedicated directory under ``src/comnumpy/``
- A ``__init__.py`` file
- Python modules implementing your algorithms
- Unit tests in the ``tests/`` directory
- Documentation under ``docs/`` (e.g., ``docs/examples/`` or ``docs/documentation/``)

You can use existing submodules like ``optical`` or ``mimo`` as templates.

Reporting Issues
----------------

If you encounter a bug or have a feature request, please open an issue on GitHub. Be sure to include:

- A clear description of the problem or proposal
- Steps to reproduce (for bugs)
- Your Python version and operating system

