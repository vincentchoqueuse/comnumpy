Installation
============

To get started with **comnumpy**, you need **Python 3.11+** and ``pip`` installed on your system.

.. code-block:: bash

    pip install comnumpy

This will install all required dependencies (``numpy``, ``scipy``, ``matplotlib``, etc.).


Developer Installation
----------------------

If you want to contribute to **comnumpy**, you can install it in **editable mode**:

.. code-block:: bash

    git clone https://github.com/vincentchoqueuse/comnumpy.git
    cd comnumpy
    pip install -e .

This lets you modify the source code and test changes without reinstalling the package.

.. note::

   Contributors are encouraged to implement and document new submodules to extend the library’s capabilities
   (e.g., ``comnumpy.optical``, ``comnumpy.mimo``, ``comnumpy.nonlinear``).

Please refer to the `contributing` section for guidelines on submitting pull requests and style conventions.
