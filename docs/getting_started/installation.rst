Installation
============

To get started with **comnumpy**, you need **Python 3.11+** and ``pip`` installed on your system.

As the library has not yet reached a stable release, it is not available on PyPI.
In the meantime, you can install the latest development version directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/vincentchoqueuse/comnumpy.git

This will fetch the latest version and install all required dependencies (``numpy``, ``scipy``, ``matplotlib``, etc.).

.. note::

   Once the library reaches a stable release, it will be published on `PyPI <https://pypi.org/>`_ for easier installation.


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
