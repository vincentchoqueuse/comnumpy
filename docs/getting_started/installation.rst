Installation
============

To get started with **comnumpy**, you need to have **Python 3.8+** and ``pip`` installed on your system.

Until the library reaches a mature and stable state, it is not yet available on PyPI.  
In the meantime, you can install the latest development version directly from the GitHub repository using the following command:

.. code-block:: bash

    pip install git+https://github.com/vincentchoqueuse/comnumpy.git

This will clone the latest version of the project and install all necessary dependencies (such as ``numpy``, ``scipy``, ``matplotlib``, etc.).

.. note::

   Once the library reaches a mature state, it will be officially released on `PyPI <https://pypi.org/>`_ for easier installation.


Developer Installation
----------------------

If you wish to contribute to the development of **comnumpy**, you can install it in **editable mode**:

.. code-block:: bash

    git clone https://github.com/vincentchoqueuse/comnumpy.git
    cd comnumpy
    pip install -e .[dev]

This will allow you to modify the source code and test changes without reinstalling the package.  
The `[dev]` extra installs development dependencies such as `pytest`, `black`, and `sphinx`.

.. note::

   Contributors are encouraged to implement and document new submodules under ``comnumpy`` to extend the library’s capabilities  
   (e.g., ``comnumpy.optical``, ``comnumpy.mimo``, ``comnumpy.nonlinear``).

Please refer to the `contributing` section for guidelines on submitting pull requests and style conventions.
