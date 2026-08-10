Examples
========

These tutorials are meant to be read in order. Each one introduces the tools
the next one uses, so that nothing is ever used before it has been explained,
and nothing is explained twice.

.. list-table::
   :header-rows: 1
   :widths: 8 30 62

   * - #
     - Tutorial
     - What it introduces
   * - 0
     - :doc:`../getting_started/first_simulation`
     - The ``Sequential`` chain, the blocks it is made of, a tap to look
       inside it, and one error rate.
   * - 1
     - :doc:`awgn`
     - A Monte Carlo study: first as an ordinary ``for`` loop, then the same
       thing with ``sweep``. Also ``seed``, ``set_params`` and
       ``plot_error_rate``.
   * - 2
     - :doc:`ofdm`
     - A frequency-selective channel, and the two ways to equalize it:
       one matrix inversion, or one division per subcarrier.
   * - 3
     - :doc:`ofdm_papr`
     - What an OFDM waveform costs the amplifier, and the CCDF that measures
       it.
   * - 4
     - :doc:`multipath`
     - The standardized fading channels of 3GPP, the delay spread, and how
       long the cyclic prefix of tutorial 2 actually has to be.
   * - 5
     - :doc:`mimo`
     - Several antennas, and the five detectors that separate the streams --
       from the pseudo-inverse to the sphere decoder.
   * - 6
     - :doc:`alamouti`
     - Diversity without channel knowledge at the transmitter: space-time
       coding, measured against its closed form.
   * - 7
     - :doc:`shaping`
     - Sending some constellation points more often than others, and the
       1.53 dB it recovers.
   * - 8
     - :doc:`optical_fiber_nonlinearity`
     - A nonlinear channel: fibre propagation, and digital back-propagation
       to undo it.

If you are looking for one specific thing rather than a course, every page is
self-contained enough to be read alone -- it will simply point back at the
tutorial where a tool was introduced instead of explaining it again.

The example scripts are also available on GitHub: `https://github.com/vincentchoqueuse/comnumpy/tree/main/examples <https://github.com/vincentchoqueuse/comnumpy/tree/main/examples>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   awgn
   ofdm
   ofdm_papr
   multipath
   mimo
   alamouti
   shaping
   optical_fiber_nonlinearity
