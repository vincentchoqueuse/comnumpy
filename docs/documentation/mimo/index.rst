MIMO
====


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   channels
   coding
   detectors
   compensators
   utils
   validators

Notes
-----

The `mimo` module implements : 

* classical frequency flat and selective MIMO channels,
* space-time block codes, taken from a registry as constellations are
  taken from ``get_alphabet``, and described by their linear dispersion
  matrices so that orthogonality is verified rather than declared,
* classical detectors for frequency flat MIMO communication, 
* compensation algorithm for channel estimation and equalization.
