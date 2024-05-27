
XArray-Histogram documentation
==============================

This package allow to compute histograms from XArray data, taking advantage of its label and dimensions management.
It relies on the `Boost Histogram <https://boost-histogram.readthedocs.io>`_ library for the computations.

It is essentially a thin wrapper using directly `Boost Histogram <https://boost-histogram.readthedocs.io>`_ on loaded data, or `Dask-histogram <https://dask-histogram.readthedocs.io>`_ on data contained in dask arrays. It thus features optimised performance, as well as lazy computation and easy up-scaling thanks to dask.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   install

   usage

.. toctree::
   :maxdepth: 1

   api

Source code: `<https://github.com/Descanonge/xarray-histogram>`__

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
