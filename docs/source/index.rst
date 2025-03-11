
.. currentmodule:: xarray_histogram

##############################
XArray-Histogram documentation
##############################

This package allows to compute histograms from and to XArray data. It relies on
the :external+boost-histogram:doc:`Boost Histogram <index>` library giving
better performances compared to :func:`numpy.histogram` and the existing
:external+xhistogram:doc:`xhistogram <index>`. It also brings features such as
integer/discrete bins or periodic bins.

Dask arrays are supported.

Vectorized manipulation and analysis of the resulting histogram(s) is provided
via an XArray accessor.

Installation
============

* Soon from PyPI ... 🚧

* From source::

   git clone https://github.com/Descanonge/xarray-histogram
   cd xarray-histogram
   pip install -e .

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage

   accessor

   performances

.. toctree::
   :maxdepth: 1

   api

Links
=====

Source code: `<https://github.com/Descanonge/xarray-histogram>`__

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
