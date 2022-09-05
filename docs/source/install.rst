
Installation
============

From PyPi::

  pypi install xarray-histogram


From source::

  git clone https://github.com/Descanonge/xarray-histogram
  cd xarray-histogram
  pypi install -e .


Requirements
------------

- Python >= 3.7
- numpy
- xarray
- `boost-histogram <https://github.com/scikit-hep/boost-histogram>`_

Optional dependencies
---------------------

If not available all data will be eagerly loaded.

- `dask <https://www.dask.org/>`_
- `dask-histogram <https://github.com/dask-contrib/dask-histogram>`_
