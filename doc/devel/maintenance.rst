.. _maintenance:

***********
Maintenance
***********

This page collects the routine maintenance tasks that come up between
releases. For the release procedure itself see :ref:`release-process`,
and for the general workflow see :ref:`development-workflow`.


Updating dependencies
=====================

*quantities* has a single runtime dependency (NumPy) and a handful of
optional ones (``pytest`` for tests, ``sphinx`` for docs, ``ruff`` for
linting, ``scipy`` as an optional backend for numerical integration).

When NumPy releases a new minor version:

1. Add the new version to the ``numpy-version`` matrix in
   ``.github/workflows/test.yml`` (both the ``test`` and ``type-check``
   jobs). Update the ``exclude`` list to reflect supported
   Python × NumPy combinations as needed.
2. Run the test suite locally against the new NumPy::

     pip install "numpy==X.Y.*"
     pytest

3. Fix any regressions and record the new supported version in
   ``CHANGES.txt``.

When dropping support for an old Python or NumPy version, also bump
``requires-python`` and the ``numpy`` lower bound in ``pyproject.toml``
and update the ``target-version`` of the ``[tool.ruff]`` config.


Rebuilding the documentation locally
====================================

The published documentation is rebuilt automatically by Read the Docs on
every push to ``master``. To rebuild locally during development::

  pip install -e ".[doc]"
  cd doc
  make html

The HTML is written to ``doc/_build/html/``. Open ``index.html`` in your
browser to preview.

The Read the Docs build is configured by ``.readthedocs.yaml`` (Python
version, build OS, requirements file) and ``doc/conf.py`` (Sphinx
extensions, theme, project metadata).


Coordinating with downstream packagers
======================================

*quantities* is packaged in several downstream channels:

* `PyPI <https://pypi.org/project/quantities/>`__ — the primary
  distribution, updated by the maintainer at release time (see
  :ref:`release-process`).
* `conda-forge <https://github.com/conda-forge/quantities-feedstock>`__ —
  the conda-forge bot opens a PR on the feedstock automatically when a
  new PyPI release appears.
* `Spack <https://packages.spack.io/package.html?name=py-quantities>`__
  — the recipe lives in the
  `spack-packages <https://github.com/spack/spack-packages>`__ repository
  at ``repos/spack_repo/builtin/packages/py_quantities/package.py``.
  Updates are made by opening a PR against ``spack/spack-packages`` after
  a release.

For other downstream packagers we rely on the packagers' own
update workflows. `Repology
<https://repology.org/project/python:quantities/versions>`__ shows a
useful cross-distribution view of which version each packager is
currently shipping; please open an issue if you find a downstream
package that has fallen significantly behind.
