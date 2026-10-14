.. _development-workflow:

********************
Development Workflow
********************

This page documents the integration workflow, code-style policy, versioning
scheme, and deprecation policy used by *quantities*.


Branching and integration
=========================

The ``master`` branch is the integration branch. All development happens in
topic branches on each contributor's personal fork of the repository
(including for maintainers) and is merged into ``master`` via pull request.
Branches are not pushed directly to the upstream repository.

Pull requests should:

* be based on an up-to-date copy of ``master``,
* contain tests covering any behavioural change,
* keep unrelated changes out (one logical change per PR).

Trivial single-commit PRs are fine; longer changes should be broken into
logically coherent commits.


Code review
===========

Every pull request must be reviewed by at least one maintainer before being
merged. CI (GitHub Actions) must pass, including:

* the ``pytest`` matrix across the supported Python and NumPy versions,
* the ``mypy`` type-check job,
* the ``ruff`` lint job (currently advisory — see below),
* the ``doctest`` of ``README.rst``.

Reviewers look for: correctness, test coverage, dimensional-correctness of
new units or operations, and compatibility with the documented API.


Code style
==========

The project follows :pep:`8`. The ``ruff`` linter is configured in
``pyproject.toml`` and is run in CI. While the existing codebase is brought
into full conformance the lint job is **advisory** (``continue-on-error:
true``); new code should nevertheless pass ``ruff check`` cleanly.

Docstrings follow the NumPy documentation standard described in
:ref:`documenting-quantities`.

To run the linter locally::

  pip install -e ".[lint]"
  ruff check quantities


.. _versioning-policy:

Versioning
==========

*quantities* follows `Semantic Versioning 2.0.0`_. Public API changes are
reflected in the version number as follows:

* a **patch** bump (``0.16.3`` → ``0.16.4``) for bug fixes that do not
  change the public API;
* a **minor** bump (``0.16.x`` → ``0.17.0``) for backwards-compatible
  feature additions;
* a **major** bump (``0.x`` → ``1.0``) for backwards-incompatible changes,
  once the project reaches a 1.0 release.

The current major version is ``0``, so per SemVer §4 anything may change
between minor releases. In practice the project treats minor releases as
allowed to contain breaking changes only when accompanied by a documented
deprecation period (see below).

Version numbers are derived from git tags by `setuptools_scm`_; tagged
releases of the form ``vX.Y.Z`` are pushed to GitHub and uploaded to PyPI
(see :ref:`release-process`).

.. _Semantic Versioning 2.0.0: https://semver.org/spec/v2.0.0.html
.. _setuptools_scm: https://github.com/pypa/setuptools_scm/


Deprecation policy
==================

Public API elements that are scheduled for removal or change should be
deprecated before being removed:

1. In the release in which the deprecation is decided:

   * emit a :class:`DeprecationWarning` (or a more specific subclass such
     as :class:`FutureWarning`) from the deprecated entry point;
   * mark the entry in the docstring with the Sphinx ``deprecated::``
     directive, naming the version in which the deprecation was introduced
     and the replacement to use;
   * record the deprecation in ``CHANGES.txt`` under the release notes for
     that release.

2. Keep the deprecated entry point working for **at least one minor
   release** before it is removed.

3. In the release that removes the deprecated entry point, record the
   removal in ``CHANGES.txt``.

For small, low-impact API tweaks where no caller is plausibly affected, the
deprecation step may be skipped at the maintainer's discretion; this should
still be called out in ``CHANGES.txt``.
