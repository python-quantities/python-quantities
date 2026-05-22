# Contributing to quantities

Contributions to *quantities*, whether bug reports, bug fixes, new units or constants, documentation improvements, or new features, are welcome.

## Reporting bugs and requesting features

Please use the [GitHub issue
tracker](https://github.com/python-quantities/python-quantities/issues).

For bug reports, please include:

- the version of *quantities* and of NumPy you are using,
- your Python version and operating system,
- a minimal example that reproduces the problem,
- the actual output and the output you expected.

## Submitting changes

Development happens on GitHub via pull requests against the `master` branch.
A short walkthrough — from cloning the repository to opening your first PR —
is in the [Onboarding guide][onboarding] in the developer documentation,
and the [Development workflow][workflow] page covers branching, review,
versioning and the deprecation policy.

Before opening a pull request:

1. Make sure the test suite passes locally (`pytest`).
2. Add a test for any bug fix or new feature.
3. Add an entry to `CHANGES.txt` describing the change.

[onboarding]: https://python-quantities.readthedocs.io/en/latest/devel/onboarding.html
[workflow]: https://python-quantities.readthedocs.io/en/latest/devel/workflow.html

## Licensing of contributions

*quantities* is distributed under the BSD 3-Clause License (see
[`doc/user/license.rst`](doc/user/license.rst)).
By submitting a contribution (for example by opening
a pull request) you agree that your contribution is licensed under the same
license and that you have the right to make it under those terms.

No separate Contributor License Agreement (CLA) is required.
