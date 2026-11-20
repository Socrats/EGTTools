# Contributing to EGTtools

EGTtools aims to make evolutionary game theory methods and models reproducible and available to a
wider audience. Contributions of every size are welcome.

## Ways to contribute

- **Report a bug or a wrong result**, using the [issue forms](https://github.com/Socrats/EGTTools/issues/new/choose).
- **Propose a method or model**, ideally with the paper that defines it.
- **Improve the documentation and examples**.
- **Add tests**, in particular tests that check results against exact or published values.
- **Implement methods, models or fixes**.
- **Improve the build and CI** for more platforms and architectures.

For anything larger than a small fix, open an issue or a discussion first, so we can agree on the
approach before you invest time in it.

## Development setup

Clone the repository with its submodules (pybind11 and vcpkg):

```bash
git clone --recurse-submodules https://github.com/Socrats/EGTTools.git
cd EGTTools
```

Create an environment with Python ≥ 3.11 and build the C++ extension in place:

```bash
pip install numpy scipy matplotlib seaborn networkx plotly pytest \
    scikit-build "cmake>=3.27,<4" ninja setuptools-scm
python setup.py build_ext --inplace
```

The tests use the in-place build automatically. The C++ dependencies come from the bundled vcpkg;
see [Build from source](https://egttools.readthedocs.io/en/latest/installation.html#build-from-source)
for alternatives (e.g. conda) and platform notes.

Notes:

- After changing a C++ file, rebuild with the same command. On macOS (Apple Silicon), re-sign the
  extension after each rebuild: `find src -name "*.so" | xargs codesign --sign - --force`.
- If a build fails after updating your compiler or SDK, delete `_skbuild/` and rebuild.

## Running the tests

```bash
python -m pytest tests -m "not slow"   # fast suite, run by CI on pull requests and on master
python -m pytest tests -m slow         # long-running tests
ctest --test-dir _skbuild/<platform>/cmake-build/tests   # C++ test programs
```

## Writing correct numerical code

Results computed with EGTtools end up in papers, so correctness is the first requirement:

- **Specify what you implement.** Processes and indicators are defined in `docs/theory/`, with
  numbered definitions and references (see `docs/theory/pairwise_comparison.md`). A new method
  should state which definition it implements, or add one.
- **Test against independent results.** Compare with a direct transcription of the equations,
  exact solutions, or published values; check Monte Carlo estimates against exact results within
  a few standard errors and with fixed seeds (`tests/test_pairwise_comparison_equations.py` shows
  the pattern).
- **Document behaviour changes.** If results change, explain why in the CHANGELOG.

## Style

- Python: [PEP 8](https://peps.python.org/pep-0008/), formatted with `black` and `isort`
  (see `.pre-commit-config.yaml`; run `pre-commit install` to apply them on commit).
  Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/) format.
- C++: C++17, formatted with the repository's `.clang-format`.
- Prefer clear names and small functions over comments that explain unclear code.

## Changelog, stubs and documentation

- Add an entry under *Unreleased* in `CHANGELOG.md` for every user-visible change.
- If you change the C++ bindings, rebuild the extension and regenerate the type stubs with the
  pinned generator: `pip install pybind11-stubgen==2.5.1`, then
  `pybind11-stubgen egttools.numerical.numerical_ -o src`. CI fails if the committed stubs differ.
  Keep function signatures enabled in the bindings (no `disable_function_signatures()`): the
  stubs and the docs read them from the docstrings.
- Update the docstrings and, where relevant, the user guide and examples in `docs/`.

## Compatibility

EGTtools is still at version 0.x. Backward compatibility is the default, but when justified (a cleaner design or better performance) there will backward breaks. They however must be marked as breaking in the
CHANGELOG with migration instructions, and must update every example and test that uses the old
behaviour.

## Pull requests

1. Fork the repository and create a branch for your change.
2. Keep the pull request focused on one change, with tests.
3. Make sure the test workflow passes; the pull request template lists the remaining checks.

## Signing off commits

To avoid copyright problems, every commit in a pull request must be signed off by its author.
Use `git commit -s`, which adds a `Signed-off-by` line to the commit message. The sign-off
certifies that you have read and accepted the Developer Certificate of Origin:

> Developer Certificate of Origin
> Version 1.1
>
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
> 1 Letterman Drive
> Suite D4700
> San Francisco, CA, 94129
>
> Everyone is permitted to copy and distribute verbatim copies of this
> license document, but changing it is not allowed.
>
>
> Developer's Certificate of Origin 1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I
>     have the right to submit it under the open source license
>     indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best
>     of my knowledge, is covered under an appropriate open source
>     license and I have the right under that license to submit that
>     work with modifications, whether created in whole or in part
>     by me, under the same open source license (unless I am
>     permitted to submit under a different license), as indicated
>     in the file; or
>
> (c) The contribution was provided directly to me by some other
>     person who certified (a), (b) or (c) and I have not modified
>     it.
>
> (d) I understand and agree that this project and the contribution
>     are public and that a record of the contribution (including all
>     personal information I submit with it, including my sign-off) is
>     maintained indefinitely and may be redistributed consistent with
>     this project or the open source license(s) involved.

This Developer Certificate of Origin (DCO) is also used by the Linux kernel in
order to attest that you have the rights to submit the code in your patches.
The code still remains yours (or of whoever it was originally). This is pretty
much all the extra that is needed.

The code remains yours (or that of whoever wrote it); the sign-off only certifies that you have the
right to contribute it.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to
uphold it.

## Attribution

These guidelines were originally adapted from the
[AI-Toolbox contributing guidelines](https://github.com/Svalorzen/AI-Toolbox/blob/master/CONTRIBUTING.md).
