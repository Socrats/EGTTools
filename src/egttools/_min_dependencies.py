"""All minimum dependencies for egttools."""

# Authors: Copied from scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from collections import defaultdict

# scipy and cython should by in sync with pyproject.toml
NUMPY_MIN_VERSION = "1.26.3"
SCIPY_MIN_VERSION = "1.12.0"
PYTEST_MIN_VERSION = "7.1.2"


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "build, install, docs, tests"),
    "scipy": (SCIPY_MIN_VERSION, "build, install, docs, tests"),
    "matplotlib": ("3.3.4", "build, install, docs, tests"),
    "seaborn": ("0.9.0", "build, install, docs, tests"),
    "networkx": ("3.2.1", "build, install, docs, tests"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "sphinx": ("7.3.7", "docs"),
    "sphinx_rtd_theme": ("0.5.1", "docs"),
    "sphinx-autodoc-typehints": ("1.12.0", "docs"),
    "sphinxcontrib-bibtex": ("2.5.0", "docs"),
    "GitPython": ("3.1.31", "docs"),
    "PyGithub": ("1.58.0", "docs"),
    "requests": ("2.29.0", "docs"),
    "numpydoc": ("1.2.0", "docs"),
    "docutils": ("0.19", "docs"),
    "prompt-toolkit": ("3.0.38", "docs"),
    "nbsphinx": ("0.8.7", "docs"),
    "recommonmark": ("0.7.1", "docs"),
    "ipykernel": ("6.22.0", "docs"),
    "ipywidgets": ("8.0.6", "docs"),
    # XXX: Pin conda-lock to the latest released version (needs manual update
    # from time to time)
    "conda-lock": ("2.5.7", "maintenance"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)