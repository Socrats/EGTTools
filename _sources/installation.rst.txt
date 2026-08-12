Installation
============

EGTtools is distributed via **PyPI** (pip wheels) and **conda** (Anaconda.org).
Prebuilt packages are available for Linux (x86_64), macOS (x86_64 and arm64),
and Windows (x86_64 and arm64), for **Python 3.10 – 3.13**.

.. list-table::
   :header-rows: 1
   :widths: 25 30 20 15

   * - Platform
     - Architectures
     - Python versions
     - OpenMP
   * - Linux
     - x86_64
     - 3.10 – 3.13
     - ✅
   * - macOS
     - x86_64, arm64 (M1–M4)
     - 3.10 – 3.13
     - ✅
   * - Windows
     - x86_64, arm64
     - 3.10 – 3.13
     - ❌


Install with conda or mamba (recommended)
------------------------------------------

The conda package bundles all native dependencies (BLAS/LAPACK, OpenMP) and
works out of the box on Linux and macOS.  It is hosted on the maintainer's
personal `Anaconda.org channel <https://anaconda.org/socrats/egttools>`_.

**miniforge**, **mambaforge**, and **miniconda** are all free for any use,
including large organisations.

.. code-block:: bash

    conda install -c socrats egttools

Or with **mamba** (faster dependency solver):

.. code-block:: bash

    mamba install -c socrats egttools

To avoid typing ``-c socrats`` every time, add the channel permanently to your
conda configuration:

.. code-block:: bash

    conda config --add channels socrats
    conda config --set channel_priority strict

After that, ``conda install egttools`` works without a channel flag.

.. note::

    EGTtools is **not** in the ``conda-forge`` or ``defaults`` channels — it is
    maintained on the author's personal Anaconda.org channel so the author
    retains full ownership of the recipe and release schedule.


Install with pip
----------------

.. code-block:: bash

    pip install egttools

To upgrade to the latest release:

.. code-block:: bash

    pip install -U egttools

.. note::

    On **macOS** inside a conda environment, installing via pip can trigger
    ABI mismatches between pip-provided and conda-provided NumPy/SciPy.
    Using the conda package above avoids this entirely.
    If you must use pip, install the heavy dependencies via conda first:

    .. code-block:: bash

        conda install numpy scipy matplotlib networkx seaborn plotly
        pip install egttools --no-deps


Build from source
-----------------

Requirements
^^^^^^^^^^^^

* Linux, macOS (Monterey or later), or Windows
* `CMake <https://cmake.org>`_ ≥ 3.27
* A C++17-capable compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019+)
* `Eigen <https://eigen.tuxfamily.org>`_ ≥ 3.4 (or 5.x)
* `Boost <https://www.boost.org>`_ ≥ 1.82 (header-only multiprecision)
* Python ≥ 3.10

The easiest way to satisfy the C++ dependencies is via **vcpkg** (bundled as a
git submodule) or via conda.

With vcpkg (default for local development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone --recurse-submodules https://github.com/Socrats/EGTTools.git
    cd EGTTools
    pip install .

With conda dependencies (``SKIP_VCPKG=ON``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Create and activate an environment with all build and runtime deps
    conda create -n egtenv python=3.11 numpy scipy matplotlib networkx \
        seaborn plotly eigen boost-cpp libblas liblapack pybind11 \
        scikit-build cmake ninja
    conda activate egtenv

    git clone --recurse-submodules https://github.com/Socrats/EGTTools.git
    cd EGTTools
    SKIP_VCPKG=ON pip install . --no-build-isolation

Development mode
^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -e .  # editable install; rebuilds the extension on demand


Python distributions
--------------------

conda / mamba
    Use ``conda install -c socrats egttools`` as described above.
    For Apple Silicon (arm64), the conda package is the recommended installation
    method since it links against native arm64 BLAS/LAPACK and libomp.

PyPy
    Recent versions of PyPy are supported by the
    `pybind11 project <https://github.com/pybind/pybind11>`_ and should thus
    also be supported by EGTtools.

Other
    For any other Python distribution, ``pip install egttools`` should work.
    Please open a `GitHub issue <https://github.com/Socrats/EGTtools/issues>`_
    if you encounter problems.


Troubleshooting
---------------

If you run into problems, please create a
`GitHub issue <https://github.com/Socrats/EGTtools/issues>`_ or write to
`elias.fernandez.domingos@ulb.be <mailto:elias.fernandez.domingos@ulb.be>`_.

Outdated pip
^^^^^^^^^^^^

If installation hangs or errors immediately, update pip first:

.. code-block:: bash

    pip install --upgrade pip

ImportError after install
^^^^^^^^^^^^^^^^^^^^^^^^^

If ``import egttools`` fails with an ``ImportError``, the most common cause on
macOS is a Python/NumPy ABI mismatch between pip and conda packages.
Use the conda package or the ``--no-deps`` + conda-deps approach above.
