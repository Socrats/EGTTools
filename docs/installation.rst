Installation
============

From PyPi
------

EGTtools can be installed using PyPi on Linux, macOS, and Windows::

    pip install egttools

To update your installed version to the latest release, add ``-U`` (or ``--upgrade``) to the command::

    pip install -U egttools

.. note::

    Currently, only the Linux build supports OpenMP parallelization for numerical simulations. This should normally be
    ok for most applications, since numerical simulations are heavy (computationally) and should be run on a
    High Power Computing (HPC) clusters
    which normally run Linux distributions. We are investigating how to provide support for OpenMP in both Windows
    and Mac OSX. In the meantime, if you really want to run numerical simulations on either of these two platforms,
    you should follow the compilation instructions below and try to link OpenMP for your platform yourself.
    Please, if you manage to do so, open an issue or a pull request with your solutions.

.. warning::

    The arm64 and universal2::arm64 have not been tested upstream on CI, so please report any issues or bugs you
    may encounter.

.. warning::

    For Apple M1 (arm64) you should install using ``pip install egttools --no-deps`` so that pip does not
    install the dependencies of the package. This is necessary since there is no Scipy wheel for architecture arm64
    available on PyPi yet.
    To install the package dependencies you should create a virtual environment
    with `miniforge <https://github.com/conda-forge/miniforge>`_. Once you have miniforge installed you can do the
    following (assuming that you are in the base miniforge environment)::

        conda create -n egtenv python=3.9
        conda activate egtenv
        conda install numpy
        conda install scipy
        conda install matplotlib
        conda install networkx

Build from source
-----------------

To build `egttools` from source you need:

* A recent version of Linux (only tested on Ubuntu), MacOSX (Mojave or above) or Windows
* `CMake <https://cmake.org>` version 3.17 or higher
* `C++ 17 <https://en.cppreference.com/w/cpp/17>`
* `Eigen <https://eigen.tuxfamily.org/index.php?title=Main_Page>` 3.3.*
* `Boost <https://www.boost.org/>` 1.80.*
* **Python** 3.7 or higher

.. warning::

    **Boost** is required in order for EGTtools to use multiprecision integers and
    floating point numbers with higher precision. You may still be able to compile EGTtools without Boost,
    but we highly recommend don't.


Once you install these libraries, you can follow the following steps.

To **install all required packages** run::

    python -m venv egttools-env
    source egttools-env/bin/activate
    pip install -r requirements.txt

Or with anaconda::

    conda env create -f environment.yml
    conda activate egttools-env

Also, to make your virtual environment visible to jupyter::

    conda install ipykernel # or pip install ipykernel
    python -m ipykernel install --user --name=egttools-env

You can **build EGTtools** by running::

    pip install build
    cd <path>
    python -m build

Where ``<path>`` represents the path to the EGTtools folder. If you are running this while inside the EGTtools folder,
then ``<path>`` is simply ``./``.

Finally, you can install EGTtools in **development** mode, this will allow the installation to update with new
modifications to the package::

    python -m pip install -e <path>


If you don't want development mode, you can skip the option ```-e```.

Python distributions
--------------------

Anaconda
    If you use the Anaconda distribution of Python, you can use the same ``pip`` command in a terminal of the appropriate Anaconda environment, either activated through the `Anaconda Navigator <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_ or `conda tool <https://conda.io/activation>`_.

PyPy
    Recent versions of PyPy are supported by the `pybind11 project <https://github.com/pybind/pybind11>`_ and should thus also be supported by EGTtools.

Other
    For other distributions of Python, we are expecting that our package is compatible with the Python versions that are out there and that ``pip`` can handle the installation. If you are using yet another Python distribution, we are definitely interested in hearing about it, so that we can add it to this list!



Troubleshooting
---------------

It is possible that you run into problems when trying to install or use EGTtools. This may happen because
you are running on a different platform or configuration than what we have listed, or simply because we have
not considered your particular scenario/environment.

If this is the case, and you do run into problems,
please create a `GitHub issue <https://github.com/Socrats/EGTtools/issues>`_,
or write `me <mailto:elias.fernandez.domingos@ulb.be>`_ a quick email.
We would be very happy to solve these problems, so that future users can avoid them and we can expand the use of our
library.


Pip version
^^^^^^^^^^^

If the standard way to install EGTtools results in an error or takes a long time,
try updating ``pip`` to the latest version by running ::

    pip install --upgrade pip

If you do not have ``pip`` installed, you can follow these instructions to
install pip: https://pip.pypa.io/en/stable/installing/